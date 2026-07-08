import os
import gc
import time
import sys

ROOT_PATH = "/sdcard/mp_deployment_source/"
try:
    if ROOT_PATH not in sys.path:
        sys.path.insert(0, ROOT_PATH)
except Exception:
    pass
try:
    for _mod in ("libs.PipeLine", "libs.PlatTasks", "libs.AIBase", "libs.AI2D", "libs.Utils"):
        if _mod in sys.modules:
            del sys.modules[_mod]
except Exception:
    pass

from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *


# =========================
# Match-day configuration
# =========================

# Put this file, deploy_config.json and the kmodel under this directory on K230.

# lcd is the usual LCKFB/JLC K230 screen. Use "hdmi" or "lt9611" for HDMI.
DISPLAY_MODE = "lcd"
DISPLAY_SIZE = None

# 1280x720 gives the detector more pixels before resize. For more FPS, try [640, 360].
RGB888P_SIZE = [1280, 720]
CAMERA_FPS = 30
SENSOR_ID = None
HMIRROR = None
VFLIP = None

# 0 means auto-pick the best visible drug room. Send T1..T8 by UART to select.
DEFAULT_TARGET_ID = 0

# Selection and lock tuning.
SELECT_MIN_SCORE = 0.45
MIN_BOX_AREA_RATIO = 0.00020
HISTORY_SIZE = 7
LOCK_MIN_HITS = 4
LOST_RESET_FRAMES = 5
SMOOTH_ALPHA = 0.62

# Delivery decision. The car should stop when state == 3.
CENTER_TOLERANCE_RATIO = 0.045
ARRIVE_BOX_HEIGHT_RATIO = 0.35

# Optional rough distance estimate: distance_mm ~= DISTANCE_K_BY_HEIGHT / box_h_px.
# Calibrate it on your car. Example: at 1000 mm, box height is 90 px -> K = 90000.
DISTANCE_K_BY_HEIGHT = 90000

# Line tracking. This follows Car_Mode's weighted multi-sensor idea:
# a few fixed ROI "virtual sensors" find the line and report weighted error.
ENABLE_LINE_TRACK = True
LINE_DETECT_EVERY_N_FRAMES = 4
LINE_THRESHOLDS = [(15, 100, 25, 127, -20, 90)]
LINE_MIN_PIXELS = 160
LINE_MIN_AREA = 120
LINE_MIN_DENSITY = 0.12
LINE_CENTER_PENALTY = 0.18
LINE_MEMORY_PENALTY = 0.45
LINE_USE_SPARSE_RGB_FALLBACK = True
LINE_SPARSE_STEP_X = 40
LINE_SPARSE_STEP_Y = 20
LINE_SPARSE_MIN_HITS = 1
LINE_RED_MIN = 80
LINE_RED_DOMINANCE = 0
LINE_ACCEPT_BGR_RED = True
LINE_ROI_BANDS = [
    (0.78, 0.98, 1.00),
    (0.54, 0.72, 0.65),
    (0.28, 0.46, 0.35),
]
LINE_SMOOTH_ALPHA = 0.65
LINE_LOST_RESET_FRAMES = 5
DRAW_LINE_ROIS = True

# UART protocol output. Official examples commonly use UART2 on pins 11/12.
USE_UART = True
UART_ID = 2
UART_BAUDRATE = 115200
UART_TX_PIN = 11
UART_RX_PIN = 12
SEND_EVERY_N_FRAMES = 1
PRINT_PROTOCOL_WHEN_NO_UART = True

# If the model is noisy in your venue, raise to 0.48-0.55.
CONFIDENCE_THRESHOLD_OVERRIDE = None
NMS_THRESHOLD_OVERRIDE = None

# Debug timing prints every frame when set to 1.
DEBUG_MODE = 0
PROFILE_TOTAL = 0


COLOR_NORMAL = (255, 0, 180, 255)
COLOR_TARGET = (255, 0, 255, 0)
COLOR_LOCKED = (255, 255, 40, 40)
COLOR_GUIDE = (160, 255, 255, 255)
COLOR_TEXT = (255, 255, 255, 255)
COLOR_WARN = (255, 255, 220, 0)
COLOR_LINE = (255, 40, 220, 40)
COLOR_LINE_ROI = (120, 255, 255, 0)


def join_path(root, name):
    if root.endswith("/"):
        return root + name
    return root + "/" + name


def file_exists(path):
    try:
        os.stat(path)
        return True
    except Exception:
        return False


def load_deploy_config():
    config_path = join_path(ROOT_PATH, "deploy_config.json")
    if not file_exists(config_path):
        config_path = "deploy_config.json"
    return read_json(config_path), config_path


def resolve_model_path(model_name):
    full_path = join_path(ROOT_PATH, model_name)
    if file_exists(full_path):
        return full_path
    return model_name


def flatten_anchors(deploy_conf, model_type):
    anchors = []
    if model_type == "AnchorBaseDet":
        raw = deploy_conf.get("anchors", [])
        for group in raw:
            for item in group:
                anchors.append(item)
    return anchors


def label_to_id(label):
    try:
        return int(label)
    except Exception:
        return -1


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def clone_candidate(src):
    dst = {}
    for key in src:
        dst[key] = src[key]
    return dst


class TargetStabilizer:
    def __init__(self):
        self.history = []
        self.smooth = None
        self.lost_count = 0

    def reset(self):
        self.history = []
        self.smooth = None
        self.lost_count = 0

    def update(self, candidate):
        if candidate is None:
            self.lost_count += 1
            if self.lost_count >= LOST_RESET_FRAMES:
                self.reset()
            return None, 0, 0

        self.lost_count = 0
        drug_id = candidate["drug_id"]
        self.history.append(drug_id)
        while len(self.history) > HISTORY_SIZE:
            self.history.pop(0)

        hits = 0
        for item in self.history:
            if item == drug_id:
                hits += 1
        locked = 1 if hits >= LOCK_MIN_HITS else 0

        if self.smooth is None or self.smooth["drug_id"] != drug_id:
            self.smooth = clone_candidate(candidate)
        else:
            self._smooth_candidate(candidate)

        return self.smooth, locked, hits

    def _smooth_candidate(self, candidate):
        a = SMOOTH_ALPHA
        b = 1.0 - SMOOTH_ALPHA
        for key in ("x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "area", "err_x", "err_y"):
            self.smooth[key] = int(self.smooth[key] * b + candidate[key] * a)
        for key in ("score", "rank"):
            self.smooth[key] = self.smooth[key] * b + candidate[key] * a
        self.smooth["label"] = candidate["label"]
        self.smooth["drug_id"] = candidate["drug_id"]


class FpsMeter:
    def __init__(self):
        self.last_ms = self._ticks_ms()
        self.count = 0
        self.fps = 0

    def _ticks_ms(self):
        try:
            return time.ticks_ms()
        except Exception:
            return int(time.time() * 1000)

    def _diff_ms(self, now, old):
        try:
            return time.ticks_diff(now, old)
        except Exception:
            return now - old

    def update(self):
        self.count += 1
        now = self._ticks_ms()
        diff = self._diff_ms(now, self.last_ms)
        if diff >= 1000:
            self.fps = int(self.count * 1000 / diff)
            self.count = 0
            self.last_ms = now
        return self.fps


class LineTracker:
    def __init__(self, frame_size):
        self.frame_w = frame_size[0]
        self.frame_h = frame_size[1]
        self.rois = self._build_rois()
        self.lost_count = 0
        self.enabled = True
        self.error_reported = False
        self.last_offset = 0.0
        self.last_angle = 0.0
        self.debug_blob_hits = 0
        self.debug_red_hits = 0
        self.debug_sample_count = 0
        self.last = self._empty_result()

    def _empty_result(self):
        return {
            "seen": 0,
            "err_x": 0,
            "angle": 0,
            "cx": self.frame_w // 2,
            "cy": 0,
            "width": 0,
            "area": 0,
            "points": [],
            "rois": self.rois,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "blob_hits": self.debug_blob_hits,
            "red_hits": self.debug_red_hits,
            "sample_count": self.debug_sample_count,
            "src": "?",
        }

    def _ensure_frame_size(self, img_obj):
        size = self._infer_frame_size(img_obj)
        if size is None:
            return
        w, h = size
        if w <= 0 or h <= 0:
            return
        if w == self.frame_w and h == self.frame_h:
            return
        self.frame_w = w
        self.frame_h = h
        self.rois = self._build_rois()
        print("Line frame size:", self.frame_w, self.frame_h)

    def _infer_frame_size(self, img_obj):
        try:
            shape = img_obj.shape
            if len(shape) == 4 and shape[0] == 1 and shape[1] == 3:
                return int(shape[3]), int(shape[2])
            if len(shape) == 4 and shape[0] == 1 and shape[3] == 3:
                return int(shape[2]), int(shape[1])
            if len(shape) == 3 and shape[0] == 3:
                return int(shape[2]), int(shape[1])
            if len(shape) == 3 and shape[2] == 3:
                return int(shape[1]), int(shape[0])
            if len(shape) == 2:
                return int(shape[1]), int(shape[0])
        except Exception:
            pass

        try:
            return int(img_obj.width()), int(img_obj.height())
        except Exception:
            pass

        try:
            return int(img_obj.width), int(img_obj.height)
        except Exception:
            return None

    def _build_rois(self):
        rois = []
        for band in LINE_ROI_BANDS:
            y_ratio, h_ratio, weight = band
            y = int(self.frame_h * y_ratio)
            h = int(self.frame_h * h_ratio)
            if y < 0:
                y = 0
            if y + h > self.frame_h:
                h = self.frame_h - y
            if h <= 0:
                continue
            rois.append({
                "rect": (0, y, self.frame_w, h),
                "weight": weight,
            })
        return rois

    def update(self, img_obj):
        if not ENABLE_LINE_TRACK or not self.enabled or img_obj is None:
            self.last = self._empty_result()
            return self.last

        self._ensure_frame_size(img_obj)
        points = []
        total_weight = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_area = 0
        max_width = 0
        self.debug_blob_hits = 0
        self.debug_red_hits = 0
        self.debug_sample_count = 0

        for roi_def in self.rois:
            rect = roi_def["rect"]
            weight = roi_def["weight"]
            if self._has_find_blobs(img_obj):
                blob = self._find_best_blob(img_obj, rect, self.frame_w * 0.5)
                if blob is not None:
                    self.debug_blob_hits += 1
            else:
                blob = None
            if blob is None and LINE_USE_SPARSE_RGB_FALLBACK:
                blob = self._sparse_red_blob(img_obj, rect)
            if blob is None:
                continue
            x, y, w, h = self._blob_rect(blob)
            cx = x + w // 2
            cy = y + h // 2
            area = self._blob_area(blob, w, h)
            points.append((cx, cy, w, h, area, rect))
            sum_x += cx * weight
            sum_y += cy * weight
            total_weight += weight
            sum_area += area
            if w > max_width:
                max_width = w

        if total_weight <= 0:
            self.lost_count += 1
            if self.lost_count >= LINE_LOST_RESET_FRAMES:
                self.last = self._empty_result()
            else:
                self.last["seen"] = 0
                self.last["blob_hits"] = self.debug_blob_hits
                self.last["red_hits"] = self.debug_red_hits
                self.last["sample_count"] = self.debug_sample_count
            return self.last

        self.lost_count = 0
        cx = int(sum_x / total_weight)
        cy = int(sum_y / total_weight)
        err_x = cx - self.frame_w // 2
        angle = self._estimate_angle(points)
        self.last_offset = float(err_x) / float(max(1, self.frame_w // 2)) * 100.0
        self.last_angle = float(angle)

        result = {
            "seen": 1,
            "err_x": err_x,
            "angle": angle,
            "cx": cx,
            "cy": cy,
            "width": max_width,
            "area": sum_area,
            "points": points,
            "rois": self.rois,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "blob_hits": self.debug_blob_hits,
            "red_hits": self.debug_red_hits,
            "sample_count": self.debug_sample_count,
        }
        self.last = self._smooth(result)
        return self.last

    def _has_find_blobs(self, img_obj):
        try:
            return callable(getattr(img_obj, "find_blobs", None))
        except Exception:
            return False

    def _sparse_red_blob(self, img_obj, rect):
        x0, y0, rw, rh = rect
        x1 = x0 + rw
        y1 = y0 + rh
        hits = 0
        sum_x = 0
        sum_y = 0
        min_x = x1
        min_y = y1
        max_x = x0
        max_y = y0

        for y in range(y0, y1, LINE_SPARSE_STEP_Y):
            for x in range(x0, x1, LINE_SPARSE_STEP_X):
                pixel = self._get_rgb_pixel(img_obj, x, y)
                if pixel is None:
                    return None
                self.debug_sample_count += 1
                r, g, b = pixel
                if self._is_red_pixel(r, g, b):
                    self.debug_red_hits += 1
                    hits += 1
                    sum_x += x
                    sum_y += y
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y

        if hits < LINE_SPARSE_MIN_HITS:
            return None

        cx = int(sum_x / hits)
        cy = int(sum_y / hits)
        w = max(LINE_SPARSE_STEP_X, max_x - min_x + LINE_SPARSE_STEP_X)
        h = max(LINE_SPARSE_STEP_Y, max_y - min_y + LINE_SPARSE_STEP_Y)
        area = hits * LINE_SPARSE_STEP_X * LINE_SPARSE_STEP_Y
        return (cx - w // 2, cy - h // 2, w, h, area)

    def _get_rgb_pixel(self, img_obj, x, y):
        try:
            shape = img_obj.shape
            if len(shape) == 4 and shape[0] == 1 and shape[1] == 3:
                return int(img_obj[0][0][y][x]), int(img_obj[0][1][y][x]), int(img_obj[0][2][y][x])
            if len(shape) == 4 and shape[0] == 1 and shape[3] == 3:
                value = img_obj[0][y][x]
                return int(value[0]), int(value[1]), int(value[2])
            if len(shape) == 3 and shape[0] == 3:
                return int(img_obj[0][y][x]), int(img_obj[1][y][x]), int(img_obj[2][y][x])
            if len(shape) == 3 and shape[2] == 3:
                value = img_obj[y][x]
                return int(value[0]), int(value[1]), int(value[2])
        except Exception:
            pass

        try:
            value = img_obj.get_pixel(x, y)
        except Exception:
            return None

        try:
            if len(value) >= 3:
                return int(value[0]), int(value[1]), int(value[2])
        except Exception:
            pass

        try:
            raw = int(value)
            if raw <= 0xFFFF:
                r = ((raw >> 11) & 0x1F) << 3
                g = ((raw >> 5) & 0x3F) << 2
                b = (raw & 0x1F) << 3
            else:
                r = (raw >> 16) & 0xFF
                g = (raw >> 8) & 0xFF
                b = raw & 0xFF
            return r, g, b
        except Exception:
            return None

    def _is_red_pixel(self, r, g, b):
        if self._is_red_order(r, g, b):
            return True
        if LINE_ACCEPT_BGR_RED and self._is_red_order(b, g, r):
            return True
        return False

    def _is_red_order(self, r, g, b):
        if r < LINE_RED_MIN:
            return False
        if r < g + LINE_RED_DOMINANCE:
            return False
        if r < b + LINE_RED_DOMINANCE:
            return False
        return True

    def _blob_density(self, blob):
        pixels = self._blob_area(blob, max(1, self._blob_rect(blob)[2]), max(1, self._blob_rect(blob)[3]))
        try:
            density = blob.density()
            return float(density)
        except Exception:
            pass
        _x, _y, w, h = self._blob_rect(blob)
        return float(pixels) / float(max(1, w * h))

    def _find_best_blob(self, img_obj, rect, frame_center_x):
        try:
            blobs = img_obj.find_blobs(
                LINE_THRESHOLDS,
                roi=rect,
                pixels_threshold=LINE_MIN_PIXELS,
                area_threshold=LINE_MIN_AREA,
                merge=True,
            )
        except TypeError:
            try:
                blobs = img_obj.find_blobs(LINE_THRESHOLDS, roi=rect, pixels_threshold=LINE_MIN_PIXELS)
            except Exception as e:
                self._report_find_blobs_error(e)
                return None
        except Exception as e:
            self._report_find_blobs_error(e)
            return None

        best = None
        best_score = -1
        x0, _y0, rw, _rh = rect
        roi_center_x = x0 + rw * 0.5
        predicted_x = frame_center_x + self.last_offset / 100.0 * frame_center_x
        predicted_x = clamp(predicted_x, x0, x0 + rw)
        for blob in blobs:
            x, y, w, h = self._blob_rect(blob)
            area = self._blob_area(blob, w, h)
            density = self._blob_density(blob)
            if density < LINE_MIN_DENSITY:
                continue
            cx = x + w // 2
            center_penalty = abs(cx - roi_center_x) * LINE_CENTER_PENALTY
            memory_penalty = abs(cx - predicted_x) * LINE_MEMORY_PENALTY
            score = area + w * 5 + h * 2 + density * 120 - center_penalty - memory_penalty
            if score > best_score:
                best_score = score
                best = blob
        return best

    def _report_find_blobs_error(self, error):
        if not self.error_reported:
            print("line find_blobs unavailable, using sparse RGB fallback:", error)
            self.error_reported = True

    def _blob_rect(self, blob):
        try:
            if len(blob) >= 5:
                return int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3])
        except Exception:
            pass
        try:
            return int(blob.x()), int(blob.y()), int(blob.w()), int(blob.h())
        except Exception:
            return int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3])

    def _blob_area(self, blob, w, h):
        try:
            if len(blob) >= 5:
                return int(blob[4])
        except Exception:
            pass
        try:
            return int(blob.pixels())
        except Exception:
            return int(w * h)

    def _estimate_angle(self, points):
        if len(points) < 2:
            return 0
        top = points[0]
        bottom = points[-1]
        dx = bottom[0] - top[0]
        dy = bottom[1] - top[1]
        if dy == 0:
            return 0
        # Small-angle approximation is enough for steering and avoids extra math deps.
        angle = int(dx * 57 / dy)
        return clamp(angle, -60, 60)

    def _smooth(self, result):
        if self.last is None or not self.last["seen"]:
            return result
        a = LINE_SMOOTH_ALPHA
        b = 1.0 - LINE_SMOOTH_ALPHA
        result["err_x"] = int(self.last["err_x"] * b + result["err_x"] * a)
        result["angle"] = int(self.last["angle"] * b + result["angle"] * a)
        result["cx"] = int(self.last["cx"] * b + result["cx"] * a)
        result["cy"] = int(self.last["cy"] * b + result["cy"] * a)
        result["width"] = int(self.last["width"] * b + result["width"] * a)
        result["area"] = int(self.last["area"] * b + result["area"] * a)
        return result


class UartBridge:
    def __init__(self):
        self.uart = None
        self.rx_buffer = ""
        self.enabled = False
        if USE_UART:
            self._init_uart()

    def _init_uart(self):
        try:
            from machine import UART
            try:
                from machine import FPIOA
                if UART_TX_PIN is not None or UART_RX_PIN is not None:
                    fpioa = FPIOA()
                    if UART_TX_PIN is not None:
                        tx_func = getattr(fpioa, "UART%d_TXD" % UART_ID)
                        fpioa.set_function(UART_TX_PIN, tx_func)
                    if UART_RX_PIN is not None:
                        rx_func = getattr(fpioa, "UART%d_RXD" % UART_ID)
                        fpioa.set_function(UART_RX_PIN, rx_func)
            except Exception as e:
                print("FPIOA setup skipped:", e)

            uart_const = getattr(UART, "UART%d" % UART_ID, UART_ID)
            try:
                self.uart = UART(
                    uart_const,
                    baudrate=UART_BAUDRATE,
                    bits=UART.EIGHTBITS,
                    parity=UART.PARITY_NONE,
                    stop=UART.STOPBITS_ONE,
                )
            except TypeError:
                self.uart = UART(uart_const, UART_BAUDRATE)
            self.enabled = True
            print("UART ready: id=%d baud=%d" % (UART_ID, UART_BAUDRATE))
        except Exception as e:
            self.uart = None
            self.enabled = False
            print("UART disabled:", e)

    def write_packet(self, payload):
        line = self._wrap_checksum(payload)
        if self.uart:
            try:
                self.uart.write(line)
                return
            except Exception as e:
                print("UART write failed:", e)
        if PRINT_PROTOCOL_WHEN_NO_UART:
            print(line.strip())

    def read_commands(self):
        if not self.uart:
            return []
        try:
            n = self.uart.any()
            if not n:
                return []
            data = self.uart.read()
        except Exception:
            return []

        text = self._bytes_to_text(data)
        if not text:
            return []
        self.rx_buffer += text

        lines = []
        while "\n" in self.rx_buffer:
            idx = self.rx_buffer.find("\n")
            line = self.rx_buffer[:idx].strip()
            self.rx_buffer = self.rx_buffer[idx + 1:]
            if line:
                lines.append(line)

        if len(self.rx_buffer) > 64:
            self.rx_buffer = self.rx_buffer[-64:]
        return lines

    def _bytes_to_text(self, data):
        if data is None:
            return ""
        try:
            return data.decode()
        except Exception:
            out = ""
            try:
                for item in data:
                    out += chr(item)
            except Exception:
                out = str(data)
            return out

    def _wrap_checksum(self, payload):
        checksum = 0
        for ch in payload:
            checksum ^= ord(ch)
        return "$%s*%02X\r\n" % (payload, checksum & 0xFF)


def parse_command(line, current_target):
    raw = line.strip()
    if raw.startswith("$"):
        star = raw.find("*")
        if star >= 0:
            raw = raw[1:star]
        else:
            raw = raw[1:]
    cmd = raw.strip().upper()
    cmd = cmd.replace(" ", "")

    if cmd in ("A", "AUTO", "T0", "TARGET0", "TARGET,0"):
        return 0, 1

    if cmd in ("R", "RESET", "C", "CLEAR"):
        return current_target, 2

    if cmd.startswith("TARGET,"):
        num = cmd[7:]
    elif cmd.startswith("TARGET"):
        num = cmd[6:]
    elif cmd.startswith("T,"):
        num = cmd[2:]
    elif cmd.startswith("T"):
        num = cmd[1:]
    else:
        return current_target, 0

    try:
        target = int(num)
        if 0 <= target <= 8:
            return target, 1
    except Exception:
        pass
    return current_target, 0


def result_to_candidates(res, labels):
    candidates = []
    if res is None:
        return candidates

    boxes = res.get("boxes", [])
    scores = res.get("scores", [])
    idxs = res.get("idx", [])
    try:
        count = len(boxes)
    except Exception:
        count = 0

    min_area = int(RGB888P_SIZE[0] * RGB888P_SIZE[1] * MIN_BOX_AREA_RATIO)
    center_x = RGB888P_SIZE[0] // 2
    center_y = RGB888P_SIZE[1] // 2

    for i in range(count):
        try:
            cls_idx = int(idxs[i])
            score = float(scores[i])
            box = boxes[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
        except Exception:
            continue

        if cls_idx < 0 or cls_idx >= len(labels):
            continue
        if score < SELECT_MIN_SCORE:
            continue

        x1 = clamp(x1, 0, RGB888P_SIZE[0] - 1)
        y1 = clamp(y1, 0, RGB888P_SIZE[1] - 1)
        x2 = clamp(x2, 0, RGB888P_SIZE[0] - 1)
        y2 = clamp(y2, 0, RGB888P_SIZE[1] - 1)
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue

        area = w * h
        if area < min_area:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        label = labels[cls_idx]
        drug_id = label_to_id(label)
        err_x = cx - center_x
        err_y = cy - center_y

        norm_area = float(area) / float(RGB888P_SIZE[0] * RGB888P_SIZE[1])
        norm_center = abs(float(err_x)) / float(center_x)
        rank = score * 1000.0 + min(norm_area * 900.0, 260.0) - norm_center * 140.0

        candidates.append({
            "label": label,
            "drug_id": drug_id,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "area": area,
            "cx": cx,
            "cy": cy,
            "err_x": err_x,
            "err_y": err_y,
            "rank": rank,
        })
    return candidates


def pick_candidate(candidates, target_id):
    best = None
    best_rank = -1000000.0
    for cand in candidates:
        if target_id > 0 and cand["drug_id"] != target_id:
            continue
        rank = cand["rank"]
        if target_id > 0:
            rank += 420.0
        if rank > best_rank:
            best_rank = rank
            best = cand
    return best


def get_pipeline_line_frame(pl):
    getter = getattr(pl, "get_line_frame", None)
    if getter is None:
        try:
            pl._line_debug_status = "no-method"
        except Exception:
            pass
        if not getattr(pl, "_line_getter_missing_reported", False):
            print("PipeLine get_line_frame missing; project libs/PipeLine.py was not loaded")
            try:
                pl._line_getter_missing_reported = True
            except Exception:
                pass
        return None
    frame = getter()
    try:
        pl._line_debug_status = getattr(pl, "line_status", "unknown")
    except Exception:
        pass
    if frame is None:
        count = getattr(pl, "_line_none_count", 0) + 1
        try:
            pl._line_none_count = count
        except Exception:
            pass
        if count == 1 or count % 30 == 0:
            print("PipeLine line frame is None, status:", getattr(pl, "_line_debug_status", "?"))
    elif not getattr(pl, "_line_frame_reported", False):
        try:
            print("PipeLine line frame:", frame.width(), frame.height(), frame)
        except Exception:
            print("PipeLine line frame:", frame)
        try:
            pl._line_frame_reported = True
        except Exception:
            pass
    return frame


def estimate_distance(candidate):
    if candidate is None or candidate["h"] <= 0 or DISTANCE_K_BY_HEIGHT <= 0:
        return 0
    return int(DISTANCE_K_BY_HEIGHT / candidate["h"])


def build_payload(frame_id, target_id, candidate, locked, hits, fps):
    if candidate is None:
        return "MV,%d,0,%d,0,0,0,0,0,0,0,0,0,0,0,%d" % (frame_id, target_id, fps)

    center_tol = int(RGB888P_SIZE[0] * CENTER_TOLERANCE_RATIO)
    arrive_h = int(RGB888P_SIZE[1] * ARRIVE_BOX_HEIGHT_RATIO)
    arrived = locked and abs(candidate["err_x"]) <= center_tol and candidate["h"] >= arrive_h
    state = 3 if arrived else (2 if locked else 1)
    score1000 = int(candidate["score"] * 1000)
    distance_mm = estimate_distance(candidate)

    return "MV,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (
        frame_id,
        state,
        target_id,
        candidate["drug_id"],
        score1000,
        candidate["cx"],
        candidate["cy"],
        candidate["w"],
        candidate["h"],
        candidate["err_x"],
        candidate["err_y"],
        candidate["area"],
        distance_mm,
        hits,
        fps,
    )


def build_line_payload(frame_id, line_result, fps):
    if line_result is None or not line_result["seen"]:
        return "LN,%d,0,0,0,0,0,0,%d" % (frame_id, fps)
    return "LN,%d,1,%d,%d,%d,%d,%d,%d" % (
        frame_id,
        line_result["err_x"],
        line_result["angle"],
        line_result["cx"],
        line_result["width"],
        line_result["area"],
        fps,
    )


def source_size_from_line(line_result):
    if line_result is not None and "frame_w" in line_result and "frame_h" in line_result:
        return line_result["frame_w"], line_result["frame_h"]
    return RGB888P_SIZE[0], RGB888P_SIZE[1]


def scaled_rect(candidate, display_size):
    dw = display_size[0]
    dh = display_size[1]
    rw = RGB888P_SIZE[0]
    rh = RGB888P_SIZE[1]
    x = int(candidate["x1"] * dw // rw)
    y = int(candidate["y1"] * dh // rh)
    w = int(candidate["w"] * dw // rw)
    h = int(candidate["h"] * dh // rh)
    return x, y, w, h


def scale_xy(x, y, display_size, source_size=None):
    if source_size is None:
        source_size = RGB888P_SIZE
    return int(x * display_size[0] // source_size[0]), int(y * display_size[1] // source_size[1])


def scale_rect_xywh(rect, display_size, source_size=None):
    x, y, w, h = rect
    if source_size is None:
        source_size = RGB888P_SIZE
    sx, sy = scale_xy(x, y, display_size, source_size)
    sw = int(w * display_size[0] // source_size[0])
    sh = int(h * display_size[1] // source_size[1])
    return sx, sy, sw, sh


def draw_candidate(draw_img, candidate, display_size, selected_id, locked):
    x, y, w, h = scaled_rect(candidate, display_size)
    is_selected = selected_id == candidate["drug_id"]
    color = COLOR_TARGET if is_selected else COLOR_NORMAL
    thickness = 4 if is_selected else 2
    if locked and is_selected:
        color = COLOR_LOCKED
        thickness = 5

    draw_img.draw_rectangle(x, y, w, h, color=color, thickness=thickness)
    score100 = int(candidate["score"] * 100)
    text = "%s %d%%" % (candidate["label"], score100)
    ty = y - 28
    if ty < 0:
        ty = y + 2
    draw_img.draw_string_advanced(x, ty, 22, text, color=color)


def draw_line_overlay(draw_img, line_result, display_size):
    if line_result is None:
        return
    source_size = source_size_from_line(line_result)

    if DRAW_LINE_ROIS:
        for roi_def in line_result["rois"]:
            x, y, w, h = scale_rect_xywh(roi_def["rect"], display_size, source_size)
            draw_img.draw_rectangle(x, y, w, h, color=COLOR_LINE_ROI, thickness=1)

    if not line_result["seen"]:
        text = "LINE LOST src:%s b:%d r:%d/%d" % (
            line_result.get("src", "?"),
            line_result.get("blob_hits", 0),
            line_result.get("red_hits", 0),
            line_result.get("sample_count", 0),
        )
        draw_img.draw_string_advanced(8, display_size[1] - 30, 22, text, color=COLOR_WARN)
        return

    lx, ly = scale_xy(line_result["cx"], line_result["cy"], display_size, source_size)
    draw_img.draw_circle(lx, ly, 7, color=COLOR_LINE, fill=True)
    draw_img.draw_line(display_size[0] // 2, display_size[1], lx, ly, color=COLOR_LINE, thickness=3)

    for point in line_result["points"]:
        px, py = scale_xy(point[0], point[1], display_size, source_size)
        draw_img.draw_circle(px, py, 5, color=COLOR_LINE, fill=True)

    text = "LINE ex:%d ang:%d b:%d r:%d/%d" % (
        line_result["err_x"],
        line_result["angle"],
        line_result.get("blob_hits", 0),
        line_result.get("red_hits", 0),
        line_result.get("sample_count", 0),
    )
    draw_img.draw_string_advanced(8, display_size[1] - 30, 22, text, color=COLOR_LINE)


def draw_overlay(draw_img, candidates, selected, target_id, locked, hits, fps, display_size, line_result):
    draw_img.clear()
    dw = display_size[0]
    dh = display_size[1]
    cx = dw // 2
    center_tol = int(dw * CENTER_TOLERANCE_RATIO)

    draw_img.draw_line(cx, 0, cx, dh, color=COLOR_GUIDE, thickness=1)
    draw_img.draw_line(cx - center_tol, dh - 70, cx - center_tol, dh, color=COLOR_GUIDE, thickness=2)
    draw_img.draw_line(cx + center_tol, dh - 70, cx + center_tol, dh, color=COLOR_GUIDE, thickness=2)
    draw_line_overlay(draw_img, line_result, display_size)

    selected_id = -1
    if selected is not None:
        selected_id = selected["drug_id"]

    for cand in candidates:
        draw_candidate(draw_img, cand, display_size, selected_id, locked)

    if selected is None:
        state_text = "T:%d SEARCH fps:%d" % (target_id, fps)
        draw_img.draw_string_advanced(8, 6, 24, state_text, color=COLOR_WARN)
        return

    center_ok = "OK" if abs(selected["err_x"]) <= int(RGB888P_SIZE[0] * CENTER_TOLERANCE_RATIO) else "ALIGN"
    lock_text = "LOCK" if locked else "SEEN"
    target_text = "AUTO" if target_id == 0 else str(target_id)
    state_text = "T:%s %s id:%d s:%d ex:%d %s fps:%d" % (
        target_text,
        lock_text,
        selected["drug_id"],
        int(selected["score"] * 100),
        selected["err_x"],
        center_ok,
        fps,
    )
    draw_img.draw_string_advanced(8, 6, 24, state_text, color=COLOR_TEXT)

    if locked:
        draw_img.draw_string_advanced(8, 36, 22, "stable:%d/%d dist:%dmm" % (
            hits,
            HISTORY_SIZE,
            estimate_distance(selected),
        ), color=COLOR_LOCKED)


def main():
    deploy_conf, config_path = load_deploy_config()
    labels = deploy_conf["categories"]
    model_type = deploy_conf["model_type"]
    anchors = flatten_anchors(deploy_conf, model_type)
    kmodel_path = resolve_model_path(deploy_conf["kmodel_path"])
    model_input_size = deploy_conf["img_size"]

    confidence_threshold = deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    if CONFIDENCE_THRESHOLD_OVERRIDE is not None:
        confidence_threshold = CONFIDENCE_THRESHOLD_OVERRIDE
    if NMS_THRESHOLD_OVERRIDE is not None:
        nms_threshold = NMS_THRESHOLD_OVERRIDE

    nms_option = deploy_conf.get("nms_option", False)

    print("Loaded config:", config_path)
    print("Model:", kmodel_path)
    print("Labels:", labels)
    print("Model type:", model_type)
    print("Line mode:", "disabled" if not ENABLE_LINE_TRACK else "shared RGBP888 sparse red tracking")

    bridge = UartBridge()
    stabilizer = TargetStabilizer()
    line_tracker = LineTracker(RGB888P_SIZE)
    fps_meter = FpsMeter()
    runtime_target = DEFAULT_TARGET_ID
    frame_id = 0

    pl = None
    det_app = None
    try:
        pl = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_mode=DISPLAY_MODE,
            display_size=DISPLAY_SIZE,
        )
        pl.create(sensor_id=SENSOR_ID, hmirror=HMIRROR, vflip=VFLIP, fps=CAMERA_FPS)
        display_size = pl.get_display_size()

        det_app = DetectionApp(
            "video",
            kmodel_path,
            labels,
            model_input_size,
            anchors,
            model_type,
            confidence_threshold,
            nms_threshold,
            RGB888P_SIZE,
            display_size,
            debug_mode=DEBUG_MODE,
        )
        det_app.nms_option = nms_option
        det_app.config_preprocess()

        while True:
            with ScopedTiming("total", PROFILE_TOTAL):
                for line in bridge.read_commands():
                    new_target, accepted = parse_command(line, runtime_target)
                    if accepted:
                        runtime_target = new_target
                        stabilizer.reset()
                        if accepted == 2:
                            print("Vision lock reset")
                        else:
                            print("Target set:", runtime_target)

                frame_id += 1
                fps = fps_meter.update()

                img = pl.get_frame()
                line_result = None
                if ENABLE_LINE_TRACK and frame_id % LINE_DETECT_EVERY_N_FRAMES == 0:
                    line_result = line_tracker.update(img)
                    line_result["src"] = "ai"
                else:
                    line_result = line_tracker.last
                res = det_app.run(img)

                candidates = result_to_candidates(res, labels)
                picked = pick_candidate(candidates, runtime_target)
                selected, locked, hits = stabilizer.update(picked)

                if frame_id % SEND_EVERY_N_FRAMES == 0:
                    payload = build_payload(frame_id, runtime_target, selected, locked, hits, fps)
                    bridge.write_packet(payload)
                    line_payload = build_line_payload(frame_id, line_result, fps)
                    bridge.write_packet(line_payload)

                draw_overlay(pl.osd_img, candidates, selected, runtime_target, locked, hits, fps, display_size, line_result)
                pl.show_image()
                gc.collect()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            if det_app is not None:
                det_app.deinit()
        except Exception as e:
            print("det_app deinit failed:", e)
        try:
            if pl is not None:
                pl.destroy()
        except Exception as e:
            print("pipeline destroy failed:", e)


main()
