import gc
import math
import os
import time

try:
    import ujson as json
except Exception:
    import json

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager


# =========================
# Combined snapshot pipeline
# =========================

ROOT_PATH = "/sdcard/mp_deployment_source/"

# Keep this equal to the line-debug program that already worked.
CAM_WIDTH = 480
CAM_HEIGHT = 272

# Use "virt" for CanMV IDE preview. Use "lcd" on the real ST7701 screen.
DISPLAY_MODE = "virt"
DISPLAY_WIDTH = CAM_WIDTH
DISPLAY_HEIGHT = CAM_HEIGHT
SHOW_UI = True

CAMERA_FPS = 30
HMIRROR = False
VFLIP = False

# Digit detector. It runs every few frames so line tracking stays responsive.
ENABLE_DIGIT_DETECT = True
DIGIT_DETECT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD_OVERRIDE = None
NMS_THRESHOLD_OVERRIDE = None

# Target selection.
DEFAULT_TARGET_ID = 0
SELECT_MIN_SCORE = 0.45
MIN_BOX_AREA_RATIO = 0.00020
HISTORY_SIZE = 7
LOCK_MIN_HITS = 4
LOST_RESET_FRAMES = 5
SMOOTH_ALPHA = 0.62

CENTER_TOLERANCE_RATIO = 0.045
ARRIVE_BOX_HEIGHT_RATIO = 0.35
DISTANCE_K_BY_HEIGHT = 35000

# Line tracking. These values are copied from the working line_debug_main.py.
LINE_THRESHOLD = [(15, 100, 25, 127, -20, 90)]
LINE_ROI_BANDS = [
    (0.78, 0.98, 1.00),
    (0.54, 0.72, 0.65),
    (0.28, 0.46, 0.35),
]
LINE_MIN_PIXELS = 160
LINE_MIN_AREA = 120
LINE_MIN_DENSITY = 0.12
LINE_CENTER_PENALTY = 0.18
LINE_MEMORY_PENALTY = 0.45
OFFSET_FULL_SCALE = 100
OFFSET_FILTER_ALPHA = 0.55
ANGLE_FILTER_ALPHA = 0.55
LOST_HOLD_FRAMES = 6

# UART text protocol, compatible with the original main.py packet style.
USE_UART = True
UART_ID = 2
UART_BAUDRATE = 115200
UART_TX_PIN = 11
UART_RX_PIN = 12
SEND_EVERY_N_FRAMES = 1
PRINT_PROTOCOL_WHEN_NO_UART = True

DEBUG_MODE = 0


WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
GREEN = (0, 220, 0)
RED = (255, 0, 0)
YELLOW = (255, 220, 0)
MAGENTA = (255, 0, 220)
CYAN = (0, 220, 255)


def ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def ticks_diff(a, b):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(a, b)
    return a - b


def check_exitpoint():
    try:
        if hasattr(os, "exitpoint"):
            os.exitpoint()
    except Exception:
        pass


def print_exception(exc):
    try:
        import sys
        sys.print_exception(exc)
    except Exception:
        print(exc)


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
    with open(config_path, "r") as f:
        return json.load(f), config_path


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


def align_up(value, align):
    return (value + align - 1) // align * align


def center_pad_param(src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w * dst_h >= src_h * dst_w:
        padded_h = int(src_w * dst_h / dst_w)
        pad_h = max(0, padded_h - src_h)
        top = pad_h // 2
        bottom = pad_h - top
        return top, bottom, 0, 0
    padded_w = int(src_h * dst_w / dst_h)
    pad_w = max(0, padded_w - src_w)
    left = pad_w // 2
    right = pad_w - left
    return 0, 0, left, right


def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def value(obj, name, default=0):
    attr = getattr(obj, name, None)
    if attr is None:
        return default
    return attr() if callable(attr) else attr


def label_to_id(label):
    try:
        return int(label)
    except Exception:
        return -1


def clone_candidate(src):
    dst = {}
    for key in src:
        dst[key] = src[key]
    return dst


class LineResult:
    def __init__(self):
        self.valid = False
        self.offset = 0
        self.angle = 0
        self.lost_frames = 0
        self.quality = 0
        self.width_ratio = 0
        self.cx = CAM_WIDTH // 2
        self.cy = 0
        self.width = 0
        self.area = 0
        self.bands = []


class LineTracker:
    def __init__(self):
        self.last_offset = 0.0
        self.last_angle = 0.0
        self.lost_frames = 0

    def process(self, img):
        result = LineResult()
        w = img.width()
        h = img.height()
        center_x = w * 0.5

        centers = []
        near = None
        far = None
        widest = 0
        area_total = 0

        for idx, band in enumerate(LINE_ROI_BANDS):
            y0r, y1r, weight = band
            y0 = int(y0r * h)
            bh = max(1, int((y1r - y0r) * h))
            roi = (0, y0, w, bh)
            blob = self._best_line_blob(img, roi, center_x)

            if blob is None:
                result.bands.append((None, y0 + bh // 2, 0, roi))
                continue

            cx = value(blob, "cx")
            cy = value(blob, "cy")
            bw = value(blob, "w")
            pixels = value(blob, "pixels")
            density = self._blob_density(blob)
            result.bands.append((cx, cy, bw, roi))
            widest = max(widest, bw)
            area_total += pixels

            centers.append((cx, cy, weight, pixels, density))
            if idx == 0:
                near = (cx, cy)
            if idx == len(LINE_ROI_BANDS) - 1:
                far = (cx, cy)

        if centers:
            weighted = 0.0
            total = 0.0
            y_weighted = 0.0
            for cx, cy, weight, _pixels, _density in centers:
                weighted += cx * weight
                y_weighted += cy * weight
                total += weight
            raw_cx = weighted / total
            raw_offset = (raw_cx - center_x) / center_x * OFFSET_FULL_SCALE

            raw_angle = self.last_angle
            if near is not None and far is not None:
                dx = far[0] - near[0]
                dy = abs(near[1] - far[1])
                if dy > 1:
                    raw_angle = math.degrees(math.atan2(dx, dy))
            elif len(centers) >= 2:
                bottom = centers[0]
                top = centers[-1]
                dx = top[0] - bottom[0]
                dy = abs(bottom[1] - top[1])
                if dy > 1:
                    raw_angle = math.degrees(math.atan2(dx, dy))

            self.last_offset = OFFSET_FILTER_ALPHA * raw_offset + (1.0 - OFFSET_FILTER_ALPHA) * self.last_offset
            self.last_angle = ANGLE_FILTER_ALPHA * raw_angle + (1.0 - ANGLE_FILTER_ALPHA) * self.last_angle
            self.lost_frames = 0
            result.valid = True
            result.quality = self._line_quality(centers, near is not None)
            result.cx = int(raw_cx)
            result.cy = int(y_weighted / total)
        else:
            self.lost_frames += 1
            result.valid = self.lost_frames <= LOST_HOLD_FRAMES
            result.quality = max(0, 35 - self.lost_frames * 10) if result.valid else 0
            result.cx = int(center_x + self.last_offset / OFFSET_FULL_SCALE * center_x)
            result.cy = int(h * 0.85)

        result.offset = int(clip(self.last_offset, -100, 100))
        result.angle = int(clip(self.last_angle, -90, 90))
        result.lost_frames = self.lost_frames
        result.width_ratio = int(clip(widest * 100 / max(1, w), 0, 100))
        result.width = widest
        result.area = area_total
        return result

    def _blob_density(self, blob):
        pixels = value(blob, "pixels")
        bw = max(1, value(blob, "w"))
        bh = max(1, value(blob, "h"))
        return value(blob, "density", float(pixels) / float(bw * bh))

    def _line_quality(self, centers, has_near):
        band_score = min(100, int(len(centers) * 100 / max(1, len(LINE_ROI_BANDS))))
        density_avg = 0.0
        for _cx, _cy, _weight, _pixels, density in centers:
            density_avg += density
        density_avg /= max(1, len(centers))
        density_score = int(clip(density_avg * 160, 0, 100))
        near_bonus = 20 if has_near else -20
        return int(clip(band_score * 0.55 + density_score * 0.45 + near_bonus, 0, 100))

    def _best_line_blob(self, img, roi, frame_center_x):
        blobs = img.find_blobs(
            LINE_THRESHOLD,
            roi=roi,
            pixels_threshold=LINE_MIN_PIXELS,
            area_threshold=LINE_MIN_AREA,
            merge=True,
        )
        if not blobs:
            return None

        best = None
        best_score = -1
        x0, _y0, rw, _rh = roi
        roi_center_x = x0 + rw * 0.5
        predicted_x = frame_center_x + self.last_offset / OFFSET_FULL_SCALE * frame_center_x
        predicted_x = clip(predicted_x, x0, x0 + rw)
        for blob in blobs:
            pixels = value(blob, "pixels")
            bw = max(1, value(blob, "w"))
            bh = max(1, value(blob, "h"))
            cx = value(blob, "cx")
            density = self._blob_density(blob)
            if density < LINE_MIN_DENSITY:
                continue
            center_penalty = abs(cx - roi_center_x) * LINE_CENTER_PENALTY
            memory_penalty = abs(cx - predicted_x) * LINE_MEMORY_PENALTY
            score = pixels + bw * 5 + bh * 2 + density * 120 - center_penalty - memory_penalty
            if score > best_score:
                best = blob
                best_score = score
        return best


class SnapshotAnchorDetector:
    def __init__(self, deploy_conf):
        import nncase_runtime as nn
        import ulab.numpy as np
        import aicube

        self.nn = nn
        self.np = np
        self.aicube = aicube
        self.labels = deploy_conf["categories"]
        self.model_type = deploy_conf["model_type"]
        self.model_input_size = deploy_conf["img_size"]
        self.anchors = flatten_anchors(deploy_conf, self.model_type)
        self.strides = [8, 16, 32]
        self.confidence_threshold = deploy_conf["confidence_threshold"]
        self.nms_threshold = deploy_conf["nms_threshold"]
        self.nms_option = deploy_conf.get("nms_option", False)

        if CONFIDENCE_THRESHOLD_OVERRIDE is not None:
            self.confidence_threshold = CONFIDENCE_THRESHOLD_OVERRIDE
        if NMS_THRESHOLD_OVERRIDE is not None:
            self.nms_threshold = NMS_THRESHOLD_OVERRIDE

        self.rgb_size = [align_up(CAM_WIDTH, 16), CAM_HEIGHT]
        self.kmodel_path = resolve_model_path(deploy_conf["kmodel_path"])
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(self.kmodel_path)
        self.ai2d = nn.ai2d()
        self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        top, bottom, left, right = center_pad_param(self.rgb_size, self.model_input_size)
        self.ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
        self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)

        out = np.ones((1, 3, self.model_input_size[1], self.model_input_size[0]), dtype=np.uint8)
        self.ai2d_output_tensor = nn.from_numpy(out)
        self.builder = None
        self.builder_shape = None
        print("Digit model:", self.kmodel_path)
        print("Digit labels:", self.labels)

    def detect(self, img):
        nchw = self._image_to_nchw(img)
        if nchw is None:
            return []

        in_shape = nchw.shape
        if self.builder is None or self.builder_shape != in_shape:
            self.builder = self.ai2d.build(
                [1, 3, in_shape[2], in_shape[3]],
                [1, 3, self.model_input_size[1], self.model_input_size[0]],
            )
            self.builder_shape = in_shape

        input_tensor = self.nn.from_numpy(nchw)
        self.builder.run(input_tensor, self.ai2d_output_tensor)
        self.kpu.set_input_tensor(0, self.ai2d_output_tensor)
        self.kpu.run()

        results = []
        for i in range(self.kpu.outputs_size()):
            out = self.kpu.get_output_tensor(i)
            results.append(out.to_numpy())
            del out

        del input_tensor
        return self._postprocess(results)

    def _image_to_nchw(self, img):
        try:
            rgb = img.to_rgb888()
        except Exception:
            rgb = img
        try:
            arr = rgb.to_numpy_ref()
        except Exception as exc:
            print("digit to_numpy_ref failed:", exc)
            return None

        shape = arr.shape
        if len(shape) == 4 and shape[0] == 1 and shape[1] == 3:
            return arr
        if len(shape) == 3 and shape[0] == 3:
            return arr.reshape((1, shape[0], shape[1], shape[2]))
        if len(shape) == 3 and shape[2] == 3:
            flat = arr.reshape((shape[0] * shape[1], shape[2]))
            trans = flat.transpose()
            copied = trans.copy()
            return copied.reshape((1, shape[2], shape[0], shape[1]))
        print("unsupported digit image shape:", shape)
        return None

    def _postprocess(self, results):
        if not results or len(results) < 3:
            return []
        if self.model_type == "AnchorBaseDet":
            return self.aicube.anchorbasedet_post_process(
                results[0],
                results[1],
                results[2],
                self.model_input_size,
                self.rgb_size,
                self.strides,
                len(self.labels),
                self.confidence_threshold,
                self.nms_threshold,
                self.anchors,
                self.nms_option,
            )
        print("unsupported model type:", self.model_type)
        return []

    def deinit(self):
        try:
            del self.builder
            del self.ai2d_output_tensor
            del self.ai2d
            del self.kpu
            self.nn.shrink_memory_pool()
        except Exception:
            pass


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


def det_boxes_to_candidates(det_boxes, labels):
    candidates = []
    if not det_boxes:
        return candidates

    min_area = int(CAM_WIDTH * CAM_HEIGHT * MIN_BOX_AREA_RATIO)
    center_x = CAM_WIDTH // 2
    center_y = CAM_HEIGHT // 2

    for det in det_boxes:
        try:
            cls_idx = int(det[0])
            score = float(det[1])
            x1 = int(det[2])
            y1 = int(det[3])
            x2 = int(det[4])
            y2 = int(det[5])
        except Exception:
            continue

        if cls_idx < 0 or cls_idx >= len(labels):
            continue
        if score < SELECT_MIN_SCORE:
            continue

        x1 = int(clip(x1, 0, CAM_WIDTH - 1))
        y1 = int(clip(y1, 0, CAM_HEIGHT - 1))
        x2 = int(clip(x2, 0, CAM_WIDTH - 1))
        y2 = int(clip(y2, 0, CAM_HEIGHT - 1))
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

        norm_area = float(area) / float(CAM_WIDTH * CAM_HEIGHT)
        norm_center = abs(float(err_x)) / float(max(1, center_x))
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


def estimate_distance(candidate):
    if candidate is None or candidate["h"] <= 0 or DISTANCE_K_BY_HEIGHT <= 0:
        return 0
    return int(DISTANCE_K_BY_HEIGHT / candidate["h"])


def build_payload(frame_id, target_id, candidate, locked, hits, fps):
    if candidate is None:
        return "MV,%d,0,%d,0,0,0,0,0,0,0,0,0,0,0,%d" % (frame_id, target_id, fps)

    center_tol = int(CAM_WIDTH * CENTER_TOLERANCE_RATIO)
    arrive_h = int(CAM_HEIGHT * ARRIVE_BOX_HEIGHT_RATIO)
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
    if line_result is None or not line_result.valid:
        return "LN,%d,0,0,0,0,0,0,%d" % (frame_id, fps)
    err_px = int(line_result.offset * (CAM_WIDTH // 2) / OFFSET_FULL_SCALE)
    return "LN,%d,1,%d,%d,%d,%d,%d,%d" % (
        frame_id,
        err_px,
        line_result.angle,
        line_result.cx,
        line_result.width,
        line_result.area,
        fps,
    )


def draw_text(img, x, y, msg, color=WHITE, size=16):
    if hasattr(img, "draw_string_advanced"):
        try:
            img.draw_string_advanced(int(x), int(y), int(size), str(msg), color=color)
            return
        except Exception:
            pass
    img.draw_string(int(x), int(y), str(msg), color=color)


def draw_line_result(img, result, fps):
    w = img.width()
    h = img.height()
    img.draw_line(w // 2, 0, w // 2, h, color=GRAY, thickness=1)
    for cx, cy, _bw, roi in result.bands:
        img.draw_rectangle(roi[0], roi[1], roi[2], roi[3], color=YELLOW, thickness=1)
        if cx is not None:
            img.draw_cross(int(cx), int(cy), color=GREEN, size=6, thickness=2)
    if result.valid:
        img.draw_line(w // 2, h - 1, int(result.cx), int(result.cy), color=GREEN, thickness=2)
        draw_text(img, 2, 2, "LINE off:%d a:%d q:%d fps:%d" % (
            result.offset,
            result.angle,
            result.quality,
            int(fps),
        ), GREEN, 15)
    else:
        draw_text(img, 2, 2, "LINE LOST lost:%d fps:%d" % (result.lost_frames, int(fps)), RED, 15)


def draw_digit_result(img, candidates, selected, target_id, locked, hits):
    selected_id = -1
    if selected is not None:
        selected_id = selected["drug_id"]

    for cand in candidates:
        color = MAGENTA
        thickness = 2
        if cand["drug_id"] == selected_id:
            color = GREEN if locked else CYAN
            thickness = 3
        img.draw_rectangle(cand["x1"], cand["y1"], cand["w"], cand["h"], color=color, thickness=thickness)
        draw_text(img, cand["x1"], max(18, cand["y1"]) - 18, "%s %d" % (
            cand["label"],
            int(cand["score"] * 100),
        ), color, 15)

    y = 20
    if selected is None:
        draw_text(img, 2, y, "DIGIT T:%d SEARCH" % target_id, YELLOW, 15)
        return

    lock_text = "LOCK" if locked else "SEEN"
    target_text = "AUTO" if target_id == 0 else str(target_id)
    draw_text(img, 2, y, "DIGIT T:%s %s id:%d ex:%d h:%d %d/%d" % (
        target_text,
        lock_text,
        selected["drug_id"],
        selected["err_x"],
        selected["h"],
        hits,
        HISTORY_SIZE,
    ), WHITE, 15)


def init_camera():
    sensor = Sensor(width=CAM_WIDTH, height=CAM_HEIGHT)
    sensor.reset()
    sensor.set_framesize(width=CAM_WIDTH, height=CAM_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)
    try:
        if HMIRROR and hasattr(sensor, "set_hmirror"):
            sensor.set_hmirror(1)
        if VFLIP and hasattr(sensor, "set_vflip"):
            sensor.set_vflip(1)
    except Exception as exc:
        print("mirror/flip skipped:", exc)
    return sensor


def init_display():
    if not SHOW_UI:
        return None
    mode = DISPLAY_MODE.lower()
    if mode == "lcd":
        try:
            Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
            return Display
        except Exception as exc:
            print("LCD display init failed, fallback to VIRT:", exc)
    Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
    return Display


def show_image(display, img):
    if display is None:
        return
    x = max(0, (DISPLAY_WIDTH - img.width()) // 2)
    y = max(0, (DISPLAY_HEIGHT - img.height()) // 2)
    try:
        display.show_image(img, x=x, y=y)
    except TypeError:
        display.show_image(img)


def main():
    sensor = None
    display = None
    detector = None
    try:
        deploy_conf = None
        config_path = None
        labels = []
        if ENABLE_DIGIT_DETECT:
            try:
                deploy_conf, config_path = load_deploy_config()
                labels = deploy_conf["categories"]
                print("Loaded config:", config_path)
            except Exception as exc:
                print("Digit config load failed, line only:", exc)
                print_exception(exc)

        sensor = init_camera()
        display = init_display()
        MediaManager.init()
        sensor.run()

        if ENABLE_DIGIT_DETECT and deploy_conf is not None:
            try:
                detector = SnapshotAnchorDetector(deploy_conf)
            except Exception as exc:
                print("Digit detector disabled:", exc)
                print_exception(exc)
                detector = None

        bridge = UartBridge()
        tracker = LineTracker()
        stabilizer = TargetStabilizer()
        runtime_target = DEFAULT_TARGET_ID

        frame_id = 0
        fps = 0.0
        last_t = ticks_ms()
        candidates = []
        selected = None
        locked = 0
        hits = 0

        print("combined started: sensor RGB565 snapshot + line + digit")
        while True:
            check_exitpoint()
            for line in bridge.read_commands():
                new_target, accepted = parse_command(line, runtime_target)
                if accepted:
                    runtime_target = new_target
                    stabilizer.reset()
                    selected = None
                    locked = 0
                    hits = 0
                    if accepted == 2:
                        print("Vision lock reset")
                    else:
                        print("Target set:", runtime_target)

            img = sensor.snapshot()
            if img is None:
                continue

            frame_id += 1
            now = ticks_ms()
            dt = ticks_diff(now, last_t)
            last_t = now
            if dt > 0:
                fps = 0.90 * fps + 0.10 * (1000.0 / dt)

            line_result = tracker.process(img)

            if detector is not None and (frame_id % DIGIT_DETECT_EVERY_N_FRAMES == 0 or selected is None):
                try:
                    det_boxes = detector.detect(img)
                    candidates = det_boxes_to_candidates(det_boxes, labels)
                    picked = pick_candidate(candidates, runtime_target)
                    selected, locked, hits = stabilizer.update(picked)
                except Exception as exc:
                    print("digit detect failed:", exc)
                    print_exception(exc)
                    candidates = []
                    selected, locked, hits = stabilizer.update(None)

            if frame_id % SEND_EVERY_N_FRAMES == 0:
                bridge.write_packet(build_payload(frame_id, runtime_target, selected, locked, hits, int(fps)))
                bridge.write_packet(build_line_payload(frame_id, line_result, int(fps)))

            if SHOW_UI:
                draw_line_result(img, line_result, fps)
                draw_digit_result(img, candidates, selected, runtime_target, locked, hits)
                show_image(display, img)

            if frame_id % 20 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            if detector is not None:
                detector.deinit()
        except Exception:
            pass
        try:
            if sensor is not None:
                sensor.stop()
        except Exception:
            pass
        try:
            if SHOW_UI:
                Display.deinit()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


main()
