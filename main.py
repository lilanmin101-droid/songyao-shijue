import os
import gc
import time

from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *


# =========================
# Match-day configuration
# =========================

# Put this file, deploy_config.json and the kmodel under this directory on K230.
ROOT_PATH = "/sdcard/mp_deployment_source/"

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


def draw_overlay(draw_img, candidates, selected, target_id, locked, hits, fps, display_size):
    draw_img.clear()
    dw = display_size[0]
    dh = display_size[1]
    cx = dw // 2
    center_tol = int(dw * CENTER_TOLERANCE_RATIO)

    draw_img.draw_line(cx, 0, cx, dh, color=COLOR_GUIDE, thickness=1)
    draw_img.draw_line(cx - center_tol, dh - 70, cx - center_tol, dh, color=COLOR_GUIDE, thickness=2)
    draw_img.draw_line(cx + center_tol, dh - 70, cx + center_tol, dh, color=COLOR_GUIDE, thickness=2)

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

    bridge = UartBridge()
    stabilizer = TargetStabilizer()
    fps_meter = FpsMeter()
    runtime_target = DEFAULT_TARGET_ID
    frame_id = 0

    pl = None
    det_app = None
    try:
        pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE, display_size=DISPLAY_SIZE)
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
                res = det_app.run(img)
                candidates = result_to_candidates(res, labels)
                picked = pick_candidate(candidates, runtime_target)
                selected, locked, hits = stabilizer.update(picked)

                if frame_id % SEND_EVERY_N_FRAMES == 0:
                    payload = build_payload(frame_id, runtime_target, selected, locked, hits, fps)
                    bridge.write_packet(payload)

                draw_overlay(pl.osd_img, candidates, selected, runtime_target, locked, hits, fps, display_size)
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
