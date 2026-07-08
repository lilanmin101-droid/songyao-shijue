import gc
import math
import time

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager


CAM_WIDTH = 480
CAM_HEIGHT = 272
DISPLAY_WIDTH = CAM_WIDTH
DISPLAY_HEIGHT = CAM_HEIGHT

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

WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
GREEN = (0, 220, 0)
RED = (255, 0, 0)
YELLOW = (255, 220, 0)


def ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def ticks_diff(a, b):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(a, b)
    return a - b


def value(obj, name, default=0):
    attr = getattr(obj, name, None)
    if attr is None:
        return default
    return attr() if callable(attr) else attr


def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class LineResult:
    def __init__(self):
        self.valid = False
        self.offset = 0
        self.angle = 0
        self.lost_frames = 0
        self.quality = 0
        self.width_ratio = 0
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

            centers.append((cx, cy, weight, pixels, density))
            if idx == 0:
                near = (cx, cy)
            if idx == len(LINE_ROI_BANDS) - 1:
                far = (cx, cy)

        if centers:
            weighted = 0.0
            total = 0.0
            for cx, _cy, weight, _pixels, _density in centers:
                weighted += cx * weight
                total += weight
            raw_offset = (weighted / total - center_x) / center_x * OFFSET_FULL_SCALE

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
        else:
            self.lost_frames += 1
            result.valid = self.lost_frames <= LOST_HOLD_FRAMES
            result.quality = max(0, 35 - self.lost_frames * 10) if result.valid else 0

        result.offset = int(clip(self.last_offset, -100, 100))
        result.angle = int(clip(self.last_angle, -90, 90))
        result.lost_frames = self.lost_frames
        result.width_ratio = int(clip(widest * 100 / max(1, w), 0, 100))
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
        draw_text(img, 2, 2, "LINE off:%d a:%d q:%d fps:%d" % (
            result.offset,
            result.angle,
            result.quality,
            int(fps),
        ), GREEN, 15)
    else:
        draw_text(img, 2, 2, "LINE LOST lost:%d fps:%d" % (result.lost_frames, int(fps)), RED, 15)


def main():
    sensor = Sensor(width=CAM_WIDTH, height=CAM_HEIGHT)
    sensor.reset()
    sensor.set_framesize(width=CAM_WIDTH, height=CAM_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
    MediaManager.init()
    sensor.run()

    tracker = LineTracker()
    fps = 0.0
    last = ticks_ms()
    frame = 0
    print("line debug started: sensor RGB565 snapshot")

    try:
        while True:
            img = sensor.snapshot()
            result = tracker.process(img)
            now = ticks_ms()
            dt = ticks_diff(now, last)
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1000.0 / dt)
            draw_line_result(img, result, fps)
            Display.show_image(img)
            frame += 1
            if frame % 20 == 0:
                gc.collect()
    finally:
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            Display.deinit()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


main()
