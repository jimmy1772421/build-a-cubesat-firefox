#!/usr/bin/env python3
import time

import cv2
import numpy as np
from picamera2 import Picamera2


CAMERA_RES = (4608, 2592)
WORK_RES = (1152, 648)
DISPLAY_RES = (1280, 720)
BOX_WIDTH_FULLRES = 1060
BOX_HEIGHT_FULLRES = 1040
BRIGHT_THRESHOLD = 200
MIN_BLOB_AREA = 50


def lock_camera(picam2: Picamera2) -> None:
    time.sleep(0.8)
    md = picam2.capture_metadata()
    controls = {"AeEnable": False, "AwbEnable": False}
    exposure = md.get("ExposureTime")
    gain = md.get("AnalogueGain")
    colour_gains = md.get("ColourGains")
    if exposure is not None:
        controls["ExposureTime"] = int(exposure)
    if gain is not None:
        controls["AnalogueGain"] = float(gain)
    if colour_gains is not None:
        controls["ColourGains"] = tuple(colour_gains)
    picam2.set_controls(controls)
    time.sleep(0.2)


def clamp_box(x: int, y: int, width: int, height: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_w - width))
    y = max(0, min(y, image_h - height))
    return x, y, width, height


def bright_mask_from_image(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, safe = cv2.threshold(gray, BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN, kernel)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, kernel)
    return safe


def largest_blob_center(mask: np.ndarray) -> tuple[int, int] | None:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_idx = None
    best_area = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA:
            continue
        if area > best_area:
            best_area = area
            best_idx = idx
    if best_idx is None:
        return None
    cx, cy = centroids[best_idx]
    return int(round(cx)), int(round(cy))


def tracking_box(fullres_bgr: np.ndarray) -> tuple[int, int, int, int]:
    image_h, image_w = fullres_bgr.shape[:2]
    width = min(max(1, BOX_WIDTH_FULLRES), image_w)
    height = min(max(1, BOX_HEIGHT_FULLRES), image_h)

    work_bgr = cv2.resize(fullres_bgr, WORK_RES, interpolation=cv2.INTER_AREA)
    mask = bright_mask_from_image(work_bgr)
    center = largest_blob_center(mask)
    if center is None:
        return clamp_box((image_w - width) // 2, (image_h - height) // 2, width, height, image_w, image_h)

    work_h, work_w = mask.shape[:2]
    scale_x = image_w / work_w
    scale_y = image_h / work_h
    cx = int(round(center[0] * scale_x))
    cy = int(round(center[1] * scale_y))
    return clamp_box(int(round(cx - width / 2.0)), int(round(cy - height / 2.0)), width, height, image_w, image_h)


def main() -> None:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAMERA_RES})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    lock_camera(picam2)

    print("box_test running")
    print("press q to quit")

    while True:
        rgb = picam2.capture_array()
        fullres = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        x, y, w, h = tracking_box(fullres)

        display = cv2.resize(fullres, DISPLAY_RES, interpolation=cv2.INTER_AREA)
        scale_x = display.shape[1] / fullres.shape[1]
        scale_y = display.shape[0] / fullres.shape[0]
        dx = int(round(x * scale_x))
        dy = int(round(y * scale_y))
        dw = int(round(w * scale_x))
        dh = int(round(h * scale_y))

        cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)
        cv2.putText(display, f"box={w}x{h}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("box_test_camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
