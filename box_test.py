#!/usr/bin/env python3
import time
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2


CAMERA_RES = (4608, 2592)
PROC_RES = (960, 540)
BOX_WIDTH_FULLRES = 1060
BOX_HEIGHT_FULLRES = 1040
SAFE_THRESHOLD = 200

DIFF_THRESHOLD = 25
KERNEL_SIZE = 5
MIN_BLOB_AREA = 250


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


def preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def fullres_to_proc(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, PROC_RES, interpolation=cv2.INTER_AREA)


def compute_mask(ref_g: np.ndarray, cur_proc_bgr: np.ndarray) -> np.ndarray:
    cur_g = preprocess_gray(cur_proc_bgr)
    diff = cv2.absdiff(ref_g, cur_g)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_component(mask: np.ndarray) -> Optional[dict]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_area = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA:
            continue
        if area > best_area:
            best_area = area
            best = {
                "x": int(stats[idx, cv2.CC_STAT_LEFT]),
                "y": int(stats[idx, cv2.CC_STAT_TOP]),
                "w": int(stats[idx, cv2.CC_STAT_WIDTH]),
                "h": int(stats[idx, cv2.CC_STAT_HEIGHT]),
                "area": area,
            }
    return best


def scale_component(component: Optional[dict], src_shape: tuple[int, int], dst_shape: tuple[int, int, int]) -> Optional[dict]:
    if component is None:
        return None
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return {
        "x": int(round(component["x"] * scale_x)),
        "y": int(round(component["y"] * scale_y)),
        "w": max(1, int(round(component["w"] * scale_x))),
        "h": max(1, int(round(component["h"] * scale_y))),
        "area": component["area"],
    }


def clamp_box(x: int, y: int, width: int, height: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_w - width))
    y = max(0, min(y, image_h - height))
    return x, y, width, height


def fixed_box(component: Optional[dict], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    image_h, image_w = image_shape[:2]
    scale_x = image_w / 4608.0
    scale_y = image_h / 2592.0
    width = min(max(1, int(round(BOX_WIDTH_FULLRES * scale_x))), image_w)
    height = min(max(1, int(round(BOX_HEIGHT_FULLRES * scale_y))), image_h)

    if component is None:
        return clamp_box((image_w - width) // 2, (image_h - height) // 2, width, height, image_w, image_h)

    cx = component["x"] + component["w"] / 2.0
    cy = component["y"] + component["h"] / 2.0
    return clamp_box(int(round(cx - width / 2.0)), int(round(cy - height / 2.0)), width, height, image_w, image_h)


def safe_mask_from_image(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, safe = cv2.threshold(gray, SAFE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN, kernel)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, kernel)
    return safe


def landing_zone_box(bgr: np.ndarray) -> tuple[int, int, int, int]:
    image_h, image_w = bgr.shape[:2]
    scale_x = image_w / 4608.0
    scale_y = image_h / 2592.0
    width = min(max(1, int(round(BOX_WIDTH_FULLRES * scale_x))), image_w)
    height = min(max(1, int(round(BOX_HEIGHT_FULLRES * scale_y))), image_h)

    safe = safe_mask_from_image(bgr)
    kernel = np.ones((height, width), np.uint8)
    centers = cv2.erode(safe, kernel, iterations=1)
    if np.count_nonzero(centers) == 0:
        return clamp_box((image_w - width) // 2, (image_h - height) // 2, width, height, image_w, image_h)

    distance = cv2.distanceTransform(centers, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(distance)
    cx, cy = max_loc
    return clamp_box(int(round(cx - width / 2.0)), int(round(cy - height / 2.0)), width, height, image_w, image_h)


def main() -> None:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAMERA_RES})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    lock_camera(picam2)

    ref_fullres = None
    ref_g = None

    print("box_test running")
    print("keys: r=set reference, c=clear reference, q=quit")

    while True:
        rgb = picam2.capture_array()
        cur_fullres = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cur_proc = fullres_to_proc(cur_fullres)

        display = cv2.resize(cur_fullres, (1280, 720), interpolation=cv2.INTER_AREA)
        mask_display = cv2.resize(safe_mask_from_image(cur_fullres), (1280, 720), interpolation=cv2.INTER_NEAREST)

        if ref_g is not None:
            mask = compute_mask(ref_g, cur_proc)
            diff_display = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("box_test_diff", diff_display)

        x, y, w, h = landing_zone_box(cur_fullres)
        scale_x = display.shape[1] / cur_fullres.shape[1]
        scale_y = display.shape[0] / cur_fullres.shape[0]
        dx = int(round(x * scale_x))
        dy = int(round(y * scale_y))
        dw = int(round(w * scale_x))
        dh = int(round(h * scale_y))
        cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)

        status = "ref=OFF" if ref_g is None else "ref=ON"
        cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"box={w}x{h}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("box_test_camera", display)
        cv2.imshow("box_test_safe", mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ref_fullres = cur_fullres.copy()
            ref_g = preprocess_gray(fullres_to_proc(ref_fullres))
            print("reference set")
        if key == ord("c"):
            ref_fullres = None
            ref_g = None
            print("reference cleared")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
