#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

FLAG_SET_REF = "/tmp/change_set_reference"
FLAG_CLEAR_REF = "/tmp/change_clear_reference"
FLAG_SAVE = "/tmp/change_save_outputs"
FLAG_QUIT = "/tmp/change_quit"

SAVE_ROOT_DIR = "/home/firefox33/build-a-cubesat-firefox/Images"
LIVE_DIR = "/tmp/change_live"
RUNTIME_DIR = os.path.join(SAVE_ROOT_DIR, "_mission_runtime")

CAMERA_RES = (4608, 2592)
PROC_RES = (960, 540)
SHOW_GUI = False

DIFF_THRESHOLD = 25
KERNEL_SIZE = 5
MIN_BLOB_AREA = 250
<<<<<<< Updated upstream
USE_CLAHE = False
BRIGHT_WIN = 80
=======
USE_CLAHE = False    # can make halos worse; start False
BRIGHT_WIN =80 #size of brightest square

# Optional: how often to write live images to /tmp
>>>>>>> Stashed changes
LIVE_WRITE_EVERY_N_FRAMES = 3

RUNTIME_REF_PATH = os.path.join(RUNTIME_DIR, "reference_fullres.png")

os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)


def lock_camera(picam2: Picamera2):
    time.sleep(0.8)
    md = picam2.capture_metadata()
    exposure = md.get("ExposureTime")
    gain = md.get("AnalogueGain")
    colour_gains = md.get("ColourGains")

    controls = {"AeEnable": False, "AwbEnable": False}
    if exposure is not None:
        controls["ExposureTime"] = int(exposure)
    if gain is not None:
        controls["AnalogueGain"] = float(gain)
    if colour_gains is not None:
        controls["ColourGains"] = tuple(colour_gains)

    picam2.set_controls(controls)
    time.sleep(0.2)
    print("Camera AE/AWB locked.")


def preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def fullres_to_proc(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, PROC_RES, interpolation=cv2.INTER_AREA)


def compute_change(ref_g: np.ndarray, cur_proc_bgr: np.ndarray):
    cur_g = preprocess_gray(cur_proc_bgr)
    diff = cv2.absdiff(ref_g, cur_g)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            cleaned[labels == idx] = 255

    overlay = cur_proc_bgr.copy()
    overlay[cleaned > 0] = (0, 0, 255)
    changed_pct = 100.0 * float((cleaned > 0).mean())
    return cleaned, overlay, changed_pct


def brightest_square(bgr: np.ndarray, win: int = BRIGHT_WIN):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(win, win), normalize=True)
    _, _, _, max_loc = cv2.minMaxLoc(mean)
    x, y = max_loc
    score = float(mean[y, x])
    return x, y, win, score


def consume_flag(path: str) -> bool:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return True
    return False


def save_runtime_reference(ref_fullres_bgr: np.ndarray):
    cv2.imwrite(RUNTIME_REF_PATH, ref_fullres_bgr)


def load_runtime_reference():
    if not os.path.exists(RUNTIME_REF_PATH):
        return None, None, None
    ref_fullres_bgr = cv2.imread(RUNTIME_REF_PATH, cv2.IMREAD_COLOR)
    if ref_fullres_bgr is None:
        return None, None, None
    ref_proc_bgr = fullres_to_proc(ref_fullres_bgr)
    ref_g = preprocess_gray(ref_proc_bgr)
    ref_bright = brightest_square(ref_proc_bgr, win=BRIGHT_WIN)
    return ref_fullres_bgr, ref_g, ref_bright


def clear_runtime_reference():
    if os.path.exists(RUNTIME_REF_PATH):
        try:
            os.remove(RUNTIME_REF_PATH)
        except OSError:
            pass


def save_bundle(ref_fullres_bgr, cur_fullres_bgr, ref_proc_bgr, cur_proc_bgr, mask, overlay, changed_pct, ref_bright, cur_bright):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SAVE_ROOT_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "ref.png"), ref_proc_bgr)
    cv2.imwrite(os.path.join(out_dir, "cur.png"), cur_proc_bgr)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, "ref_fullres.png"), ref_fullres_bgr)
    cv2.imwrite(os.path.join(out_dir, "cur_fullres.png"), cur_fullres_bgr)

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"changed_pct={changed_pct:.6f}\n")
        handle.write(f"camera_res={CAMERA_RES}\n")
        handle.write(f"proc_res={PROC_RES}\n")
        handle.write(f"diff_threshold={DIFF_THRESHOLD}\n")
        handle.write(f"kernel_size={KERNEL_SIZE}\n")
        handle.write(f"min_blob_area={MIN_BLOB_AREA}\n")
        handle.write(f"use_clahe={USE_CLAHE}\n")
        handle.write(f"bright_win={BRIGHT_WIN}\n")
        if ref_bright is not None:
            rx, ry, rs, rscore = ref_bright
            handle.write(f"ref_bright_x={rx}\nref_bright_y={ry}\nref_bright_s={rs}\nref_bright_score={rscore:.6f}\n")
        if cur_bright is not None:
            bx, by, bs, bscore = cur_bright
            handle.write(f"cur_bright_x={bx}\ncur_bright_y={by}\ncur_bright_s={bs}\ncur_bright_score={bscore:.6f}\n")

    print(f"Saved bundle to: {out_dir}")


def write_live(ref_set: bool, cur_proc_bgr, mask, overlay, changed_pct: float):
    cv2.imwrite(os.path.join(LIVE_DIR, "cur.jpg"), cur_proc_bgr)
    cv2.imwrite(os.path.join(LIVE_DIR, "overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(LIVE_DIR, "mask.png"), mask)
    with open(os.path.join(LIVE_DIR, "status.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"ref_set={ref_set}\n")
        handle.write(f"changed_pct={changed_pct:.6f}\n")


def annotate_bright_square(image: np.ndarray, bright):
    if bright is None:
        return
    x, y, size, score = bright
    cv2.rectangle(image, (x, y), (x + size, y + size), (0, 255, 255), 2)
    cv2.putText(
        image,
        f"bright={score:.1f}",
        (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


<<<<<<< Updated upstream
=======
def brightest_square(bgr, win = 80):
	"""

	"""
# -------------------------
# Main
# -------------------------
>>>>>>> Stashed changes
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAMERA_RES})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    lock_camera(picam2)

    ref_fullres_bgr, ref_g, ref_bright = load_runtime_reference()
    if ref_g is not None:
        print("Loaded persisted reference image.")

    print("\nRunning change detector.")
    print(f"Set reference flag: {FLAG_SET_REF}")
    print(f"Clear reference flag: {FLAG_CLEAR_REF}")
    print(f"Save bundle flag: {FLAG_SAVE}")
    print(f"Quit flag: {FLAG_QUIT}")

    frame_i = 0

    while True:
        if consume_flag(FLAG_QUIT):
            print("Quitting.")
            break

        rgb = picam2.capture_array()
        cur_fullres_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cur_proc_bgr = fullres_to_proc(cur_fullres_bgr)

        if consume_flag(FLAG_CLEAR_REF):
            ref_fullres_bgr = None
            ref_g = None
            ref_bright = None
            clear_runtime_reference()
            print("Reference cleared.")

        if consume_flag(FLAG_SET_REF):
            ref_fullres_bgr = cur_fullres_bgr.copy()
            ref_proc_bgr = cur_proc_bgr.copy()
            ref_g = preprocess_gray(ref_proc_bgr)
            ref_bright = brightest_square(ref_proc_bgr, win=BRIGHT_WIN)
            save_runtime_reference(ref_fullres_bgr)
            print("Reference set and persisted.")

        if ref_g is None:
            h, w = cur_proc_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            overlay = cur_proc_bgr.copy()
            changed_pct = 0.0
        else:
            mask, overlay, changed_pct = compute_change(ref_g, cur_proc_bgr)

        cur_bright = brightest_square(cur_proc_bgr, win=BRIGHT_WIN)
        annotate_bright_square(overlay, cur_bright)
        cv2.rectangle(mask, (cur_bright[0], cur_bright[1]), (cur_bright[0] + cur_bright[2], cur_bright[1] + cur_bright[2]), 255, 2)

        if consume_flag(FLAG_SAVE):
            if ref_g is None or ref_fullres_bgr is None:
                print(f"Cannot save: reference not set. Run: touch {FLAG_SET_REF}")
            else:
                ref_proc_bgr = fullres_to_proc(ref_fullres_bgr)
                ref_annotated = ref_proc_bgr.copy()
                annotate_bright_square(ref_annotated, ref_bright)
                save_bundle(
                    ref_fullres_bgr,
                    cur_fullres_bgr,
                    ref_annotated,
                    cur_proc_bgr,
                    mask,
                    overlay,
                    changed_pct,
                    ref_bright,
                    cur_bright,
                )

        frame_i += 1
        if frame_i % LIVE_WRITE_EVERY_N_FRAMES == 0:
            write_live(ref_g is not None, cur_proc_bgr, mask, overlay, changed_pct)

        if SHOW_GUI:
            cv2.imshow("camera", cur_proc_bgr)
            cv2.imshow("overlay", overlay)
            cv2.imshow("mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                ref_fullres_bgr = cur_fullres_bgr.copy()
                ref_g = preprocess_gray(cur_proc_bgr)
                ref_bright = brightest_square(cur_proc_bgr, win=BRIGHT_WIN)
                save_runtime_reference(ref_fullres_bgr)
            if key == ord("c"):
                ref_fullres_bgr = None
                ref_g = None
                ref_bright = None
                clear_runtime_reference()
            if key == ord("s") and ref_fullres_bgr is not None and ref_g is not None:
                ref_proc_bgr = fullres_to_proc(ref_fullres_bgr)
                ref_annotated = ref_proc_bgr.copy()
                annotate_bright_square(ref_annotated, ref_bright)
                save_bundle(
                    ref_fullres_bgr,
                    cur_fullres_bgr,
                    ref_annotated,
                    cur_proc_bgr,
                    mask,
                    overlay,
                    changed_pct,
                    ref_bright,
                    cur_bright,
                )

        time.sleep(0.01)

    picam2.stop()
    if SHOW_GUI:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
