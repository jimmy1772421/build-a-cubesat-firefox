#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2

# -------------------------
# "Service-like" control files
# -------------------------
FLAG_SET_REF   = "/tmp/change_set_reference"
FLAG_CLEAR_REF = "/tmp/change_clear_reference"
FLAG_SAVE      = "/tmp/change_save_outputs"
FLAG_QUIT      = "/tmp/change_quit"

# -------------------------
# Output settings
# -------------------------
SAVE_ROOT_DIR = "/home/firefox33/build-a-cubesat-firefox/Images"  # <-- change this
LIVE_DIR = "/tmp/change_live"  # always updated live frames (for headless viewing)

# Folder-per-save: SAVE_ROOT_DIR/YYYYmmdd_HHMMSS/{ref,cur,mask,overlay}.png
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)

# -------------------------
# Camera settings
# -------------------------
RES = (960, 540)     # processing resolution (increase for quality, decrease for speed)
SHOW_GUI = True      # set False if no display (headless)

# -------------------------
# Change detection tuning
# -------------------------
DIFF_THRESHOLD = 25
KERNEL_SIZE = 5
MIN_BLOB_AREA = 250
USE_CLAHE = False    # can make halos worse; start False

# Brightest-square tuning (raw image)
BRIGHT_WIN = 80      # size of the brightest square (pixels). Try 50–150.

# Optional: how often to write live images to /tmp
LIVE_WRITE_EVERY_N_FRAMES = 3

# -------------------------
# Helpers
# -------------------------
def lock_camera(picam2: Picamera2):
    """Lock AE/AWB to reduce false positives from auto-adjustment."""
    time.sleep(0.8)
    md = picam2.capture_metadata()
    exposure = md.get("ExposureTime", None)
    gain = md.get("AnalogueGain", None)
    colour_gains = md.get("ColourGains", None)

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
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        g = clahe.apply(g)
    return g

def compute_change(ref_g: np.ndarray, cur_bgr: np.ndarray):
    cur_g = preprocess_gray(cur_bgr)
    diff = cv2.absdiff(ref_g, cur_g)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    k = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # remove tiny blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            cleaned[labels == i] = 255

    overlay = cur_bgr.copy()
    overlay[cleaned > 0] = (0, 0, 255)

    changed_pct = 100.0 * float((cleaned > 0).mean())
    return cleaned, overlay, changed_pct

def brightest_square(bgr, win=80):
    """
    Find the window with the highest mean brightness in the raw image.
    Returns (x, y, win, score).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(win, win), normalize=True)
    _, _, _, maxLoc = cv2.minMaxLoc(mean)
    x, y = maxLoc
    score = float(mean[y, x])
    return x, y, win, score

def consume_flag(path: str) -> bool:
    """Return True if file exists, and remove it (one-shot command)."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return True
    return False

def save_bundle(save_root: str, ref_bgr, cur_bgr, mask, overlay, changed_pct: float, ref_bright, cur_bright):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_root, stamp)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "ref.png"), ref_bgr)
    cv2.imwrite(os.path.join(out_dir, "cur.png"), cur_bgr)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

    # small metadata text file
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"changed_pct={changed_pct:.6f}\n")
        f.write(f"diff_threshold={DIFF_THRESHOLD}\n")
        f.write(f"kernel_size={KERNEL_SIZE}\n")
        f.write(f"min_blob_area={MIN_BLOB_AREA}\n")
        f.write(f"use_clahe={USE_CLAHE}\n")
        f.write(f"res={RES}\n")
        f.write(f"bright_win={BRIGHT_WIN}\n")

        if ref_bright is not None:
            rx, ry, rs, rscore = ref_bright
            f.write(f"ref_bright_x={rx}\nref_bright_y={ry}\nref_bright_s={rs}\nref_bright_score={rscore:.6f}\n")

        if cur_bright is not None:
            bx, by, bs, bscore = cur_bright
            f.write(f"cur_bright_x={bx}\ncur_bright_y={by}\ncur_bright_s={bs}\ncur_bright_score={bscore:.6f}\n")

    print(f"✅ Saved bundle to: {out_dir}")

def write_live(ref_set: bool, cur_bgr, mask, overlay, changed_pct: float):
    # Always update latest live outputs for headless viewing
    cv2.imwrite(os.path.join(LIVE_DIR, "cur.jpg"), cur_bgr)
    cv2.imwrite(os.path.join(LIVE_DIR, "overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(LIVE_DIR, "mask.png"), mask)
    with open(os.path.join(LIVE_DIR, "status.txt"), "w") as f:
        f.write(f"ref_set={ref_set}\n")
        f.write(f"changed_pct={changed_pct:.6f}\n")

# -------------------------
# Main
# -------------------------
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": RES})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    lock_camera(picam2)

    print("\nRunning change detector (no ROS).")
    print("Terminal commands:")
    print(f"  Set reference:   touch {FLAG_SET_REF}")
    print(f"  Clear reference: touch {FLAG_CLEAR_REF}")
    print(f"  Save bundle:     touch {FLAG_SAVE}")
    print(f"  Quit:            touch {FLAG_QUIT}")
    print("Keyboard shortcuts (GUI windows focused): r=set ref, c=clear, s=save, q=quit")
    print(f"Live outputs: {LIVE_DIR}/overlay.jpg  {LIVE_DIR}/mask.png  {LIVE_DIR}/status.txt\n")

    ref_bgr = None
    ref_g = None
    ref_bright = None
    cur_bright = None

    frame_i = 0

    while True:
        if consume_flag(FLAG_QUIT):
            print("Quitting.")
            break

        rgb = picam2.capture_array()
        cur_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # handle commands
        if consume_flag(FLAG_CLEAR_REF):
            ref_bgr = None
            ref_g = None
            ref_bright = None
            print("Reference cleared.")

        if consume_flag(FLAG_SET_REF):
            ref_bgr = cur_bgr.copy()
            ref_g = preprocess_gray(ref_bgr)

            # record brightest square in reference + draw it on ref_bgr
            ref_bright = brightest_square(ref_bgr, win=BRIGHT_WIN)
            rx, ry, rs, rscore = ref_bright
            cv2.rectangle(ref_bgr, (rx, ry), (rx+rs, ry+rs), (0, 255, 255), 2)
            cv2.putText(ref_bgr, f"ref_bright={rscore:.1f}",
                        (rx, max(0, ry-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

            print("Reference set (brightest square recorded).")

        # compute outputs
        if ref_g is None:
            h, w = cur_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            overlay = cur_bgr.copy()
            changed_pct = 0.0
        else:
            mask, overlay, changed_pct = compute_change(ref_g, cur_bgr)

        # Brightest square on current RAW image + draw on overlay + mask
        cur_bright = brightest_square(cur_bgr, win=BRIGHT_WIN)
        bx, by, bs, bscore = cur_bright

        cv2.rectangle(overlay, (bx, by), (bx+bs, by+bs), (0, 255, 255), 2)
        cv2.putText(overlay, f"bright={bscore:.1f}",
                    (bx, max(0, by-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(mask, (bx, by), (bx+bs, by+bs), 255, 2)

        # save on command
        if consume_flag(FLAG_SAVE):
            if ref_bgr is None:
                print("⚠️ Cannot save: reference not set. Run: touch", FLAG_SET_REF)
            else:
                save_bundle(SAVE_ROOT_DIR, ref_bgr, cur_bgr, mask, overlay, changed_pct, ref_bright, cur_bright)

        # stream outputs (GUI)
        if SHOW_GUI:
            cv2.imshow("camera (raw)", cur_bgr)
            cv2.imshow("change overlay", overlay)
            cv2.imshow("change mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit (keyboard).")
                break
            elif key == ord('r'):
                ref_bgr = cur_bgr.copy()
                ref_g = preprocess_gray(ref_bgr)

                ref_bright = brightest_square(ref_bgr, win=BRIGHT_WIN)
                rx, ry, rs, rscore = ref_bright
                cv2.rectangle(ref_bgr, (rx, ry), (rx+rs, ry+rs), (0, 255, 255), 2)
                cv2.putText(ref_bgr, f"ref_bright={rscore:.1f}",
                            (rx, max(0, ry-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)

                print("Reference set (keyboard, brightest recorded).")
            elif key == ord('c'):
                ref_bgr = None
                ref_g = None
                ref_bright = None
                print("Reference cleared (keyboard).")
            elif key == ord('s'):
                if ref_bgr is None:
                    print("⚠️ Cannot save: reference not set.")
                else:
                    save_bundle(SAVE_ROOT_DIR, ref_bgr, cur_bgr, mask, overlay, changed_pct, ref_bright, cur_bright)

        # stream outputs (headless-friendly files)
        frame_i += 1
        if frame_i % LIVE_WRITE_EVERY_N_FRAMES == 0:
            write_live(ref_g is not None, cur_bgr, mask, overlay, changed_pct)

        # small pacing (optional)
        time.sleep(0.01)

    picam2.stop()
    if SHOW_GUI:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
