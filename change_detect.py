
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

def consume_flag(path: str) -> bool:
    """Return True if file exists, and remove it (one-shot command)."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return True
    return False

def save_bundle(save_root: str, ref_bgr, cur_bgr, mask, overlay, changed_pct: float):
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
    print(f"Live outputs: {LIVE_DIR}/overlay.jpg  {LIVE_DIR}/mask.png  {LIVE_DIR}/status.txt\n")

    ref_bgr = None
    ref_g = None

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
            print("Reference cleared.")

        if consume_flag(FLAG_SET_REF):
            ref_bgr = cur_bgr.copy()
            ref_g = preprocess_gray(ref_bgr)
            print("Reference set.")

        # compute outputs
        if ref_g is None:
            h, w = cur_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            overlay = cur_bgr
            changed_pct = 0.0
        else:
            mask, overlay, changed_pct = compute_change(ref_g, cur_bgr)

        # save on command
        if consume_flag(FLAG_SAVE):
            if ref_bgr is None:
                print("⚠️ Cannot save: reference not set. Run: touch", FLAG_SET_REF)
            else:
                save_bundle(SAVE_ROOT_DIR, ref_bgr, cur_bgr, mask, overlay, changed_pct)

        # stream outputs (GUI)
        if SHOW_GUI:
            cv2.imshow("camera (raw)", cur_bgr)
            cv2.imshow("change overlay", overlay)
            cv2.imshow("change mask", mask)
            # also allow keyboard control if you want
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit (keyboard).")
                break
            elif key == ord('r'):
                ref_bgr = cur_bgr.copy()
                ref_g = preprocess_gray(ref_bgr)
                print("Reference set (keyboard).")
            elif key == ord('c'):
                ref_bgr = None
                ref_g = None
                print("Reference cleared (keyboard).")
            elif key == ord('s'):
                if ref_bgr is None:
                    print("⚠️ Cannot save: reference not set.")
                else:
                    save_bundle(SAVE_ROOT_DIR, ref_bgr, cur_bgr, mask, overlay, changed_pct)

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
