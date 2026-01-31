"""
Reads IMU acceleration. If a "shake" is detected, pauses, captures a photo,
saves it with a descriptive filename, and optionally git-pushes it.
"""

# AUTHOR:
# DATE:

import time
import os
import math
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from git import Repo
from picamera2 import Picamera2

# -------------------
# VARIABLES (EDIT ME)
# -------------------

# Shake threshold uses CHANGE in acceleration magnitude (m/s^2).
# Start around 3–6 for gentle shakes; 8–12 for harder shakes.
THRESHOLD = 6.0

# Prevent taking many photos from one shake
COOLDOWN_SEC = 2.0

# How long to wait after shake before capturing (lets rig settle)
SETTLE_SEC = 0.5

REPO_PATH = "/home/firefox33/build-a-cubesat-firefox"  # your repo root
FOLDER_PATH = "Images"  # folder INSIDE repo (no leading slash)
NAME = "ElijahD"        # FirstNameLastInitial

# -------------------
# Init IMU + Camera
# -------------------
i2c = board.I2C()
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)  # not used for shake detect, but OK to init

picam2 = Picamera2()


def git_push():
    """
    Stages, commits, and pushes new images to your GitHub repo.
    """
    try:
        repo = Repo(REPO_PATH)
        origin = repo.remote('origin')
        origin.pull()
        repo.git.add(os.path.join(REPO_PATH, FOLDER_PATH))
        repo.index.commit('New Photo')
        origin.push()
        print("pushed changes to git")
    except Exception as e:
        print(f"⚠️ Couldn't upload to git: {e}")


def img_gen(name: str) -> str:
    """
    Generates a new image name like: REPO_PATH/Images/Name_HHMMSS.jpg
    """
    t = time.strftime("_%H%M%S")
    img_dir = os.path.join(REPO_PATH, FOLDER_PATH)
    os.makedirs(img_dir, exist_ok=True)
    return os.path.join(img_dir, f"{name}{t}.jpg")


def accel_mag(ax, ay, az) -> float:
    """Acceleration magnitude in m/s^2."""
    return math.sqrt(ax*ax + ay*ay + az*az)


def take_photo():
    # Configure camera ONCE
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let auto exposure settle

    print("Camera started.")
    print("Mag sample (FYI):", mag.magnetic)

    prev_mag = None
    last_trigger_time = 0.0

    while True:
        ax, ay, az = accel_gyro.acceleration  # m/s^2

        m = accel_mag(ax, ay, az)
        jerk = 0.0 if prev_mag is None else abs(m - prev_mag)
        prev_mag = m

        now = time.time()

        # Trigger when change in acceleration is large AND we aren't in cooldown
        if jerk > THRESHOLD and (now - last_trigger_time) > COOLDOWN_SEC:
            last_trigger_time = now
            print(f"🚨 Shake detected! jerk={jerk:.2f} m/s^2  (ax,ay,az)=({ax:.2f},{ay:.2f},{az:.2f})")

            # Pause to settle, then capture
            time.sleep(SETTLE_SEC)

            img_path = img_gen(NAME)
            picam2.capture_file(img_path)
            print(f"📸 Photo saved: {img_path}")

            # Optional: push to GitHub
            git_push()

        else:
            # Debug print (comment out once it works)
            print(f"still... jerk={jerk:.2f} mag={m:.2f}  ax={ax:.2f} ay={ay:.2f} az={az:.2f}")

        time.sleep(0.05)  # 20 Hz loop


def main():
    take_photo()


if __name__ == '__main__':
    main()
