"""
The Python code you will write for this module should read
acceleration data from the IMU. When a reading comes in that surpasses
an acceleration threshold (indicating a shake), your Pi should pause,
trigger the camera to take a picture, then save the image with a
descriptive filename. You may use GitHub to upload your images automatically,
but for this activity it is not required.

The provided functions are only for reference, you do not need to use them. 
You will need to complete the take_photo() function and configure the VARIABLES section
"""

#AUTHOR: 
#DATE:

#import libraries
import time
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from git import Repo
from picamera2 import Picamera2
import os

#VARIABLES
THRESHOLD = 15      #Any desired value from the accelerometer
REPO_PATH = "/home/firefox33/build-a-cubesat-firefox"     #Your github repo path: ex. /home/pi/FlatSatChallenge
FOLDER_PATH = "Images"   #Your image folder path in your GitHub repo: ex. /Images

#imu and camera initialization
i2c = board.I2C()
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)
picam2 = Picamera2()


def git_push():
    """
    Stages, commits, and pushes new images to your GitHub repo.
    """
    try:
        print(f"[git_push] Using REPO_PATH={REPO_PATH}")
        print(f"[git_push] Using FOLDER_PATH={FOLDER_PATH}")

        repo = Repo(REPO_PATH)
        origin = repo.remote('origin')
        print('[git_push] Found remote:', origin.name)

        origin.pull()
        print('[git_push] Pulled changes')

        folder_full = os.path.join(REPO_PATH, FOLDER_PATH)
        print('[git_push] Adding folder:', folder_full)
        repo.git.add(folder_full)

        if repo.is_dirty():
            repo.index.commit('New Photo')
            print('[git_push] Made the commit')
            origin.push()
            print('[git_push] Pushed changes')
        else:
            print('[git_push] No changes to commit')

    except Exception as e:
        print("Couldn't upload to git. Error was:")
        print(repr(e))


def img_gen(name):
    """
    This function is complete. Generates a new image name.

    Parameters:
        name (str): your name ex. MasonM
    """
    t = time.strftime("_%H%M%S")
    imgname = (f'{REPO_PATH}/{FOLDER_PATH}/{name}{t}.jpg')
    return imgname


def take_photo():
    """
    This function is NOT complete. Takes a photo when the FlatSat is shaken.
    Replace psuedocode with your own code.
    """
    config =picam2.create_still_configuration()
    picam2.configure(config)
    while True:
        accelx, accely, accelz = accel_gyro.acceleration

        #CHECKS IF READINGS ARE ABOVE THRESHOLD
        if accelx > THRESHOLD or accelz > THRESHOLD or accely > THRESHOLD:
            #PAUSE
            print("above threshold -taking photo") 
            time.sleep(0.1)
            picam2.start() 
            time.sleep(0.5)
            name = "ElijahD"     #First Name, Last Initial  ex. MasonM
            #TAKE PHOT
            img_path = img_gen(name)
            picam2.capture_file(img_path) 
            print("photo taken")
            #PUSH PHOTO TO GITHUB
            git_push()
            print("photo push") 
            time.sleep(0.2) 
        
        #debugging code
        print(f"nothing is happening {accelx}. {accely} {accelz}")

        time.sleep(0.2)          
        
        #PAUSE


def main():
    take_photo()


if __name__ == '__main__':
    main()