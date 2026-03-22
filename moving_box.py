#!/usr/bin/env python3
import cv2
import numpy as np


WINDOW_NAME = "moving_box"
WIDTH = 1280
HEIGHT = 720
BOX_W = 220
BOX_H = 200
DX = 7
DY = 5


def main() -> None:
    x = 120
    y = 90
    dx = DX
    dy = DY

    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        x2 = x + BOX_W
        y2 = y + BOX_H
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            break

        x += dx
        y += dy

        if x <= 0 or x + BOX_W >= WIDTH:
            dx *= -1
            x = max(0, min(x, WIDTH - BOX_W))

        if y <= 0 or y + BOX_H >= HEIGHT:
            dy *= -1
            y = max(0, min(y, HEIGHT - BOX_H))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
