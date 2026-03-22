#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np

from mission import landing_zone_box, scale_box


WIDTH = 4608
HEIGHT = 2592
OUTPUT_DIR = Path("landing_box_test_output")


def build_test_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

    # Add hazards across most of the field.
    rng = np.random.default_rng(42)
    for _ in range(250):
        x = int(rng.integers(0, WIDTH))
        y = int(rng.integers(0, HEIGHT))
        r = int(rng.integers(8, 30))
        cv2.circle(image, (x, y), r, (0, 0, 0), -1)

    for _ in range(45):
        x1 = int(rng.integers(0, WIDTH - 220))
        y1 = int(rng.integers(0, HEIGHT - 220))
        w = int(rng.integers(60, 220))
        h = int(rng.integers(60, 220))
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 0, 0), -1)

    # Force an obvious off-center safe zone in the lower-right quadrant.
    safe_x1, safe_y1 = 2850, 1350
    safe_x2, safe_y2 = 4250, 2350
    cv2.rectangle(image, (safe_x1, safe_y1), (safe_x2, safe_y2), (255, 255, 255), -1)

    return image, (safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1)


def build_dummy_bm() -> np.ndarray:
    bm = np.zeros((540, 960), dtype=np.uint8)
    cv2.rectangle(bm, (590, 285), (820, 455), 255, -1)
    return bm


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image, expected_box = build_test_image()
    box = landing_zone_box(image)
    x, y, w, h = box

    boxed = image.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 12)

    expected_boxed = image.copy()
    ex, ey, ew, eh = expected_box
    cv2.rectangle(expected_boxed, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 12)

    bm = build_dummy_bm()
    bm_box = scale_box(box, image.shape[:2], bm.shape[:2])
    bx, by, bw, bh = bm_box
    bm_boxed = cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bm_boxed, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    cv2.imwrite(str(OUTPUT_DIR / "landing_input.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / "landing_boxed.png"), boxed)
    cv2.imwrite(str(OUTPUT_DIR / "landing_expected_boxed.png"), expected_boxed)
    cv2.imwrite(str(OUTPUT_DIR / "bm_boxed.png"), bm_boxed)

    summary = {
        "image_size": {"width": WIDTH, "height": HEIGHT},
        "landing_box": {"x": x, "y": y, "width": w, "height": h},
        "expected_box": {"x": ex, "y": ey, "width": ew, "height": eh},
        "bm_size": {"width": bm.shape[1], "height": bm.shape[0]},
        "bm_box": {"x": bx, "y": by, "width": bw, "height": bh},
        "note": "The raw input image contains no guide rectangle. The intended landing zone is the large white region in the lower-right quadrant.",
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
