#!/usr/bin/env python3
import random
from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 4608
HEIGHT = 2592
SEED = 42
OUTPUT_PATH = Path("landing_test.png")
OBVIOUS_OUTPUT_PATH = Path("landing_test_obvious.png")


def main() -> None:
    random.seed(SEED)

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for _ in range(140):
        r = random.randint(6, 18)
        x = random.randint(r, WIDTH - r)
        y = random.randint(r, HEIGHT - r)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

    for _ in range(28):
        w = random.randint(40, 140)
        h = random.randint(40, 140)
        x = random.randint(0, WIDTH - w)
        y = random.randint(0, HEIGHT - h)
        draw.ellipse((x, y, x + w, y + h), fill=(0, 0, 0))

    for _ in range(10):
        w = random.randint(140, 320)
        h = random.randint(140, 320)
        x = random.randint(0, WIDTH - w)
        y = random.randint(0, HEIGHT - h)
        draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))

    corridor_margin = 180
    draw.rectangle((WIDTH // 2 - 260, corridor_margin, WIDTH // 2 + 260, HEIGHT - corridor_margin), fill=(255, 255, 255))

    image.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH.resolve()}")

    obvious = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(obvious)

    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(255, 255, 255))

    for _ in range(220):
        r = random.randint(14, 34)
        x = random.randint(r, WIDTH - r)
        y = random.randint(r, HEIGHT - r)
        if WIDTH // 2 - 700 < x < WIDTH // 2 + 700 and HEIGHT // 2 - 520 < y < HEIGHT // 2 + 520:
            continue
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

    for _ in range(60):
        w = random.randint(80, 260)
        h = random.randint(80, 260)
        x = random.randint(0, WIDTH - w)
        y = random.randint(0, HEIGHT - h)
        if WIDTH // 2 - 760 < x < WIDTH // 2 + 520 and HEIGHT // 2 - 560 < y < HEIGHT // 2 + 480:
            continue
        draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))

    draw.rounded_rectangle(
        (WIDTH // 2 - 620, HEIGHT // 2 - 470, WIDTH // 2 + 620, HEIGHT // 2 + 470),
        radius=120,
        fill=(255, 255, 255),
    )

    obvious.save(OBVIOUS_OUTPUT_PATH)
    print(f"Saved {OBVIOUS_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
