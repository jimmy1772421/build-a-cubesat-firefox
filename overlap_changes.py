#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required to run overlap_changes.py") from exc


REPO_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = REPO_ROOT / "Images"
OUTPUT_DIR = IMAGES_DIR / "combined"


def git_pull() -> None:
    result = subprocess.run(
        ["git", "pull", "--rebase"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git pull failed"
        raise RuntimeError(message)


def discover_mask_entries() -> list[dict]:
    entries: list[dict] = []

    for pass_dir in sorted(IMAGES_DIR.glob("P[0-9][0-9][0-9]")):
        mask_path = pass_dir / f"BM-{pass_dir.name}.png"
        if not mask_path.exists():
            continue
        base_path = pass_dir / "current.png"
        if not base_path.exists():
            base_path = pass_dir / "reference.png"
        entries.append(
            {
                "id": pass_dir.name,
                "mask": mask_path,
                "base": base_path if base_path.exists() else None,
                "source": "pass",
            }
        )

    if entries:
        return entries

    for stamp_dir in sorted(IMAGES_DIR.glob("20*")):
        if not stamp_dir.is_dir():
            continue
        mask_path = stamp_dir / "mask.png"
        if not mask_path.exists():
            continue
        base_path = stamp_dir / "cur_fullres.png"
        if not base_path.exists():
            base_path = stamp_dir / "cur.png"
        entries.append(
            {
                "id": stamp_dir.name,
                "mask": mask_path,
                "base": base_path if base_path.exists() else None,
                "source": "timestamp",
            }
        )

    return entries


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.array(image, dtype=np.uint8)


def build_outputs(entries: list[dict]) -> dict:
    latest = entries[-1]
    if latest["base"] is not None:
        base_image = load_image(latest["base"])
    else:
        first_mask = Image.open(latest["mask"]).convert("L")
        base_image = Image.merge("RGBA", (first_mask, first_mask, first_mask, Image.new("L", first_mask.size, 255)))

    size = base_image.size
    count_map = np.zeros((size[1], size[0]), dtype=np.uint16)

    for entry in entries:
        mask = load_mask(entry["mask"], size=size)
        count_map += (mask > 0).astype(np.uint16)

    combined_mask = np.where(count_map > 0, 255, 0).astype(np.uint8)
    max_count = int(count_map.max())
    if max_count == 0:
        heat = np.zeros_like(combined_mask)
    else:
        heat = np.round((count_map / max_count) * 255.0).astype(np.uint8)

    base_array = np.array(base_image, dtype=np.uint8)
    overlay = base_array.copy()
    changed = count_map > 0
    if changed.any():
        # Red overlay with preserved image detail underneath.
        overlay[changed, 0] = np.clip(overlay[changed, 0] * 0.35, 0, 255).astype(np.uint8)
        overlay[changed, 1] = np.clip(overlay[changed, 1] * 0.35, 0, 255).astype(np.uint8)
        overlay[changed, 2] = 255

    return {
        "base": base_image,
        "combined_mask": Image.fromarray(combined_mask, mode="L"),
        "count_map": Image.fromarray(heat, mode="L"),
        "overlay": Image.fromarray(overlay, mode="RGBA"),
        "max_count": max_count,
        "size": {"width": size[0], "height": size[1]},
    }


def save_outputs(entries: list[dict], outputs: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs["combined_mask"].save(OUTPUT_DIR / "combined_mask.png")
    outputs["count_map"].save(OUTPUT_DIR / "combined_count.png")
    outputs["overlay"].save(OUTPUT_DIR / "combined_overlay.png")

    summary = {
        "entries_used": [
            {
                "id": entry["id"],
                "mask": str(entry["mask"].relative_to(REPO_ROOT)),
                "base": str(entry["base"].relative_to(REPO_ROOT)) if entry["base"] is not None else None,
                "source": entry["source"],
            }
            for entry in entries
        ],
        "entry_count": len(entries),
        "max_overlap_count": outputs["max_count"],
        "image_size": outputs["size"],
        "outputs": {
            "combined_mask": str((OUTPUT_DIR / "combined_mask.png").relative_to(REPO_ROOT)),
            "combined_count": str((OUTPUT_DIR / "combined_count.png").relative_to(REPO_ROOT)),
            "combined_overlay": str((OUTPUT_DIR / "combined_overlay.png").relative_to(REPO_ROOT)),
        },
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull the repo and overlap all mission change maps")
    parser.add_argument("--no-pull", action="store_true", help="Skip git pull before processing")
    args = parser.parse_args(argv)

    if not args.no_pull:
        git_pull()

    entries = discover_mask_entries()
    if not entries:
        raise SystemExit("No pass binary maps or timestamp masks were found under Images/")

    outputs = build_outputs(entries)
    save_outputs(entries, outputs)

    print(f"Combined {len(entries)} change maps into {OUTPUT_DIR}")
    print(f"Max overlap count: {outputs['max_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
