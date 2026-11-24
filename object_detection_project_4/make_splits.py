"""Create a fixed train/val/test split of the Potholes dataset.

The dataset ships without a splits.json (despite the task PDF), so we make our
own once, with a fixed seed, and commit the result. Every later step (proposal
evaluation, training, AP on test) reads the same file, so no test image can
leak into decisions made on train/val.

Standard library only, so it also runs with the HPC's system python3 (3.9).

Usage:
    python make_splits.py [--data-root DIR] [--out FILE] [--seed N]
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/dtu/datasets1/02516/potholes")
FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
DEFAULT_SEED = 0


def list_images(data_root: Path) -> List[str]:
    """Image file names that have both an image and an annotation file."""
    images = {p.stem for p in (data_root / "images").glob("*.png")}
    annotations = {p.stem for p in (data_root / "annotations").glob("*.xml")}
    missing = images ^ annotations
    if missing:
        raise ValueError(f"Images without annotation or vice versa: {sorted(missing)}")
    return [f"{stem}.png" for stem in images]


def make_splits(names: List[str], seed: int) -> Dict[str, List[str]]:
    """Shuffle deterministically and cut into train/val/test by FRACTIONS."""
    # Sort first: directory listing order differs between machines, and the
    # shuffle is only reproducible if it always starts from the same order.
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)

    n_test = round(len(shuffled) * FRACTIONS["test"])
    n_val = round(len(shuffled) * FRACTIONS["val"])
    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", type=Path, help="default: splits.json next to this script")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    out = args.out or Path(__file__).resolve().parent / "splits.json"

    names = list_images(args.data_root)
    if not names:
        raise SystemExit(f"No images found in {args.data_root / 'images'}")

    splits = make_splits(names, args.seed)
    splits_with_seed = {"seed": args.seed, **splits}
    out.write_text(json.dumps(splits_with_seed, indent=2) + "\n")

    sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    print(f"{len(names)} images -> {sizes}. Written to {out}")


if __name__ == "__main__":
    main()
