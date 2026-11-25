"""Recall vs. number of proposals N on the train split, for several proposal sets.

Writes one CSV row per N with a recall column per proposal set, so the curves
can be plotted (plot_recall.py) and the elbow chosen. numpy only, so it runs in
the HPC venv.

Usage:
    python recall_curves.py --out recall_curves.csv \
        baseline=../data/potholes/proposals quality_full=../data/potholes/proposals_quality_full
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_proposals import ANNOTATIONS_DIR, IOU_THRESHOLD, SPLITS_FILE, compute_IoU
from groundtruth import parse_boxes

MAX_N = 3000
STEP = 10


def first_hit_per_gt(proposals_dir: Path, image_names: list) -> np.ndarray:
    """For every train GT box: index of the first proposal with IoU >= threshold (inf if none).

    A GT box counts as found for N proposals exactly when its first hit is < N,
    so one pass gives the whole recall curve.
    """
    first_hits = []
    for name in image_names:
        stem = Path(name).stem
        boxes = parse_boxes(ANNOTATIONS_DIR / f"{stem}.xml")
        gt = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1 in boxes])
        hits = compute_IoU(np.load(proposals_dir / f"{stem}.npy"), gt) >= IOU_THRESHOLD
        first = np.where(hits.any(axis=0), hits.argmax(axis=0), np.inf)
        first_hits.append(first)
    return np.concatenate(first_hits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sets", nargs="+", help="label=proposals_dir")
    parser.add_argument("--out", type=Path, default=Path("recall_curves.csv"))
    args = parser.parse_args()

    train = json.loads(SPLITS_FILE.read_text())["train"]
    ns = np.arange(STEP, MAX_N + 1, STEP)

    columns = {}
    for spec in args.sets:
        label, directory = spec.split("=", 1)
        hits = first_hit_per_gt(Path(directory), train)
        columns[label] = [(hits < n).mean() for n in ns]
        print(f"{label}: {len(hits)} GT boxes, recall@all {np.isfinite(hits).mean():.3f}")

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", *columns])
        for i, n in enumerate(ns):
            writer.writerow([int(n), *(f"{columns[label][i]:.4f}" for label in columns)])
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
