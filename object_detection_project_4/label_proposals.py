"""Task 4.1.4: give every proposal its best IoU with any ground-truth pothole.

For each image (all splits) this saves labels/<stem>.npz with
    boxes   (P, 4)  all proposals, same order as the .npy file
    max_iou (P,)    best IoU of each proposal with any GT box
The 0/1 label is NOT stored: the dataloader applies the thresholds, so they can
change without rerunning this script. Only box coordinates are compared, so no
images are loaded.
"""
import json

import numpy as np

from annotations import parse_boxes
from config import (ANNOTATIONS_DIR, LABELS_DIR, N_PROPOSALS, NEG_IOU, POS_IOU,
                    PROPOSALS_DIR, SPLITS_FILE)
from evaluate_proposals import compute_IoU


def label_image(stem: str) -> np.ndarray:
    """Best IoU with any GT box for each proposal of one image."""
    proposals = np.load(PROPOSALS_DIR / f"{stem}.npy")                          # (P, 4)
    boxes = parse_boxes(ANNOTATIONS_DIR / f"{stem}.xml")
    gt = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1 in boxes])          # (G, 4)

    iou = compute_IoU(proposals, gt)   # (P, G): row = proposal, column = pothole
    max_iou = iou.max(axis=1)          # per proposal: its best pothole match

    np.savez(LABELS_DIR / f"{stem}.npz", boxes=proposals, max_iou=max_iou)
    return max_iou


def main() -> None:
    splits = json.loads(SPLITS_FILE.read_text())
    train = set(splits["train"])
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_ignored = n_done = n_skipped = 0
    for name in splits["train"] + splits["val"] + splits["test"]:
        stem = name[:-4]  # "potholes106.png" -> "potholes106"
        if not (PROPOSALS_DIR / f"{stem}.npy").exists():
            n_skipped += 1  # e.g. locally, where only a few sample images have proposals
            continue
        max_iou = label_image(stem)
        n_done += 1

        if name in train:
            top = max_iou[:N_PROPOSALS]
            n_pos += int((top >= POS_IOU).sum())
            n_neg += int((top < NEG_IOU).sum())
            n_ignored += int(((top >= NEG_IOU) & (top < POS_IOU)).sum())

    print(f"Labelled {n_done} images -> {LABELS_DIR} (skipped {n_skipped} without proposals)")
    total = n_pos + n_neg + n_ignored
    if total:
        print(f"Train, first {N_PROPOSALS} proposals per image ({total} proposals):")
        print(f"  pothole    (IoU >= {POS_IOU}): {n_pos:>7}  ({n_pos / total:.1%})")
        print(f"  ignored    ({NEG_IOU} <= IoU < {POS_IOU}): {n_ignored:>7}  ({n_ignored / total:.1%})")
        print(f"  background (IoU < {NEG_IOU}): {n_neg:>7}  ({n_neg / total:.1%})")


if __name__ == "__main__":
    main()
