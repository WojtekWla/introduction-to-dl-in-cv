"""Recompute each proposal's IoU with the ground truth and drop the ignore band.

predict.py runs with keep_all=True because 4.3 needs every proposal, so its .npz
includes crops with 0.3 <= IoU < 0.5 - the ambiguous band that training drops.
Those rows carry label 0, so leaving them in inflates the false-positive count
with crops the classifier was never trained to reject.

    python filter_ignore_band.py outputs/transfer_val.npz

Writes outputs/transfer_val_clean.npz with the same keys, ignore band removed.
"""
import sys
from pathlib import Path

import numpy as np

from annotations import parse_boxes
from config import ANNOTATIONS_DIR, NEG_IOU, POS_IOU
from evaluate_proposals import compute_IoU

IN_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/transfer_val.npz")

data = np.load(IN_PATH)
boxes = data["boxes"]
image_idx = data["image_idx"]
image_names = data["image_names"]
scores = data["scores"]
labels = data["labels"]

# ---- recompute max IoU per proposal, image by image ----------------------
max_iou = np.zeros(len(boxes), dtype=np.float64)

for i, name in enumerate(image_names):
    stem = Path(str(name)).stem
    gt = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1
                   in parse_boxes(ANNOTATIONS_DIR / f"{stem}.xml")], dtype=np.float64)

    rows = image_idx == i
    if len(gt) == 0:          # image with no annotated pothole
        continue              # every proposal stays at IoU 0 -> background
    max_iou[rows] = compute_IoU(boxes[rows].astype(np.float64), gt).max(axis=1)

# ---- apply the same rule training used ------------------------------------
keep = (max_iou >= POS_IOU) | (max_iou < NEG_IOU)
dropped = (~keep).sum()

print(f"{IN_PATH}")
print(f"  proposals        {len(boxes)}")
print(f"  ignore band      {dropped}  ({dropped/len(boxes):.2%})  "
       f"{NEG_IOU} <= IoU < {POS_IOU}")
print(f"  kept             {keep.sum()}")

# sanity: the recomputed labels must match the ones in the file
recomputed = (max_iou >= POS_IOU).astype(labels.dtype)
mismatch = (recomputed[keep] != labels[keep]).sum()
print(f"  label mismatch   {mismatch}   (must be 0)")

out_path = IN_PATH.with_name(IN_PATH.stem + "_clean.npz")
np.savez_compressed(
    out_path,
    image_idx=image_idx[keep],
    boxes=boxes[keep],
    scores=scores[keep],
    labels=labels[keep],
    max_iou=max_iou[keep],
    image_names=image_names,
)
print(f"\nwrote {out_path}")

pos = int(labels[keep].sum())
neg = int(keep.sum()) - pos
print(f"  pothole {pos}, background {neg}  "
      f"(always-background accuracy = {neg/(pos+neg):.4f})")
