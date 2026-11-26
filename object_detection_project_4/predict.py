"""Score every proposal of a split with a trained model.

    python predict.py checkpoints/transfer.pt test
    python predict.py checkpoints/scratch.pt val

Saves outputs/<checkpoint>_<split>.npz, one row per proposal:
    image_idx    (N,)   index into image_names
    boxes        (N,4)  xmin, ymin, xmax, ymax in original image pixels
    scores       (N,)   P(pothole), between 0 and 1
    labels       (N,)   IoU-derived label, for checking only
    image_names  (M,)   file name of each image

This is the handover to 4.3. It does not threshold, suppress or evaluate
anything - NMS and AP do that.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PotholeProposals, make_transform
from models import PotholeCNN, PotholeResNet18

CHECKPOINT = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/transfer.pt")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- model
# size and imagenet_norm come from the checkpoint, not from this file:
# scoring crops differently than training did silently ruins the scores.
checkpoint = torch.load(CHECKPOINT, map_location=device)
size = checkpoint["size"]
imagenet_norm = checkpoint["imagenet_norm"]

if checkpoint["model"] == "scratch":
    model = PotholeCNN()
else:
    model = PotholeResNet18()

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# ---------------------------------------------------------------- data
# keep_all=True: every proposal is scored, including the 0.3-0.5 IoU band that
# training drops. NMS needs the complete set - a box it never saw cannot be
# suppressed, and a box never scored cannot be detected.
dataset = PotholeProposals(
    SPLIT,
    make_transform(train=False, size=size, imagenet_norm=imagenet_norm),
    keep_all=True,
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"checkpoint {CHECKPOINT}  (epoch {checkpoint['epoch']}, F1 {checkpoint['f1']:.4f})")
print(f"split      {SPLIT}: {len(dataset.image_names)} images, {len(dataset)} proposals")
print(f"input      {size}x{size}  imagenet_norm={imagenet_norm}")

# ---------------------------------------------------------------- scoring
all_scores = []
all_boxes = []
all_image_idx = []
all_labels = []

with torch.no_grad():
    for crops, labels, boxes, image_idx in loader:
        crops = crops.to(device)
        logits = model(crops)

        # column 1 is P(pothole). Keep the probability, not argmax: 4.3 needs
        # a continuous score to rank detections and sweep the threshold.
        scores = torch.softmax(logits, dim=1)[:, 1]

        all_scores.append(scores.cpu().numpy())
        all_boxes.append(boxes.numpy())
        all_image_idx.append(image_idx.numpy())
        all_labels.append(labels.numpy())

scores = np.concatenate(all_scores)
boxes = np.concatenate(all_boxes)
image_idx = np.concatenate(all_image_idx)
labels = np.concatenate(all_labels)

# ---------------------------------------------------------------- save
Path("outputs").mkdir(exist_ok=True)
out_path = Path("outputs") / f"{CHECKPOINT.stem}_{SPLIT}.npz"
np.savez_compressed(
    out_path,
    image_idx=image_idx,
    boxes=boxes,
    scores=scores,
    labels=labels,
    image_names=np.array(dataset.image_names),
)

print(f"\nwrote {out_path}")
print(f"scores: min {scores.min():.3f}  median {np.median(scores):.3f}  "
      f"max {scores.max():.3f}")
print(f"above 0.5: {(scores > 0.5).sum()} of {len(scores)} proposals")