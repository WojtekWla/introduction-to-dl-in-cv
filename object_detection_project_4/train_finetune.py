"""Stage 2: fine-tune the ResNet18 - unfreeze layer4 and continue from stage 1.

    python train_finetune.py

Starts from checkpoints/transfer.pt (the frozen-backbone head), unfreezes the
last residual block, and trains at a 10x lower learning rate. This is the
"finetune the network" of task 4.2.3 - stage 1 only trained a new head on top
of features that never moved.

Saves checkpoints/finetune.pt and finetune_history.csv
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PotholeProposals, balanced_sampler, make_transform
from models import PotholeResNet18

EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-4              # 10x lower than stage 1: adjusting pretrained weights
SIZE = 224
IMAGENET_NORM = True
RESUME = "checkpoints/transfer.pt"

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path("checkpoints").mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
train_set = PotholeProposals(
    "train",
    make_transform(train=True, size=SIZE, imagenet_norm=IMAGENET_NORM),
    add_gt_boxes=True,
)
val_set = PotholeProposals(
    "val",
    make_transform(train=False, size=SIZE, imagenet_norm=IMAGENET_NORM),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          sampler=balanced_sampler(train_set, 0.25), num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"device {device}")
print(f"train  {len(train_set)} crops, {train_set.class_counts()}")
print(f"val    {len(val_set)} crops, {val_set.class_counts()}")

# ---------------------------------------------------------------- model
model = PotholeResNet18()

checkpoint = torch.load(RESUME, map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
print(f"resumed {RESUME}  (epoch {checkpoint['epoch']}, F1 {checkpoint['f1']:.4f})")

model.unfreeze_layer4()          # the whole point of this script
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable now {trainable/1e6:.2f}M  (was {156004/1e6:.2f}M frozen)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# ---------------------------------------------------------------- training
history = []
best_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    start = time.perf_counter()

    model.train()
    train_loss = 0.0
    seen = 0
    for crops, labels, boxes, image_idx in train_loader:
        crops, labels = crops.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(crops)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
    train_loss = train_loss / seen

    model.eval()
    correct = total = tp = pred_pos = real_pos = 0
    with torch.no_grad():
        for crops, labels, boxes, image_idx in val_loader:
            crops, labels = crops.to(device), labels.to(device)
            preds = model(crops).argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            pred_pos += (preds == 1).sum().item()
            real_pos += (labels == 1).sum().item()

    accuracy = correct / total
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / real_pos if real_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    seconds = round(time.perf_counter() - start, 1)

    print(f"epoch {epoch:>2}/{EPOCHS}  loss {train_loss:.4f}  "
          f"val acc {accuracy:.4f}  P {precision:.3f}  R {recall:.3f}  "
          f"F1 {f1:.4f}  [{seconds}s]")

    history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": accuracy,
                    "val_precision": precision, "val_recall": recall, "val_f1": f1,
                    "seconds": seconds})

    if f1 > best_f1:
        best_f1 = f1
        torch.save({"model_state": model.state_dict(), "model": "transfer",
                    "size": SIZE, "imagenet_norm": IMAGENET_NORM,
                    "epoch": epoch, "f1": f1}, "checkpoints/finetune.pt")
        print(f"         saved checkpoints/finetune.pt")

with open("checkpoints/finetune_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)

print(f"\nbest val pothole F1: {best_f1:.4f}")
