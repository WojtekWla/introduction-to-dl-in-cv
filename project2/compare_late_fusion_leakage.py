"""
Runs the LateFusion model from fusion.py on both the leaky (ufc10) and
no-leakage (ucf101_noleakage) datasets, saving loss/accuracy curves and a
test-set confusion matrix for each run.

The model classes are duplicated from fusion.py (not imported) because
fusion.py builds its dataset loaders unconditionally at module import time
using a hardcoded local path, which would crash on import here.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import FrameVideoDataset

DATASETS = [
    ("/dtu/datasets1/02516/ufc10", "leakage"),
    ("/dtu/datasets1/02516/ucf101_noleakage", "noleakage"),
]

BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
SEED = 0

project_dir = Path(__file__).resolve().parent
checkpoint_dir = project_dir / "checkpoints"
plots_dir = project_dir / "plots"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)


# ---- model (copied from fusion.py) ----

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


def _stage(in_ch, out_ch, stride):
    return nn.Sequential(
        ResidualBlock(in_ch, out_ch, stride=stride),
        ResidualBlock(out_ch, out_ch, stride=1),
    )


class Encoder(nn.Module):
    def __init__(self, in_ch, feature_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = _stage(32, 64, stride=2)
        self.layer2 = _stage(64, 128, stride=2)
        self.layer3 = _stage(128, 256, stride=2)
        self.layer4 = _stage(256, feature_dim, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


class LateFusion(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super().__init__()
        self.encoder = Encoder(in_ch=3, feature_dim=feature_dim)
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, self.feature_dim).mean(1)
        return self.classifier(x)


# ---- train / eval ----

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(clips)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.inference_mode()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        logits = model(clips)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def get_class_names(root_dir, split="train"):
    df = pd.read_csv(Path(root_dir) / "metadata" / f"{split}.csv")
    return df.drop_duplicates("label").sort_values("label")["action"].tolist()


def compute_confusion_matrix(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history, title_prefix, save_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"{title_prefix} loss")
    plt.savefig(plots_dir / f"{save_prefix}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title(f"{title_prefix} accuracy")
    plt.savefig(plots_dir / f"{save_prefix}_acc.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment(data_root, tag, device, epochs=EPOCHS):
    print(f"\n==== Running LateFusion on '{tag}' ({data_root}) ====")
    set_seed(SEED)

    train_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    train_dataset = FrameVideoDataset(root_dir=data_root, split="train", transform=train_tf, stack_frames=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                               pin_memory=True, persistent_workers=True)

    val_dataset = FrameVideoDataset(root_dir=data_root, split="val", transform=eval_tf, stack_frames=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                             pin_memory=True, persistent_workers=True)

    test_dataset = FrameVideoDataset(root_dir=data_root, split="test", transform=eval_tf, stack_frames=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                              pin_memory=True, persistent_workers=True)

    model = LateFusion(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = checkpoint_dir / f"latefusion_{tag}_best.pt"
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_accuracy = -float("inf")

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_accuracy
        print(f"[{tag}] epoch {epoch + 1:3d}/{epochs} | train loss {train_loss:.4f} acc {train_acc * 100:.2f}% "
              f"| val loss {val_loss:.4f} acc {val_acc * 100:.2f}% | {epoch_time:.1f}s"
              f"{'  *best*' if is_best else ''}")

        if is_best:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "seed": SEED,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[{tag}] TEST | loss {test_loss:.4f} acc {test_acc * 100:.2f}%")

    plot_curves(history, title_prefix=f"LateFusion ({tag})", save_prefix=f"latefusion_{tag}")

    class_names = get_class_names(data_root, split="train")
    preds, labels = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(preds, labels, num_classes=len(class_names))
    plot_confusion_matrix(
        cm, class_names,
        title=f"LateFusion ({tag}) confusion matrix - test",
        save_path=plots_dir / f"latefusion_{tag}_confusion_matrix.png",
    )

    return {"tag": tag, "best_val_accuracy": best_val_accuracy, "test_loss": test_loss, "test_acc": test_acc}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    results = [run_experiment(data_root, tag, device) for data_root, tag in DATASETS]

    print("\n==== Summary: leakage vs no-leakage (LateFusion) ====")
    for r in results:
        print(f"{r['tag']:>10s} | best val acc {r['best_val_accuracy'] * 100:.2f}% "
              f"| test loss {r['test_loss']:.4f} | test acc {r['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
