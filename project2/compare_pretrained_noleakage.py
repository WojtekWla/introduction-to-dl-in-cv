"""
Runs two transfer-learning models on the no-leakage dataset only:
  - LateFusionPretrained: late fusion over frames using an ImageNet-pretrained
    ResNet18 as the per-frame encoder (mostly frozen, only layer4 + head trained)
  - CNN3DPretrained: a Kinetics-400-pretrained r3d_18 3D CNN (stem/layer1/layer2
    frozen, only layer3/layer4 + head trained)

Saves loss/accuracy curves and a test-set confusion matrix for each model.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.video import r3d_18, R3D_18_Weights

from datasets import FrameVideoDataset

DATA_ROOT = "/dtu/datasets1/02516/ucf101_noleakage"

BATCH_SIZE = 16
SEED = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

project_dir = Path(__file__).resolve().parent
checkpoint_dir = project_dir / "checkpoints"
plots_dir = project_dir / "plots"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)


# ---- models ----

class LateFusionPretrained(nn.Module):
    """LateFusion with an ImageNet-pretrained ResNet18 as the per-frame encoder."""

    def __init__(self, num_classes=10):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # freeze everything except layer4; same "freeze early generic
        # layers" strategy used for the pretrained 3D CNN below
        for name, param in self.encoder.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, self.feature_dim).mean(1)
        return self.classifier(x)


class CNN3DPretrained(nn.Module):
    """3D CNN with a Kinetics-400-pretrained r3d_18 backbone."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        for module in (self.backbone.stem, self.backbone.layer1, self.backbone.layer2):
            for param in module.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        return self.backbone(x)


MODELS = {
    "latefusion_pretrained": {
        "build": lambda: LateFusionPretrained(num_classes=10),
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "lr": 1e-4, "epochs": 30,
    },
    "cnn3d_pretrained": {
        "build": lambda: CNN3DPretrained(num_classes=10),
        "mean": KINETICS_MEAN, "std": KINETICS_STD,
        "lr": 1e-4, "epochs": 30,
    },
}


# ---- train / eval (shared) ----

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


def run_experiment(name, cfg, device):
    print(f"\n==== Running {name} on noleakage ({DATA_ROOT}) ====")
    set_seed(SEED)

    train_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    train_dataset = FrameVideoDataset(root_dir=DATA_ROOT, split="train", transform=train_tf, stack_frames=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                               pin_memory=True, persistent_workers=True)

    val_dataset = FrameVideoDataset(root_dir=DATA_ROOT, split="val", transform=eval_tf, stack_frames=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                             pin_memory=True, persistent_workers=True)

    test_dataset = FrameVideoDataset(root_dir=DATA_ROOT, split="test", transform=eval_tf, stack_frames=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                              pin_memory=True, persistent_workers=True)

    model = cfg["build"]().to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    checkpoint_path = checkpoint_dir / f"{name}_noleakage_best.pt"
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_accuracy = -float("inf")

    for epoch in range(cfg["epochs"]):
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
        print(f"[{name}] epoch {epoch + 1:3d}/{cfg['epochs']} | train loss {train_loss:.4f} acc {train_acc * 100:.2f}% "
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
    print(f"[{name}] TEST | loss {test_loss:.4f} acc {test_acc * 100:.2f}%")

    plot_curves(history, title_prefix=f"{name} (noleakage)", save_prefix=f"{name}_noleakage")

    class_names = get_class_names(DATA_ROOT, split="train")
    preds, labels = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(preds, labels, num_classes=len(class_names))
    plot_confusion_matrix(
        cm, class_names,
        title=f"{name} (noleakage) confusion matrix - test",
        save_path=plots_dir / f"{name}_noleakage_confusion_matrix.png",
    )

    return {"name": name, "best_val_accuracy": best_val_accuracy, "test_loss": test_loss, "test_acc": test_acc}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    results = [run_experiment(name, cfg, device) for name, cfg in MODELS.items()]

    print("\n==== Summary: pretrained models on noleakage ====")
    for r in results:
        print(f"{r['name']:>22s} | best val acc {r['best_val_accuracy'] * 100:.2f}% "
              f"| test loss {r['test_loss']:.4f} | test acc {r['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
