from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

from datasets import FrameVideoDataset, FlowVideoDataset


device = None


class DinoSpatial(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov3-vits16-pretrain-lvd1689m"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.config.hidden_size, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb):
        # FrameVideoDataset(stack_frames=True) returns [B, C, T, H, W].
        batch, channels, frames, height, width = rgb.shape
        rgb = rgb.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

        with torch.no_grad():
            embeddings = self.backbone(pixel_values=rgb).pooler_output

        video_features = embeddings.reshape(batch, frames, -1).mean(dim=1)
        return self.classifier(video_features)


class FlowStream(nn.Module):
    def __init__(self, num_classes, n_flows=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_flows, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.linear(self.conv(x))


class DualStream(nn.Module):
    def __init__(self, num_classes, n_flows=9):
        super().__init__()
        self.spatial = DinoSpatial(num_classes)
        self.temporal = FlowStream(num_classes, n_flows)

    def forward(self, rgb, flow):
        spatial_logits = self.spatial(rgb)
        temporal_logits = self.temporal(flow)
        return (spatial_logits + temporal_logits) / 2


class DualStreamDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, n_flows=9):
        self.rgb_dataset = FrameVideoDataset(
            root_dir=root_dir, split=split, transform=transform, stack_frames=True
        )
        self.flow_dataset = FlowVideoDataset(root_dir=root_dir, split=split, n_flows=n_flows)
        if len(self.rgb_dataset) != len(self.flow_dataset):
            raise ValueError("RGB and flow datasets have different lengths")

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        rgb, label = self.rgb_dataset[idx]
        flow, flow_label = self.flow_dataset[idx]
        if label != flow_label:
            raise ValueError(f"RGB and flow labels do not match at index {idx}")
        return rgb, flow, label


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for rgb, flow, labels in tqdm(loader, leave=False):
        rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(rgb, flow)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_video(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for rgb, flow, labels in tqdm(loader, leave=False):
        rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)
        logits = model(rgb, flow)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, lr=1e-3, num_epochs=50, weight_decay=1e-4):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, best_state = -1.0, None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate_video(model, val_loader, loss_fn)
        scheduler.step()

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.3f} video acc {val_acc:.3f}"
            f"{'  *best*' if is_best else ''}"
        )

    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    torch.manual_seed(0)
    np.random.seed(0)

    # root_dir = "/dtu/datasets1/02516/ucf101_noleakage"
    root_dir = "/Users/makro/Projects/DTU/Vision/video_classification_project_2/ufc10"
    batch_size = 4
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # No independent random flipping: the RGB and optical-flow inputs must agree.
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def make_loader(split, shuffle=False):
        return DataLoader(
            DualStreamDataset(root_dir=root_dir, split=split, transform=rgb_transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            persistent_workers=True,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    model = DualStream(num_classes=10).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    model, history = fit(model, train_loader, val_loader)

    test_loss, test_acc = evaluate_video(
        model, test_loader, nn.CrossEntropyLoss(label_smoothing=0.1)
    )
    print(f"test loss {test_loss:.3f} | test video acc {test_acc:.3f}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "dual_stream_dinov3.pt")

    epochs = range(1, len(history["train_loss"]) + 1)
    for metric in ("loss", "acc"):
        plt.figure()
        plt.plot(epochs, history[f"train_{metric}"], label=f"train {metric}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"val {metric}")
        plt.xlabel("epoch")
        plt.ylabel("accuracy" if metric == "acc" else "loss")
        plt.legend()
        plt.title(f"DualStream DINOv3 {metric}")
        plt.savefig(out_dir / f"dual_stream_dinov3_{metric}.png", dpi=150, bbox_inches="tight")
    plt.show()
