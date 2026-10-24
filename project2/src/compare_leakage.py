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
from train_3dcnn import Model, train, evaluate, set_seed, seed_worker, LR, WEIGHT_DECAY, SEED, EPOCHS

# datasets to compare: (root_dir, short tag used in filenames/titles)
DATASETS = [
    ("/dtu/datasets1/02516/ufc10", "leakage"),
    ("/dtu/datasets1/02516/ucf101_noleakage", "noleakage"),
]

BATCH_SIZE = 16

project_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = project_dir / "checkpoints"
plots_dir = project_dir / "plots"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)


def get_class_names(root_dir, split="train"):
    df = pd.read_csv(Path(root_dir) / "metadata" / f"{split}.csv")
    return df.drop_duplicates("label").sort_values("label")["action"].tolist()


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
    print(f"\n==== Running 3D CNN on '{tag}' ({data_root}) ====")
    set_seed(SEED)

    frame_transform = transforms.Compose(
        [transforms.Resize((112, 112)), transforms.ToTensor()]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_dataset = FrameVideoDataset(root_dir=data_root, split="train", transform=frame_transform)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=train_generator,
    )

    val_dataset = FrameVideoDataset(root_dir=data_root, split="val", transform=frame_transform)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )

    test_dataset = FrameVideoDataset(root_dir=data_root, split="test", transform=frame_transform)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = checkpoint_dir / f"3dcnn_{tag}_best.pt"
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_accuracy = -float("inf")

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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

    # evaluate the best (by val acc) checkpoint on the held-out test split
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[{tag}] TEST | loss {test_loss:.4f} acc {test_acc * 100:.2f}%")

    plot_curves(history, title_prefix=f"3D CNN ({tag})", save_prefix=f"3dcnn_{tag}")

    class_names = get_class_names(data_root, split="train")
    preds, labels = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(preds, labels, num_classes=len(class_names))
    plot_confusion_matrix(
        cm, class_names,
        title=f"3D CNN ({tag}) confusion matrix - test",
        save_path=plots_dir / f"3dcnn_{tag}_confusion_matrix.png",
    )

    return {"tag": tag, "best_val_accuracy": best_val_accuracy, "test_loss": test_loss, "test_acc": test_acc}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    results = [run_experiment(data_root, tag, device) for data_root, tag in DATASETS]

    print("\n==== Summary: leakage vs no-leakage ====")
    for r in results:
        print(f"{r['tag']:>10s} | best val acc {r['best_val_accuracy'] * 100:.2f}% "
              f"| test loss {r['test_loss']:.4f} | test acc {r['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
