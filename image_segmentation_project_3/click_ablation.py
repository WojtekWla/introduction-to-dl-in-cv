import os
import csv

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import v2

from seg_datasets import PH2, PH2_Annotated, build_loaders
from models import EncDecCNN, UNet, dice_score, evaluate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname("/zhome/2b/7/233718/idlcv/project_3/image_segmentation_project_3")
ABLATION_DIR = os.path.join(PROJECT_DIR, "click_ablation")
MODELS_DIR  = os.path.join(ABLATION_DIR, "models")
FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")
METRICS_DIR = os.path.join(ABLATION_DIR, "metrics")
for _dir in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    os.makedirs(_dir, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'
print("Running on:", device)

NUM_EPOCHS = int(os.environ.get("EPOCHS", 50))
print("Epochs:", NUM_EPOCHS)

SIZE = 256
BATCH_SIZE = 6

LEARNING_RATE  = 3e-3
WEIGHT_DECAY   = 1e-4
OPTIMIZER_NAME = "adamw"
GRAD_CLIP_NORM = None

FULL_SUPERVISION_VAL_DICE = {"EncDec": 0.9171, "UNet": 0.9169}


def build_optimizer(model):
    if OPTIMIZER_NAME == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unknown OPTIMIZER_NAME: {OPTIMIZER_NAME}")


train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
joint_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomAffine(degrees=25, translate=(0.1, 0.1)),
])

ph2_eval, _ = build_loaders(PH2, PH2.list_cases, size=SIZE, batch_size=BATCH_SIZE)
train_eval_loader = ph2_eval["train"]

loss_fn = nn.BCEWithLogitsLoss()

MODELS = {
    "EncDec": EncDecCNN,
    "UNet": UNet,
}

CLICK_COUNTS = [1, 3, 10, 25, 50, 100]
SAMPLING_STRATEGIES = ["random", "biased"]

SUMMARY_PATH = os.path.join(METRICS_DIR, "summary.csv")
LEADERBOARD_PATH = os.path.join(METRICS_DIR, "leaderboard.txt")
SUMMARY_FIELDS = ["model", "strategy", "n_clicks", "best_val_dice", "test_dice", "test_iou",
                  "test_accuracy", "test_sensitivity", "test_specificity"]


def run_experiment(model_name, model_cls, strategy, n_clicks, num_epochs):
    tag = f"{model_name}_{strategy}_N{n_clicks}"
    print(f"\n===== Training {tag} =====", flush=True)

    ph2_annotated_loaders, _ = build_loaders(
        PH2_Annotated, PH2_Annotated.list_cases, size=SIZE, batch_size=BATCH_SIZE,
        train_transform=train_transform, joint_transform=joint_transform,
        pos_clicks=n_clicks, neg_clicks=n_clicks, sampling_strategy=strategy)
    train_loader = ph2_annotated_loaders["train"]
    val_loader = ph2_annotated_loaders["val"]
    test_loader = ph2_annotated_loaders["test"]

    torch.manual_seed(SEED)
    model = model_cls().to(device)
    optimizer = build_optimizer(model)

    train_losses, val_losses, val_dices = [], [], []
    best_dice = -1.0
    best_state, best_sample = None, None

    for epoch in tqdm(range(num_epochs), unit="epoch", desc=tag):
        model.train()
        train_loss = 0.0
        click_sample = None
        for i, (data, target) in enumerate(train_loader):
            if i == 0:
                click_sample = (data[0].detach().cpu().clone(), target[0, 0].detach().cpu().clone())
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            labeled = target != -1
            if not labeled.any():
                continue
            loss = loss_fn(output[labeled], target[labeled])
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        sample = None
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item() * data.size(0)
                val_dice += dice_score(output, target) * data.size(0)
                if i == 0:
                    sample = (data[0].cpu(), target[0, 0].cpu(),
                              (torch.sigmoid(output[0, 0]) > 0.5).float().cpu())
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(f"Epoch {epoch + 1}/{num_epochs}  loss train: {train_loss:.4f}"
              f"\tval: {val_loss:.4f}\tval Dice: {val_dice:.4f}", flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_sample = sample
            best_click_sample = click_sample

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_DIR, f"best_{tag}.pt"))

    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{tag} - loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"loss_{tag}.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(val_dices, label="val Dice")
    ax.axhline(FULL_SUPERVISION_VAL_DICE[model_name], color="gray", ls="--",
               label=f"full supervision ({FULL_SUPERVISION_VAL_DICE[model_name]:.3f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1)
    ax.set_title(f"{tag} - validation accuracy (Dice)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"accuracy_{tag}.png"), dpi=120)
    plt.close(fig)

    click_img, click_points = best_click_sample
    img, mask, pred = best_sample
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(click_img.permute(1, 2, 0).numpy())
    for value, color in ((1, "lime"), (0, "red")):
        y, x = torch.where(click_points == value)
        ax[0].scatter(x.numpy(), y.numpy(), c=color, s=15)
    ax[0].set_title(f"train image + {n_clicks}p/{n_clicks}n clicks ({strategy})")
    ax[1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1); ax[1].set_title("ground truth (val)")
    ax[2].imshow(pred.numpy(), cmap="gray", vmin=0, vmax=1); ax[2].set_title("prediction (best epoch)")
    for a in ax:
        a.axis("off")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"prediction_{tag}.png"), dpi=120)
    plt.close(fig)

    metrics = {split: evaluate(model, loader, device)
               for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]}
    metrics_path = os.path.join(METRICS_DIR, f"metrics_{tag}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"{tag} (best val Dice = {best_dice:.4f})\n")
        f.write(f"hyperparams: lr={LEARNING_RATE}  weight_decay={WEIGHT_DECAY}  optimizer={OPTIMIZER_NAME}"
                f"  grad_clip={GRAD_CLIP_NORM}  epochs={num_epochs}  n_clicks={n_clicks}  strategy={strategy}\n")
        f.write(f"full-supervision reference val Dice: {FULL_SUPERVISION_VAL_DICE[model_name]:.4f}\n")
        for split, m in metrics.items():
            line = f"{split:5s} " + "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
            print(line, flush=True)
            f.write(line + "\n")

    return {"tag": tag, "model": model_name, "strategy": strategy, "n_clicks": n_clicks,
            "best_val_dice": best_dice, "test": metrics["test"]}


def update_summary(result):
    rows = []
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if not (r["model"] == result["model"] and r["strategy"] == result["strategy"]
                                     and r["n_clicks"] == str(result["n_clicks"]))]
    t = result["test"]
    rows.append({"model": result["model"], "strategy": result["strategy"], "n_clicks": result["n_clicks"],
                 "best_val_dice": f"{result['best_val_dice']:.4f}", "test_dice": f"{t['dice']:.4f}",
                 "test_iou": f"{t['iou']:.4f}", "test_accuracy": f"{t['accuracy']:.4f}",
                 "test_sensitivity": f"{t['sensitivity']:.4f}", "test_specificity": f"{t['specificity']:.4f}"})
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def tag_of(row):
    return f"{row['model']}_{row['strategy']}_N{row['n_clicks']}"


def plot_comparison(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_name in MODELS:
        for strategy in SAMPLING_STRATEGIES:
            pts = sorted(((int(r["n_clicks"]), float(r["best_val_dice"])) for r in rows
                          if r["model"] == model_name and r["strategy"] == strategy), key=lambda p: p[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"{model_name}, {strategy}")
        ax.axhline(FULL_SUPERVISION_VAL_DICE[model_name], ls="--", alpha=0.5,
                   label=f"{model_name} full supervision")
    ax.set_xscale("log")
    ax.set_xlabel("clicks per class (N)"); ax.set_ylabel("val Dice"); ax.set_ylim(0, 1)
    ax.set_title("PH2 - click count vs. val Dice, by model and sampling strategy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "summary_comparison.png"), dpi=120)
    plt.close(fig)


def print_and_save_leaderboard(rows):
    rows_sorted = sorted(rows, key=lambda r: float(r["best_val_dice"]), reverse=True)
    header = f"{'rank':<5}{'config':<30}{'val_dice':<10}{'test_dice':<10}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows_sorted, 1):
        lines.append(f"{i:<5}{tag_of(r):<30}{r['best_val_dice']:<10}{r['test_dice']:<10}")
    lines.append("")
    lines.append("Full-supervision reference (from loss_ablation/finetuning/ph/, not re-run here):")
    for model_name, dice in FULL_SUPERVISION_VAL_DICE.items():
        lines.append(f"  {model_name}: val Dice {dice:.4f}")
    text = "\n".join(lines)
    print("\nLeaderboard (ranked by val Dice):")
    print(text)
    with open(LEADERBOARD_PATH, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    rows = None
    for model_name, model_cls in MODELS.items():
        for strategy in SAMPLING_STRATEGIES:
            for n_clicks in CLICK_COUNTS:
                result = run_experiment(model_name, model_cls, strategy, n_clicks, NUM_EPOCHS)
                rows = update_summary(result)

    print(f"\nUpdated summary ({len(rows)} run(s) so far): {SUMMARY_PATH}")
    plot_comparison(rows)
    print(f"Updated comparison figure: {os.path.join(FIGURES_DIR, 'summary_comparison.png')}")
    print_and_save_leaderboard(rows)
