"""
IDLCV Project 3 - Loss function ablation (DRIVE and PH2 segmentation).

What this does
---------------
DRIVE vessel masks are heavily class-imbalanced: vessel (foreground) pixels
are a small fraction of the image, background dominates. PH2 lesion masks
are much more balanced (large, smooth lesions) - the course's own baseline
reached ~0.92 Dice on PH2 with plain, untuned CrossEntropy, no reweighting
needed. DATASET below ("DRIVE" or "PH2") picks which dataset this
invocation runs against - size/batch_size switch automatically to match
each dataset's established settings (see the DATASET block). This script
trains segmentation architectures already defined in models.py - EncDecCNN
(plain encoder-decoder), UNet (same backbone + skip connections), and
UNetConvT (UNet with a learnable ConvTranspose2d decoder) - with pixel-wise
losses, to see which loss best handles each dataset's imbalance:

  1. CrossEntropy - plain nn.BCEWithLogitsLoss(). Binary cross-entropy per
     pixel, no class weighting. The baseline.
  2. Focal        - torchvision.ops.sigmoid_focal_loss(). Down-weights
     already-easy (confidently correct) pixels via (1-p)^gamma so training
     focuses on hard pixels, and reweights classes via alpha.
  3. WeightedCE   - nn.BCEWithLogitsLoss(pos_weight=...). Plain BCE, but the
     foreground term is scaled up by the train-set background/foreground
     pixel ratio, so missing a vessel pixel costs more than missing a
     background pixel.

Architecture, data augmentation, optimizer and image resolution are kept
IDENTICAL to EncoderDecoderCNN_aug.py and are NOT swept here - this is a
loss-function-only ablation, by design (see the project instructions: full
ablations across every axis are not expected, for GPU budget reasons).
Model selection within a run is always by validation Dice, never by
validation loss - the three losses live on different numeric scales, so
loss values are not comparable across runs.

BEST_CONFIGS is an explicit list of (model, lr, loss, focal_alpha)
combinations to run THIS invocation, in one `python loss_ablation.py` call
- see the BEST_CONFIGS block for what's included per DATASET and why. This
is NOT a fresh grid search (especially not DRIVE at SIZE=576, which is ~4x
the compute of 256x256 per epoch) - edit BEST_CONFIGS directly to add/
remove entries. summary.csv/leaderboard.txt/summary_comparison.png
accumulate across invocations, keyed on dataset+model+lr+loss, so running
PH2 then DRIVE (or vice versa) in the same output folder adds rows for
both instead of overwriting - no need to move the folder aside just for a
dataset switch (optimizer/weight-decay changes still aren't part of the
key, so those still need the manual-rename workflow from past runs).

Output layout (all paths below are the single source of truth - change here,
nowhere else, if you want a different layout):
  loss_ablation/
    models/  best_<Model>_lr<LR>_<Loss>.pt       best-val-Dice checkpoint
    figures/ loss_<Model>_lr<LR>_<Loss>.png       train/val loss curve
             accuracy_<Model>_lr<LR>_<Loss>.png   val Dice ("accuracy") curve
             prediction_<Model>_lr<LR>_<Loss>.png image | GT | prediction, best epoch
             summary_comparison.png              test Dice, every run so far, ranked by val Dice
    metrics/ metrics_<Model>_lr<LR>_<Loss>.txt     5 metrics x {train,val,test} + resolved hyperparams
             summary.csv                          one row per (model, lr, loss) run so far
             leaderboard.txt                      same rows, ranked by val Dice (selection metric)

Quick smoke test:  EPOCHS=1 python loss_ablation.py
"""
import os
import csv

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms as transforms
from torchvision.transforms import v2

from seg_datasets import DRIVE, PH2, build_loaders
from models import EncDecCNN, UNet, UNetConvT, dice_score, evaluate

import matplotlib
matplotlib.use("Agg")                       # cluster has no display -> save plots to file
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Visible paths - everything this script reads/writes lives under these.
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname("/zhome/2b/7/233718/idlcv/project_3/image_segmentation_project_3")
ABLATION_DIR = os.path.join(PROJECT_DIR, "loss_ablation")
MODELS_DIR  = os.path.join(ABLATION_DIR, "models")   # best_<Model>_<Loss>.pt checkpoints
FIGURES_DIR = os.path.join(ABLATION_DIR, "figures")  # curves + qualitative predictions
METRICS_DIR = os.path.join(ABLATION_DIR, "metrics")  # per-run metrics.txt + summary.csv
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

DATASET = "PH2"      # one of: "DRIVE", "PH2" - which dataset BEST_CONFIGS runs against this invocation
if DATASET == "DRIVE":
    DATASET_CLS, DATASET_IDS_FN = DRIVE, DRIVE.list_ids
    SIZE = 576        # near-native DRIVE resolution - see module docstring
    BATCH_SIZE = 2    # drop to 1 if this OOMs at 576x576
elif DATASET == "PH2":
    DATASET_CLS, DATASET_IDS_FN = PH2, PH2.list_cases
    SIZE = 256        # PH2 lesions are large/smooth - no thin-structure reason to go near-native like DRIVE
    BATCH_SIZE = 6    # matches EncoderDecoderCNN_aug.py's PH2 batch size
else:
    raise ValueError(f"Unknown DATASET: {DATASET!r}")

# ---------------------------------------------------------------------------
# Optimization / regularization knobs - fixed across every entry in
# BEST_CONFIGS below (lr varies per-config instead; see BEST_CONFIGS).
# ---------------------------------------------------------------------------
WEIGHT_DECAY   = 1e-4    # L2 regularization via the optimizer; best value found so far
OPTIMIZER_NAME = "adamw"  # try: "adam", "adamw", "sgd" (uses momentum=0.9 below) - best found so far
GRAD_CLIP_NORM = None    # try: 1.0, 5.0 - set a float to enable torch.nn.utils.clip_grad_norm_
# Dropout is not wired into EncDecCNN/UNet (models.py). To try it, add
# nn.Dropout2d(p=...) inside conv_block() / after an encoder ReLU and rerun.
# Augmentation strength (flip/rotation/translation/color jitter) is also a
# natural ablation axis - the settings below just reuse what
# EncoderDecoderCNN_aug.py already uses, unchanged.


def build_optimizer(model, lr):
    if OPTIMIZER_NAME == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_NAME == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unknown OPTIMIZER_NAME: {OPTIMIZER_NAME}")


# ---------------------------------------------------------------------------
# Data: DATASET (DRIVE or PH2), same augmentation as EncoderDecoderCNN_aug.py
# for both (not an ablation axis here - kept fixed so the loss function is
# the only thing that changes between runs).
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
joint_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomAffine(degrees=25, translate=(0.1, 0.1)),
])

data_loaders, data_split = build_loaders(DATASET_CLS, DATASET_IDS_FN, size=SIZE, batch_size=BATCH_SIZE,
                                          train_transform=train_transform, joint_transform=joint_transform)
train_loader, val_loader, test_loader = data_loaders["train"], data_loaders["val"], data_loaders["test"]

# Train split evaluated WITHOUT augmentation, same rationale as EncoderDecoderCNN_aug.py
data_eval, _ = build_loaders(DATASET_CLS, DATASET_IDS_FN, size=SIZE, batch_size=BATCH_SIZE)
train_eval_loader = data_eval["train"]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
FOCAL_ALPHA = 0.75  # weight on the foreground (vessel) class. RetinaNet's default (0.25)
                     # DOWN-weights foreground, which hurts on an already-rare class like
                     # this one - try 0.5 / 0.75 / 0.9 (higher = more weight on foreground).
FOCAL_GAMMA = 2.0    # focusing strength - higher values focus more on hard/misclassified
                     # pixels and less on easy ones already classified correctly; try 1.0, 3.0, 5.0.

POS_WEIGHT_OVERRIDE = None  # set e.g. POS_WEIGHT_OVERRIDE = 5.0 to bypass the computed value below


def compute_pos_weight(loader):
    """Background/foreground pixel ratio over the (unaugmented) train split.
    Feeds nn.BCEWithLogitsLoss(pos_weight=...) so missing a vessel pixel is
    penalized `pos_weight` times more than missing a background pixel."""
    fg = bg = 0.0
    for _, target in loader:
        fg += target.sum().item()
        bg += (1 - target).sum().item()
    return bg / fg


pos_weight_value = POS_WEIGHT_OVERRIDE if POS_WEIGHT_OVERRIDE is not None else compute_pos_weight(train_eval_loader)
print(f"pos_weight (train-split background/foreground pixel ratio): {pos_weight_value:.3f}")


def make_loss_fns(focal_alpha=FOCAL_ALPHA):
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    return {
        "CrossEntropy": nn.BCEWithLogitsLoss(),
        "Focal": lambda logits, target: sigmoid_focal_loss(
            logits, target, alpha=focal_alpha, gamma=FOCAL_GAMMA, reduction="mean"),
        "WeightedCE": nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor),
    }


MODELS = {
    "EncDec": EncDecCNN,
    "UNet": UNet,
    "UNetConvT": UNetConvT,
}

# ---------------------------------------------------------------------------
# Explicit list of configs to run THIS invocation against DATASET. Edit this
# list to add/remove configs - each runs once, in order, in this invocation.
# focal_alpha is only used when loss == "Focal" (ignored otherwise).
#
# For DATASET="PH2": PH2 lesions are large/smooth with much milder class
# imbalance than DRIVE vessels - this set reuses the DRIVE-tuned champions
# (UNet/UNetConvT + Focal) as-is, PLUS a CrossEntropy entry per architecture,
# specifically to test whether the DRIVE finding "Focal beats CrossEntropy,
# and EncDec collapses under CrossEntropy without skip connections" still
# holds once the severe imbalance that motivated it is mostly gone - per the
# course baseline, plain CrossEntropy alone reached ~0.92 Dice on PH2 for
# both EncDec and UNet, so a collapse here would be surprising, not expected.
# For DATASET="DRIVE": this is the 576x576 best-configs-so-far list.
# ---------------------------------------------------------------------------
if DATASET == "PH2":
    BEST_CONFIGS = [
        {"label": "DRIVE champion, reused as-is",       "model": "UNet",      "lr": 3e-3, "loss": "Focal",       "focal_alpha": 0.75},
        {"label": "does CrossEntropy still lag on PH2?","model": "UNet",      "lr": 3e-3, "loss": "CrossEntropy","focal_alpha": None},
        {"label": "best ConvTranspose variant, reused", "model": "UNetConvT", "lr": 3e-3, "loss": "Focal",       "focal_alpha": 0.75},
        {"label": "does EncDec still collapse on PH2?", "model": "EncDec",    "lr": 3e-3, "loss": "CrossEntropy","focal_alpha": None},
    ]
else:
    BEST_CONFIGS = [
        {"label": "overall champion @256",       "model": "UNet",      "lr": 3e-3, "loss": "Focal",      "focal_alpha": 0.75},
        {"label": "2nd-best champion @256",      "model": "UNet",      "lr": 5e-3, "loss": "Focal",      "focal_alpha": 0.90},
        {"label": "best ConvTranspose variant",  "model": "UNetConvT", "lr": 3e-3, "loss": "Focal",      "focal_alpha": 0.75},
        {"label": "does EncDec recover at 576?", "model": "EncDec",    "lr": 1e-3, "loss": "WeightedCE", "focal_alpha": None},
    ]

SUMMARY_PATH = os.path.join(METRICS_DIR, "summary.csv")
LEADERBOARD_PATH = os.path.join(METRICS_DIR, "leaderboard.txt")
SUMMARY_FIELDS = ["dataset", "model", "lr", "loss", "best_val_dice", "test_dice", "test_iou",
                  "test_accuracy", "test_sensitivity", "test_specificity"]


def run_experiment(model_name, model_cls, lr, loss_name, loss_fn, num_epochs):
    lr_str = f"{lr:.0e}"
    tag = f"{DATASET}_{model_name}_lr{lr_str}_{loss_name}"
    print(f"\n===== Training {tag} =====", flush=True)

    torch.manual_seed(SEED)  # same init per architecture regardless of lr/loss - isolates their effect
    model = model_cls().to(device)
    optimizer = build_optimizer(model, lr)

    train_losses, val_losses, val_dices = [], [], []
    best_dice = -1.0
    best_state, best_sample = None, None

    for epoch in tqdm(range(num_epochs), unit="epoch", desc=tag):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
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

        # Model selection is always by val Dice, never val loss: BCE, focal and
        # weighted-BCE live on different numeric scales and aren't comparable.
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_sample = sample

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_DIR, f"best_{tag}.pt"))

    # --- Figures ---
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
    ax.set_xlabel("epoch"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1)
    ax.set_title(f"{tag} - validation accuracy (Dice)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"accuracy_{tag}.png"), dpi=120)
    plt.close(fig)

    img, mask, pred = best_sample
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(img.permute(1, 2, 0).numpy()); ax[0].set_title("image")
    ax[1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1); ax[1].set_title("ground truth")
    ax[2].imshow(pred.numpy(), cmap="gray", vmin=0, vmax=1); ax[2].set_title("prediction (best epoch)")
    for a in ax:
        a.axis("off")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"prediction_{tag}.png"), dpi=120)
    plt.close(fig)

    # --- Metrics: 5 metrics x {train, val, test}, best-val-Dice checkpoint ---
    metrics = {split: evaluate(model, loader, device)
               for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]}
    metrics_path = os.path.join(METRICS_DIR, f"metrics_{tag}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"{tag} (best val Dice = {best_dice:.4f})\n")
        f.write(f"hyperparams: lr={lr}  weight_decay={WEIGHT_DECAY}  optimizer={OPTIMIZER_NAME}"
                f"  grad_clip={GRAD_CLIP_NORM}  epochs={num_epochs}\n")
        for split, m in metrics.items():
            line = f"{split:5s} " + "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
            print(line, flush=True)
            f.write(line + "\n")

    return {"tag": tag, "dataset": DATASET, "model": model_name, "lr": lr_str, "loss": loss_name,
            "best_val_dice": best_dice, "test": metrics["test"]}


def update_summary(result):
    """Upsert this run's row into summary.csv (keyed on dataset+model+lr+loss) and return all rows so far."""
    rows = []
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("dataset", "DRIVE")  # old summary.csv files (pre-DATASET) were always DRIVE
        if not r["dataset"]:
            r["dataset"] = "DRIVE"
    rows = [r for r in rows if not (r["dataset"] == result["dataset"] and r["model"] == result["model"]
                                     and r["lr"] == result["lr"] and r["loss"] == result["loss"])]
    t = result["test"]
    rows.append({"dataset": result["dataset"], "model": result["model"], "lr": result["lr"], "loss": result["loss"],
                 "best_val_dice": f"{result['best_val_dice']:.4f}", "test_dice": f"{t['dice']:.4f}",
                 "test_iou": f"{t['iou']:.4f}", "test_accuracy": f"{t['accuracy']:.4f}",
                 "test_sensitivity": f"{t['sensitivity']:.4f}", "test_specificity": f"{t['specificity']:.4f}"})
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def tag_of(row):
    return f"{row.get('dataset', 'DRIVE')}_{row['model']}_lr{row['lr']}_{row['loss']}"


def plot_comparison(rows):
    """Bar chart of every run seen so far (across all invocations), ranked by val Dice."""
    rows_sorted = sorted(rows, key=lambda r: float(r["best_val_dice"]), reverse=True)
    tags = [tag_of(r) for r in rows_sorted]
    val_vals = [float(r["best_val_dice"]) for r in rows_sorted]
    test_vals = [float(r["test_dice"]) for r in rows_sorted]

    x = np.arange(len(tags))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(tags)), 4))
    ax.bar(x - width / 2, val_vals, width, label="val Dice (selection)")
    ax.bar(x + width / 2, test_vals, width, label="test Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1)
    ax.set_title("DRIVE - all runs so far, ranked by val Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "summary_comparison.png"), dpi=120)
    plt.close(fig)


def print_and_save_leaderboard(rows):
    """Rank every run seen so far by val Dice (the selection metric) - never by test Dice,
    or picking the 'best' config would amount to tuning on the test set."""
    rows_sorted = sorted(rows, key=lambda r: float(r["best_val_dice"]), reverse=True)
    header = f"{'rank':<5}{'config':<38}{'val_dice':<10}{'test_dice':<10}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows_sorted, 1):
        lines.append(f"{i:<5}{tag_of(r):<38}{r['best_val_dice']:<10}{r['test_dice']:<10}")
    text = "\n".join(lines)
    print("\nLeaderboard (ranked by val Dice):")
    print(text)
    with open(LEADERBOARD_PATH, "w") as f:
        f.write(text + "\n")
    best = rows_sorted[0]
    print(f"\nBest so far: {tag_of(best)}  (val Dice {best['best_val_dice']}, test Dice {best['test_dice']})")


if __name__ == "__main__":
    rows = None
    for cfg in BEST_CONFIGS:
        model_name = cfg["model"]
        if model_name not in MODELS:
            raise ValueError(f"BEST_CONFIGS: unknown model {model_name!r}, must be one of {list(MODELS)}")
        alpha = cfg["focal_alpha"] if cfg["focal_alpha"] is not None else FOCAL_ALPHA
        loss_fns = make_loss_fns(alpha)
        loss_name = cfg["loss"]
        if loss_name not in loss_fns:
            raise ValueError(f"BEST_CONFIGS: unknown loss {loss_name!r}, must be one of {list(loss_fns)}")
        print(f"\n[{cfg['label']}]")
        result = run_experiment(model_name, MODELS[model_name], cfg["lr"], loss_name, loss_fns[loss_name], NUM_EPOCHS)
        rows = update_summary(result)

    print(f"\nUpdated summary ({len(rows)} run(s) so far): {SUMMARY_PATH}")
    plot_comparison(rows)
    print(f"Updated comparison figure: {os.path.join(FIGURES_DIR, 'summary_comparison.png')}")
    print_and_save_leaderboard(rows)
