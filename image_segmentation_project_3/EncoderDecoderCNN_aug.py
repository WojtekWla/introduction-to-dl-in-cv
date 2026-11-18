import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2

from seg_datasets import PH2, PH2_Annotated, DRIVE, build_loaders, make_splits
from models import EncDecCNN, UNet, dice_score, evaluate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(42)
np.random.seed(42)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'
    
print("Running on:", device)

NUM_EPOCHS = int(os.environ.get("EPOCHS", 100))
print("Epochs:", NUM_EPOCHS)

size = 256
size_hr = 576

train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

joint_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomAffine(degrees=25, translate=(0.1, 0.1)),
])

ph2_loaders, ph2_split = build_loaders(PH2, PH2.list_cases, size=size, batch_size=6,
                                       train_transform=train_transform,
                                       joint_transform=joint_transform)

ph2_annotated_loaders, _ = build_loaders(PH2_Annotated, PH2_Annotated.list_cases,
                                       pos_clicks=10, neg_clicks=10, sampling_strategy="biased",
                                       size=size, batch_size=6,
                                       train_transform=train_transform,
                                       joint_transform=joint_transform)
ph2_annotation_tag = os.path.basename(ph2_annotated_loaders["train"].dataset.annotation_dir)

drive_loaders, drive_split = build_loaders(DRIVE, DRIVE.list_ids, size=size, batch_size=2,
                                           train_transform=train_transform,
                                           joint_transform=joint_transform)

drive_hr_loaders, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size_hr, batch_size=2,
                                    train_transform=train_transform,
                                    joint_transform=joint_transform)

train_ph2, val_ph2, test_ph2 = ph2_loaders["train"], ph2_loaders["val"], ph2_loaders["test"]

train_drive, val_drive, test_drive = drive_loaders["train"], drive_loaders["val"], drive_loaders["test"]

train_drive_hr, val_drive_hr, test_drive_hr = (drive_hr_loaders["train"],
                                               drive_hr_loaders["val"], drive_hr_loaders["test"])

ph2_eval, _ = build_loaders(PH2, PH2.list_cases, size=size, batch_size=6)

drive_eval, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size, batch_size=2)
drive_hr_eval, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size_hr, batch_size=2)


def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=50, tag="model"):
    train_losses, val_losses, val_dices = [], [], []
    best_dice = 0.0
    run_dir = os.path.join(PROJECT_DIR, "runs", tag)
    result_dir = os.path.join(PROJECT_DIR, "results", tag)
    epoch_dir = os.path.join(result_dir, "epochs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(epoch_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "best.pt")
    with open(os.path.join(run_dir, "history.csv"), "w") as history:
        history.write("epoch,train_loss,val_loss,val_dice\n")
    with open(os.path.join(run_dir, "train.log"), "w") as log:
        log.write(f"Training {tag}\n")

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        train_loss = 0.0
        model.train()
        for minibatch_no, (data, target) in enumerate(train_loader):
            if minibatch_no == 0:
                train_image = data[0].detach().cpu().clone()
                train_points = target[0, 0].detach().cpu().clone()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            labeled = target != -1
            if not labeled.any():
                continue
            loss = loss_fn(output[labeled], target[labeled])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        val_loss, val_dice = 0.0, 0.0
        model.eval()
        for minibatch_no, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            if minibatch_no == 0:
                val_image = data[0].detach().cpu().clone()
                val_mask = target[0, 0].detach().cpu().clone()
                val_pred = (torch.sigmoid(output[0, 0]) > 0.5).cpu()
            val_loss += loss_fn(output, target).item() * data.size(0)
            val_dice += dice_score(output, target) * data.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss   /= len(val_loader.dataset)
        val_dice   /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        message = "Epoch {e}/{n}  loss train: {tl:.4f}\t val: {vl:.4f}\t val Dice: {vd:.4f}".format(
              e=epoch + 1, n=num_epochs, tl=train_loss, vl=val_loss, vd=val_dice)
        print(message, flush=True)
        with open(os.path.join(run_dir, "train.log"), "a") as log:
            log.write(message + "\n")
        with open(os.path.join(run_dir, "history.csv"), "a") as history:
            history.write(f"{epoch + 1},{train_loss},{val_loss},{val_dice}\n")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(train_image.permute(1, 2, 0).numpy())
        if (train_points == -1).any():
            for value, color in ((1, "green"), (0, "red")):
                y, x = torch.where(train_points == value)
                ax[0].scatter(x.numpy(), y.numpy(), c=color, s=20)
        ax[0].set_title("Training image + clicks")
        ax[1].imshow(val_image.permute(1, 2, 0).numpy())
        ax[1].set_title("Validation image")
        ax[2].imshow(val_mask.numpy(), cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("Ground truth")
        ax[3].imshow(val_pred.numpy(), cmap="gray", vmin=0, vmax=1)
        ax[3].set_title("Prediction")
        for axis in ax:
            axis.axis("off")
        fig.suptitle(f"{tag} — epoch {epoch + 1}")
        fig.tight_layout()
        fig.savefig(os.path.join(epoch_dir, f"epoch_{epoch + 1:03d}.png"), dpi=120)
        plt.close(fig)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best val Dice {best_dice:.4f}, saved {checkpoint_path}", flush=True)


    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_losses, label='train')
    ax[0].plot(val_losses,   label='val')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[1].plot(val_dices, label='val Dice')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('Dice'); ax[1].legend()
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "curves.png"), dpi=120)
    plt.close(fig)

    print(f"Best val Dice: {best_dice:.4f}  (model: {checkpoint_path}, plot: {result_dir}/curves.png)", flush=True)
    return {"train_loss": train_losses, "val_loss": val_losses,
            "val_dice": val_dices, "best_dice": best_dice}

print("\nU-Net shapes, 256x256 input:")
UNet().to(device)(torch.zeros(1, 3, size, size, device=device), verbose=True)
print("\nU-Net shapes, 576x576 input:")
UNet().to(device)(torch.zeros(1, 3, size_hr, size_hr, device=device), verbose=True)

loss_fn = nn.BCEWithLogitsLoss()

experiments = [
    # ("encdec_ph2",      EncDecCNN, train_ph2,      val_ph2),   # PH2 already done (unchanged)
    # ("encdec_drive",    EncDecCNN, train_drive,    val_drive),
    # ("encdec_drive576", EncDecCNN, train_drive_hr, val_drive_hr),
    # ("unet_ph2",        UNet,      train_ph2,      val_ph2),   # PH2 already done (unchanged)
    # ("unet_drive",      UNet,      train_drive,    val_drive),
    # ("unet_drive576",   UNet,      train_drive_hr, val_drive_hr),
        (f"unet_ph2_{ph2_annotation_tag}", UNet, ph2_annotated_loaders["train"], ph2_annotated_loaders["val"]),
]

for tag, model_cls, tr_loader, va_loader in experiments:
    print(f"\n===== Training {tag} =====", flush=True)
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, loss_fn, tr_loader, va_loader, num_epochs=NUM_EPOCHS, tag=tag)

eval_sets = {
    f"ph2_{ph2_annotation_tag}": [("train", ph2_eval["train"]), ("val", ph2_annotated_loaders["val"]),
                      ("test", ph2_annotated_loaders["test"])],
    "ph2":      [("train", ph2_eval["train"]), ("val", val_ph2), ("test", test_ph2)],
    "drive":    [("train", drive_eval["train"]), ("val", val_drive), ("test", test_drive)],
    "drive576": [("train", drive_hr_eval["train"]), ("val", val_drive_hr), ("test", test_drive_hr)],
}
for tag, model_cls, _, _ in experiments:
    model = model_cls().to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "runs", tag, "best.pt"), map_location=device))
    print(f"\n{tag} (best val Dice model)")
    metrics_path = os.path.join(PROJECT_DIR, "results", tag, "metrics.txt")
    with open(metrics_path, "w") as metrics:
        metrics.write(f"{tag} (best val Dice model)\n")
    for split, loader in eval_sets[tag.split("_", 1)[1]]:
        m = evaluate(model, loader, device)
        message = f"{split:5s} " + "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
        print(message, flush=True)
        with open(metrics_path, "a") as metrics:
            metrics.write(message + "\n")
