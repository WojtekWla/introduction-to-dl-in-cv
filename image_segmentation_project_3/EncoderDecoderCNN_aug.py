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

import matplotlib
matplotlib.use("Agg")                       # cluster has no display -> save plots to file
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

# NEW: number of epochs in one place. Quick test run:  EPOCHS=1 python EncoderDecoderCNN.py
NUM_EPOCHS = int(os.environ.get("EPOCHS", 100))
print("Epochs:", NUM_EPOCHS)

size = 256          # PH2 and the standard DRIVE runs
size_hr = 576       # NEW: DRIVE near-native resolution (565x584 -> 576x576, divisible by 16)

# image only (like ColorJitter): plain transforms.Compose
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# image AND mask together (rotation, translation, flips): v2.Compose
joint_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomAffine(degrees=25, translate=(0.1, 0.1)),
])

# PH2: augmented (train split only)
ph2_loaders, ph2_split = build_loaders(PH2, PH2.list_cases, size=size, batch_size=6,
                                       train_transform=train_transform,
                                       joint_transform=joint_transform)

ph2_annotated_loaders, _ = build_loaders(PH2_Annotated, PH2_Annotated.list_cases,
                                       pos_clicks=10, neg_clicks=10, sampling_strategy="biased",
                                       size=size, batch_size=6,
                                       train_transform=train_transform,
                                       joint_transform=joint_transform)
ph2_annotation_tag = os.path.basename(ph2_annotated_loaders["train"].dataset.annotation_dir)

# DRIVE: augmented (train split only), same transforms as PH2
drive_loaders, drive_split = build_loaders(DRIVE, DRIVE.list_ids, size=size, batch_size=2,
                                           train_transform=train_transform,
                                           joint_transform=joint_transform)

# DRIVE at near-native resolution, same split (same seed), augmented
drive_hr_loaders, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size_hr, batch_size=2,
                                    train_transform=train_transform,
                                    joint_transform=joint_transform)

#Set split for all datasets
train_ph2, val_ph2, test_ph2 = ph2_loaders["train"], ph2_loaders["val"], ph2_loaders["test"]

train_drive, val_drive, test_drive = drive_loaders["train"], drive_loaders["val"], drive_loaders["test"]

train_drive_hr, val_drive_hr, test_drive_hr = (drive_hr_loaders["train"],             # NEW
                                               drive_hr_loaders["val"], drive_hr_loaders["test"])

#PH2 eval split - needed as initially its called with augmentation
ph2_eval, _ = build_loaders(PH2, PH2.list_cases, size=size, batch_size=6)

#DRIVE eval splits - train split evaluated WITHOUT augmentation
drive_eval, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size, batch_size=2)
drive_hr_eval, _ = build_loaders(DRIVE, DRIVE.list_ids, size=size_hr, batch_size=2)

#Exercise 2 - Implement Simple Encoder Decoder CNN + Evaluation Metrics

class EncDecCNN(nn.Module):
    def __init__(self):
        super(EncDecCNN, self).__init__()
        self.encoder = nn.Sequential(
            #First Part - Downsampling - Conv + Batch Norm + ReLU + Pooling
            nn.Conv2d(3,8,3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),        #256 -> 128

            nn.Conv2d(8,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),        #128 -> 64

            nn.Conv2d(16,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),        #64 -> 32

            nn.Conv2d(16,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),        #32 -> 16

            # bottleneck (same as in UNet)
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            #Second Part - Upsampling - Upsampling + Conv + Batch Norm + ReLU, head outputs logits
            nn.Upsample(scale_factor=2),            #16->32
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),           # 32 -> 64
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),            # 64 -> 128
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),                        # 128 -> 256
            #Head output raw logits sigmoid in loss/metrics
            #No ReLU as that would make negative outputs impossible therefore no background logits
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


#Exercise 3 - U-Net (same depth/channels as EncDecCNN + skip connections)

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = conv_block(3, 8)        # 256
        self.enc2 = conv_block(8, 16)       # 128
        self.enc3 = conv_block(16, 16)      # 64
        self.enc4 = conv_block(16, 32)      # 32
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 32)    # 16

        # Decoder: input channels = upsampled + skip (concatenated)
        self.up = nn.Upsample(scale_factor=2)
        self.dec4 = conv_block(32 + 32, 16)     # 32
        self.dec3 = conv_block(16 + 16, 16)     # 64
        self.dec2 = conv_block(16 + 16, 8)      # 128
        self.dec1 = conv_block(8 + 8, 8)        # 256

        # Head: 1 channel raw logits
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x, verbose=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        # skip connections: torch.cat joins upsampled map + matching encoder map
        d4 = self.dec4(torch.cat([self.up(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        out = self.head(d1)

        if verbose:
            for name, t in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("bottleneck", b),
                            ("d4", d4), ("d3", d3), ("d2", d2), ("d1", d1), ("out", out)]:
                print(f"{name:10s} {tuple(t.shape)}")
        return out


def dice_score(logits, target, eps=1e-7):
    """Dice of a batch
    torch.sigmoid(logits) converts output of the model into probabilities
    if prob > 0.5 = True -> converted to 1 with float()
    inter then calculates product of pred and target -> only True if both = 1 and sums them across all pixels
    eps - prevents division by zero
    .item extracts final scalar value
    """
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


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
        #Train
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
            #data has shape (batch, channels, height, width) -> data.size(0) = batch size
            train_loss += loss.item() * data.size(0)   #batch average loss x images in batch = batch total

        #Validate
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

        train_loss /= len(train_loader.dataset) #Dividing total in batch by total in split
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

        # Save best model by val dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best val Dice {best_dice:.4f}, saved {checkpoint_path}", flush=True)


    # Plot after all epoch
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

def evaluate(model, loader, eps=1e-7):
    """All five metrics over a whole split, from pixel counts (TP, FP, FN, TN)."""
    model.eval()
    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = (torch.sigmoid(model(data)) > 0.5).float()
            tp += (pred * target).sum().item()
            fp += (pred * (1 - target)).sum().item()
            fn += ((1 - pred) * target).sum().item()
            tn += ((1 - pred) * (1 - target)).sum().item()
    return {
        "dice":        (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou":         (tp + eps) / (tp + fp + fn + eps),
        "accuracy":    (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": (tp + eps) / (tp + fn + eps),   # recall: share of true foreground found
        "specificity": (tn + eps) / (tn + fp + eps),   # share of true background kept as background
    }


# verify shapes at each layer (copy this output into the report)
print("\nU-Net shapes, 256x256 input:")
UNet().to(device)(torch.zeros(1, 3, size, size, device=device), verbose=True)
print("\nU-Net shapes, 576x576 input:")                                                  # NEW
UNet().to(device)(torch.zeros(1, 3, size_hr, size_hr, device=device), verbose=True)

loss_fn = nn.BCEWithLogitsLoss()

# AUG RUN: only the four DRIVE experiments, now with augmentation: (tag, model class, train loader, val loader)
experiments = [
    # ("encdec_ph2",      EncDecCNN, train_ph2,      val_ph2),   # PH2 already done (unchanged)
    # ("encdec_drive",    EncDecCNN, train_drive,    val_drive),
    # ("encdec_drive576", EncDecCNN, train_drive_hr, val_drive_hr),
    # ("unet_ph2",        UNet,      train_ph2,      val_ph2),   # PH2 already done (unchanged)
    # ("unet_drive",      UNet,      train_drive,    val_drive),
    # ("unet_drive576",   UNet,      train_drive_hr, val_drive_hr),
        (f"unet_ph2_{ph2_annotation_tag}", UNet, ph2_annotated_loaders["train"], ph2_annotated_loaders["val"]),
]

#Actually Training (fresh model + optimizer for every experiment)
for tag, model_cls, tr_loader, va_loader in experiments:
    print(f"\n===== Training {tag} =====", flush=True)
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, loss_fn, tr_loader, va_loader, num_epochs=NUM_EPOCHS, tag=tag)

# Evaluation: best model (highest val Dice), five metrics, all three splits
# Train splits are evaluated WITHOUT augmentation (ph2_eval, drive_eval, drive_hr_eval)
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
        m = evaluate(model, loader)
        message = f"{split:5s} " + "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
        print(message, flush=True)
        with open(metrics_path, "a") as metrics:
            metrics.write(message + "\n")
