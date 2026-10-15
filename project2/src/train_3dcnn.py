import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os

from pathlib import Path
from .datasets import FrameVideoDataset

# based on the folder structure on DTU HPC server
data_root = Path("/dtu/datasets1/02516/ufc10/")

# the 3d CNN definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv3d(3, 16, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


# validation inference to evaluate
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = clips.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_loss += loss.item() * clips.size(0)
        total_samples += clips.size(0)

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    #print(f"Accuracy: {average_accuracy * 100:.2f}%")
    return average_loss, average_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

LR = 3e-4
EPOCHS = 200
SEED = 42
WEIGHT_DECAY = 1e-4

project_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

job_id = os.environ.get("LSB_JOBID", "local")
checkpoint_path = checkpoint_dir / f"drop_adamw_3dcnn_{job_id}_best.pt"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    set_seed(SEED)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    frame_transform = transforms.Compose(
        [transforms.Resize((112, 112)), transforms.ToTensor()]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)


    # set and load dataset for training
    train_dataset = FrameVideoDataset(root_dir=data_root, split="train", transform=frame_transform)
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True,
                                worker_init_fn=seed_worker,
                                generator=train_generator,
                              )

    # set and load dataset for validation
    val_dataset = FrameVideoDataset(root_dir=data_root, split="val", transform=frame_transform)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            )


    print("Running training for 3d CNN")
    print(f"Using: Optimizer: {optimizer} | Loss function: {criterion} | Device: {device} ")

    #save current validation best accuracy
    best_val_accuracy = -float("inf")

    for epoch in range(EPOCHS):
        start = time.time()
        avg_loss, avg_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start
        print(f"EPOCH: {epoch + 1}/{EPOCHS}")
        print(f"Loss: {avg_loss:.4f} ----- Accuracy: {avg_acc*100:.2f}% --- Time for epoch: {epoch_time:.2f} seconds")

        print(f"Validation: Loss: {val_loss:.4f}. |  Accuracy: {val_accuracy * 100:.2f}%")
        print("---------------------------- \n")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "seed": SEED,
                },
                checkpoint_path,
            )




if __name__ == "__main__":
    main()
