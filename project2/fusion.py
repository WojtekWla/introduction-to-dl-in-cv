"""
The following code compares both late and ealry fusion for video classification on the
UFC-101 dataset provided
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import FrameImageDataset, FrameVideoDataset
import random

#Data Exploration
import pandas as pd
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPLORE = False

if EXPLORE:
    frames_dir = r'C:\Users\ebjen\OneDrive\Desktop\DTU Sem1\Intro to DL for CV\Project2\ucf101\ufc10\frames'
    phases = ['train', 'val', 'test']

    data_rows = []

    for p in phases:
        phase_path = os.path.join(frames_dir, p)
        if os.path.exists(phase_path):
            for cls in os.listdir(phase_path):
                class_path = os.path.join(phase_path, cls)

                if os.path.isdir(class_path):
                    for vid in os.listdir(class_path):
                        video_path = os.path.join(class_path, vid)

                        if os.path.isdir(video_path):
                            frame_count = len(os.listdir(video_path))

                            data_rows.append({
                                'name': vid,
                                'class': cls,
                                'frames': frame_count,
                                'phase': p
                            })

    df = pd.DataFrame(data_rows)

    if not df.empty:
        # Print the sample counts per phase
        for p in phases:
            count = len(df[df['phase'] == p])
            print(f'Samples in {p} set: {count}')

        # Plot 1: Class counts
        def plot_class_count(df):
            fig = plt.figure(figsize=(20, 7))
            ax = sns.countplot(x='class', hue='phase', data=df)
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f"{p.get_height():.0f}", (p.get_x() + 0.05, p.get_height() + 0.5))

            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

        plot_class_count(df)

        # Plot 2: Frame distributions
        def plot_frames_dist(df):
            ax = sns.displot(df, x='frames', hue='phase', kde=True, element='poly')
            plt.show()

        plot_frames_dist(df)
    else:
        print("No data found inside train/val/test.")


#Late Fusion Algorithm
#individual 2D CNN for each frame then combine near top w2ith average pooling flattening
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = '/dtu/datasets1/02516/ufc10'
num_classes = 10
batch_size = 8


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        #First Convolution
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        #Second Convolution
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        #Dimension handling with linear projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))            # Conv -> BN  (NO relu yet)
        out = out + self.shortcut(x)               # F(x) + x   <- the skip
        return self.relu(out)                      # ReLU AFTER the merge


#ResidualBlock Class above is a single learning unit
#You need to arrange mulitple of them into a pipeline (group blocks into stages)
#Each stage shrinks image and widens channels (loses spatial detail for semantic knowledge)
def _stage(in_ch, out_ch, stride):
    return nn.Sequential(
        ResidualBlock(in_ch, out_ch, stride=stride),   # 1st: downsample + widen
        #only first block downsamples second one hardcoded stride = 1
        ResidualBlock(out_ch, out_ch, stride=1),        # 2nd: self.shortcut stays identity 
    )

#shared between both late and early fusion
class Encoder(nn.Module):
    def __init__(self, in_ch, feature_dim = 256):
        super().__init__()
        #Entry point of image (raw input to 32 channels)
        #in_ch = 3 if late fusion or 3xT if early
        #After stem, out_channel is 32 for both models
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        #downsampling with stride 2 reduces spatial size and increases field of view
        self.layer1 = _stage(32,64,stride=2)   # 112 -> 56
        self.layer2 = _stage(64,128,stride=2)   # 56  -> 28
        self.layer3 = _stage(128,256,stride=2)   # 28  -> 14
        self.layer4 = _stage(256,feature_dim,stride=2)   # 14  -> 7
        
        self.pool = nn.AdaptiveAvgPool2d(1)   # GAP -> [N, feature_dim, 1, 1]
        self.feature_dim = feature_dim

    def forward(self, x):                 # x: [N, in_ch, 112, 112]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)    # [N, feature_dim]


#LateFusion Wrapper
class LateFusion(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super().__init__()
        self.encoder = Encoder(in_ch=3, feature_dim=feature_dim)   # one RGB frame
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):                                   # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        #B*T = independant single frame images without no idea they belong to clips
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # -> [B*T, C, H, W]  frames separate
        x = self.encoder(x)                                     # -> [B*T, feature_dim]
        #Takes out T dim and then averages over it with mean(1)
        x = x.view(B, T, self.feature_dim).mean(1)              # -> [B, feature_dim]  FUSE over T
        return self.classifier(x)                               # -> [B, num_classes]

#EarlyFusion Wrapper
class EarlyFusion(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256, num_frames = 10):
        super().__init__()
        #all framees fed into encoder input channel at once
        self.encoder = Encoder(in_ch=3 * num_frames, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        B,C,T,H,W = x.shape
        #T folded into channel dim -> encoder sees B images each with 30 channels (since 10 frames)
        #Time fused immediately before encoding
        x = x.reshape(B, C*T, H, W)
        x = self.encoder(x)
        return self.classifier(x)


#To incorporate dataloader
def to_posix(ds):
    if hasattr(ds, 'video_paths'):
        ds.video_paths = [p.replace('\\', '/') for p in ds.video_paths]
    return ds


#Preprocessing of data fixed resize and data augmentation for training set
train_tf = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
eval_tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

#Stack frames stacks single [C,T,H,W] -> [B,C,T,H,W] expected as input in the late fusion
trainset = to_posix(FrameVideoDataset(root_dir=root_dir, split='train', transform=train_tf, stack_frames=True))   
#num_workers for parrallel processing 
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

valset = to_posix(FrameVideoDataset(root_dir=root_dir, split='val', transform=eval_tf, stack_frames=True))
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

def train(model, optimizer, scheduler=None, num_epochs=10):
    #Loss function defined (softmax across classes)
    loss_funct = nn.CrossEntropyLoss()
    history = []
    best_val, best_state = 0.0, None

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        #Build in method all nn.Module objects inherit
        model.train()
        #accumulate correct predictions and error per epoch
        train_correct = 0
        train_loss = []
        #per batch
        for clips, target in tqdm(train_loader, total=len(train_loader)):
            clips, target = clips.to(device), target.to(device)   # [B, C, T, H, W]
            optimizer.zero_grad()
            output = model(clips)
            loss = loss_funct(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_correct += (target == output.argmax(1)).sum().cpu().item()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        test_correct = 0
        test_total = 0
        for clips, labels in val_loader:
            clips = clips.to(device)
            with torch.no_grad():
                output = model(clips)
            predicted = output.argmax(1).cpu()
            test_correct += (labels == predicted).sum().item()
            test_total += labels.size(0)

        tr_acc = train_correct / len(trainset) * 100
        val_acc = test_correct / test_total * 100
        #store data per epoch in hisotry list
        history.append({'loss': float(np.mean(train_loss)),'train': tr_acc, 'val': val_acc})

        if val_acc > best_val:                                       
            best_val = val_acc                                       
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  

        print(f"Loss train: {np.mean(train_loss):.3f}\t",
            f"Accuracy train: {train_correct / len(trainset) * 100:.1f}%\t",
            f"test (video): {test_correct / test_total * 100:.1f}%")

    return history, best_state, best_val

#Same weight init and shuffle order to compare both models
def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

if __name__ == '__main__':
    NUM_EPOCHS = 50

    runs = {
        "late":  lambda: LateFusion(num_classes),
        "early": lambda: EarlyFusion(num_classes, num_frames=10),
    }

    results = {}
    #name is late or early, build is the lambda = the value
    for name, build in runs.items():
        set_seed(0)                                  
        model = build().to(device)
        #adamW optimizer used decays weights slightly towards zero to prevent weight from growing huge avoid overfititng
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        #adaptive change to learning rate decaying to zero at end
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        print(f"\n Training Model {name} fusion")
        history, best_state, best_val = train(model, optimizer, scheduler, num_epochs=NUM_EPOCHS)

        torch.save(best_state, f"best_{name}.pt")      # save best weights for later
        results[name] = {'history': history, 'best_val': best_val}
        print(f"{name} best val: {best_val:.1f}%")

        del model, optimizer                           # free GPU before next model
        torch.cuda.empty_cache()

    print("\nBest val accuracy")
    for name, r in results.items():
        print(f"{name:5s}: {r['best_val']:.1f}%")

    # comparison plot for the report
    for name, r in results.items():
        plt.plot([h['val'] for h in r['history']], label=f"{name} (val)")
    plt.xlabel("epoch"); plt.ylabel("val accuracy (%)")
    plt.legend(); plt.title("Late vs early fusion")
    plt.savefig("late_vs_early.png", dpi=150)
    print("\nSaved plot to late_vs_early.png")