import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from datasets import FrameVideoDataset, FrameImageDataset


device = None
class SingleFrame(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.seq(x)  


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for frames, labels in tqdm(loader, leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(frames)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate_video(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for clips, labels in loader:
        B, C, T, H, W = clips.shape
        x = clips.permute(0, 2, 1, 3, 4)
        x = x.reshape(B*T, C, H, W).to(device)
        probs = F.softmax(model(x), dim=1).view(B, T, -1).mean(1)
        labels = labels.to(device)
        total_loss += F.nll_loss(torch.log(probs + 1e-8), labels, reduction='sum').item()
        correct += (probs.argmax(1) == labels).sum().item()

        n += B
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate_video_single_frame(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for clips, labels in tqdm(loader, leave=False):
        B, C, T, H, W = clips.shape
        x = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(device)
        probs = F.softmax(model(x), dim=1).view(B, T, -1).mean(1)
        labels = labels.to(device)
        total_loss += F.nll_loss(torch.log(probs + 1e-8), labels, reduction='sum').item()
        correct += (probs.argmax(1) == labels).sum().item()
        n += B
    return total_loss / n, correct / n

def fit(model, train_loader, val_loader, lr, num_epochs=10, weight_decay=1e-4):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        va_loss, va_acc = evaluate_video(model, val_loader, loss_fn)
        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f'epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} '
              f'| val loss {va_loss:.3f} video acc {va_acc:.3f}'
              f'{"  *best*" if va_acc == best_acc else ""}')

    model.load_state_dict(best_state)

    return model, history


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU available")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("MPS available")
        device = torch.device('mps')
    else:
        print("Running on cpu")
        device = torch.device('cpu')

    torch.manual_seed(0)
    np.random.seed(0)

    # root_dir = "/dtu/datasets1/02516/ufc10"
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    test_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='test', stack_frames=True),
        batch_size=64, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
    )

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        FrameImageDataset(root_dir=root_dir, split='train', transform=train_tf),
        batch_size=64, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
    )

    validate_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='val', stack_frames=True),
        batch_size=64, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
    )

    model = SingleFrame(num_classes=10).to(device)
    model, hist_scratch = fit(model, train_loader, validate_loader, 0.001, num_epochs=30)

    test_loss, test_acc = evaluate_video(model, test_loader, nn.CrossEntropyLoss(label_smoothing=0.1))
    print(f'test loss {test_loss:.3f} | test video acc {test_acc:.3f}')

    epochs = range(1, len(hist_scratch['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, hist_scratch['train_loss'], label='train loss')
    plt.plot(epochs, hist_scratch['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('SingleFrame loss')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/single_frame_loss.png', dpi=150, bbox_inches='tight')

    plt.figure()
    plt.plot(epochs, hist_scratch['train_acc'], label='train acc')
    plt.plot(epochs, hist_scratch['val_acc'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('SingleFrame accuracy')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/single_frame_acc.png', dpi=150, bbox_inches='tight')
    plt.show()

