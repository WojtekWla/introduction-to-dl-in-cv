import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import copy
from datasets import FrameVideoDataset

device = None
class LateFusion(nn.Module):
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
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0,2,1,3,4)
        x = x.reshape(B*T, C, H, W)

        x = self.conv(x)
        x = x.view(B, T, -1)
        x = x.mean(1)

        return self.linear(x)

def train_one_epoch(model, loader, optmizer, loss_fn):
    model.train()

    train_loss, correct, n = 0.0, 0, 0

    for frames, labels in tqdm(loader, leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optmizer.zero_grad()
        res = model(frames)
        loss = loss_fn(res, labels)
        loss.backward()
        optmizer.step()

        train_loss += loss.item() * labels.size(0)
        correct += (res.argmax(1) == labels).sum().item()

        n += labels.size(0)
    return train_loss / n, correct / n

@torch.no_grad()
def evaluate_video(model, loader, loss_fn):
    model.eval()
    eval_loss, correct, n = 0.0, 0, 0
    for clips, labels in tqdm(loader, leave=False):
        clips = clips.to(device)
        labels = labels.to(device)
        res = model(clips)
        loss = loss_fn(res, labels)

        eval_loss += loss.item() * labels.size(0)
        correct += (res.argmax(1) == labels).sum().item()
        n += labels.size(0)

    return eval_loss/n, correct/n

def fit(model, train_loader, eval_loader, lr, num_epochs=10, weight_decay=1e-4):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epochs)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn)
        eval_loss, eval_accuracy = evaluate_video(model, eval_loader, loss_fn)
        scheduler.step()
        
        for k, v in zip(history, [train_loss, train_accuracy, eval_loss, eval_accuracy]):
            history[k].append(v)

        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(f'epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_accuracy:.3f} '
            f'| val loss {eval_loss:.3f} video acc {eval_accuracy:.3f}'
            f'{"  *best*" if eval_accuracy == best_acc else ""}')

    model.load_state_dict(best_state)
    return model, history


if __name__ == '__main__':
    batch_size = 16
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
    root_dir = "/Users/wojciechwladzinski/projects/uni/introduction-to-dl-in-cv/data/ufc10"

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor()])

    test_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='test', stack_frames=True, transform=eval_tf),
        batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
    )

    train_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='train', transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
    )

    validate_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='val', stack_frames=True, transform=eval_tf),
        batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
    )

    model = LateFusion(num_classes=10).to(device)
    model, hist_scratch = fit(model, train_loader, validate_loader, 0.001, num_epochs=50)

    test_loss, test_acc = evaluate_video(model, test_loader, nn.CrossEntropyLoss(label_smoothing=0.1))
    print(f'test loss {test_loss:.3f} | test video acc {test_acc:.3f}')

    epochs = range(1, len(hist_scratch['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, hist_scratch['train_loss'], label='train loss')
    plt.plot(epochs, hist_scratch['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('LateFusion loss')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/late_fusion_loss.png', dpi=150, bbox_inches='tight')

    plt.figure()
    plt.plot(epochs, hist_scratch['train_acc'], label='train acc')
    plt.plot(epochs, hist_scratch['val_acc'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('LateFusion accuracy')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/late_fusion_acc.png', dpi=150, bbox_inches='tight')
    plt.show()