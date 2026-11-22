import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import copy
from datasets import FrameVideoDataset

device = None

# Kinetics-400 normalization stats used by torchvision's pretrained video models
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


class CNN3DPretrained(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # freeze the early, generic layers (stem + layer1 + layer2); only
        # layer3, layer4, and the new head get fine-tuned on the small
        # no-leakage training set, to limit how much capacity can overfit
        for module in (self.backbone.stem, self.backbone.layer1, self.backbone.layer2):
            for param in module.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()

    train_loss, correct, n = 0.0, 0, 0
    for clips, labels in tqdm(loader, leave=False):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()

        res = model(clips)
        loss = loss_fn(res, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        correct += (res.argmax(1) == labels).sum().item()

        n += labels.size(0)

    return train_loss / n, correct / n

@torch.no_grad()
def evaluate_video(model, loader, loss_fn):
    model.eval()
    eval_loss, correct, n = 0.0, 0, 0
    for clips, labels in tqdm(loader, leave=False):
        clips, labels = clips.to(device), labels.to(device)

        res = model(clips)
        loss = loss_fn(res, labels)

        eval_loss += loss.item() * labels.size(0)
        correct += (res.argmax(1) == labels).sum().item()
        n += labels.size(0)

    return eval_loss / n, correct / n

def fit(model, train_loader, val_loader, lr=0.001, num_epochs=10, weight_decay=1e-4):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params=trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate_video(model, val_loader, loss_fn)
        scheduler.step()

        for k, v in zip(history, [train_loss, train_acc, val_loss, val_acc]):
            history[k].append(v)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f'epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} '
            f'| val loss {val_loss:.3f} video acc {val_acc:.3f}'
            f'{"  *best*" if val_acc == best_acc else ""}')

    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    print("Transfer learning cnn 3D")
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

    root_dir = "/dtu/datasets1/02516/ucf101_noleakage"

    train_tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(KINETICS_MEAN, KINETICS_STD),
    ])
    eval_tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(KINETICS_MEAN, KINETICS_STD),
    ])

    test_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='test', stack_frames=True, transform=eval_tf),
        batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True
    )

    train_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='train', transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True
    )

    validate_loader = DataLoader(
        FrameVideoDataset(root_dir=root_dir, split='val', stack_frames=True, transform=eval_tf),
        batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True
    )

    model = CNN3DPretrained(num_classes=10).to(device)
    # lower lr than the scratch CNN3D: fine-tuning pretrained weights, not training from scratch
    model, hist_scratch = fit(model, train_loader, validate_loader, 0.0001, num_epochs=30)

    test_loss, test_acc = evaluate_video(model, test_loader, nn.CrossEntropyLoss(label_smoothing=0.1))
    print(f'test loss {test_loss:.3f} | test video acc {test_acc:.3f}')

    epochs = range(1, len(hist_scratch['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, hist_scratch['train_loss'], label='train loss')
    plt.plot(epochs, hist_scratch['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('3D CNN (pretrained) loss')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/3D_CNN_pretrained_loss.png', dpi=150, bbox_inches='tight')

    plt.figure()
    plt.plot(epochs, hist_scratch['train_acc'], label='train acc')
    plt.plot(epochs, hist_scratch['val_acc'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('3D CNN (pretrained) accuracy')
    plt.savefig('/zhome/2b/7/233718/idlcv/output/3D_CNN_pretrained_acc.png', dpi=150, bbox_inches='tight')
    plt.show()
