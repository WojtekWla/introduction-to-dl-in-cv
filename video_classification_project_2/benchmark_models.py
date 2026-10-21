from early_fusion import EarlyFusion
from late_fusion import LateFusion
from single_frame_cnn import SingleFrame
from cnn3D import CNN3D
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from datasets import FrameVideoDataset, FrameImageDataset
import matplotlib.pyplot as plt

OUTPUT_DIR = '/zhome/2b/7/233718/idlcv/output'
ROOT_DIR = '/dtu/datasets1/02516/ufc10'
NUM_CLASSES = 10

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

def fit(model, train_loader, val_loader, eval_fn, lr=0.001, num_epochs=10, weight_decay=1e-4):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = eval_fn(model, val_loader, loss_fn)
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


def data_loader(model_name):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    test_loader, train_loader, validate_loader = None, None, None
    if model_name == "single_frame":
        test_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='test', stack_frames=True),
            batch_size=64, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
        
        train_loader = DataLoader(
            FrameImageDataset(root_dir=ROOT_DIR, split='train'),
            batch_size=64, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
        )
        
        validate_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='val', stack_frames=True),
            batch_size=64, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
    elif model_name == "early_fusion":
        test_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='test', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
        
        train_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='train', transform=train_tf),
            batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
        )
    
        validate_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='val', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
    elif model_name == "late_fusion":
        test_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='test', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )

        train_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='train', transform=train_tf),
            batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
        )

        validate_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='val', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
    else:
        test_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='test', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )
        
        train_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='train', transform=train_tf),
            batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True
        )

        validate_loader = DataLoader(
            FrameVideoDataset(root_dir=ROOT_DIR, split='val', stack_frames=True),
            batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
        )

    return test_loader, train_loader, validate_loader
    

if __name__ == "__main__":
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

    models = {
        "single_frame": SingleFrame(num_classes=NUM_CLASSES),
        "early_fusion": EarlyFusion(num_classes=NUM_CLASSES,  n_frames=10),
        "late_fusion": LateFusion(num_classes=NUM_CLASSES),
        "cnn3D": CNN3D(num_classes=NUM_CLASSES)
    }

    log_path = f'{OUTPUT_DIR}/benchmark_results.log'
    with open(log_path, 'w') as f:
        f.write('model,test_loss,test_acc\n')

    for name, model in models.items():
        print(f"Evaluating {name}")
        model = model.to(device)
        test_loader, train_loader, validate_loader = data_loader(name)


        eval_fn = evaluate_video_single_frame if name == "single_frame" else evaluate_video
        model, hist_scratch = fit(model, train_loader, validate_loader, eval_fn, 0.001, num_epochs=50)
        
        test_loss, test_acc = eval_fn(model, test_loader, nn.CrossEntropyLoss(label_smoothing=0.1))
        print(f'test loss {test_loss:.3f} | test video acc {test_acc:.3f}')

        with open(log_path, 'a') as f:
            f.write(f'{name},{test_loss:.4f},{test_acc:.4f}\n')

        epochs = range(1, len(hist_scratch['train_loss']) + 1)
        plt.figure()
        plt.plot(epochs, hist_scratch['train_loss'], label='train loss')
        plt.plot(epochs, hist_scratch['val_loss'], label='val loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f'{name} loss')
        plt.savefig(f'{OUTPUT_DIR}/{name}_loss.png', dpi=150, bbox_inches='tight')
    
        plt.figure()
        plt.plot(epochs, hist_scratch['train_acc'], label='train acc')
        plt.plot(epochs, hist_scratch['val_acc'], label='val acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.title(f'{name} accuracy')
        plt.savefig(f'{OUTPUT_DIR}/{name}_acc.png', dpi=150, bbox_inches='tight')
