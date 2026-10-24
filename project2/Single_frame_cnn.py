"""
Single-frame CNN basecase for Project 2
2D classifier without time awareness followed by averaging per frame probabilites

Compares results with custom architecture and using pre-trained model
    python single_frame.py --arch scratch    
    python single_frame.py --arch resnet18     # ImageNet-pretrained
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
#datsets.py functions
from datasets import FrameImageDataset, FrameVideoDataset

#Dataloader in datasets.py gives us frames of shape [B, 3, H, W] after batching 
#B is the label assigned to the frame (0-9) ~ 10 classes actions

#check if GPU available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = '/dtu/datasets1/02516/ufc10'
num_classes = 10
batch_size = 32

# Windows fix: glob in datasets.py returns backslash paths, but datasets.py
# parses them with split('/'). Flip the stored paths to forward slashes so its
# split('/') finds the right video_name. datasets.py itself is left untouched.
def to_posix(ds):
    if hasattr(ds, 'frame_paths'):
        ds.frame_paths = [p.replace('\\', '/') for p in ds.frame_paths]
    if hasattr(ds, 'video_paths'):
        ds.video_paths = [p.replace('\\', '/') for p in ds.video_paths]
    return ds


#Model Definition
#Batch extracted are random image frames from any of the clips
#With batch of 32 frame images -> input [32, 3, 64, 64] 
class single_frame(nn.Module):
    def __init__(self, num_classes = 10):
        super(single_frame, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3,16,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   #maxpooling(2x2 window) halves both H&W 64x64->32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  #32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  #16x16 -> 8x8
        )

        self.fully_connected = nn.Sequential(
            #8x8 is spatial size after convolution and we flatten it into 1 vector stack given last out channels - 64
            nn.Linear(8 * 8 * 64, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.convolutional(x)
        # reshape x so it becomes flat, except for the first dimension (minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# An ImageNet-pretrained ResNet18 with the head swapped for our 10 classes (transfer learning)
def pretrained_resnet18(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


#Training on individual frames and evaluating on the full video

def make_loader(pretrained):
    if pretrained:
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   #mean average set , stand dev set for imagenet
        train_tf = transforms.Compose([transforms.Resize((112, 112)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), norm])
        eval_tf = transforms.Compose([transforms.Resize((112, 112)),
                                      transforms.ToTensor(), norm])
    else:
        train_tf = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
        eval_tf = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor()])

     # Per-frame dataset for training
    #looks inside directory for train folder and applies the above tranf
    trainset = to_posix(FrameImageDataset(root_dir=root_dir, split='train', transform=train_tf))
    #pytorch dataloader acting as delivery mechanism
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
 
    # Per-video dataset for evaluation (stacks the 10 frames [C, T, H, W])
    valset = to_posix(FrameVideoDataset(root_dir=root_dir, split='val',transform=eval_tf, stack_frames=True))
    val_loader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=4)
 
    return trainset, train_loader, val_loader


#Training
def train(model, optimizer, trainset, train_loader, val_loader, num_epochs=10):
    loss_fun = nn.CrossEntropyLoss()
    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': []}
 
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        #train on individual frames
        model.train()
        train_correct = 0
        train_loss = []
        for data, target in tqdm(train_loader, total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()
 
            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
 
        # evaluate per video: softmax 10 frames, average, argmax
        model.eval()
        test_correct = 0
        test_total = 0
        for clips, labels in val_loader:              # clips: [B, C, T, H, W]
            B, C, Tn, H, W = clips.shape
            x = clips.permute(0, 2, 1, 3, 4).reshape(B * Tn, C, H, W).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)     # [B*T, num_classes]
            probs = probs.view(B, Tn, -1).mean(1)      # average over frames
            predicted = probs.argmax(1).cpu()
            test_correct += (labels == predicted).sum().item()
            test_total += B
 
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['test_acc'].append(test_correct / test_total)
        out_dict['train_loss'].append(np.mean(train_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1] * 100:.1f}%\t",
              f"test (video): {out_dict['test_acc'][-1] * 100:.1f}%")
    return out_dict


#Run both and compare
#Run both and compare
if __name__ == '__main__':
    # Scratch CNN
    trainset, train_loader, val_loader = make_loader(pretrained=False)
    model = single_frame(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out_scratch = train(model, optimizer, trainset, train_loader, val_loader, num_epochs=10)

    # Pretrained ResNet18
    trainset, train_loader, val_loader = make_loader(pretrained=True)
    model = pretrained_resnet18(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    out_pretrained = train(model, optimizer, trainset, train_loader, val_loader, num_epochs=10)

    print(f"\nBest val video accuracy: scratch: {max(out_scratch['test_acc'])*100:.1f}%"
          f"   pretrained: {max(out_pretrained['test_acc'])*100:.1f}%")

    # Plot both test (video-level) accuracies on one figure
    epochs = np.arange(1, len(out_scratch['test_acc']) + 1)
    plt.figure()
    plt.plot(epochs, out_scratch['test_acc'], 'r', label='Scratch CNN')
    plt.plot(epochs, out_pretrained['test_acc'], 'b', label='Pretrained ResNet18')
    plt.legend()
    plt.xlabel('Epoch number')
    plt.ylabel('Video-level accuracy')
    plt.title('Single-frame baseline: scratch vs pretrained')
    plt.savefig('single_frame_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()