"""
First Model: CNN constructed by scratch to prcess pothole proposal crops
Input: 3 x 128 x 128 crop 
Ouput: 2 logits
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

num_classes = 2

class PotholeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> 256 x 1 x 1
            nn.Flatten(),             # -> 256
            nn.Dropout(0.3),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self,x):
        x = self.convolution(x)
        return self.classifier(x)


class PotholeResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # freeze every weight
        for param in self.backbone.parameters():
            param.requires_grad = False

        # replace the 1000-class head with own
        # so the new head is the only thing that learns in stage 1.
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 254),
            nn.ReLU(),
            nn.Linear(254,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
        )

    def unfreeze_layer4(self):
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone.conv1(x)       # 3 x 224 x 224 -> 64 x 112 x 112
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)     # -> 64 x 56 x 56

        x = self.backbone.layer1(x)      # -> 64 x 56 x 56
        x = self.backbone.layer2(x)      # -> 128 x 28 x 28
        x = self.backbone.layer3(x)      # -> 256 x 14 x 14
        x = self.backbone.layer4(x)      # -> 512 x 7 x 7

        x = self.backbone.avgpool(x)     # -> 512 x 1 x 1
        x = torch.flatten(x, 1)          # -> 512
        x = self.backbone.fc(x)          # -> 2
        return x


if __name__ == "__main__":
    model = PotholeCNN()
    out = model(torch.randn(4, 3, 128, 128))
    n = sum(p.numel() for p in model.parameters())
    print(f"scratch  output {tuple(out.shape)}   parameters {n/1e6:.2f}M")

    model = PotholeResNet18()
    out = model(torch.randn(4, 3, 224, 224))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"stage 1  output {tuple(out.shape)}   "
          f"{total/1e6:.2f}M params, {trainable} trainable")

    model.unfreeze_layer4()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"stage 2  after unfreeze_layer4(): {trainable/1e6:.2f}M trainable")