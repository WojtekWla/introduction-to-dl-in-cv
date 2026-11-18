import torch
import torch.nn as nn


class EncDecCNN(nn.Module):
    def __init__(self):
        super(EncDecCNN, self).__init__()
        self.encoder = nn.Sequential(
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

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
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
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = conv_block(3, 8)        # 256
        self.enc2 = conv_block(8, 16)       # 128
        self.enc3 = conv_block(16, 16)      # 64
        self.enc4 = conv_block(16, 32)      # 32
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 32)    # 16

        self.up = nn.Upsample(scale_factor=2)
        self.dec4 = conv_block(32 + 32, 16)     # 32
        self.dec3 = conv_block(16 + 16, 16)     # 64
        self.dec2 = conv_block(16 + 16, 8)      # 128
        self.dec1 = conv_block(8 + 8, 8)        # 256

        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x, verbose=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

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


class UNetConvT(nn.Module):
    def __init__(self):
        super(UNetConvT, self).__init__()
        self.enc1 = conv_block(3, 8)        # 256
        self.enc2 = conv_block(8, 16)       # 128
        self.enc3 = conv_block(16, 16)      # 64
        self.enc4 = conv_block(16, 32)      # 32
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 32)    # 16

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(32 + 32, 16)     # 32
        self.dec3 = conv_block(16 + 16, 16)     # 64
        self.dec2 = conv_block(16 + 16, 8)      # 128
        self.dec1 = conv_block(8 + 8, 8)        # 256

        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x, verbose=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)

        if verbose:
            for name, t in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("bottleneck", b),
                            ("d4", d4), ("d3", d3), ("d2", d2), ("d1", d1), ("out", out)]:
                print(f"{name:10s} {tuple(t.shape)}")
        return out


def dice_score(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


def evaluate(model, loader, device, eps=1e-7):
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
        "sensitivity": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    }
