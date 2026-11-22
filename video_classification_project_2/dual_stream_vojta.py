import copy
import csv
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ============================== CONFIGURATION ===============================
DATA_DIR = Path('/dtu/datasets1/02516/ucf101_noleakage')
OUTPUT_DIR = Path('outputs')

# System variants to train: {name: start from ImageNet-pretrained weights?}.
VARIANTS = {'pretrained': True, 'scratch': False}
# 42 for the main run; the extra repetitions of the pretrained variant used 43 and 44.
SEED = 42
# True: the test split is evaluated once per trained model (use for the final run).
# While experimenting, set False so that the test split is never even looked at.
EVALUATE_TEST = True

# Data
RGB_FRAMES = 10                   # uniformly sampled frames per video
FLOW_IMAGES = 9                   # flows between consecutive frames (RGB_FRAMES - 1)
FLOW_CHANNELS = 2 * FLOW_IMAGES   # (u, v) per flow -> 18 input channels
IMAGE_SIZE = 224                  # frames and flows are resized to IMAGE_SIZE x IMAGE_SIZE (224 = native flow size)
LOAD_THREADS = 8
MAX_TRAIN = MAX_VAL = MAX_TEST = None   # e.g. 100 / 40 / 40 for a quick smoke test

# Training
EPOCHS = 30
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-2
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1             # training loss only; limits over-confidence on a tiny train set


def learning_rate(pretrained):
    """Fine-tuning pretrained weights needs a smaller step than training from scratch."""
    return 1e-4 if pretrained else 1e-3


EXPECTED_CLASSES = {
    'BodyWeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope',
    'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups',
}


# ================================== DATA ====================================
def read_samples(split, limit=None):
    """Parse metadata/{split}.csv into (frame_paths, flow_paths, label) tuples.

    Layout: frames/{split}/{action}/{video}/frame_{1..10}.jpg and
            flows/{split}/{action}/{video}/flow_{i}_{i+1}.npy  (i = 1..9).
    The CSV `label` column is authoritative. Also returns {label: action name}.
    """
    csv_path = DATA_DIR / 'metadata' / f'{split}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    samples, label_to_action = [], {}
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = csv.DictReader(handle)
        if not {'video_name', 'action', 'label'}.issubset(rows.fieldnames or []):
            raise ValueError(f'{csv_path}: missing CSV columns')
        for row in rows:
            name, action, label = row['video_name'], row['action'], int(row['label'])
            if label_to_action.setdefault(label, action) != action:
                raise ValueError(f'{split}: label {label} maps to two different actions')
            frame_dir = DATA_DIR / 'frames' / split / action / name
            flow_dir = DATA_DIR / 'flows' / split / action / name
            frames = [frame_dir / f'frame_{i}.jpg' for i in range(1, RGB_FRAMES + 1)]
            flows = [flow_dir / f'flow_{i}_{i + 1}.npy' for i in range(1, FLOW_IMAGES + 1)]
            missing = [str(p) for p in frames + flows if not p.is_file()]
            if missing:
                raise FileNotFoundError(f'{name} ({split}) missing data: {missing[:3]}')
            samples.append((frames, flows, label))
    random.Random(SEED).shuffle(samples)     # only matters when a MAX_* limit is set
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f'No samples found in {csv_path}')
    return samples, label_to_action


def flow_to_chw(array, path):
    """Validate a raw flow array (2,H,W) or (H,W,2) and return it as float32 (2,H,W) = (u,v)."""
    if array.ndim != 3:
        raise ValueError(f'Expected 3D optical-flow array, got {array.shape}: {path}')
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    elif array.shape[0] != 2:
        raise ValueError(f'Expected HxWx2 or 2xHxW, got {array.shape}: {path}')
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f'Non-finite flow values: {path}')
    return array


def load_video(sample):
    """Decode one video: uint8 RGB (10,3,S,S) and float16 flow (18,S,S) in pixel units."""
    frame_paths, flow_paths, _ = sample
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            frames.append(np.asarray(image, dtype=np.uint8).copy())
    rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    flows = torch.from_numpy(np.stack(
        [flow_to_chw(np.load(p, allow_pickle=False), p) for p in flow_paths]))   # 9,2,H,W
    _, _, h, w = flows.shape
    flows = nn.functional.interpolate(flows, size=(IMAGE_SIZE, IMAGE_SIZE),
                                      mode='bilinear', align_corners=False)
    # Flow is a displacement in pixels, so resizing the image rescales its magnitude too.
    flows[:, 0] *= IMAGE_SIZE / w
    flows[:, 1] *= IMAGE_SIZE / h
    return rgb, flows.reshape(FLOW_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).half()


def random_resized_crop(rgb, flow):
    """One random crop (50-100% of the area, aspect 3:4..4:3) for a whole video.

    The same crop is applied to all frames and flows, then resized back to IMAGE_SIZE.
    Zooming in magnifies motion, so the flow values are rescaled by the zoom factor.
    """
    s = IMAGE_SIZE
    area = random.uniform(0.5, 1.0) * s * s
    ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
    cw = min(s, int(round((area * ratio) ** 0.5)))
    ch = min(s, int(round((area / ratio) ** 0.5)))
    x0, y0 = random.randint(0, s - cw), random.randint(0, s - ch)
    resize = dict(size=(s, s), mode='bilinear', align_corners=False)
    rgb = nn.functional.interpolate(rgb[..., y0:y0 + ch, x0:x0 + cw], **resize)
    flow = nn.functional.interpolate(flow[None, :, y0:y0 + ch, x0:x0 + cw], **resize)[0]
    flow[0::2] *= s / cw
    flow[1::2] *= s / ch
    return rgb, flow


def mirror(rgb, flow):
    """Horizontally mirror a video: flip frames and flow, and negate the horizontal flow u.

    u sits in the even channels of the [u1,v1,u2,v2,...] stack.
    """
    flow = flow.flip(-1)
    flow[0::2] = -flow[0::2]
    return rgb.flip(-1), flow


class VideoDataset(Dataset):
    """All decoded videos of one split, held in RAM (a few hundred MB) so epochs are fast.

    Item: rgb (10,3,S,S) in [-0.5,0.5], flow (18,S,S) roughly in [-3,3], label.
    The training split gets random augmentation; val/test are returned unchanged.
    """

    def __init__(self, samples, train):
        self.train = train
        self.labels = torch.tensor([s[2] for s in samples])
        with ThreadPoolExecutor(LOAD_THREADS) as pool:
            videos = list(pool.map(load_video, samples))
        self.rgb = torch.stack([v[0] for v in videos])     # N,10,3,S,S uint8
        self.flow = torch.stack([v[1] for v in videos])    # N,18,S,S   float16 (pixels)
        self.flow_scale = 1.0                              # set once from the train split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = self.rgb[idx].float() / 255.0 - 0.5
        flow = (self.flow[idx].float() / self.flow_scale).clamp(-3, 3)
        if self.train:
            rgb, flow = random_resized_crop(rgb, flow)
            if random.random() < 0.5:
                rgb, flow = mirror(rgb, flow)
        return rgb, flow, self.labels[idx]


def estimate_flow_scale(dataset, max_videos=50):
    """Flow normaliser: the 99th percentile of |flow| in pixels, measured on train only."""
    step = max(1, len(dataset) // max_videos)
    values = dataset.flow[::step].float().abs().numpy().ravel()
    scale = float(np.percentile(values, 99))
    print(f'Flow |value| (train, px): mean={values.mean():.3f} p99={scale:.3f} '
          f'max={values.max():.3f}', flush=True)
    return max(scale, 1e-3)


def load_data():
    """Read, validate and decode all three splits. Returns (datasets, class_names)."""
    splits = {}
    for split, limit in [('train', MAX_TRAIN), ('val', MAX_VAL), ('test', MAX_TEST)]:
        splits[split] = read_samples(split, limit)
    mapping = splits['train'][1]
    class_names = [action for _, action in sorted(mapping.items())]
    if set(class_names) != EXPECTED_CLASSES or sorted(mapping) != list(range(10)):
        raise ValueError(f'Unexpected classes in CSV: {mapping}')
    if any(split_mapping != mapping for _, split_mapping in splits.values()):
        raise ValueError('train/val/test CSVs disagree on the label -> action mapping')
    print('Videos: ' + ', '.join(f'{k}={len(v[0])}' for k, v in splits.items()), flush=True)

    first_flow = splits['train'][0][0][1][0]
    raw = np.load(first_flow, allow_pickle=False)
    print(f'Raw flow file {first_flow.name}: shape={raw.shape} dtype={raw.dtype}', flush=True)
    first_frame = splits['train'][0][0][0][0]
    with Image.open(first_frame) as image:
        print(f'Raw frame {first_frame.name}: size (W x H) = {image.size}', flush=True)

    datasets = {split: VideoDataset(samples, train=(split == 'train'))
                for split, (samples, _) in splits.items()}
    scale = estimate_flow_scale(datasets['train'])
    for dataset in datasets.values():
        dataset.flow_scale = scale
    return datasets, class_names


# ================================== MODELS ==================================
def build_backbone(in_channels, pretrained):
    """ResNet18 without its classifier: (B,in_channels,H,W) -> (B,512) feature vector.

    pretrained=True starts from ImageNet weights, otherwise from random weights.
    For the flow stack (in_channels = 18) the first conv layer is rebuilt for 18 inputs:
      * pretrained: the ImageNet RGB filters are averaged over colour and copied to every
        channel, scaled by 3/in_channels so the response magnitude matches the original conv
        (cross-modality pre-training; there is no ImageNet model for flow);
      * from scratch: random init, the same one torchvision uses for every ResNet conv layer.
    """
    import torchvision
    if pretrained:
        try:
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as error:
            raise RuntimeError('Could not load ImageNet ResNet18 weights. Cache them once on '
                               f'the login node (compute nodes may lack internet): {error}')
    else:
        net = torchvision.models.resnet18(weights=None)
    if in_channels != 3:
        old = net.conv1
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True)
                                       .repeat(1, in_channels, 1, 1) * 3 / in_channels)
            else:
                nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    net.fc = nn.Identity()
    return net, 512


class SpatialNet(nn.Module):
    """Spatial stream: a 2D CNN on every RGB frame; the per-frame scores are averaged."""

    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.backbone, dim = build_backbone(3, pretrained)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(dim, num_classes))
        self.imagenet_norm = pretrained          # only pretrained weights expect ImageNet stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb, flow):                       # rgb (B,T,3,H,W); flow is unused
        b, t = rgb.shape[:2]
        x = rgb.flatten(0, 1)                           # frames become a big batch: B*T,3,H,W
        if self.imagenet_norm:
            x = (x + 0.5 - self.mean) / self.std
        scores = self.head(self.backbone(x))            # B*T,C
        return scores.reshape(b, t, -1).mean(dim=1)     # B,C  (average over time)


class TemporalNet(nn.Module):
    """Temporal stream: one 2D CNN whose first layer sees all 18 flow channels (early fusion)."""

    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.backbone, dim = build_backbone(FLOW_CHANNELS, pretrained)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(dim, num_classes))

    def forward(self, rgb, flow):                       # flow (B,18,H,W); rgb is unused
        return self.head(self.backbone(flow))


# ================================= TRAINING =================================
def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    for rgb, flow, labels in loader:
        rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(rgb, flow)
            loss = nn.functional.cross_entropy(
                logits, labels, label_smoothing=LABEL_SMOOTHING if training else 0.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count += labels.numel()
        total_loss += loss.item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return {'loss': total_loss / count, 'accuracy': correct / count}


def fit(tag, model, lr, loaders, device, out_dir):
    """Train one stream for EPOCHS and restore its best-validation-accuracy weights."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history, best_acc, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        train = run_epoch(model, loaders['train'], device, optimizer)
        val = run_epoch(model, loaders['val'], device)
        scheduler.step()
        history.append({'epoch': epoch, 'train': train, 'val': val})
        print(f'[{tag}] epoch {epoch:02d}/{EPOCHS} | '
              f'train loss {train["loss"]:.3f} acc {train["accuracy"]:.1%} | '
              f'val loss {val["loss"]:.3f} acc {val["accuracy"]:.1%} | '
              f'{time.time() - started:.1f}s', flush=True)
        if val['accuracy'] > best_acc:
            best_acc, best_epoch = val['accuracy'], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'epoch': best_epoch, 'val_accuracy': best_acc},
               out_dir / f'{tag.split("/")[-1]}_best.pt')
    print(f'[{tag}] best val acc {best_acc:.1%} at epoch {best_epoch}', flush=True)
    return history, best_epoch


# ================================ EVALUATION ================================
@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax class scores (N,C) and labels (N,); the loader must not shuffle."""
    model.eval()
    probs, labels = [], []
    for rgb, flow, y in loader:
        probs.append(model(rgb.to(device), flow.to(device)).softmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


def evaluate(models, loaders, device, num_classes):
    """Accuracy of each stream and of the fusion on val (and on test if EVALUATE_TEST).

    Fusion = average of the two softmax outputs ("class score fusion").
    Returns (results, test confusion matrix of the fusion or None).
    """
    results = {name: {} for name in ('spatial', 'temporal', 'fusion')}
    confusion = None
    for split in (['val', 'test'] if EVALUATE_TEST else ['val']):
        probs = {}
        for name, model in models.items():
            probs[name], labels = predict_probs(model, loaders[split], device)
        probs['fusion'] = (probs['spatial'] + probs['temporal']) / 2
        for name, p in probs.items():
            results[name][f'{split}_accuracy'] = accuracy(p, labels)
        if split == 'test':
            confusion = np.zeros((num_classes, num_classes), dtype=int)
            for true, pred in zip(labels.tolist(), probs['fusion'].argmax(dim=1).tolist()):
                confusion[true, pred] += 1
    return results, confusion


# ================================ REPORTING =================================
def make_figures(histories, confusion, class_names, title, out_dir):
    """Training curves of both streams and the test confusion matrix of the fused model."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipped plots.', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel in [(axes[0], 'loss', 'Cross-entropy loss'),
                            (axes[1], 'accuracy', 'Accuracy')]:
        for name, color in [('spatial', 'tab:blue'), ('temporal', 'tab:orange')]:
            epochs = [r['epoch'] for r in histories[name]]
            ax.plot(epochs, [r['train'][key] for r in histories[name]], color=color,
                    ls='--', label=f'{name} train')
            ax.plot(epochs, [r['val'][key] for r in histories[name]], color=color,
                    label=f'{name} val')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.grid(True, alpha=.25)
    axes[1].legend()
    fig.savefig(out_dir / 'curves.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    if confusion is None:                     # no test evaluation in this run
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap='Blues')
    ticks = range(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=60, ha='right')
    ax.set_yticks(ticks, class_names)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, int(confusion[i, j]), ha='center', va='center',
                    color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    ax.set(xlabel='Predicted', ylabel='True', title=f'{title}: fusion, test set')
    fig.savefig(out_dir / 'confusion_test.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


def results_table(all_results):
    """Markdown table: val (and test, if evaluated) accuracy of each stream and the fusion."""
    split = 'val / test' if EVALUATE_TEST else 'val'
    lines = [f'Accuracy ({split})', '',
             '| Variant | Spatial | Temporal | Fusion |', '|---|---|---|---|']
    for variant, results in all_results.items():
        cells = []
        for stream in ('spatial', 'temporal', 'fusion'):
            r = results[stream]
            cells.append(f'{r["val_accuracy"]:.1%}'
                         + (f' / {r["test_accuracy"]:.1%}' if 'test_accuracy' in r else ''))
        lines.append(f'| {variant} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ================================== MAIN ====================================
def run_variant(name, pretrained, datasets, class_names, device):
    """Train both streams of one variant, fuse them, evaluate, and write its outputs."""
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'\n===== Variant "{name}" (ImageNet pretrained: {pretrained}) =====', flush=True)

    loaders = {split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == 'train'),
                                 drop_last=(split == 'train'), pin_memory=True)
               for split, ds in datasets.items()}
    num_classes = len(class_names)
    models = {'spatial': SpatialNet(num_classes, pretrained).to(device),
              'temporal': TemporalNet(num_classes, pretrained).to(device)}

    histories, best_epochs = {}, {}
    for stream, model in models.items():           # each stream has its own training run
        histories[stream], best_epochs[stream] = fit(
            f'{name}/{stream}', model, learning_rate(pretrained), loaders, device, out_dir)

    results, confusion = evaluate(models, loaders, device, num_classes)
    print(f'--- "{name}" results (best-validation checkpoints) ---\n'
          + results_table({name: results}), flush=True)

    (out_dir / 'metrics.json').write_text(json.dumps({
        'variant': name, 'imagenet_pretrained': pretrained, 'results': results,
        'best_epoch': best_epochs, 'class_names': class_names,
        'confusion_test_fusion': None if confusion is None else confusion.tolist(),
        'settings': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'image_size': IMAGE_SIZE,
                     'learning_rate': learning_rate(pretrained), 'weight_decay': WEIGHT_DECAY,
                     'dropout': DROPOUT, 'label_smoothing': LABEL_SMOOTHING, 'seed': SEED},
        'history': histories}, indent=2))
    make_figures(histories, confusion, class_names, name, out_dir)
    return results


def main():
    beginning = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LSF job: {os.environ.get("LSB_JOBID", "local")}; device: {device}; '
          f'data: {DATA_DIR}', flush=True)
    if device.type != 'cuda':
        raise RuntimeError('No CUDA GPU found. This script is intended for the DTU GPU job.')
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {DATA_DIR}')

    print('Decoding all videos into RAM...', flush=True)
    datasets, class_names = load_data()
    print(f'Data ready after {time.time() - beginning:.0f}s.', flush=True)

    all_results = {}
    for name, pretrained in VARIANTS.items():
        all_results[name] = run_variant(name, pretrained, datasets, class_names, device)
        torch.cuda.empty_cache()
        # Rewrite the summary after every variant so a later failure keeps earlier results.
        (OUTPUT_DIR / 'summary.json').write_text(json.dumps(all_results, indent=2))
        (OUTPUT_DIR / 'summary.md').write_text(results_table(all_results) + '\n')

    print('\n===== SUMMARY =====\n' + results_table(all_results), flush=True)
    print(f'Finished in {time.time() - beginning:.0f}s. '
          f'Files under {OUTPUT_DIR.resolve()}', flush=True)


if __name__ == '__main__':
    main()
