"""Proposal-crop dataset for training (4.2) and testing (4.3) the pothole classifier.

Each item is one proposal of one image:
    crop       transformed image crop (tensor, e.g. 3x224x224)
    label      1 = pothole (best IoU >= pos_iou), 0 = background (< neg_iou)
    box        (xmin, ymin, xmax, ymax) in original image pixels
    image_idx  index into dataset.image_names (for grouping detections per image in 4.3)

Proposals with neg_iou <= IoU < pos_iou are dropped for training; keep_all=True keeps
every proposal (needed for NMS + AP in 4.3, where nothing may be thrown away).

Typical use:
    train = PotholeProposals("train", make_transform(train=True))
    loader = DataLoader(train, batch_size=64, sampler=balanced_sampler(train), num_workers=4)
    val = PotholeProposals("val", make_transform(train=False))

Run `python dataset.py` for a summary and an image of example crops.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from evaluate_proposals import ANNOTATIONS_DIR, SPLITS_FILE
from groundtruth import parse_boxes
from label_proposals import LABELS_DIR, NEG_IOU, POS_IOU
from proposals import IMAGES_DIR, N_PROPOSALS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(train: bool, size: int = 224, imagenet_norm: bool = True) -> Callable:
    """Crop -> fixed-size tensor. Pretrained ImageNet models need imagenet_norm=True."""
    steps = [transforms.Resize((size, size))]
    if train:
        # A pothole flipped left-right is still a pothole; cheap extra data.
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    if imagenet_norm:
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


class PotholeProposals(Dataset):
    def __init__(
        self,
        split: str,
        transform: Callable,
        n_proposals: int = N_PROPOSALS,
        pos_iou: float = POS_IOU,
        neg_iou: float = NEG_IOU,
        keep_all: bool = False,
        add_gt_boxes: bool = False,
        images_dir: Path = IMAGES_DIR,
        labels_dir: Path = LABELS_DIR,
        annotations_dir: Path = ANNOTATIONS_DIR,
        splits_file: Path = SPLITS_FILE,
    ):
        """
        split:        "train", "val" or "test" (from splits.json)
        n_proposals:  use the first N proposals of each image
        keep_all:     keep the ignored IoU band too (labelled by pos_iou); use for 4.3
        add_gt_boxes: add every ground-truth box as an extra positive (R-CNN trick;
                      covers the ~9% of potholes no proposal matches). Training only.
        """
        self.transform = transform
        self.images_dir = Path(images_dir)
        self._cache: dict = {}  # image_idx -> PIL image; images are small, reloading per crop is slow

        names = json.loads(Path(splits_file).read_text())[split]
        self.image_names = []
        boxes, labels, image_idx = [], [], []
        for name in names:
            stem = Path(name).stem
            label_file = Path(labels_dir) / f"{stem}.npz"
            if not label_file.exists():  # e.g. locally, where only sample images are labelled
                continue
            data = np.load(label_file)
            b = data["boxes"][:n_proposals]
            iou = data["max_iou"][:n_proposals]

            if keep_all:
                keep = np.ones(len(iou), dtype=bool)
            else:
                keep = (iou >= pos_iou) | (iou < neg_iou)
            b, lab = b[keep], (iou[keep] >= pos_iou).astype(np.int64)

            if add_gt_boxes:
                gt = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1
                               in parse_boxes(Path(annotations_dir) / f"{stem}.xml")], dtype=b.dtype)
                b = np.concatenate([b, gt])
                lab = np.concatenate([lab, np.ones(len(gt), dtype=np.int64)])

            idx = len(self.image_names)
            self.image_names.append(name)
            boxes.append(b)
            labels.append(lab)
            image_idx.append(np.full(len(lab), idx, dtype=np.int64))

        if not self.image_names:
            raise FileNotFoundError(f"No labelled images for split '{split}' in {labels_dir}. "
                                    "Run proposals.py and label_proposals.py first.")
        self.boxes = np.concatenate(boxes).astype(np.float32)
        self.labels = np.concatenate(labels)
        self.image_idx = np.concatenate(image_idx)

    def __len__(self) -> int:
        return len(self.labels)

    def _image(self, idx: int) -> Image.Image:
        if idx not in self._cache:
            # PIL + RGB: cv2 would give BGR, which silently breaks ImageNet-pretrained models.
            self._cache[idx] = Image.open(self.images_dir / self.image_names[idx]).convert("RGB")
        return self._cache[idx]

    def __getitem__(self, i: int):
        image = self._image(int(self.image_idx[i]))
        x0, y0, x1, y1 = self.boxes[i]
        # Round outward and keep at least 1 px, inside the image.
        left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        right = min(image.width, max(left + 1, int(np.ceil(x1))))
        bottom = min(image.height, max(top + 1, int(np.ceil(y1))))
        crop = image.crop((left, top, right, bottom))
        return (self.transform(crop), int(self.labels[i]),
                torch.from_numpy(self.boxes[i]), int(self.image_idx[i]))

    def class_counts(self) -> dict:
        n_pos = int(self.labels.sum())
        return {"pothole": n_pos, "background": len(self.labels) - n_pos}


def balanced_sampler(dataset: PotholeProposals, pos_fraction: float = 0.25,
                     num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sampler that makes on average `pos_fraction` of every batch potholes.

    Without it ~97% of crops are background and the network learns "always background".
    Default epoch length: every positive is drawn about once per epoch.
    """
    counts = dataset.class_counts()
    n_pos, n_neg = counts["pothole"], counts["background"]
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes to balance, got {counts}")
    weights = np.where(dataset.labels == 1, pos_fraction / n_pos, (1 - pos_fraction) / n_neg)
    if num_samples is None:
        num_samples = int(n_pos / pos_fraction)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=num_samples, replacement=True)


def main() -> None:
    """Summary per split + an image of example crops to check the labels by eye."""
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    for split in ("train", "val", "test"):
        try:
            ds = PotholeProposals(split, make_transform(train=False))
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"{split:>5}: {len(ds.image_names)} images, {len(ds)} crops, {ds.class_counts()}")

    train = PotholeProposals("train", make_transform(train=False, size=96, imagenet_norm=False))
    loader = DataLoader(train, batch_size=64, sampler=balanced_sampler(train))
    crops, labels, _, _ = next(iter(loader))
    print(f"one balanced batch: {int(labels.sum())}/{len(labels)} potholes")

    out = LABELS_DIR.parent / "example_crops.png"
    # Top row(s): potholes, bottom: background. Should be obvious at a glance.
    save_image(torch.cat([crops[labels == 1][:16], crops[labels == 0][:16]]), out, nrow=8, padding=2)
    print(f"Saved {out}  (first rows: label 1 = pothole, rest: label 0 = background)")


if __name__ == "__main__":
    main()
