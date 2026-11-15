"""
IDLCV Project 3 (Segmentation) - Task 1: PH2 + DRIVE data loaders.
Built from Project3.ipynb. Run on the DTU HPC, where /dtu/datasets1/02516 exists:

Set DATA_ROOT in the environment to point somewhere else (e.g. for a local test).
"""

import os
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF            # NEW
from torchvision import tv_tensors                         # NEW


DATA_ROOT = os.environ.get("DATA_ROOT", "/dtu/datasets1/02516")


# Reproducible train / val / test split  (same seed -> same split every run)

def make_splits(items, val_frac=0.15, test_frac=0.15, seed=42):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    return {
        "test":  items[:n_test],
        "val":   items[n_test:n_test + n_val],
        "train": items[n_test + n_val:],
    }


def _binary_mask(label_pil, size):
    """Nearest-neighbour resize + threshold -> float tensor (1,H,W) in {0,1}."""
    label_pil = label_pil.convert("L").resize((size, size), Image.NEAREST)
    arr = (np.asarray(label_pil) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)




import torch.nn.functional as F

def _annotated_points(label_pil, size, pos=10, neg=10, sampling="random"):
    mask = _binary_mask(label_pil, size)[0]
    points = torch.full_like(mask, -1)

    if sampling == "random":
        pos_y, pos_x = torch.where(mask == 1)
        neg_y, neg_x = torch.where(mask == 0)

    elif sampling == "biased":
        m = mask[None, None]

        # Erode the lesion by approximately 10 pixels
        eroded = 1 - F.max_pool2d(1 - m, kernel_size=21, stride=1, padding=10)[0, 0]

        pos_y, pos_x = torch.where(eroded == 1)
        neg_y, neg_x = torch.where(mask == 0)
        
    else:
        raise ValueError(f"Unsupported sampling strategy: {sampling}")

    if pos > len(pos_x) or neg > len(neg_x) or pos < 0 or neg < 0:
        raise ValueError("Invalid number of clicks")

    pos_idx = torch.randperm(len(pos_x))[:pos]
    neg_idx = torch.randperm(len(neg_x))[:neg]

    points[pos_y[pos_idx], pos_x[pos_idx]] = 1
    points[neg_y[neg_idx], neg_x[neg_idx]] = 0

    return points.unsqueeze(0)



def _apply_joint(joint_transform, image, label):                       # NEW
    """Run a torchvision.transforms.v2 pipeline on image AND mask together.
    The mask is wrapped as tv_tensors.Mask, so v2 uses the same random
    parameters for both and nearest-neighbour interpolation for the mask."""
    img = tv_tensors.Image(TF.pil_to_tensor(image))
    mask = tv_tensors.Mask(torch.from_numpy(np.array(label))[None])
    img, mask = joint_transform(img, mask)
    image = TF.to_pil_image(img.as_subclass(torch.Tensor))
    label = Image.fromarray(mask.as_subclass(torch.Tensor)[0].numpy())
    return image, label


# PH2 - skin lesion segmentation
class PH2(torch.utils.data.Dataset):
    def __init__(self, split, transform, size=256,
                 data_path=os.path.join(DATA_ROOT, "PH2_Dataset_images"),
                 joint_transform=None):                                 # NEW
        self.transform = transform
        self.joint_transform = joint_transform                          # NEW
        self.size = size
        self.image_paths = [
            os.path.join(data_path, c, f"{c}_Dermoscopic_Image", f"{c}.bmp")
            for c in split]
        self.label_paths = [
            os.path.join(data_path, c, f"{c}_lesion", f"{c}_lesion.bmp")
            for c in split]

    @staticmethod
    def list_cases(data_path=os.path.join(DATA_ROOT, "PH2_Dataset_images")):
        return sorted(d for d in os.listdir(data_path) if d.startswith("IMD"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")          # CHANGED
        if self.joint_transform is not None:                            # NEW
            image, label = _apply_joint(self.joint_transform, image, label)
        X = self.transform(image)
        Y = _binary_mask(label, self.size)
        return X, Y


class PH2_Annotated(PH2):
    def __init__(self, split, transform, size=256,
                 data_path=os.path.join(DATA_ROOT, "PH2_Dataset_images"),
                 joint_transform=None, pos_clicks=10, neg_clicks=10,
                 sampling_strategy="random"):
        super().__init__(split, transform, size, data_path, joint_transform)
        self.pos_clicks = pos_clicks
        self.neg_clicks = neg_clicks
        self.sampling_strategy = sampling_strategy
        self.annotation_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ph2_annotated",
            f"{pos_clicks}p_{neg_clicks}n_{sampling_strategy}")
        os.makedirs(self.annotation_dir, exist_ok=True)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        case_id = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        annotation_path = os.path.join(self.annotation_dir, f"{case_id}.npy")

        if os.path.exists(annotation_path):
            Y = torch.from_numpy(np.load(annotation_path, allow_pickle=False))
            if Y.shape != (1, self.size, self.size):
                raise ValueError(f"Annotation size mismatch: {annotation_path}")
        else:
            Y = _annotated_points(label, self.size, self.pos_clicks,
                                      self.neg_clicks, self.sampling_strategy)
            np.save(annotation_path, Y.numpy())

        image = image.resize((self.size, self.size), Image.BILINEAR)
        if self.joint_transform is not None:
            # Zero padding must decode to unlabeled (-1), not a negative click.
            label = Image.fromarray((Y[0].numpy() + 1).astype(np.uint8))
            image, label = _apply_joint(self.joint_transform, image, label)
            Y = torch.from_numpy(np.array(label)).unsqueeze(0).float() - 1
        X = self.transform(image)
        return X, Y


# DRIVE - retinal blood vessel segmentation
class DRIVE(torch.utils.data.Dataset):
    def __init__(self, split, transform, size=256,
                 data_path=os.path.join(DATA_ROOT, "DRIVE", "training"),
                 return_fov=False, joint_transform=None):               # NEW
        if return_fov and joint_transform is not None:                  # NEW
            raise ValueError("return_fov=True cannot be combined with joint_transform.")
        self.transform = transform
        self.joint_transform = joint_transform                          # NEW
        self.size = size
        self.return_fov = return_fov
        self.image_paths = [
            os.path.join(data_path, "images", f"{i}_training.tif") for i in split]
        self.label_paths = [
            os.path.join(data_path, "1st_manual", f"{i}_manual1.gif") for i in split]
        self.fov_paths = [
            os.path.join(data_path, "mask", f"{i}_training_mask.gif") for i in split]

    @staticmethod
    def list_ids(data_path=os.path.join(DATA_ROOT, "DRIVE", "training")):
        imgs = glob.glob(os.path.join(data_path, "images", "*_training.tif"))
        return sorted(os.path.basename(p).split("_")[0] for p in imgs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")          # CHANGED
        if self.joint_transform is not None:                            # NEW
            image, label = _apply_joint(self.joint_transform, image, label)
        X = self.transform(image)
        Y = _binary_mask(label, self.size)
        if self.return_fov:
            fov = _binary_mask(Image.open(self.fov_paths[idx]), self.size)
            return X, Y, fov
        return X, Y


# Convenience: build the three DataLoaders for either dataset.
# Resize((size,size)) and ToTensor are ALWAYS applied to the image.
#   train_transform : IMAGE-ONLY transforms.Compose for the train split
#   eval_transform  : IMAGE-ONLY transforms.Compose for val/test (normally None)
#   joint_transform : transforms.v2.Compose applied to IMAGE + MASK, train only
def build_loaders(dataset_cls, ids_fn, size=256, batch_size=6, seed=42,
                  num_workers=0, train_transform=None, eval_transform=None,
                  joint_transform=None, pos_clicks=10, neg_clicks=10,
                  sampling_strategy="random"):
    def make_tf(extra):
        if extra is None:
            extra = []
        elif isinstance(extra, transforms.Compose):
            extra = extra.transforms
        return transforms.Compose([transforms.Resize((size, size)),
                                   *extra,
                                   transforms.ToTensor()])
    train_tf = make_tf(train_transform)
    eval_tf = make_tf(eval_transform)
    sp = make_splits(ids_fn(), seed=seed)
    loaders = {k: DataLoader((PH2 if dataset_cls is PH2_Annotated and k != "train" else dataset_cls)(
                                         v, transform=train_tf if k == "train" else eval_tf,
                                         size=size,
                                         joint_transform=joint_transform if k == "train" else None,
                                         **(dict(pos_clicks=pos_clicks, neg_clicks=neg_clicks,
                                                 sampling_strategy=sampling_strategy)
                                            if dataset_cls is PH2_Annotated and k == "train" else {})),
                             batch_size=batch_size,
                             shuffle=(k == "train"), num_workers=num_workers)
               for k, v in sp.items()}
    return loaders, sp


if __name__ == "__main__":
    # Example usage / quick check (run on the cluster where the data is mounted)
    from torchvision.transforms import v2
    ph2_joint = v2.Compose([v2.RandomHorizontalFlip(),
                            v2.RandomAffine(degrees=25, translate=(0.1, 0.1))])
    for name, cls, ids_fn, img_tf, joint in [
            ("PH2", PH2, PH2.list_cases,
             transforms.Compose([transforms.ColorJitter(0.2, 0.2)]), ph2_joint),
            ("DRIVE", DRIVE, DRIVE.list_ids, None, None)]:
        loaders, sp = build_loaders(cls, ids_fn, size=256,
                                    batch_size=2, num_workers=0,
                                    train_transform=img_tf,
                                    joint_transform=joint)
        X, Y = next(iter(loaders["train"]))
        print(f"[{name}] train/val/test = "
              f"{len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])} | "
              f"X {tuple(X.shape)} | Y {tuple(Y.shape)} | "
              f"mask values {sorted(set(Y.unique().tolist()))}")
