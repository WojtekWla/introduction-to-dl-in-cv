# Project 4: Pothole detection

Data: `/dtu/datasets1/02516/potholes` on the HPC (665 images, one class).
Scripts read it from there. To use a local copy, set `DATA_ROOT` to the folder
with `images/` and `annotations/`.

Outputs go to `../data/potholes/` (on the HPC: `~/idlcv/data/potholes/`).

## Setup on the HPC (once)

Copy this folder to the HPC (e.g. `~/idlcv/project4`). The dataset needs the
`labels` folder in `../data/potholes/` next to it. Easiest: get `labels.tgz`
from the team and unpack it:

```
mkdir -p ~/idlcv/data/potholes && tar xzf labels.tgz -C ~/idlcv/data/potholes
```

Or make it yourself (needs `pip install opencv-contrib-python`):

```
mkdir -p logs
bsub < jobs/proposals.sh        # Selective Search for all images, ~40 min
python label_proposals.py       # after the job finished, a few seconds
```

Check: `python dataset.py` prints `train: 466 images, 430503 crops, {'pothole': 13530, ...}`.

## Using the data (4.2 / 4.3)

Needs only torch + torchvision (no OpenCV). Put your script in this folder and
import the dataset:

```python
from torch.utils.data import DataLoader
from dataset import PotholeProposals, balanced_sampler, make_transform

train = PotholeProposals("train", make_transform(train=True))
train_loader = DataLoader(train, batch_size=64, sampler=balanced_sampler(train), num_workers=4)

val = PotholeProposals("val", make_transform(train=False))
test = PotholeProposals("test", make_transform(train=False), keep_all=True)  # for NMS + AP
```

Each item is `(crop, label, box, image_idx)`. Label 1 = pothole, 0 = background.
`dataset.image_names[image_idx]` is the image file.

Options:
- `make_transform(train, size=224, imagenet_norm=True)`: use `imagenet_norm=False` for the from-scratch model.
- `balanced_sampler(ds, pos_fraction=0.25)`: share of potholes per batch (only ~3% of crops are potholes).
- `PotholeProposals(..., n_proposals=1000, pos_iou=0.5, neg_iou=0.3, add_gt_boxes=False)`.

The defaults (N, thresholds, paths) are in `config.py`. Change them there, not in your script.
- `keep_all=True`: keep every proposal (for testing; nothing may be dropped before NMS).

## Decisions

- **Split:** `splits.json`, our own (the dataset has none): 466 train / 66 val / 133 test. Don't use test until the final AP.
- **Proposals:** Selective Search, Quality mode, full resolution, first **1000** per image. Recall on train: 0.906 (`figures/recall_vs_n.png`). Shrinking to 300 px dropped it to 0.777.
- **Labels:** IoU ≥ 0.5 pothole, < 0.3 background, in between ignored.
- Generate proposals on the HPC only: on a Mac their order differs per run, which changes "the first 1000".

## Files

| File | What it does |
|---|---|
| `dataset.py` | Dataset, sampler and transforms for training and testing |
| `config.py` | Shared paths and settings (N, IoU thresholds) |
| `splits.json` | Train/val/test image lists |
| `annotations.py` | Reads the XML boxes |
| `proposals.py` | Selective Search, saves boxes per image (needs OpenCV) |
| `label_proposals.py` | Best IoU of every proposal with a pothole (task 4.1.4) |
| `evaluate_proposals.py` | Proposal recall for several N (task 4.1.3) |
| `recall_curves.py`, `plot_recall.py` | Recall-vs-N plot |
| `groundtruth.py` | View images with their boxes (needs a screen, run locally) |
| `make_splits.py` | How `splits.json` was made |
| `jobs/proposals.sh` | HPC batch job for the proposals |
| `download.py` | Old, not needed anymore |

Tests: `uv run --with pytest pytest object_detection_project_4/`
