# Object Detection Project 4 — Potholes

Dataset lives on the DTU HPC cluster at `/dtu/datasets1/02516/potholes` (SSH
host alias `gbar`), with `images/potholesNNN.png` and
`annotations/potholesNNN.xml` (Pascal VOC-style boxes): 665 images, 1740
boxes, one class (`pothole`).

## Train/val/test split

The dataset has no `splits.json` (the task PDF mentions one, but it isn't on
the cluster), so `splits.json` in this folder is our own fixed 70/10/20 split
by image (seed 0: 466 train, 66 val, 133 test). Use it for everything: tune on
train/val only, report AP on test only. It was made with:

```
python make_splits.py            # on the HPC; or --data-root DIR elsewhere
```

Test: `uv run --with pytest pytest object_detection_project_4/test_make_splits.py`

## Usage

Run in this order:

### 1. `download.py`

Downloads a random sample of image/annotation pairs from the cluster via
`scp` into `data/potholes/{images,annotations}` (relative to the repo root).

```
python download.py [n]      # n defaults to 10
```

### 2. `groundtruth.py`

Visualizes the downloaded images: raw image on the left, same image with
ground-truth boxes drawn on the right.

```
python groundtruth.py
```

Navigate with `a`/`d` or the left/right arrow keys, `q`/ESC to quit.

### 3. `proposals.py`

Runs Selective Search on each downloaded image and saves candidate object
proposal boxes to `data/potholes/proposals/potholesNNN.npy` (one file per
image, boxes as `(xmin, ymin, xmax, ymax)` in original image coordinates).
Images are resized by their longest side before running SS for speed (never
upscaled), and proposal boxes are rescaled back before saving.

```
python proposals.py
```

### 4. `evaluate_proposals.py`

Evaluates how many of the saved proposals are actually needed: computes, for
each GT box, whether any of the first N proposals overlaps it with
IoU >= 0.5, then reports recall averaged over all downloaded images for a
range of N (plus "all" = every proposal an image has, uncapped).

```
python evaluate_proposals.py
```

Run `proposals.py` before this one — it reads the `.npy` files `proposals.py`
produces.
