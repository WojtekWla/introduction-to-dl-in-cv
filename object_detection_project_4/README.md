# Object Detection Project 4 — Potholes

Dataset lives on the DTU HPC cluster at `/dtu/datasets1/02516/potholes`, with
`images/potholesNNN.png` and `annotations/potholesNNN.xml` (Pascal VOC-style
boxes): 665 images, 1740 boxes, one class (`pothole`).

All scripts read the dataset from that path by default. To run elsewhere (e.g.
on a downloaded copy), set `DATA_ROOT` to the folder that contains `images/`
and `annotations/`:

```
DATA_ROOT=../data/potholes_sample python evaluate_proposals.py
```

Outputs (proposals) go to `data/potholes/` one level above this folder (the
repo root locally, `~/idlcv` on the HPC), never into the read-only dataset
folder.

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

### 1. `groundtruth.py`

Visualizes the images: raw image on the left, same image with ground-truth
boxes drawn on the right. Needs a display, so run it locally on a downloaded
copy via `DATA_ROOT`.

```
python groundtruth.py
```

Navigate with `a`/`d` or the left/right arrow keys, `q`/ESC to quit.

### 2. `proposals.py`

Runs Selective Search on every image (all splits: training, validation and
testing all need proposals) and saves candidate boxes to
`../data/potholes/proposals/potholesNNN.npy` (one file per image, boxes as
`(xmin, ymin, xmax, ymax)` in original image coordinates). Images are resized
by their longest side before running SS for speed (never upscaled), and
proposal boxes are rescaled back before saving.

```
python proposals.py                                   # team default: quality, full size -> ../data/potholes/proposals_quality_full
python proposals.py --mode fast --max-dim 300 --out-dir ../data/potholes/proposals   # old baseline
```

**Team choice:** Quality mode at full resolution, and the first
`N_PROPOSALS = 1000` proposals per image (recall 0.906 on train, see
`figures/recall_vs_n.png`). Both are defined at the top of `proposals.py`;
import them from there instead of hard-coding.

On the HPC, run it as a batch job instead (see `jobs/proposals.sh` for how to
pass the settings).

**Proposal order is partly random.** OpenCV's Selective Search returns the same
set of boxes every run, but ranks them with a random component
(`cv2.setRNGSeed` doesn't control it). On the HPC (Linux) a fresh process
always gives the same order, so a full `proposals.py` run is reproducible; on
macOS the order changes between runs. Since "the first N proposals" depends on
the order, generate the proposals once on the HPC and have everyone use those
saved files.

### 3. `evaluate_proposals.py`

Evaluates how many proposals are needed: for each GT box of the **train**
split, checks whether any of the first N proposals overlaps it with
IoU >= 0.5, then reports recall over all train GT boxes for a range of N (plus
"all" = every proposal an image has, uncapped). Also prints the number of
proposals per image and the recall split by GT box size (min side < 20 px vs
larger). Run `proposals.py` first.

```
python evaluate_proposals.py [--proposals-dir DIR]
```

### 4. `recall_curves.py` + `plot_recall.py`

Recall vs. N (steps of 10, up to 3000) for several proposal sets at once, to
find the elbow. Compute on the HPC (numpy only), plot locally (matplotlib):

```
python recall_curves.py --out recall_curves.csv fast_300=../data/potholes/proposals quality_full=../data/potholes/proposals_quality_full
python plot_recall.py figures/recall_curves.csv figures/recall_vs_n.png
```

Current result for all four settings: `figures/recall_vs_n.png`.

### 5. `label_proposals.py` (task 4.1.4)

For every image, saves `../data/potholes/labels/<stem>.npz` with `boxes` (all
proposals) and `max_iou` (each proposal's best IoU with any GT box). Thresholds
are applied in the dataloader: IoU >= 0.5 pothole, < 0.3 background, in between
ignored. Prints the class balance on train (first 1000 proposals: 3.0% pothole,
3.3% ignored, 93.6% background).

```
python label_proposals.py
```

`download.py` (copy a random sample via `scp`, host alias `gbar`) predates the
`DATA_ROOT` setup and is no longer needed.
