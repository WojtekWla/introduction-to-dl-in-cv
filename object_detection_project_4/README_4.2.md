# 4.2 — Proposal classifier

Trains the CNN that scores object proposals, and writes the per-proposal scores
that 4.3 turns into detections.

The detector is R-CNN: 4.1 produced ~1000 candidate boxes per image with
Selective Search and labelled each by its IoU with the ground truth. This part
trains a 2-class CNN (pothole / background) on the cropped boxes. 4.3 then
scores every proposal of a test image, applies NMS, and computes AP. The network
never predicts coordinates — it only ranks boxes that already exist.

## Files

| File | What it is |
|---|---|
| `models.py` | both model definitions, nothing else |
| `train_scratch.py` | trains the from-scratch CNN |
| `train_transfer.py` | trains the ImageNet ResNet18 |
| `predict.py` | scores every proposal of a split → `.npz` for 4.3 |

They import `dataset.py`, `config.py` and `annotations.py`, which are **shared
and frozen** — changing `N_PROPOSALS` or the IoU thresholds there invalidates
everyone's results.

## Running it

On the HPC, with the labels unpacked as the main README describes:

```bash
python models.py            # sanity check, no data needed
python train_scratch.py
python train_transfer.py
python predict.py checkpoints/transfer.pt val
python predict.py checkpoints/transfer.pt test
```

No arguments on the training scripts — settings are constants at the top of each
file. `predict.py` takes a checkpoint and a split.

Run `predict.py` on **val** as well as test: val is what the threshold for 4.3
is chosen from, and test must not be touched until that is settled.

`python models.py` should print:

```
scratch  output (4, 2)   parameters 0.42M
stage 1  output (4, 2)   11.33M params, 156004 trainable
stage 2  after unfreeze_layer4(): 8.55M trainable
```

If `trainable` in stage 1 is not 156004, the backbone freeze is broken.

### GPU note

On a V100 (compute capability 7.0) the default torch wheel has no matching
kernels and `conv2d` fails with "unable to find an engine to execute this
computation". Install the cu126 build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

A100 nodes (`02516sh`) do not need this.

## Model 1 — `PotholeCNN`, from scratch

Four conv blocks, each `Conv3x3 → BatchNorm → ReLU → MaxPool2`, halving the
resolution and doubling the channels:

```
input 3 x 128 x 128
  block 1   3 ->  32      128 -> 64
  block 2  32 ->  64       64 -> 32
  block 3  64 -> 128       32 -> 16
  block 4 128 -> 256       16 ->  8
head
  AdaptiveAvgPool2d(1)   256 x 8 x 8 -> 256 x 1 x 1
  Flatten                -> 256
  Dropout(0.3)
  Linear(256, 100) + ReLU
  Linear(100, 2)         -> 2 logits
```

0.42M parameters, all trainable. `AdaptiveAvgPool2d(1)` means the input size is
free — the head never sees the spatial dimensions, so the same model runs at
128 or 224 without a code change.

`bias=False` is not set on the convs; BatchNorm's own shift makes those biases
redundant, but they are harmless.

## Model 2 — `PotholeResNet18`, transfer learning

torchvision ResNet18 with ImageNet weights. `__init__` does three things, in
this order:

1. load the pretrained network
2. freeze every parameter (`requires_grad = False`)
3. replace the 1000-class `fc` with a new head

The order matters: a layer created *after* the freeze is trainable by default,
so the new head is the only thing that learns.

```
input 3 x 224 x 224
  conv1 + bn1 + relu        -> 64 x 112 x 112
  maxpool                   -> 64 x 56 x 56
  layer1                    -> 64 x 56 x 56     frozen
  layer2                    -> 128 x 28 x 28    frozen
  layer3                    -> 256 x 14 x 14    frozen
  layer4                    -> 512 x 7 x 7      frozen
  avgpool + flatten         -> 512
  fc (new, trainable):
      Linear(512, 254) + ReLU
      Linear(254, 100) + ReLU
      Dropout(0.5)
      Linear(100, 2)        -> 2 logits
```

11.33M parameters, **156,004 trainable**.

Two optional methods for a second stage:

- `unfreeze_layer4()` → 8.55M trainable, fine-tunes the last residual block
- `unfreeze_all()` → everything trainable

If you use either, drop the learning rate to 1e-4 or lower — you are adjusting
pretrained weights, not learning from scratch.

## Settings that must match the model

These are not free choices. They are set at the top of each training script and
read back out of the checkpoint by `predict.py`.

| | scratch | transfer |
|---|---|---|
| `SIZE` | 128 (or 224) | 224 |
| `IMAGENET_NORM` | **False** | **True** |
| `LR` | 1e-3 | 1e-3 frozen / 1e-4 fine-tuned |
| `EPOCHS` | 15 | 8 |

`IMAGENET_NORM` is the one that bites. The pretrained weights expect
ImageNet-normalised pixels; feed them raw [0,1] values and the model still
trains, just worse, with no error — which looks like "pretraining didn't help".

If you compare the two models, run them at the **same** `SIZE`, or the result is
confounded by input resolution rather than being about pretraining.

## What training does

Both scripts:

- use `balanced_sampler(train_set, 0.25)` so batches are ~25% pothole instead of
  the natural 3%. An epoch is `n_pos / 0.25` ≈ 59k crops, not all 431k.
- use `add_gt_boxes=True` on train only — adds the ground-truth boxes as
  positives, covering the ~9% of potholes no proposal matched. Never on val or
  test: that would leak the answer into the metric.
- validate with **no sampler**, on the natural 3% positive rate, because that is
  the distribution at test time.
- checkpoint on the best **pothole F1**, never accuracy. Predicting background
  for every crop scores ~96.9% accuracy on this val set and is worthless.

## Outputs

```
checkpoints/scratch.pt              best-F1 weights + size + imagenet_norm
checkpoints/scratch_history.csv     per-epoch loss, acc, P, R, F1, seconds
checkpoints/transfer.pt
checkpoints/transfer_history.csv
outputs/<checkpoint>_<split>.npz    per-proposal scores
```

The `.npz` is the handover to 4.3, one row per proposal:

| key | shape | meaning |
|---|---|---|
| `image_idx` | (N,) | index into `image_names` |
| `boxes` | (N, 4) | xmin, ymin, xmax, ymax, original image pixels |
| `scores` | (N,) | P(pothole), continuous 0–1 |
| `labels` | (N,) | IoU-derived label, for checking only |
| `image_names` | (M,) | file name per image index |

`predict.py` uses `keep_all=True`, so every proposal is scored including the
0.3–0.5 IoU band that training drops — NMS cannot suppress a box it never saw.
The score is `softmax(logits)[:, 1]`, not `argmax`: AP needs a continuous score
to rank detections and sweep the threshold.

`predict.py` does not threshold, suppress or evaluate anything. That is 4.3.

## Results

| model | val accuracy | precision | recall | F1 | epochs |
|---|---|---|---|---|---|
| scratch | | | | | |
| transfer (frozen) | | | | | |
| transfer (layer4) | | | | | |

Fill from the last rows of the `_history.csv` files.
