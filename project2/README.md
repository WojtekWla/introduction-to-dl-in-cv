# Project 2: Video Classification

This directory contains the current 3D CNN baseline and the DTU HPC job used to
train it on the UCF10 frame dataset.

## Structure

```text
project2/
├── src/
│   ├── datasets.py
│   └── train_3dcnn.py
├── jobs/
│   └── train_3dcnn.sh
├── logs/          # LSF stdout and stderr (ignored by Git)
└── checkpoints/   # best checkpoint from each run
```

## Current Experiment

- Input clips: 10 RGB frames resized to 112 x 112 pixels
- Model: four 3D convolution blocks with 16, 32, 64, and 128 channels
- Regularization: batch normalization and dropout 0.4
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- Batch size: 16
- Epochs: 200
- Seed: 42
- Checkpoint selection: highest validation accuracy

The HPC dataset root is `/dtu/datasets1/02516/ufc10`. The dataset class returns
clips in PyTorch's expected 3D convolution layout: `[channels, time, height,
width]`. The data loader adds the batch dimension.

## DTU HPC Run

The job expects Python 3.11.13 and the virtual environment at
`$HOME/.venvs/idlcv-p2`. That environment must contain at least PyTorch,
torchvision, pandas, NumPy, and Pillow.

Place this directory at `$HOME/idlcv/project2`, then submit from inside it:

```bash
bsub -app c02516_1g.10gb < jobs/train_3dcnn.sh
```

LSF output is written to `logs/`. The training script saves the best validation
checkpoint as `checkpoints/drop_adamw_3dcnn_<job-id>_best.pt`.
