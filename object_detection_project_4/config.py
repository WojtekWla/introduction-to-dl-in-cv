"""Shared paths and team settings for Project 4.

Kept free of OpenCV so the dataset (4.2/4.3) works without it.
"""
import os
from pathlib import Path

# Dataset (read-only on the HPC). Set DATA_ROOT to use a local copy.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/dtu/datasets1/02516/potholes"))
IMAGES_DIR = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"

SPLITS_FILE = Path(__file__).resolve().parent / "splits.json"

# Our outputs, next to the project folder (the dataset folder is read-only).
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "potholes"
PROPOSALS_DIR = OUTPUT_DIR / "proposals_quality_full"
LABELS_DIR = OUTPUT_DIR / "labels"

# Team choices: first 1000 proposals per image (recall 0.906 on train, see
# figures/recall_vs_n.png); IoU >= 0.5 pothole, < 0.3 background, in between ignored.
N_PROPOSALS = 1000
POS_IOU = 0.5
NEG_IOU = 0.3
