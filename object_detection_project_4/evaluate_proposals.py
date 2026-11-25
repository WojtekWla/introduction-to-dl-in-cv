import argparse
import cv2
import numpy as np
from groundtruth import parse_boxes
from pathlib import Path
import os
import json


LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "potholes"

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/dtu/datasets1/02516/potholes"))

IMAGES_DIR = DEFAULT_DATA_ROOT / 'images'
PROPOSALS_DIR = LOCAL_DIR / 'proposals'
ANNOTATIONS_DIR = DEFAULT_DATA_ROOT / "annotations"

SPLITS_FILE = Path(__file__).resolve().parent / "splits.json"

IOU_THRESHOLD = 0.5
SMALL_BOX_SIDE = 20
RANGE = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]


def compute_IoU(proposals, ground_truth):
    x1 = np.maximum(proposals[:, None, 0], ground_truth[None, :, 0])
    y1 = np.maximum(proposals[:, None, 1], ground_truth[None, :, 1])
    x2 = np.minimum(proposals[:, None, 2], ground_truth[None, :, 2])
    y2 = np.minimum(proposals[:, None, 3], ground_truth[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    area_proposals = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    area_gt = (ground_truth[:, 2] - ground_truth[:, 0]) * (ground_truth[:, 3] - ground_truth[:, 1])
    union = area_proposals[:, None] + area_gt[None, :] - inter

    return inter / union


def main():
    parser = argparse.ArgumentParser(description="Proposal recall on the train split.")
    parser.add_argument("--proposals-dir", type=Path, default=PROPOSALS_DIR)
    args = parser.parse_args()

    splits = json.loads(SPLITS_FILE.read_text())
    training_images = splits["train"]
    images_paths = [IMAGES_DIR / image_name for image_name in training_images]
    if not images_paths:
        print(f"No training images listed in {SPLITS_FILE}.")
        return


    total_groud_truths = 0
    found_per_n = {n: 0 for n in RANGE}
    found_all = 0
    proposals_per_image = []
    # Recall with all proposals, split by GT box size: are small potholes the ones missed?
    small_found = small_total = 0

    for image_path in images_paths:
        proposal_path = args.proposals_dir / f'{image_path.stem}.npy'
        if not proposal_path.exists():
            print("Proposal not found, skipping image")
            continue
        boxes = parse_boxes(ANNOTATIONS_DIR / f"{image_path.stem}.xml")
        if not boxes:
            continue

        ground_truth_boxes = np.array([[xmin, ymin, xmax, ymax] for _, xmin, ymin, xmax, ymax in boxes])
        proposals = np.load(proposal_path)
        total_groud_truths += len(ground_truth_boxes)
        proposals_per_image.append(len(proposals))

        iou_matrix = compute_IoU(proposals, ground_truth_boxes)
        for n in RANGE:
            best_iou_per_gt = iou_matrix[:n].max(axis=0)
            found_per_n[n] += int((best_iou_per_gt >= IOU_THRESHOLD).sum())

        found_per_gt = iou_matrix.max(axis=0) >= IOU_THRESHOLD
        found_all += int(found_per_gt.sum())

        gt_sides = np.minimum(ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0],
                              ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1])
        is_small = gt_sides < SMALL_BOX_SIDE
        small_total += int(is_small.sum())
        small_found += int(found_per_gt[is_small].sum())

    print(f"{'N':>6} | {'recall':>7}")
    for n in RANGE:
        recall = found_per_n[n] / total_groud_truths if total_groud_truths else 0.0
        print(f"{n:>6} | {recall:>7.3f}")

    recall_all = found_all / total_groud_truths if total_groud_truths else 0.0
    print(f"{'all':>6} | {recall_all:>7.3f}")

    if proposals_per_image:
        print(f"proposals per image: mean {np.mean(proposals_per_image):.0f}, "
              f"median {np.median(proposals_per_image):.0f}")
    large_total = total_groud_truths - small_total
    large_found = found_all - small_found
    if small_total and large_total:
        print(f"recall (all proposals), GT min side < {SMALL_BOX_SIDE}px: "
              f"{small_found / small_total:.3f} ({small_total} boxes); "
              f">= {SMALL_BOX_SIDE}px: {large_found / large_total:.3f} ({large_total} boxes)")


if __name__ == "__main__":
    main()