import cv2
import numpy as np
from groundtruth import parse_boxes
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "potholes"
IMAGES_DIR = LOCAL_DIR / "images"
ANNOTATIONS_DIR = LOCAL_DIR / "annotations"
PROPOSALS_DIR = LOCAL_DIR / 'proposals'

IOU_THRESHOLD = 0.5
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
    images_paths = sorted(IMAGES_DIR.glob("potholes*.png"))
    if not images_paths:
        print(f"No images found in {IMAGES_DIR}. Run download.py first.")
        return


    total_groud_truths = 0
    found_per_n = {n: 0 for n in RANGE}
    found_all = 0

    for image_path in images_paths:
        proposal_path = PROPOSALS_DIR / f'{image_path.stem}.npy'
        if not proposal_path.exists():
            print("Proposal not found, skipping image")
            continue
        boxes = parse_boxes(ANNOTATIONS_DIR / f"{image_path.stem}.xml")
        if not boxes:
            continue

        ground_truth_boxes = np.array([[xmin, ymin, xmax, ymax] for _, xmin, ymin, xmax, ymax in boxes])
        proposals = np.load(proposal_path)
        total_groud_truths += len(ground_truth_boxes)

        iou_matrix = compute_IoU(proposals, ground_truth_boxes)
        for n in RANGE:
            best_iou_per_gt = iou_matrix[:n].max(axis=0)
            found_per_n[n] += int((best_iou_per_gt >= IOU_THRESHOLD).sum())

        best_iou_per_gt_all = iou_matrix.max(axis=0)
        found_all += int((best_iou_per_gt_all >= IOU_THRESHOLD).sum())

    print(f"{'N':>6} | {'recall':>7}")
    for n in RANGE:
        recall = found_per_n[n] / total_groud_truths if total_groud_truths else 0.0
        print(f"{n:>6} | {recall:>7.3f}")

    recall_all = found_all / total_groud_truths if total_groud_truths else 0.0
    print(f"{'all':>6} | {recall_all:>7.3f}")


if __name__ == "__main__":
    main()