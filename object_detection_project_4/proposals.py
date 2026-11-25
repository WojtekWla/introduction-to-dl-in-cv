import argparse
import cv2
from pathlib import Path
import numpy as np
import os

LOCAL_DIR = Path(__file__).resolve().parents[1] / 'data' / 'potholes'

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/dtu/datasets1/02516/potholes"))

IMAGES_DIR = DEFAULT_DATA_ROOT / 'images'

# Team choice (recall_vs_n.png): Quality mode at full resolution, first 1000
# proposals per image (recall 0.906 on train). Shrinking to 300 px missed small potholes.
DEFAULT_MODE = "quality"
TARGET_MAX_DIM = 0  # 0 = no resize
PROPOSALS_DIR = LOCAL_DIR / 'proposals_quality_full'
N_PROPOSALS = 1000

def resize_image(frame, max_dim=TARGET_MAX_DIM):
    if max_dim <= 0:
        return frame, 1.0
    orig_height, orig_width = frame.shape[:2]
    scale = min(1.0, max_dim / max(orig_width, orig_height))
    if scale == 1.0:
        return frame, scale
    new_width, new_height = round(orig_width * scale), round(orig_height * scale)
    return cv2.resize(frame, (new_width, new_height)), scale

def extract_proposals_for_frame(frame, selective_search_processor, mode=DEFAULT_MODE):
    selective_search_processor.setBaseImage(frame)
    if mode == "quality":
        selective_search_processor.switchToSelectiveSearchQuality()
    else:
        selective_search_processor.switchToSelectiveSearchFast()
    return selective_search_processor.process()

def rescale_to_original(rectangles, scale):
    x, y, w, h = rectangles[:, 0], rectangles[:, 1], rectangles[:, 2], rectangles[:, 3]
    xmin = x / scale
    ymin = y / scale
    xmax = (x + w) / scale
    ymax = (y + h) / scale

    return np.stack([xmin, ymin, xmax, ymax], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Selective Search proposals for every image.")
    parser.add_argument("--max-dim", type=int, default=TARGET_MAX_DIM,
                        help="resize longest side to this before SS (never upscales; 0 = no resize)")
    parser.add_argument("--mode", choices=["fast", "quality"], default=DEFAULT_MODE)
    parser.add_argument("--out-dir", type=Path, default=PROPOSALS_DIR)
    args = parser.parse_args()

    images_paths = sorted(IMAGES_DIR.glob('potholes*.png'))
    if not images_paths:
        print("Images could not be found")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    for image_path in images_paths:
        frame = cv2.imread(image_path)

        frame, scale = resize_image(frame, args.max_dim)
        rectangles = extract_proposals_for_frame(frame, ss, args.mode)
        rectangles = rescale_to_original(rectangles, scale)

        np.save(args.out_dir / f'{image_path.stem}.npy', rectangles)


if __name__ == "__main__":
    main()