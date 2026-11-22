import cv2
from pathlib import Path
import numpy as np

LOCAL_DIR = Path(__file__).resolve().parents[1] / 'data' / 'potholes'
IMAGES_DIR = LOCAL_DIR / 'images'
PROPOSALS_DIR = LOCAL_DIR / 'proposals'

TARGET_MAX_DIM = 300

def resize_image(frame):
    orig_height, orig_width = frame.shape[:2]
    scale = min(1.0, TARGET_MAX_DIM / max(orig_width, orig_height))
    if scale == 1.0:
        return frame, scale
    new_width, new_height = round(orig_width * scale), round(orig_height * scale)
    return cv2.resize(frame, (new_width, new_height)), scale

def extract_proposals_for_frame(frame, selective_search_processor):
    selective_search_processor.setBaseImage(frame)
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
    images_paths = sorted(IMAGES_DIR.glob('potholes*.png'))
    if not images_paths:
        print("Images could not be found")
        return

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    for image_path in images_paths:
        frame = cv2.imread(image_path)

        frame, scale = resize_image(frame)
        rectangles = extract_proposals_for_frame(frame, ss)
        rectangles = rescale_to_original(rectangles, scale)

        np.save(f'{PROPOSALS_DIR}/{image_path.stem}.npy', rectangles)
        

    
    

if __name__ == "__main__":
    main()