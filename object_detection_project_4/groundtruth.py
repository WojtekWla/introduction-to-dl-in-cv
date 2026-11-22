import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "potholes"
IMAGES_DIR = LOCAL_DIR / "images"
ANNOTATIONS_DIR = LOCAL_DIR / "annotations"

LEFT_KEYS = {ord("a"), 65361, 63234, 2424832}
RIGHT_KEYS = {ord("d"), 65363, 63235, 2555904}
QUIT_KEYS = {27, ord("q")}


def parse_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def build_frame(image_path):
    xml_path = ANNOTATIONS_DIR / f"{image_path.stem}.xml"
    image = cv2.imread(str(image_path))
    boxes = parse_boxes(xml_path)

    annotated = image.copy()
    for name, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(annotated, name, (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return cv2.hconcat([image, annotated])


def main():
    image_paths = sorted(IMAGES_DIR.glob("potholes*.png"))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}. Run download.py first.")
        return

    window = "Potholes ground truth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    index = 0
    while True:
        image_path = image_paths[index]
        cv2.setWindowTitle(window, f"{image_path.name} ({index + 1}/{len(image_paths)})")
        cv2.imshow(window, build_frame(image_path))

        key = cv2.waitKeyEx(0)
        if key in QUIT_KEYS:
            break
        elif key in RIGHT_KEYS:
            index = (index + 1) % len(image_paths)
        elif key in LEFT_KEYS:
            index = (index - 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
