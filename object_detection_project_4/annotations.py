"""Read the Pascal VOC-style XML annotations of the Potholes dataset."""
import xml.etree.ElementTree as ET


def parse_boxes(xml_path):
    """List of (name, xmin, ymin, xmax, ymax) for every object in one XML file."""
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
