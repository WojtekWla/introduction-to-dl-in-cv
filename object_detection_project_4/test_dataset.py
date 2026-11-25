import json

import numpy as np
import pytest
from PIL import Image

from dataset import PotholeProposals, balanced_sampler, make_transform

# One 100x80 image, one GT box, five proposals with known best IoUs.
BOXES = np.array([[10, 10, 50, 50],    # 0.9  -> pothole
                  [0, 0, 20, 20],      # 0.1  -> background
                  [30, 30, 90, 70],    # 0.4  -> ignored
                  [60, 0, 99, 79],     # 0.0  -> background
                  [10, 10, 49.6, 50.2]],  # 0.6 -> pothole (fractional coords)
                 dtype=np.float64)
MAX_IOU = np.array([0.9, 0.1, 0.4, 0.0, 0.6])


@pytest.fixture
def data(tmp_path):
    images, labels, annotations = tmp_path / "images", tmp_path / "labels", tmp_path / "annotations"
    for d in (images, labels, annotations):
        d.mkdir()
    Image.new("RGB", (100, 80), (120, 60, 30)).save(images / "potholes1.png")
    np.savez(labels / "potholes1.npz", boxes=BOXES, max_iou=MAX_IOU)
    (annotations / "potholes1.xml").write_text(
        "<annotation><object><name>pothole</name><bndbox><xmin>12</xmin><ymin>11</ymin>"
        "<xmax>48</xmax><ymax>49</ymax></bndbox></object></annotation>")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["potholes1.png", "missing.png"], "val": [], "test": []}))
    return dict(images_dir=images, labels_dir=labels, annotations_dir=annotations, splits_file=splits)


def make(data, **kwargs):
    return PotholeProposals("train", make_transform(train=False, size=32), **data, **kwargs)


def test_thresholds_drop_ignored_band(data):
    ds = make(data)
    assert len(ds) == 4  # the 0.4 proposal is dropped
    assert ds.labels.tolist() == [1, 0, 0, 1]
    assert ds.image_names == ["potholes1.png"]  # missing label file skipped


def test_keep_all_keeps_every_proposal(data):
    ds = make(data, keep_all=True)
    assert ds.labels.tolist() == [1, 0, 0, 0, 1]


def test_n_proposals_takes_first_n(data):
    assert make(data, n_proposals=2).labels.tolist() == [1, 0]


def test_add_gt_boxes_adds_positives(data):
    ds = make(data, add_gt_boxes=True)
    assert ds.labels.tolist() == [1, 0, 0, 1, 1]
    assert ds.boxes[-1].tolist() == [12, 11, 48, 49]


def test_item_is_resized_crop_with_label_box_and_image(data):
    crop, label, box, image_idx = make(data)[0]
    assert tuple(crop.shape) == (3, 32, 32)
    assert label == 1 and image_idx == 0
    assert box.tolist() == [10, 10, 50, 50]


def test_balanced_sampler_hits_target_fraction(data):
    ds = make(data)
    sampler = balanced_sampler(ds, pos_fraction=0.25, num_samples=20000)
    drawn = ds.labels[list(sampler)]
    assert drawn.mean() == pytest.approx(0.25, abs=0.02)


def test_no_labels_raises_clear_error(data, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="label_proposals.py"):
        make({**data, "labels_dir": empty})
