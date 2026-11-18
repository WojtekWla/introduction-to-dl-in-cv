import os
import torch
import torch.nn.functional as F

from seg_datasets import PH2, make_splits, _binary_mask
from PIL import Image

SIZE = 256


def eroded_count(mask):
    m = mask[None, None]
    eroded = 1 - F.max_pool2d(1 - m, kernel_size=21, stride=1, padding=10)[0, 0]
    return int(eroded.sum().item())


if __name__ == "__main__":
    cases = PH2.list_cases()
    split = make_splits(cases, seed=42)
    train_cases = split["train"]
    print(f"Scanning {len(train_cases)} training-set PH2 masks...")

    rows = []
    from seg_datasets import DATA_ROOT
    data_path = os.path.join(DATA_ROOT, "PH2_Dataset_images")

    for case in train_cases:
        label_path = os.path.join(data_path, case, f"{case}_lesion", f"{case}_lesion.bmp")
        label = Image.open(label_path).convert("L")
        mask = _binary_mask(label, SIZE)[0]
        pos = int(mask.sum().item())
        neg = int((1 - mask).sum().item())
        eroded = eroded_count(mask)
        rows.append((case, pos, neg, eroded))

    rows.sort(key=lambda r: r[3])

    print(f"\n{'case':10s}{'lesion_px':>12s}{'bg_px':>12s}{'eroded_px':>12s}")
    for case, pos, neg, eroded in rows:
        print(f"{case:10s}{pos:12d}{neg:12d}{eroded:12d}")

    min_pos = min(r[1] for r in rows)
    min_neg = min(r[2] for r in rows)
    min_eroded = min(r[3] for r in rows)
    print(f"\nSmallest raw lesion pixel count   (caps 'random' pos_clicks): {min_pos}")
    print(f"Smallest background pixel count   (caps neg_clicks, both modes): {min_neg}")
    print(f"Smallest eroded lesion pixel count (caps 'biased' pos_clicks): {min_eroded}")
    print(f"\n-> Safe max N for a symmetric pos=neg sweep across BOTH sampling modes: {min(min_pos, min_neg, min_eroded)}")
