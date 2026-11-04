"""
Sanity-check figures for the report (Task 1).

Saves one PNG per dataset showing a few training samples:
  PH2  : image | lesion mask | overlay
  DRIVE: image | vessel mask (1st_manual) | FOV mask | vessel overlay

Usage (on the cluster, next to seg_datasets_from_notebook.py):
    python visualize_data.py            # saves PNGs in the current folder
    python visualize_data.py outdir 4   # output folder, number of samples

Needs matplotlib:  pip install matplotlib
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from seg_datasets import PH2, DRIVE, make_splits

SIZE = 256


def overlay(ax, img, mask, color="autumn"):
    ax.imshow(img)
    ax.imshow(np.ma.masked_where(mask < 0.5, mask), cmap=color, alpha=0.55)


def make_figure(name, ds, n, out_dir, with_fov=False):
    n = min(n, len(ds))
    cols = 4 if with_fov else 3
    fig, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.2 * n), squeeze=False)
    for r in range(n):
        item = ds[r]
        img = item[0].permute(1, 2, 0).numpy()
        mask = item[1][0].numpy()
        axes[r][0].imshow(img)
        axes[r][1].imshow(mask, cmap="gray")
        if with_fov:
            axes[r][2].imshow(item[2][0].numpy(), cmap="gray")
            overlay(axes[r][3], img, mask)
        else:
            overlay(axes[r][2], img, mask)
        print(f"[{name}] sample {r}: foreground fraction = {mask.mean():.3f}")
    titles = (["image", "vessel mask (1st_manual)", "FOV mask (not the target)",
               "overlay"] if with_fov else ["image", "lesion mask", "overlay"])
    for c, t in enumerate(titles):
        axes[0][c].set_title(t, fontsize=10)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name}_samples.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(out_dir, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                             transforms.ToTensor()])

    ph2_train = make_splits(PH2.list_cases())["train"]
    make_figure("PH2", PH2(ph2_train, transform=tf, size=SIZE), n, out_dir)

    drive_train = make_splits(DRIVE.list_ids())["train"]
    make_figure("DRIVE", DRIVE(drive_train, transform=tf, size=SIZE,
                               return_fov=True), n, out_dir, with_fov=True)
