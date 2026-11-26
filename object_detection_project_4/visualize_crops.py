"""Show proposal crops next to their labels, to check the data by eye.

Run this before training anything: it is the cheapest way to catch a scrambled
label file, a wrong split, or boxes that no longer match their image. If the
crops titled "pothole" do not look like potholes, nothing downstream can work.

Saves a PNG instead of opening a window, so it works on a compute node.

    python visualize_crops.py                        # 8 potholes + 8 background, train
    python visualize_crops.py --split val --n 12
    python visualize_crops.py --mode random --n 10   # natural mix (~97% background)
    python visualize_crops.py --with-context         # also show the whole image + box
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.pyplot as plt
import numpy as np

from dataset import PotholeProposals, make_transform

LABEL_NAME = {0: "background", 1: "pothole"}
LABEL_COLOR = {0: "#c44e52", 1: "#2ca02c"}


def pick_indices(dataset, mode, n, rng):
    """Row(s) of dataset indices to show."""
    pos = np.flatnonzero(dataset.labels == 1)
    neg = np.flatnonzero(dataset.labels == 0)
    if mode == "positives":
        return rng.choice(pos, min(n, len(pos)), replace=False)
    if mode == "negatives":
        return rng.choice(neg, min(n, len(neg)), replace=False)
    if mode == "random":
        return rng.choice(len(dataset), min(n, len(dataset)), replace=False)
    # "balanced": one row of each class
    half = max(1, n // 2)
    return np.concatenate([
        rng.choice(pos, min(half, len(pos)), replace=False),
        rng.choice(neg, min(half, len(neg)), replace=False),
    ])


def to_image(tensor):
    """CHW float tensor in [0,1] -> HWC array matplotlib can draw."""
    return tensor.permute(1, 2, 0).clamp(0, 1).numpy()


def grid(dataset, indices, out_path, cols):
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.0 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, indices):
        crop, label, box, image_idx = dataset[int(i)]
        ax.imshow(to_image(crop))
        ax.set_title(LABEL_NAME[label], color=LABEL_COLOR[label], fontsize=9)
        w, h = box[2] - box[0], box[3] - box[1]
        ax.set_xlabel(f"{dataset.image_names[image_idx]}\n{int(w)}x{int(h)} px", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def with_context(dataset, indices, out_path, cols=4):
    """Each crop shown in place: the whole image with its proposal box drawn."""
    from matplotlib.patches import Rectangle

    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, indices):
        _, label, box, image_idx = dataset[int(i)]
        ax.imshow(dataset._image(int(image_idx)))
        x0, y0, x1, y1 = [float(v) for v in box]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=LABEL_COLOR[label], lw=2))
        ax.set_title(LABEL_NAME[label], color=LABEL_COLOR[label], fontsize=9)
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--mode", default="balanced",
                    choices=["balanced", "positives", "negatives", "random"])
    ap.add_argument("--n", type=int, default=16, help="how many crops")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--size", type=int, default=128, help="crop size shown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-context", action="store_true",
                    help="also save the whole image with the box drawn on it")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    # imagenet_norm=False: normalised tensors would show as false-colour mush.
    dataset = PotholeProposals(
        args.split,
        make_transform(train=False, size=args.size, imagenet_norm=False),
    )
    print(f"{args.split}: {len(dataset.image_names)} images, {len(dataset)} crops, "
          f"{dataset.class_counts()}")

    rng = np.random.default_rng(args.seed)
    indices = pick_indices(dataset, args.mode, args.n, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"crops_{args.split}_{args.mode}.png"
    grid(dataset, indices, out, args.cols)
    print(f"wrote {out}")

    if args.with_context:
        out = out_dir / f"crops_{args.split}_{args.mode}_context.png"
        with_context(dataset, indices[: min(8, len(indices))], out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
