"""Plot recall vs. number of proposals from recall_curves.csv (made by recall_curves.py).

Usage:
    python plot_recall.py recall_curves.csv recall_vs_n.png
"""
import csv
import sys

import matplotlib.pyplot as plt

# Categorical slots 1-4 of the validated default chart palette, in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {
    "fast_300": "Fast, 300 px",
    "fast_full": "Fast, full size",
    "quality_300": "Quality, 300 px",
    "quality_full": "Quality, full size",
}
MIN_LABEL_GAP = 0.04  # in recall units, enough for one 8pt line at this figure size


def main() -> None:
    csv_path, png_path = sys.argv[1], sys.argv[2]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    ns = [int(r["n"]) for r in rows]
    series = [k for k in rows[0] if k != "n"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ends = {}
    for color, key in zip(COLORS, series):
        recall = [float(r[key]) for r in rows]
        ax.plot(ns, recall, color=color, linewidth=2, label=LABELS.get(key, key))
        ends[key] = recall[-1]

    # Direct labels at the line ends: two colors are below 3:1 contrast on white,
    # so identity must not rely on the legend swatch alone. Push labels apart
    # (top to bottom) when lines end at nearly the same recall.
    label_y = None
    for key, end in sorted(ends.items(), key=lambda item: -item[1]):
        label_y = end if label_y is None else min(end, label_y - MIN_LABEL_GAP)
        ax.annotate(f"{LABELS.get(key, key)} ({end:.2f})", (ns[-1], label_y),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color="#333333", annotation_clip=False)

    ax.set_xlabel("Number of proposals per image (N)")
    ax.set_ylabel("Recall (IoU ≥ 0.5), train split")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, ns[-1])
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Written {png_path}")


if __name__ == "__main__":
    main()
