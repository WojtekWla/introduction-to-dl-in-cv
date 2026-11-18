import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

txt = open(sys.argv[1] if len(sys.argv) > 1 else "run_log.txt").read()

runs, cur = {}, []
for line in txt.splitlines():
    m = re.match(r"Epoch (\d+)/\d+\s+loss train: [\d.]+\s+val: [\d.]+\s+val Dice: ([\d.]+)", line)
    if m:
        if int(m.group(1)) == 1:
            cur = []
        cur.append(float(m.group(2)))
        continue
    m = re.search(r"Best val Dice: [\d.]+\s+\(model: (\w+)_best\.pt", line)
    if m:
        runs[m.group(1)] = cur
        cur = []

if not runs:
    sys.exit("No runs found in the log.")

panels = [("ph2", "PH2 (256×256)"), ("drive", "DRIVE (256×256)"),
          ("drive576", "DRIVE (576×576)")]
panels = [(k, t) for k, t in panels if f"encdec_{k}" in runs or f"unet_{k}" in runs]
styles = {"encdec": ("#1f77b4", "Encoder-decoder"), "unet": ("#d62728", "U-Net")}

fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.6),
                         sharey=True, squeeze=False)
for ax, (key, title) in zip(axes[0], panels):
    for model, (color, label) in styles.items():
        r = runs.get(f"{model}_{key}")
        if r:
            ax.plot(range(1, len(r) + 1), r, color=color, lw=1.3, label=label)
    n = max(len(runs.get(f"{m}_{key}", [])) for m in styles)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_xlim(1, max(n, 2)); ax.set_ylim(-0.03, 1)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3, lw=0.5)
axes[0][0].set_ylabel("validation Dice", fontsize=8)
axes[0][0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("val_dice_curves.pdf")
fig.savefig("val_dice_curves.png", dpi=300)

for tag, r in runs.items():
    best = max(r)
    print(f"{tag:16s} best {best:.4f} at epoch {r.index(best) + 1} of {len(r)}")
print("saved val_dice_curves.pdf / .png")
