"""Relative performance drop of each model under synthetic corruptions.

Writes figures/robustness.png and prints the relative change (%) for every corruption level.
Note: the YOLOv8s clean baseline here (0.797) comes from a different evaluation run than the
headline test mAP (0.873); compare drops within this table only.
"""
import pandas as pd

from common import MODEL_COLORS, RESULTS, plt, save, style

COLS = {"yolo_map50": "YOLOv8s", "deeplabv3_miou": "DeepLabV3", "fpn_miou": "FPN"}


def main():
    df = pd.read_csv(RESULTS / "robustness/robustness.csv")
    base = df[df.bozulma == "baseline"].iloc[0]
    rows = df[df.bozulma != "baseline"].copy()
    for c in COLS:
        rows[c + "_rel"] = 100 * (rows[c] - base[c]) / base[c]

    print(f"{'corruption':<11}{'level':>6}" + "".join(f"{v:>11}" for v in COLS.values()))
    for _, r in rows.iterrows():
        print(f"{r.bozulma:<11}{r.seviye:>6}" + "".join(f"{r[c + '_rel']:+10.1f}%" for c in COLS))

    labels = [f"{a}\n{b}" for a, b in zip(rows.bozulma, rows.seviye)]
    x = range(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for k, (c, name) in enumerate(COLS.items()):
        ax.bar([i + (k - 1) * w for i in x], rows[c + "_rel"], w, label=name, color=MODEL_COLORS[name])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="grey", lw=1)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax, "Relative change vs. clean input", None, "relative change (%)")
    save(fig, "robustness.png")


if __name__ == "__main__":
    main()
