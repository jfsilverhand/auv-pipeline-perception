"""Precision / recall / F1 of YOLOv8s on the test split as a function of the confidence threshold.

Writes figures/confidence_sweep.png and prints the F1-optimal threshold.
"""
import pandas as pd

from common import RESULTS, plt, save, style


def main():
    df = pd.read_csv(RESULTS / "detection/conf_sweep.csv")
    best = df.loc[df.f1.idxmax()]
    print(f"Best F1 = {best.f1:.3f} at conf = {best.conf:.2f} "
          f"(P = {best.precision:.3f}, R = {best.recall:.3f}, TP/FP/FN = {int(best.tp)}/{int(best.fp)}/{int(best.fn)})")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(df.conf, df.precision, marker="o", ms=3, label="precision")
    ax.plot(df.conf, df.recall, marker="o", ms=3, label="recall")
    ax.plot(df.conf, df.f1, marker="o", ms=3, label="F1", color="black")
    ax.axvline(best.conf, ls="--", lw=1, color="grey")
    ax.legend(fontsize=8, frameon=False)
    style(ax, "YOLOv8s confidence sweep (test split, IoU >= 0.5)", "confidence threshold", "score")
    save(fig, "confidence_sweep.png")


if __name__ == "__main__":
    main()
