"""Does a higher underwater image-quality score (UIQM / UCIQE) predict better model accuracy?

For each of the 10 enhancement methods, the mean change in task metric over the three models
(YOLOv8s mAP@0.5, DeepLabV3 mIoU, FPN mIoU) is compared with the no-enhancement baseline,
then correlated with UIQM and UCIQE (Spearman). Writes figures/enhancement_vs_quality.png.
"""
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, plt, save, style

TASK_COLS = ["yolo_map50", "deeplabv3_miou", "fpn_miou"]


def main():
    df = pd.read_csv(RESULTS / "enhancement/enhancement_ablation.csv")
    base = df[df.yontem == "baseline"].iloc[0]
    df["mean_delta"] = sum(df[c] - base[c] for c in TASK_COLS) / len(TASK_COLS)
    methods = df[df.yontem != "baseline"].sort_values("mean_delta", ascending=False)

    print("method                      UIQM   UCIQE  mean delta")
    for _, r in methods.iterrows():
        print(f"{r.yontem:<26} {r.uiqm:6.3f} {r.uciqe:7.4f} {r.mean_delta:+10.4f}")

    for q in ["uiqm", "uciqe"]:
        rho, p = spearmanr(methods[q], methods.mean_delta)
        print(f"Spearman({q.upper()}, mean delta) over {len(methods)} methods: rho = {rho:+.2f}, p = {p:.2f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(methods.uiqm, methods.mean_delta, color="#2e86ab")
    for _, r in methods.iterrows():
        ax.annotate(r.yontem, (r.uiqm, r.mean_delta), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=1)
    style(ax, "Image quality score vs. task accuracy change", "UIQM (higher = 'better looking')",
          "mean change vs. no enhancement")
    save(fig, "enhancement_vs_quality.png")


if __name__ == "__main__":
    main()
