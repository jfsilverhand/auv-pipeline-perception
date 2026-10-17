"""Training curves for YOLOv8s (Ultralytics results.csv) and the two segmentation models.

Prints the best epoch of each run and writes figures/training_curves.png.
"""
import pandas as pd

from common import MODEL_COLORS, RESULTS, plt, save, style


def yolo_best_epoch(df):
    # Ultralytics selects best.pt by fitness = 0.1 * mAP50 + 0.9 * mAP50-95
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return df.loc[fitness.idxmax()]


def main():
    yolo = pd.read_csv(RESULTS / "detection/yolov8s_train_results.csv")
    best = yolo_best_epoch(yolo)
    print(f"YOLOv8s: {len(yolo)} epochs run (patience 15), best fitness at epoch {int(best.epoch)}: "
          f"val mAP50={best['metrics/mAP50(B)']:.3f}, mAP50-95={best['metrics/mAP50-95(B)']:.3f}")

    seg = {}
    for name, fname in [("DeepLabV3", "deeplabv3_train_log.csv"), ("FPN", "fpn_train_log.csv")]:
        df = pd.read_csv(RESULTS / "segmentation" / fname)
        i = df["val_miou"].idxmax()
        seg[name] = df
        print(f"{name}: {len(df)} epochs run (patience 10), best val mIoU {df.val_miou[i]:.4f} at epoch {int(df.epoch[i])}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(yolo.epoch, yolo["metrics/mAP50(B)"], label="val mAP@0.5", color=MODEL_COLORS["YOLOv8s"])
    ax.plot(yolo.epoch, yolo["metrics/mAP50-95(B)"], label="val mAP@0.5:0.95", color="#7fb8d4")
    ax.axvline(best.epoch, ls="--", lw=1, color="grey")
    ax.text(best.epoch + 1, 0.15, f"best epoch {int(best.epoch)}", fontsize=8, color="grey")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "YOLOv8s (stopped at epoch 85, patience 15)", "epoch", "validation metric")

    ax = axes[1]
    for name, df in seg.items():
        ax.plot(df.epoch, df.val_miou, label=name, color=MODEL_COLORS[name])
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Segmentation fine-tuning on corrected masks", "epoch", "validation mIoU")
    save(fig, "training_curves.png")


if __name__ == "__main__":
    main()
