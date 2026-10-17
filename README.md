# Underwater pipeline perception for an AUV

Camera-based pipeline detection and segmentation for an autonomous underwater vehicle, deployed on a Jetson Orin Nano with TensorRT FP16. This is the result and analysis repository of my B.Sc. thesis (Kocaeli University, Computer Engineering, 2026), built for the AURA team vehicle.

It contains the logged outputs of the training and evaluation runs, plus code that regenerates every figure and number in this README from those files. It does **not** contain the dataset, the trained weights or the original training scripts (see [What is not here](#what-is-not-here)).

![Training curves](figures/training_curves.png)

## Pipeline

| Stage | Models | Data |
|---|---|---|
| Detection | YOLOv8s, YOLOv11s (Ultralytics, single class `pipe`) | 10,017 images after pHash near-duplicate removal; 8,792 train / 629 val / 596 test |
| Segmentation | DeepLabV3 and FPN, ResNet-34 encoder (segmentation_models_pytorch) | 2,533-image set with 1,129 hand-corrected masks; 2,025 train / 252 val / 256 test |
| Deployment | TensorRT FP16 engines on Jetson Orin Nano Super (JetPack 6.2) | Pool and sea test videos |
| Tracking | Kalman-filter tracker on top of detections (vehicle side) | Team test video |

Detection and segmentation were trained and tested on different splits, so their metrics are not directly comparable.

## Results

### Detection (test split, 596 images)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (RTX 3050 Ti Laptop) |
|---|---|---|---|---|---|
| YOLOv8s | 0.873 | 0.685 | 0.816 | 0.850 | 8.4 ms |
| YOLOv11s | 0.870 | 0.690 | 0.859 | 0.804 | 8.6 ms |
| Untrained control: COCO weights, no pipe class | 0.061 / 0.079 | | | | |

YOLOv8s was trained with SGD, seed 0, patience 15; it stopped at epoch 85 and the best checkpoint is epoch 70 (val mAP@0.5 0.906).

### Segmentation (205 pipe-containing images of the 256-image test split)

| Model | mIoU | Dice | Inference | Untrained control (random decoder) |
|---|---|---|---|---|
| DeepLabV3 | 0.773 | 0.856 | 42.0 ms | 0.145 |
| FPN | 0.782 | 0.856 | 16.1 ms | 0.036 |

On the same pipeline, hand-correcting 1,129 masks raised DeepLabV3 validation mIoU from 0.691 to 0.771.

### Jetson Orin Nano Super, TensorRT FP16 †

| Model | trtexec GPU time | Real video, 25 W | Real video, MAXN_SUPER |
|---|---|---|---|
| YOLOv8s | 7.31 ms | 23.6 FPS | 38.9 FPS |
| YOLOv11s | 7.50 ms | 23.7 FPS | 43.4 FPS |
| FPN | 9.60 ms | 15.7 FPS | 25.1 FPS |
| DeepLabV3 | 18.39 ms | 14.3 FPS | 20.9 FPS |

Average board input power (VDD_IN) during inference was 8.8 W at 25 W mode and 12.4 W at MAXN_SUPER. Locking clocks with `jetson_clocks` raised YOLOv11s from 24.7 to 43.4 FPS on the same video. The real-video FPS is far below the trtexec GPU time, so most of the frame time is spent outside the GPU (decode, pre/post-processing). I have not profiled this, so it remains a hypothesis.

† These numbers come from the thesis measurements. The raw `trtexec` and `tegrastats` logs were not retained, so they cannot be re-derived from this repository.

## What I found

1. **Pick the threshold for the tracker, not the leaderboard.** On the test split, F1 peaks at confidence 0.40 (P 0.811, R 0.834). On the validation split, the F1 curve peaks at 0.386 ([plot](figures/ultralytics_yolov8s/BoxF1_curve.png)). → `analysis/confidence_sweep.py`
2. **Classical enhancement did not help.** Ten methods (CLAHE, UDCP, Retinex, fusion, …) were applied before all three models. None improved the mean task score meaningfully: the best was +0.001 (Ancuti colour compensation), and the other nine lowered it by 0.003–0.052. Underwater image-quality scores did not predict accuracy: Spearman ρ = −0.25, p = 0.49 for UIQM, n = 10. → `analysis/enhancement_ablation.py`
3. **Gaussian noise breaks the detector first, but this is not an architecture result.** At σ = 25, YOLOv8s loses 54% of its mAP and the segmentation models lose 11–13% of their mIoU. The segmentation models were trained with GaussNoise augmentation and YOLOv8s was not, and the metrics and input sizes also differ. A controlled re-run is needed before drawing any conclusion. → `analysis/robustness.py`
4. **The cleanest-looking test subset scored lowest** (mAP@0.5 0.794, against 0.958 for another subset). Small objects do not explain it, since only 1.0% of its boxes are small. Label style (hand-drawn boxes vs. boxes derived from polygons) and lower training representation (1,188 vs. 2,515 images) are more likely causes.
5. **Hybrid detection + segmentation on video.** On a 6,550-frame video, frame time was 15.7 ms for YOLOv8s alone, 54.9 ms for FPN and 44.4 ms for the hybrid. A max-confidence ensemble raised mAP@0.5 on the 256-image set from 0.676 to 0.736, but cut throughput from 44 to 18 FPS.

## Limitations

Read these before quoting any number above.

- **Frame-level splits.** On the segmentation set, 107 of the 256 test images come from 8 videos that also appear in training. Of the remaining 141 test images, 112 have an augmented copy of the same source image in training. The segmentation test scores are therefore optimistic. The detection split was not re-audited, because the dataset is no longer available locally. The audit tool is `tools/split_leakage_check.py`.
- One training run per model (YOLO seed 0, segmentation seed 42); no confidence intervals.
- The robustness test reports a different clean YOLOv8s baseline (0.797) than the headline test number (0.873), because it came from a separate evaluation run. Compare drops only within that table.
- Only a monocular camera is used: no sonar and no multi-sensor fusion.

**What I would do differently:** split by source video from the start; give every model the same augmentation; measure power per model instead of per power mode; and keep training scripts and raw logs under version control from day one.

## Reproduce the figures

```bash
pip install -r requirements.txt
cd analysis
python run_all.py
```

This rewrites `figures/*.png` and prints the derived numbers (best epochs, best threshold, Spearman correlation, relative robustness drops).

## Repository layout

```
results/        logged outputs of the original runs (CSV / JSON / YAML)
  reports_tr/   the original evaluation reports, in Turkish
analysis/       scripts that turn results/ into figures and numbers
figures/        generated figures; ultralytics_yolov8s/ holds the original Ultralytics plots of the run
tools/          split_leakage_check.py (needs the private dataset)
docs/           PROVENANCE.md: which number comes from which file
```

Column names in some CSVs are Turkish; [`results/README.md`](results/README.md) translates them.

## What is not here

- **Dataset**: it combines team-recorded video with third-party Roboflow exports that I cannot redistribute.
- **Weights**: they were trained on that dataset.
- **Training and evaluation scripts**: the original scripts were not recovered in a publishable state. The numbers here come from their logged outputs, and the scripts in this repo only analyse those outputs.

## Thesis

N. E. Güven, *İnsansız Su Altı Araçlarında Derin Öğrenme Tabanlı Otonom Boru Hattı Takibi Sistemi* (Deep-learning-based autonomous pipeline tracking for unmanned underwater vehicles), B.Sc. thesis, Kocaeli University, 2026. In Turkish; available on request.
