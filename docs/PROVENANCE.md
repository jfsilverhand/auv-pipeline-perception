# Provenance of the README numbers

Each number in the main README is listed here with its source: either a file in `results/` or a script that derives it. Numbers marked **thesis** have no raw file in this repository.

| README claim | Value | Source |
|---|---|---|
| Detection dataset size | 8,792 / 629 / 596 | Ultralytics training log (not included; train 8,792, val 629) and `reports_tr/yolov8_test_raporu.txt` (test 596) |
| Near-duplicate removal | 10,639 → 10,017 images | thesis (pHash 8×8 DCT, Hamming ≤ 5) |
| Detection is single-class | `nc=1`, class `deniz_boru` | Ultralytics log of the same `data/dataset_20k/data.yaml` used in `detection/yolov8s_args.yaml` |
| YOLOv8s test mAP@0.5 / 0.5:0.95 / P / R | 0.873 / 0.685 / 0.816 / 0.850 | `reports_tr/yolov8_test_raporu.txt` |
| YOLOv8s inference | 8.4 ms | `reports_tr/yolov8_test_raporu.txt` |
| YOLOv8s best epoch, stop epoch | 70, 85 | `analysis/training_curves.py` on `detection/yolov8s_train_results.csv` |
| YOLOv11s test metrics, inference | 0.870 / 0.690 / 0.859 / 0.804, 8.6 ms | `reports_tr/yolov11s_test_raporu.txt` |
| Untrained detection controls | 0.061 / 0.079 | `detection/untrained_controls_yolo.json` |
| Segmentation test mIoU / Dice / ms | DeepLabV3 0.773 / 0.856 / 42.0; FPN 0.782 / 0.856 / 16.1 | `segmentation/seg_test_eval.csv` |
| Untrained segmentation controls | 0.145 / 0.036 | `comparison/untrained_vs_final.csv`, `segmentation/untrained_control_fpn.json` |
| Mask correction effect | 0.691 → 0.771 | 0.771: `segmentation/deeplabv3_train_log.csv`; 0.691: **thesis** (pre-correction checkpoint) |
| Best F1 threshold (test) | 0.822 at 0.40 | `analysis/confidence_sweep.py` on `detection/conf_sweep.csv` |
| Best F1 threshold (val) | 0.84 at 0.386 | `figures/ultralytics_yolov8s/BoxF1_curve.png` |
| Enhancement result, Spearman | best +0.001; ρ = −0.25, p = 0.49 | `analysis/enhancement_ablation.py` on `enhancement/enhancement_ablation.csv` |
| Noise drops | −54% (YOLO), −11 to −13% (seg) at σ = 25 | `analysis/robustness.py` on `robustness/robustness.csv` |
| Subset scores and box statistics | 0.794 / 0.958, 1.0% small boxes, 1,188 / 2,515 train images | `detection/test_subset_analysis.csv` |
| Video hybrid ms per frame | 15.7 / 54.9 / 44.4 | `comparison/video_hybrid.csv` |
| Ensemble | 0.676 → 0.736, 44 → 18 FPS | `comparison/ensemble.csv` |
| Split leakage on the segmentation set | 107 / 256; 112 / 141 | output of `tools/split_leakage_check.py` on the private dataset (October 2026) |
| Jetson trtexec, real-video FPS, power, jetson_clocks | see README table | **thesis**; raw `trtexec` / `tegrastats` logs not retained |

## Known inconsistencies kept on purpose

- `comparison/master_test_3model.csv` reports YOLOv8s mAP@0.5 = 0.676 and 55 FPS. That evaluation uses the 256-image segmentation test split, with boxes derived from masks, and a different timing script. It is a different measurement from the 596-image detection test (0.873, 8.4 ms), not a conflicting one.
- `robustness/robustness.csv` uses a clean YOLOv8s baseline of 0.797, from yet another evaluation run. Its protocol was not recorded.
