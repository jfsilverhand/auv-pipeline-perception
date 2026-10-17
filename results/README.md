# results/

These are the logged outputs of the original runs. No values were edited. Files were renamed to English, and the local `save_dir` path in `yolov8s_args.yaml` was replaced. Several files keep their original Turkish column names; the tables below translate them.

## detection/

| File | Content |
|---|---|
| `yolov8s_train_results.csv` | Ultralytics `results.csv` of the YOLOv8s run (85 epochs, validation metrics per epoch) |
| `yolov8s_args.yaml` | Ultralytics training arguments (`data/dataset_20k/data.yaml`, single class, SGD, seed 0) |
| `conf_sweep.csv` | Test-split precision / recall / F1 / TP / FP / FN per confidence threshold (IoU ≥ 0.5) |
| `error_categories.csv` | Test errors at conf 0.40. `kategori` = category (Tam Dogru = correct, Loc Hatasi = localisation error, FP, FN, Dup = duplicate), `adet` = count, `hata_ici_%` = share of all errors |
| `bbox_size_performance.csv` | `boyut` = box size bucket (kucuk = small, orta = medium, buyuk = large), `n_gt` = ground-truth boxes, `map50`, `recall_c40` = recall at conf 0.40. Produced by a separate evaluation script, so it is not directly comparable to the Ultralytics test mAP |
| `test_subset_analysis.csv` | Per source subset of the test split. `kaynak` = source, `test_n` / `train_n` = images, `borulu_%` = share with a pipe, `ort_alan_px2` = mean box area, `alan_orani_%` = box/image area, `nbbox_img` = boxes per image, `enboy` = aspect ratio, `kucuk/orta/buyuk_%` = small/medium/large boxes, `gt_stili` = label style (polygon-derived, bbox, borusuz = mostly empty) |
| `untrained_controls_yolo.json` | COCO-pretrained YOLOv8s / YOLOv11s evaluated on the test split without any pipe training |

## segmentation/

| File | Content |
|---|---|
| `seg_test_eval.csv` | DeepLabV3 / FPN on the 205 pipe-containing test images: val/test mIoU, Dice, P, R, pixel accuracy, inference ms, FPS. `n_borulu` = images with a pipe, `fark` = test − val |
| `deeplabv3_train_log.csv`, `fpn_train_log.csv` | Per-epoch fine-tuning log on the corrected masks (patience 10, early-stopped at epoch 31). `sure_sn` = epoch time in seconds |
| `untrained_control_fpn.json` | FPN with an ImageNet encoder and a randomly initialised decoder, no training |

## comparison/

| File | Content |
|---|---|
| `untrained_vs_final.csv` | Untrained control vs. final model (semicolon separated, decimal comma). `Kazanim` = difference, `Kazanim_Kat` = ratio |
| `master_test_3model.csv` | The three models on the 256-image segmentation test split, with YOLO boxes derived from masks. **`fps_jetson_tahmin` is a placeholder estimate (PC FPS × 0.45), not a measurement.** `fps_pc` came from a different timing script than the inference times in the main README; its timing conditions were not recorded |
| `video_hybrid.csv` | YOLO, FPN and the hybrid on one 6,550-frame video. `kare` = frames, `tespit` = frames with a detection, `ms_kare` = ms per frame end-to-end, `hata_std_x/y` = std. of tracking error in px |
| `ensemble.csv` | YOLO alone vs. two YOLO+FPN ensembles on the 256-image set (`yontem` = method) |

## enhancement/

`enhancement_ablation.csv`: one row per enhancement method (`yontem`) with its image-quality scores (`uiqm`, `uciqe`), each model's score after enhancement and `sure_sn` (processing time).

## robustness/

`robustness.csv`: `bozulma` = corruption type, `seviye` = level (blur kernel size, noise σ, brightness %, contrast factor, hue shift), followed by each model's score.

## reports_tr/

These are the original text reports in Turkish, written by the evaluation scripts at the time. They include interpretation notes from that time, which the main README supersedes.
