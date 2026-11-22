# RF-DETR Small, seed 0 (Kaggle, Tesla T4)

- Code: commit `24b361e`; settings in `run_config.json`, environment in `sessions.json`.
- Training ran in one session, 7.4 h, and finished (early stopping / epoch limit; state `done`).
- Batch 8 x grad-accum 2 (effective 16), resolution 512, lr 1e-4 / encoder 1.5e-4.
- Best checkpoint selected on val only. test1/test2 were scored once after training.
- `metrics/{split}.json`: output of `uatd.eval_coco` (COCO stats, Ultralytics-style P/R/F1,
  per-class, tiny/non-tiny bins from `configs/eval.yaml`).
- Raw predictions (~45 MB per split) and checkpoints are kept outside the repo
  (`D:\UATD\kaggle_out\rfdetr`).

| split | mAP50 | mAP75 | mAP50:95 | P | R | F1 | AP tiny | AP non-tiny |
|---|---|---|---|---|---|---|---|---|
| val   | 0.967 | 0.632 | 0.578 | 0.972 | 0.967 | 0.969 | 0.460 | 0.592 |
| test1 | 0.815 | 0.261 | 0.368 | 0.867 | 0.829 | 0.840 | 0.323 | 0.375 |
| test2 | 0.825 | 0.246 | 0.363 | 0.878 | 0.850 | 0.855 | 0.259 | 0.365 |
