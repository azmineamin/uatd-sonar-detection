# UATD Sonar Object Detection (CSE499A)

Object detection on forward-looking sonar images from the
[UATD dataset](https://doi.org/10.1038/s41597-022-01854-w) (Xie et al., *Scientific Data* 2022),
with a focus on tiny/small targets and transformer (DETR-family) detectors.
CSE499A capstone, North South University.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Data split (fixed)

The split reproduces the original Kaggle notebook (`uatd-yolo.ipynb`): `UATD_Training` is
shuffled with `random.Random(42)` and 10 % becomes `val`; `UATD_Test_1` and `UATD_Test_2`
are the two test sets. Images with any invalid box are dropped (11 images in total).
The manifests in [`data/splits/`](data/splits/) are committed and every pipeline reads them.
**Never re-split.**

| split | source | images | objects | dropped |
|---|---|---|---|---|
| train | UATD_Training | 6,835 | 11,059 | 5 |
| val | UATD_Training | 758 | 1,237 | 2 |
| test1 | UATD_Test_1 | 798 | 1,169 | 2 |
| test2 | UATD_Test_2 | 798 | 1,154 | 2 |

Notes:
- `val` is a random 10 % of consecutive sonar frames, so val scores are optimistic. Report test1 and test2.
- Test results are on the *cleaned* official test sets (4 test images removed).
- Published UATD numbers often use other splits and are not directly comparable.

Rebuild or verify the manifests (output must be identical to the committed files):

```bash
# from an extracted copy (e.g. the Kaggle mirror)
python -m uatd.splits --raw-root /kaggle/input/datasets/mridhafahadhossain/uatd-dataset --out /tmp/splits

# or straight from figshare: reads only the XMLs and BMP headers over HTTP (no 4.7 GB download)
python -m uatd.splits --remote --cache data/raw/uatd_index.json.gz --out /tmp/splits
```

## Convert to YOLO / COCO

```bash
python -m uatd.convert --raw-root <UATD root> --out data/processed --link-mode symlink
# --link-mode png  writes lossless PNG copies (faster reads on Kaggle); pixels are identical
# --smoke          converts only 6 images per split (quick check)
```

Output: `images/{split}/`, `labels/{split}/` (YOLO), `coco/{split}.json`, `uatd_yolo.yaml`.
File names and COCO `uatd_id` are split-prefixed (`test1_00247`) because raw stems collide
across folders. COCO `category_id` = YOLO class ID (0–9).

## RF-DETR (main pipeline, ICLR 2026)

Trains on all of `train` (6,835 images), selects the best checkpoint on `val` (early stopping), and is
scored on `test1` and `test2` with the shared evaluator. Test images are never used for training or selection.

```bash
pip install -e ".[rfdetr]"
python scripts/train_rfdetr.py --data data/processed --model small --epochs 100 --batch 8 --grad-accum 2 --seed 0
python scripts/predict_rfdetr.py --run runs/rfdetr/small_r512_s0 --data data/processed
# quick CPU check: add --device cpu --smoke to the training command
```

**Checkpoints and resume.** RF-DETR saves `last.ckpt` (with optimizer state) after every epoch. Re-running the
same training command resumes from the newest checkpoint that loads. `--time-budget-hours` stops training cleanly
before a Kaggle session ends. `status.json` in the run folder is `running`, `interrupted`, `done` or `failed`;
a `done` run is never retrained. On Kaggle, use [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb)
(each stage is skipped when already done; turn on *Persistence → Files only*).

**Evaluation.** `python -m uatd.eval_coco --gt <coco/split.json> --pred <predictions.json>` reports COCO mAP,
mAP50/75, AP_s/m/l, AR, per-class AP and (once `configs/eval.yaml` is frozen) UATD-relative size bins.
It also reports P, R and F1 with the Ultralytics definition used for the YOLO26n baseline (IoU 0.5, at the
confidence of maximum mean F1), so every model's P/R is comparable; `tests/test_eval.py` checks it against
Ultralytics' own code.

## Classes

`0 ball, 1 circle cage, 2 cube, 3 cylinder, 4 human body, 5 metal bucket, 6 plane, 7 rov, 8 square cage, 9 tyre`

## Roadmap

- [x] 1. Scaffold (layout, `pyproject.toml`, `.gitignore`, pre-commit/ruff)
- [x] 2. Split and convert: committed manifests, YOLO + COCO views, exact-count tests
  - [x] manifests built from the figshare data, all Section 4.3 numbers verified (`tests/test_splits.py`)
  - [x] split re-verified on the extracted BMPs; full `uatd.convert` run (9,189 images) matches all counts
- [ ] 3. Dataset analysis; freeze the tiny-object threshold in `configs/eval.yaml`
- [ ] 4. Shared COCO evaluator + efficiency tooling
  - [x] `uatd.eval_coco` (COCO metrics, per-class AP, size bins), tested
  - [ ] efficiency (params, GFLOPs, T4 latency)
- [ ] 5. Main pipeline: RF-DETR (ICLR 2026) N/S
  - [x] resumable training + prediction scripts, CPU smoke test on real UATD images
  - [ ] full training run on Kaggle
- [ ] 6. Baselines: YOLOv12, FBRT-YOLO, D-FINE/DEIM, Dome-DETR
- [ ] 7. Error analysis
- [ ] 8. Method (tiny-object idea on RF-DETR) + ablations
- [ ] 9. Final runs (3 seeds), efficiency table, figures

## Repository layout

```
configs/       data and experiment configs
data/splits/   committed split manifests + expected_counts.json
src/uatd/      package: constants, voc (cleaning rules), rawindex, splits, convert
tests/         pytest suite (exact split numbers, cleaning rules, converter)
scripts/       training / evaluation entry points
notebooks/     thin Kaggle runner
results/       small committed result summaries
```
