# CLAUDE.md — UATD Sonar Object Detection (CSE499A)

This file gives Claude Code the context for this repository. Read it fully at the start of every session.

## 1. Project in one paragraph

CSE499A capstone, North South University (student GitHub: `azmineamin`). We detect objects in **forward-looking sonar images** from the **UATD dataset**, using a recent detection pipeline, with a focus on **tiny/small object detection and/or a transformer (DETR-family) detector**, as the professor asked. The project runs Nov 2026 to May 2027, and the target is a Q1/Q2 journal or an A/A* conference.

**Out of scope:** habitat mapping, side-scan segmentation, BenthiCat, SeafloorAI, FSOD, and any other dataset. An earlier direction (`azmineamin/sonar-habitat-repro`, RSOD reproduction and BenthiCat segmentation) has been dropped. Do not carry code or plans over from it unless asked. Its `tools/uatd_to_coco.py` may be read for reference only.

## 2. Hard rules

1. **Keep the data split fixed.** Use exactly the split described in Section 4, which reproduces the Kaggle notebook `uatd-yolo.ipynb`. Never reshuffle, re-split or "fix" it. Every model is trained and evaluated on the same committed manifests.
2. **Use one evaluator for all models.** Report every number with the shared COCO evaluator (Section 6), never only with a framework's built-in metric. Built-in metrics are fine for monitoring during training.
3. **Never tune on test.** `val` is used for early stopping and model selection. `test1` and `test2` are only used for final reporting.
4. **Don't write academic prose.** The student must write all paper/report text in their own words (they have had AI-detector problems). Claude may write code, comments, READMEs, tables and figures, and may fix facts or shorten the student's text when asked. It must not draft paragraphs for the paper, literature review or report.
5. **Pipelines must come from peer-reviewed top venues.** Every detector we build on must be published (2025/2026) at an **A/A\* conference** (CORE ranking: CVPR, ICCV, ECCV, NeurIPS, ICLR, AAAI, ACM MM, IJCAI, …) or in a **Q1/Q2 journal** (SJR/JCR) from IEEE, Nature, Springer, Elsevier, MDPI, Wiley, Taylor & Francis, ACM, Hindawi, IGI or OUP. arXiv-only methods, company releases without a paper (e.g., Ultralytics YOLO26), Frontiers and PLOS papers can be read for background but **must not be the pipeline**. When adding a new method, record its venue, year and code link in Section 7.4 first, and check the CORE rank or journal quartile; if unsure, say so.
6. **Do not use RT-DETR.** A classmate is already working on RT-DETR (and RT-DETRv2). Don't train or tune RT-DETR/RT-DETRv2 as our pipeline or our own baseline. If a comparison row is needed, coordinate with the classmate and cite their numbers on the same split. Our transformer work must stay clearly different from theirs (see Section 7).
7. **Make runs reproducible.** Fix seeds, save the resolved config next to every run, log the git commit hash, and never overwrite a finished run directory.

## 3. Dataset: UATD

- Underwater Acoustic Target Detection dataset (Xie et al., *Scientific Data* 2022, doi 10.1038/s41597-022-01854-w). Tritech Gemini 1200ik multibeam forward-looking sonar. Source: figshare 21331143. Kaggle mirror used so far: `/kaggle/input/datasets/mridhafahadhossain/uatd-dataset`.
- Raw layout: `UATD_Training/`, `UATD_Test_1/`, `UATD_Test_2/`, each with an `images` folder (`.bmp`, about 3–4 MB each) and an annotations folder (Pascal-VOC `.xml`). Find the folders with a case-insensitive search for `image|images|img|jpegimages` and `annotation|annotations|xml|labels`, as the notebook does.
- 7,600 / 800 / 800 image–XML pairs; 9,200 in total.
- **Classes (this order = class IDs 0–9):**
  `ball, circle cage, cube, cylinder, human body, metal bucket, plane, rov, square cage, tyre`
  XML `<name>` is lower-cased and stripped before lookup. An unknown class name must raise an error.
- **Filenames collide across splits.** Each folder numbers its images from `00001`, so `test1/00247.bmp` and `train/00247.bmp` are different images. Anything that merges splits (COCO JSON, caches, prediction files) must use split-prefixed IDs, such as `test1_00247`.

## 4. The fixed split (reproduce exactly)

### 4.1 Algorithm (from the notebook, cells 2–4)

```python
SEED, VAL_FRACTION = 42, 0.10
# 1. For each raw folder: pair image and XML by file stem (rglob; image exts .bmp/.png/.jpg/.jpeg),
#    keep stems present in both, sort the stems as strings (zero-padded, so numeric order).
# 2. Shuffle ONLY UATD_Training:
train_pairs = sorted_training_pairs.copy()
random.Random(SEED).shuffle(train_pairs)      # depends only on list length (7600) and seed
n_val = round(len(train_pairs) * VAL_FRACTION) # 760
val   = train_pairs[:n_val]
train = train_pairs[n_val:]
test1 = UATD_Test_1 pairs (sorted), test2 = UATD_Test_2 pairs (sorted)
```

Pre-cleaning sizes: train 6,840 · val 760 · test1 800 · test2 800.

### 4.2 Cleaning rules (applied to all four splits)

For every `<object>`:
- Missing `<bndbox>`, unparseable or non-finite coordinates → object invalid.
- `xmin > xmax` or `ymin > ymax` → swap them (8 boxes in total).
- Clip to `[0, width] × [0, height]`, using the real image size read with PIL (2 boxes clipped).
- Zero area after clipping → object invalid (12 boxes).
- **If any object in an image is invalid, drop the whole image** (image and label). Do not keep a partial label set.
- Images with no objects would be kept as background, but UATD has none.

### 4.3 Expected result (tests must check these exactly)

| split | images | objects |
|---|---|---|
| train | 6,835 | 11,059 |
| val | 758 | 1,237 |
| test1 | 798 | 1,169 |
| test2 | 798 | 1,154 |

Dropped images (11):
- train: `02630, 00707, 00402, 02477, 06975`
- val: `02589, 00438`
- test1: `00247, 00614`
- test2: `00065, 00754`

Per-class object counts:

| class | train | val | test1 | test2 |
|---|---|---|---|---|
| ball | 2733 | 337 | 197 | 193 |
| circle cage | 606 | 55 | 98 | 99 |
| cube | 2390 | 254 | 172 | 166 |
| cylinder | 512 | 52 | 48 | 45 |
| human body | 1163 | 120 | 76 | 73 |
| metal bucket | 426 | 50 | 6 | 5 |
| plane | 730 | 65 | 134 | 135 |
| rov | 629 | 71 | 150 | 150 |
| square cage | 868 | 111 | 167 | 168 |
| tyre | 1002 | 122 | 121 | 120 |

### 4.4 Manifests

The first job is to generate `data/splits/{train,val,test1,test2}.txt` (one source stem per line, plus the source folder) and `data/splits/expected_counts.json`, then **commit them**. After that, every converter reads the manifests and never re-derives the split. `tests/test_splits.py` must assert every number in 4.3.

### 4.5 Caveats to keep in mind

- `val` is a random 10 % of `UATD_Training`. Consecutive sonar frames are near-duplicates, so val scores are optimistic (YOLO26n: val mAP50 0.966 vs test ≈ 0.78–0.80). That is expected; report test1 and test2.
- Published UATD numbers often use other splits (e.g., MambaDSF uses a 7:1:2 random split; RSOD uses Test_1 as val and Test_2 as test). They are **not directly comparable** with ours. Say so in every comparison table, and re-run baselines on our split rather than copying numbers.
- Removing 4 test images means we evaluate on "cleaned" official test sets. Document this.

## 5. Baseline already done (Kaggle notebook, for reference)

YOLO26 has no peer-reviewed paper, so it is **only a reference point** (and a check that our evaluator matches the notebook). It is not one of our pipelines.


YOLO26n (`yolo26n.pt`, Ultralytics 8.4.215, torch 2.11, Kaggle Tesla T4), imgsz 640, AdamW lr0 1e-3, cosine LR, 150 epochs with patience 25, AMP, seed 42. Batch 256 was requested; Ultralytics hit out-of-memory and fell back to 32. Augmentation: hsv_h=0, hsv_s=0, hsv_v=0.2, degrees=0, translate=0.1, scale=0.3, flipud=0, fliplr=0.5, mosaic=1.0, mixup=0, close_mosaic=10. 2.54 M parameters, ~4.8 ms inference per image.

| split | P | R | mAP50 | mAP75 | mAP50:95 |
|---|---|---|---|---|---|
| val | 0.968 | 0.957 | 0.966 | 0.595 | 0.561 |
| test1 | 0.848 | 0.820 | 0.777 | 0.208 | 0.329 |
| test2 | 0.852 | 0.836 | 0.804 | 0.204 | 0.337 |

Observations to build on:
- Weak classes on test: **square cage** (AP50 ≈ 0.51), **circle cage** (≈ 0.51), **cube** (≈ 0.70).
- mAP75 is very low (~0.21) compared with mAP50: **localisation precision is the main gap**, which fits the small-target / blurred-boundary story.
- pycocotools on test2 gave AP_small 0.342, AP_medium 0.337, AP_large 0.343. COCO's absolute area bins say little on these large images, so we need a dataset-relative "tiny" definition (Section 6).
- Notebook bugs not to repeat: total training time was computed by summing a cumulative `time` column (gave nonsense); GFLOPs came out as N/A; batch size was not set to a value that fits a T4.

## 6. Evaluation protocol (one shared evaluator)

`src/uatd/eval_coco.py` takes ground-truth COCO JSON and a predictions JSON (any model) and reports:
- COCO mAP@[.5:.95], mAP50, mAP75, AP_s/m/l, AR@100, using `pycocotools` (or `faster-coco-eval`, same numbers);
- per-class AP50 and AP50:95;
- **size-stratified AP on UATD-relative bins**. Compute the box-area distribution on *train*, define tiny as area ≤ the 25th percentile (and report the absolute pixel threshold), and also report COCO's absolute bins. Freeze the thresholds in `configs/eval.yaml` once computed;
- efficiency: parameters, GFLOPs at the training resolution, and latency/FPS on a T4 at batch 1 (FP16 and FP32, stated).

Report test1 and test2 separately; their mean is optional. Final results: mean ± std over 3 seeds (e.g., 0, 1, 2) for the main models.

## 7. Pipelines to build

All models consume the same manifests. Converters write YOLO-format and COCO-format views of the data into `data/processed/` (gitignored). Use symlinks or lossless PNG copies; pixel data must stay identical to the BMPs.

Every pipeline below is from a 2025/2026 A/A\* conference (rule 5), and none is RT-DETR (rule 6). Verify each install on Kaggle with a `--smoke` run before committing to it.

### 7.1 Main track — transformer: RF-DETR (ICLR 2026)
- **RF-DETR** (Robinson et al., Roboflow + CMU, ICLR 2026). Code: `roboflow/rf-detr`, `pip install rfdetr`, Apache-2.0, COCO-format data (check the docs for the expected folder layout).
- Why it is our main pipeline: it is a DETR built on a DINOv2 backbone with weight-sharing architecture search (LW-DETR lineage), so it is clearly separate from the classmate's RT-DETR work. It is designed to transfer to new domains, which suits sonar.
- Start with the N and S sizes at the default resolution, then raise the resolution once the image-size analysis is done.

### 7.2 Tiny-object track (where our contribution should go)
- **Dome-DETR** (Hu et al., ACM MM 2025). Code: `RicePasteM/Dome-DETR`. Density-oriented feature and query manipulation for tiny objects, evaluated on AI-TOD-V2 and VisDrone. Check which base detector its code builds on; if it is built on the RT-DETR family, tell the student so they can confirm there is no overlap with the classmate.
- **FBRT-YOLO** (Xiao et al., AAAI 2025). Code: `galaxy-oss/FCM`. Real-time detector with modules aimed at small targets in aerial images (VisDrone, UAVDT, AI-TOD). Serves as the CNN-side small-object baseline.
- Contribution ideas to test on top of RF-DETR once baselines and error analysis are done: high-resolution (stride-4/8) feature injection, small-object-aware query selection, a tiny-aware matching cost or loss, and sonar-specific augmentation. Every idea borrowed from a paper must come from an eligible venue (rule 5).

### 7.3 Comparison baselines (same split, same evaluator)
- **YOLOv12** (Tian et al., NeurIPS 2025). Code: `sunsmarterjie/yolov12` (also available in Ultralytics). Attention-centric YOLO; our main CNN baseline.
- **D-FINE** (Peng et al., ICLR 2025, `Peterande/D-FINE`) and **DEIM** (Huang et al., CVPR 2025, `Intellindust-AI-Lab/DEIM`): strong real-time DETR comparisons. Both descend from the RT-DETR codebase. Use them only as comparison rows, not as our main pipeline, and mention the relationship to the student.
- YOLO26n from Section 5: reference row only.
- RT-DETR: classmate's numbers only, if they agree and use the same split.

### 7.4 Eligibility register (keep up to date)

| method | venue | year | code | role |
|---|---|---|---|---|
| RF-DETR | ICLR (A\*) | 2026 | roboflow/rf-detr | main pipeline |
| Dome-DETR | ACM MM (A\*) | 2025 | RicePasteM/Dome-DETR | tiny-object |
| FBRT-YOLO | AAAI (A\*) | 2025 | galaxy-oss/FCM | tiny-object (CNN) |
| YOLOv12 | NeurIPS (A\*) | 2025 | sunsmarterjie/yolov12 | baseline |
| D-FINE | ICLR (A\*) | 2025 | Peterande/D-FINE | baseline |
| DEIM | CVPR (A\*) | 2025 | Intellindust-AI-Lab/DEIM | baseline |
| YOLO26 | none (Ultralytics release) | — | ultralytics | reference only |
| RT-DETR / v2 | — | — | — | not used (classmate) |

Not eligible as pipelines (arXiv only at last check): DEIMv2, TinyFormer, DFIR-DETR. Re-check if they get published.

Sonar papers on UATD, for related work (check the quartile/rank and the split before citing): FS2-DETR (MDPI *JMSE*, 2026), RSOD (AAAI 2026, semi-supervised). MambaDSF (arXiv 2026) and MLFANet (*Frontiers in Marine Science* 2025) are background only.

## 8. Proposed repository layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml            # package `uatd`, pinned deps per track (extras: rfdetr, dome, yolo, deim)
├── configs/
│   ├── data/uatd.yaml        # class names, raw root, processed root
│   ├── eval.yaml             # frozen size thresholds
│   ├── rfdetr/ dome/ yolo/ deim/   # one file per experiment
├── data/
│   ├── splits/               # COMMITTED manifests + expected_counts.json
│   ├── raw/  processed/      # gitignored
├── src/uatd/
│   ├── splits.py             # builds manifests (Section 4.1)
│   ├── convert.py            # XML -> YOLO txt + COCO json from manifests (Section 4.2)
│   ├── stats.py              # image-size / box-size / class stats, plots
│   ├── eval_coco.py          # shared evaluator (Section 6)
│   ├── efficiency.py         # params, GFLOPs, latency
│   └── viz.py                # GT vs prediction figures
├── scripts/                  # train_yolo.py, train_rfdetr.py, train_deim.sh, evaluate.py
├── notebooks/kaggle_runner.ipynb   # thin: clone repo, pip install -e ., call scripts
├── tests/                    # test_splits.py, test_convert.py, test_eval.py
├── results/                  # small CSV/JSON summaries (committed)
└── runs/                     # checkpoints and logs (gitignored)
```

## 9. Compute and environment

- Development happens locally in VS Code. The local machine may have no CUDA GPU, so every script must support `--device cpu --smoke` (a few images, 1 epoch) for quick checks.
- Full training runs on **Kaggle (Tesla T4, 16 GB)** or Colab. Kaggle allows roughly 30 GPU hours per week and 12 hours per session (check the current quota). Always enable checkpoint saving and `resume`.
- The notebook should stay thin: clone the repo, install it, run scripts. No logic lives only in a notebook.
- Kaggle disk reads of the BMPs were slow ("slow image access" warning). Copy the data to `/kaggle/working` or convert it to lossless PNG once per session.
- Set an explicit batch size that fits a T4 (start with 32 for YOLO26n at 640; lower for DETRs or higher resolutions). Use AMP.
- Plotting on headless machines needs `MPLBACKEND=Agg`.
- Pin package versions per track in `pyproject.toml` extras. Ultralytics, rfdetr and DEIM may need different torch versions; use separate environments if they clash.

## 10. Roadmap (do in order; tick off in README)

1. **Scaffold:** repo layout, `pyproject.toml`, `.gitignore`, pre-commit (ruff).
2. **Split and convert:** `splits.py` → committed manifests; `convert.py` → YOLO + COCO views; tests pass with the exact numbers in 4.3.
3. **Dataset analysis:** native image sizes, box width/height/area histograms per class and split, objects per image. Freeze the tiny threshold in `configs/eval.yaml`.
4. **Evaluator:** `eval_coco.py` plus efficiency tooling. If the old `best.pt` is available, re-score it to check that we get ≈ 0.777 / 0.804 mAP50 on test1/test2.
5. **Main pipeline:** RF-DETR N/S on our split, first at default resolution, then higher.
6. **Baselines:** YOLOv12 n/s, FBRT-YOLO, DEIM-D-FINE-S, then Dome-DETR. Same split, same evaluator.
7. **Error analysis:** confusion between cage classes and cube, the mAP50 vs mAP75 gap, recall on tiny boxes, qualitative failure gallery.
8. **Method:** pick one tiny-object idea from an eligible venue, implement it on RF-DETR, run ablations.
9. **Final runs:** 3 seeds for the main models, efficiency table, figures, `results/summary.csv`.

## 11. Conventions

- Python ≥ 3.10, type hints, `pathlib`, no hard-coded absolute paths (use config files and CLI flags).
- Every training script writes `runs/<track>/<exp_name>/` containing the resolved config, git hash, seed, environment info, metrics JSON and best weights.
- Results tables are generated from run JSONs by a script, never typed by hand.
- Git: small commits on feature branches. Commit only when the student asks. Never commit data, weights or secrets.
- When unsure about a paper's claim, venue or code, say so and check it; do not guess numbers.
