"""Figures and LaTeX tables for the RF-DETR project-update report (report/).

Every number comes from files written by the pipeline: the shared evaluator's
``metrics/{split}.json``, the predictions, the RF-DETR training log (``metrics.csv``,
``training_config.json``) and the processed COCO ground truth. Nothing is typed by hand
except the YOLO26n reference values, which are copied from the original notebook report
(Ultralytics validator, not our evaluator) and labelled as such.

    python scripts/make_report_assets.py \\
        --results D:/UATD/kaggle_out/rfdetr/results_small_rdefault_s0 \\
        --train-dir D:/UATD/kaggle_out/rfdetr/runs/rfdetr/small_rdefault_s0 \\
        --processed D:/UATD/processed --out report
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from uatd.constants import CLASS_NAMES  # noqa: E402
from uatd.pr_f1 import box_iou  # noqa: E402

SPLITS = ("val", "test1", "test2")
TEST_SPLITS = ("test1", "test2")

# YOLO26n reference (CLAUDE.md Section 5; original notebook, Ultralytics 8.4.215 validator).
YOLO26N = {
    "overall": {
        "val": dict(P=0.9675, R=0.9569, F1=0.9622, mAP50=0.9659, mAP75=0.5951, mAP50_95=0.5612),
        "test1": dict(P=0.8476, R=0.8201, F1=0.8336, mAP50=0.7766, mAP75=0.2077, mAP50_95=0.3292),
        "test2": dict(P=0.8521, R=0.8357, F1=0.8438, mAP50=0.8041, mAP75=0.2041, mAP50_95=0.3367),
    },
    "AP50": {
        "test1": [0.7851, 0.5118, 0.7075, 0.9909, 0.7687, 0.7233, 0.9913, 0.9727, 0.5092, 0.8051],
        "test2": [0.7645, 0.5169, 0.6986, 0.8880, 0.8603, 0.9950, 0.9949, 0.9570, 0.5215, 0.8446],
    },
    "AP50_95": {
        "test1": [0.4069, 0.1999, 0.2430, 0.4035, 0.3175, 0.2218, 0.6112, 0.4427, 0.1420, 0.3039],
        "test2": [0.3868, 0.2074, 0.2442, 0.3289, 0.3273, 0.2895, 0.6535, 0.4337, 0.1425, 0.3535],
    },
    "params_M": 2.51,
    "gflops": 5.9,
    "imgsz": 640,
}

C_RF = "#1f6f8b"
C_YOLO = "#c8763a"
SPLIT_COLORS = {"val": "#7a8b99", "test1": "#1f6f8b", "test2": "#3a9c6b"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# ----------------------------------------------------------------------------- helpers
def f3(x: float) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_table(
    path: Path, colspec: str, header: list[str], rows: list[list[str]], *, mid=(), pre: str = ""
):
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if pre:
        lines.append(pre)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in mid:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def bold_best(vals: list[float]) -> list[str]:
    best = max(v for v in vals if not np.isnan(v))
    return [rf"\textbf{{{f3(v)}}}" if v == best else f3(v) for v in vals]


def load_preds(results: Path, split: str) -> list[dict]:
    return json.loads((results / "predictions" / f"{split}.json").read_text())


# ----------------------------------------------------------------------------- figures
def fig_training(log_rows: list[dict], out: Path) -> dict:
    by_ep: dict[int, dict] = defaultdict(dict)
    for r in log_rows:
        ep = int(r["epoch"])
        for k, v in r.items():
            if v not in ("", None) and k not in ("epoch", "step"):
                by_ep[ep][k] = float(v)
    eps = sorted(by_ep)
    get = lambda k: np.array([by_ep[e].get(k, np.nan) for e in eps])  # noqa: E731
    x = np.array(eps) + 1  # 1-based epochs in the report
    m5095 = get("val/mAP_50_95")
    best_i = int(np.nanargmax(m5095))

    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    ax[0].plot(x, get("train/loss"), color=C_RF, label="total")
    ax[0].set_title("Training loss (epoch mean)")
    ax[0].set_xlabel("epoch")
    ax2 = ax[1]
    for k, lab, c in (
        ("train/loss_ce", "classification", "#8e5ea2"),
        ("train/loss_bbox", "L1 box", "#3a9c6b"),
        ("train/loss_giou", "GIoU", "#c8763a"),
    ):
        ax2.plot(x, get(k), label=lab, color=c)
    ax2.set_title("Loss components (final decoder layer)")
    ax2.set_xlabel("epoch")
    ax2.legend(frameon=False, fontsize=7)
    for k, lab, c, ls in (
        ("val/mAP_50", "mAP50", C_RF, "-"),
        ("val/mAP_75", "mAP75", "#8e5ea2", "-"),
        ("val/mAP_50_95", "mAP50:95", "#3a9c6b", "-"),
        ("val/ema_mAP_50_95", "mAP50:95 (EMA)", "#3a9c6b", ":"),
    ):
        ax[2].plot(x, get(k), label=lab, color=c, ls=ls)
    ax[2].axvline(x[best_i], color="k", lw=0.8, ls="--")
    ax[2].annotate(
        f"best: epoch {x[best_i]}",
        (x[best_i], 0.05),
        xytext=(-4, 0),
        textcoords="offset points",
        fontsize=7,
        ha="right",
    )
    ax[2].set_ylim(0, 1.02)
    ax[2].set_title("Validation (RF-DETR built-in evaluator)")
    ax[2].set_xlabel("epoch")
    ax[2].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout()
    fig.savefig(out / "fig_training_curves.png")
    plt.close(fig)

    epoch_table = [
        [
            str(x[i]),
            f3(by_ep[e].get("train/loss", np.nan)),
            f3(by_ep[e].get("train/loss_ce", np.nan)),
            f"{by_ep[e].get('train/loss_bbox', np.nan):.4f}",
            f3(by_ep[e].get("train/loss_giou", np.nan)),
            f3(by_ep[e].get("val/precision", np.nan)),
            f3(by_ep[e].get("val/recall", np.nan)),
            f3(by_ep[e].get("val/mAP_50", np.nan)),
            f3(by_ep[e].get("val/mAP_75", np.nan)),
            f3(by_ep[e].get("val/mAP_50_95", np.nan)),
        ]
        for i, e in enumerate(eps)
    ]
    return {
        "n_epochs": len(eps),
        "best_epoch": int(x[best_i]),
        "best_val_map5095_builtin": float(m5095[best_i]),
        "epoch_table": epoch_table,
    }


def fig_split_bars(metrics: dict, out: Path):
    keys = [
        ("P", "P"),
        ("R", "R"),
        ("F1", "F1"),
        ("mAP50", "mAP50"),
        ("mAP75", "mAP75"),
        ("mAP50_95", "mAP50:95"),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    w = 0.26
    xs = np.arange(len(keys))
    for i, s in enumerate(SPLITS):
        vals = [metrics[s][k] for k, _ in keys]
        b = ax.bar(xs + (i - 1) * w, vals, w, label=s, color=SPLIT_COLORS[s])
        ax.bar_label(b, fmt="%.3f", fontsize=5.5, padding=1)
    ax.set_xticks(xs, [lab for _, lab in keys])
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("RF-DETR-S on the three evaluation splits (shared evaluator)")
    fig.savefig(out / "fig_split_comparison.png")
    plt.close(fig)


def fig_vs_yolo(metrics: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    y = np.arange(len(CLASS_NAMES))
    for ax, s in zip(axes, TEST_SPLITS, strict=True):
        rf = [metrics[s]["per_class"][c]["AP50"] for c in CLASS_NAMES]
        yo = YOLO26N["AP50"][s]
        ax.barh(y + 0.2, rf, 0.4, color=C_RF, label="RF-DETR-S (ours)")
        ax.barh(y - 0.2, yo, 0.4, color=C_YOLO, label="YOLO26n (reference)")
        ax.set_yticks(y, CLASS_NAMES)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{s}: per-class AP50")
        ax.set_xlabel("AP50")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_per_class_vs_yolo.png")
    plt.close(fig)


def fig_size_bins(metrics: dict, out: Path):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    w = 0.35
    xs = np.arange(len(SPLITS))
    for i, (b, c) in enumerate((("tiny", "#c8763a"), ("non_tiny", C_RF))):
        vals = [metrics[s]["size_bins"][b]["AP50_95"] for s in SPLITS]
        bars = ax.bar(
            xs + (i - 0.5) * w,
            vals,
            w,
            color=c,
            label="tiny (area $\\leq$ 2300 px$^2$)" if b == "tiny" else "non-tiny",
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)
    ax.set_xticks(xs, SPLITS)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("AP50:95")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("AP on UATD-relative size bins")
    fig.savefig(out / "fig_size_bins.png")
    plt.close(fig)


def coco_eval(gt_path: Path, preds: list[dict]) -> COCOeval:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
    return ev


def fig_pr_curves(processed: Path, results: Path, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    cmap = plt.get_cmap("tab10")
    rec = np.linspace(0, 1, 101)
    for ax, s in zip(axes, TEST_SPLITS, strict=True):
        ev = coco_eval(processed / "coco" / f"{s}.json", load_preds(results, s))
        prec = ev.eval["precision"][0, :, :, 0, 2]  # IoU 0.5, all areas, maxDets 100
        for k, name in enumerate(CLASS_NAMES):
            p = prec[:, k]
            if (p > -1).any():
                ax.plot(
                    rec,
                    np.where(p > -1, p, np.nan),
                    color=cmap(k),
                    lw=1,
                    label=f"{name} {np.mean(p[p > -1]):.3f}",
                )
        mean = np.where(prec > -1, prec, np.nan)
        ax.plot(
            rec, np.nanmean(mean, 1), color="k", lw=2, label=f"all classes {np.nanmean(mean):.3f}"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall")
        ax.set_title(f"{s}: precision-recall at IoU 0.5")
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    axes[0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(out / "fig_pr_curves.png")
    plt.close(fig)


def confusion(gt: dict, preds: list[dict], thr: float, iou_thr: float = 0.5) -> np.ndarray:
    """Rows = predicted class (last = background), columns = true class (last = background).

    Detections with score >= thr; class-agnostic greedy one-to-one matching at IoU >= 0.5.
    """
    n = len(CLASS_NAMES)
    m = np.zeros((n + 1, n + 1), dtype=int)
    g_by: dict[int, list] = defaultdict(list)
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        g_by[a["image_id"]].append((a["category_id"], [x, y, x + w, y + h]))
    d_by: dict[int, list] = defaultdict(list)
    for p in preds:
        if p["score"] >= thr:
            x, y, w, h = p["bbox"]
            d_by[p["image_id"]].append((p["category_id"], [x, y, x + w, y + h]))
    for img in gt["images"]:
        g, d = g_by.get(img["id"], []), d_by.get(img["id"], [])
        gm, dm = set(), set()
        if g and d:
            iou = box_iou(np.array([b for _, b in g]), np.array([b for _, b in d]))
            pairs = sorted(
                (
                    (iou[i, j], i, j)
                    for i in range(len(g))
                    for j in range(len(d))
                    if iou[i, j] >= iou_thr
                ),
                reverse=True,
            )
            for _, i, j in pairs:
                if i not in gm and j not in dm:
                    gm.add(i)
                    dm.add(j)
                    m[d[j][0], g[i][0]] += 1
        for i, (c, _) in enumerate(g):
            if i not in gm:
                m[n, c] += 1
        for j, (c, _) in enumerate(d):
            if j not in dm:
                m[c, n] += 1
    return m


def fig_confusion(processed: Path, results: Path, thr: float, out: Path) -> dict:
    labels = [*CLASS_NAMES, "background"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    summary = {}
    for ax, s in zip(axes, TEST_SPLITS, strict=True):
        gt = json.loads((processed / "coco" / f"{s}.json").read_text())
        m = confusion(gt, load_preds(results, s), thr)
        summary[s] = m.tolist()
        ax.imshow(np.log1p(m), cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(
                        j,
                        i,
                        m[i, j],
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if m[i, j] > m.max() * 0.4 else "black",
                    )
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("true class")
        ax.set_ylabel("predicted class")
        ax.grid(False)
        ax.set_title(f"{s} (score $\\geq$ {thr:.3f}, IoU $\\geq$ 0.5)")
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_test.png")
    plt.close(fig)
    return summary


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        with contextlib.suppress(OSError):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fig_gallery(processed: Path, results: Path, split: str, thr: float, out: Path, n: int = 4):
    gt = json.loads((processed / "coco" / f"{split}.json").read_text())
    preds = load_preds(results, split)
    imgs = sorted(gt["images"], key=lambda i: i["id"])
    pick = random.Random(42).sample(imgs, n)
    g_by, d_by = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        g_by[a["image_id"]].append(a)
    for p in preds:
        if p["score"] >= thr:
            d_by[p["image_id"]].append(p)
    H = 700
    font = _font(18)
    tiles = []
    for im in pick:
        src = Image.open(processed / "images" / im["file_name"]).convert("RGB")
        sc = H / src.height
        base = src.resize((round(src.width * sc), H))
        pair = []
        for kind in ("GT", "pred"):
            t = base.copy()
            dr = ImageDraw.Draw(t)
            items = g_by[im["id"]] if kind == "GT" else d_by[im["id"]]
            for a in items:
                x, y, w, h = (v * sc for v in a["bbox"])
                col = (60, 220, 60) if kind == "GT" else (255, 190, 0)
                dr.rectangle([x, y, x + w, y + h], outline=col, width=3)
                lab = CLASS_NAMES[a["category_id"]]
                if kind == "pred":
                    lab += f" {a['score']:.2f}"
                dr.text((x, max(0, y - 22)), lab, fill=col, font=font)
            dr.text(
                (8, 8),
                f"{im['uatd_id']}  {'ground truth' if kind == 'GT' else 'RF-DETR'}",
                fill=(255, 255, 255),
                font=font,
            )
            pair.append(t)
        w = sum(p.width for p in pair) + 6
        canvas = Image.new("RGB", (w, H), (255, 255, 255))
        canvas.paste(pair[0], (0, 0))
        canvas.paste(pair[1], (pair[0].width + 6, 0))
        tiles.append(canvas)
    gap = 20
    rows = [tiles[i : i + 2] for i in range(0, len(tiles), 2)]
    W = max(sum(t.width for t in row) + gap * (len(row) - 1) for row in rows)
    sheet = Image.new("RGB", (W, len(rows) * (H + gap)), (255, 255, 255))
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            sheet.paste(t, (x, r * (H + gap)))
            x += t.width + gap
    sheet.save(out / f"fig_gallery_{split}.jpg", quality=90)
    return [im["uatd_id"] for im in pick]


# ----------------------------------------------------------------------------- tables
def class_counts(processed: Path) -> dict:
    out = {}
    for s in ("train", *SPLITS):
        gt = json.loads((processed / "coco" / f"{s}.json").read_text())
        c = Counter(a["category_id"] for a in gt["annotations"])
        out[s] = {
            "images": len(gt["images"]),
            "objects": len(gt["annotations"]),
            "per_class": [c.get(k, 0) for k in range(len(CLASS_NAMES))],
        }
    return out


def tables(
    metrics: dict,
    counts: dict,
    train_cfg: dict,
    run_cfg: dict,
    sessions: list,
    train_info: dict,
    conf: dict,
    gal: dict,
    out: Path,
    gen: Path,
):
    # overall
    rows = []
    for s in SPLITS:
        m = metrics[s]
        rows.append([s, *(f3(m[k]) for k in ("P", "R", "F1", "mAP50", "mAP75", "mAP50_95"))])
    write_table(
        gen / "tab_overall.tex",
        "lrrrrrr",
        ["Split", "P", "R", "F1", "mAP50", "mAP75", "mAP50:95"],
        rows,
    )

    # headline
    write_table(
        gen / "tab_headline.tex",
        "ccc",
        ["Validation mAP50", "Test 1 mAP50", "Test 2 mAP50"],
        [[f"{100 * metrics[s]['mAP50']:.2f}\\%" for s in SPLITS]],
    )

    # comparison with YOLO26n
    rows = []
    for s in SPLITS:
        for name, src in (("YOLO26n$^\\dagger$", YOLO26N["overall"][s]), ("RF-DETR-S", metrics[s])):
            rows.append(
                [
                    s if name.startswith("YOLO") else "",
                    name,
                    *(f3(src[k]) for k in ("P", "R", "F1", "mAP50", "mAP75", "mAP50_95")),
                ]
            )
    write_table(
        gen / "tab_vs_yolo.tex",
        "llrrrrrr",
        ["Split", "Model", "P", "R", "F1", "mAP50", "mAP75", "mAP50:95"],
        rows,
        mid=(2, 4),
    )

    # per-class
    for s in SPLITS:
        pc = metrics[s]["per_class"]
        rows = [
            [
                c,
                str(pc[c]["num_gt"]),
                f3(pc[c]["P"]),
                f3(pc[c]["R"]),
                f3(pc[c]["F1"]),
                f3(pc[c]["AP50"]),
                f3(pc[c]["AP50_95"]),
            ]
            for c in CLASS_NAMES
        ]
        write_table(
            gen / f"tab_per_class_{s}.tex",
            "lrrrrrr",
            ["Class", "GT", "P", "R", "F1", "AP50", "AP50:95"],
            rows,
        )

    # per-class AP50 / AP50:95 vs YOLO on tests
    rows = []
    for k, c in enumerate(CLASS_NAMES):
        r = [c]
        for s in TEST_SPLITS:
            r += [
                f3(YOLO26N["AP50"][s][k]),
                f3(metrics[s]["per_class"][c]["AP50"]),
                f3(YOLO26N["AP50_95"][s][k]),
                f3(metrics[s]["per_class"][c]["AP50_95"]),
            ]
        rows.append(r)
    write_table(
        gen / "tab_per_class_vs_yolo.tex",
        "lrrrrrrrr",
        ["Class", *["YOLO26n", "RF-DETR"] * 4],
        rows,
        pre=r" & \multicolumn{2}{c}{Test 1 AP50} & \multicolumn{2}{c}{Test 1 AP50:95}"
        r" & \multicolumn{2}{c}{Test 2 AP50} & \multicolumn{2}{c}{Test 2 AP50:95} \\"
        "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
    )

    # COCO detail
    names = [
        ("mAP50_95", "AP", "0.50:0.95", "all"),
        ("mAP50", "AP", "0.50", "all"),
        ("mAP75", "AP", "0.75", "all"),
        ("AP_small", "AP", "0.50:0.95", "small"),
        ("AP_medium", "AP", "0.50:0.95", "medium"),
        ("AP_large", "AP", "0.50:0.95", "large"),
        ("AR1", "AR@1", "0.50:0.95", "all"),
        ("AR10", "AR@10", "0.50:0.95", "all"),
        ("AR100", "AR@100", "0.50:0.95", "all"),
        ("AR_small", "AR@100", "0.50:0.95", "small"),
        ("AR_medium", "AR@100", "0.50:0.95", "medium"),
        ("AR_large", "AR@100", "0.50:0.95", "large"),
    ]
    rows = [[a, b, c, *(f3(metrics[s][k]) for s in SPLITS)] for k, a, b, c in names]
    write_table(
        gen / "tab_coco_detail.tex",
        "lllrrr",
        ["Metric", "IoU", "Area", "Val", "Test 1", "Test 2"],
        rows,
    )

    # size bins
    rows = []
    for s in SPLITS:
        b = metrics[s]["size_bins"]
        rows.append(
            [
                s,
                str(b["tiny"]["num_gt"]),
                f3(b["tiny"]["AP50"]),
                f3(b["tiny"]["AP50_95"]),
                str(b["non_tiny"]["num_gt"]),
                f3(b["non_tiny"]["AP50"]),
                f3(b["non_tiny"]["AP50_95"]),
            ]
        )
    write_table(
        gen / "tab_size_bins.tex",
        "lrrrrrr",
        [
            "Split",
            "tiny GT",
            "tiny AP50",
            "tiny AP50:95",
            "non-tiny GT",
            "non-tiny AP50",
            "non-tiny AP50:95",
        ],
        rows,
    )

    # dataset
    rows = [
        [s, f"{counts[s]['images']:,}", f"{counts[s]['objects']:,}"] for s in ("train", *SPLITS)
    ]
    rows.append(
        [
            "Total",
            f"{sum(counts[s]['images'] for s in counts):,}",
            f"{sum(counts[s]['objects'] for s in counts):,}",
        ]
    )
    write_table(gen / "tab_dataset.tex", "lrr", ["Split", "Images", "Objects"], rows, mid=(4,))
    rows = [
        [c, *(str(counts[s]["per_class"][k]) for s in ("train", *SPLITS))]
        for k, c in enumerate(CLASS_NAMES)
    ]
    write_table(
        gen / "tab_class_counts.tex", "lrrrr", ["Class", "Train", "Val", "Test 1", "Test 2"], rows
    )

    # model / training config
    mc, tc = train_cfg["model_config"], train_cfg["train_config"]
    a = run_cfg["args"]
    rows = [
        ["Model", f"{run_cfg['model']} (pretrained: {Path(mc['pretrain_weights']).name})"],
        ["Encoder", tex_escape(f"{mc['encoder']} (DINOv2, windowed), patch {mc['patch_size']}")],
        [
            "Decoder",
            f"{mc['dec_layers']} layers, hidden {mc['hidden_dim']}, "
            f"{mc['num_queries']} queries, two-stage",
        ],
        ["Input resolution", f"{mc['resolution']} $\\times$ {mc['resolution']} (square resize)"],
        [
            "Epochs",
            f"{a['epochs']} max; early stopping patience {a['patience']} "
            f"(min delta {tc['early_stopping_min_delta']})",
        ],
        [
            "Batch",
            f"{a['batch']} $\\times$ grad. accumulation {a['grad_accum']} "
            f"(effective {a['batch'] * a['grad_accum']})",
        ],
        [
            "Optimizer",
            f"AdamW; lr {a['lr']:g}, encoder lr {a['lr_encoder']:g}, "
            f"weight decay {tc['weight_decay']:g}",
        ],
        [
            "LR decay",
            f"layer-wise ViT decay {tc['lr_vit_layer_decay']}, "
            f"component decay {tc['lr_component_decay']}",
        ],
        ["EMA", f"decay {tc['ema_decay']}, tau {tc['ema_tau']}"],
        [
            "Augmentation",
            "RF-DETR defaults: multi-scale (per batch), scale jitter, horizontal flip",
        ],
        ["Gradient clipping", f"max norm {tc['clip_max_norm']}"],
        ["Precision", "mixed (AMP, fp16 on T4)"],
        ["Seed", str(a["seed"])],
        ["Checkpoint selection", "best val mAP50:95 (RF-DETR built-in evaluator)"],
    ]
    write_table(gen / "tab_train_config.tex", "p{3.6cm}p{10.4cm}", ["Setting", "Value"], rows)

    s0 = sessions[0]
    env = s0["env"]
    rows = [
        ["Hardware", f"Kaggle, {env['gpu']} (1 GPU used)"],
        [
            "Software",
            tex_escape(
                f"Python {env['python']}; PyTorch {env['packages']['torch']}; "
                f"rfdetr {env['packages']['rfdetr']}; CUDA {env['cuda']}"
            ),
        ],
        ["Code version", f"git commit \\texttt{{{s0['git']['commit'][:7]}}} (clean)"],
        [
            "Training time",
            f"{sum(x['hours'] for x in sessions):.2f} h in {len(sessions)} session(s)",
        ],
        ["Epochs run", f"{train_info['n_epochs']} (early stopping)"],
        [
            "Best epoch",
            f"{train_info['best_epoch']} (val mAP50:95 "
            f"{train_info['best_val_map5095_builtin']:.3f}, built-in evaluator)",
        ],
        [
            "Prediction threshold",
            f"score $\\geq$ {metrics['val']['threshold']}, at most 300 detections per image",
        ],
    ]
    write_table(gen / "tab_environment.tex", "p{3.6cm}p{10.4cm}", ["Item", "Value"], rows)

    # epoch log (longtable body only)
    body = "\n".join(" & ".join(r) + r" \\" for r in train_info["epoch_table"])
    (gen / "tab_epoch_log_rows.tex").write_text(body + "\n", encoding="utf-8", newline="\n")

    # numbers for inline use
    facts = {
        "metrics": {
            s: {
                k: metrics[s][k]
                for k in ("P", "R", "F1", "mAP50", "mAP75", "mAP50_95", "conf_at_max_F1")
            }
            for s in SPLITS
        },
        "train": {k: v for k, v in train_info.items() if k != "epoch_table"},
        "confusion_threshold": conf["thr"],
        "confusion": conf["matrices"],
        "gallery_images": gal,
    }
    (out / "generated" / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("report"))
    args = ap.parse_args(argv)

    fig_dir, gen = args.out / "figures", args.out / "generated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gen.mkdir(parents=True, exist_ok=True)

    metrics = {s: json.loads((args.results / "metrics" / f"{s}.json").read_text()) for s in SPLITS}
    run_cfg = json.loads((args.results / "run_config.json").read_text())
    sessions = json.loads((args.results / "sessions.json").read_text())
    train_cfg = json.loads((args.train_dir / "training_config.json").read_text())
    with open(args.train_dir / "metrics.csv", newline="") as f:
        log_rows = list(csv.DictReader(f))

    # operating point chosen on val only (never on test)
    thr = metrics["val"]["conf_at_max_F1"]

    print("training curves ...")
    train_info = fig_training(log_rows, fig_dir)
    print("bar charts ...")
    fig_split_bars(metrics, fig_dir)
    fig_vs_yolo(metrics, fig_dir)
    fig_size_bins(metrics, fig_dir)
    print("PR curves ...")
    fig_pr_curves(args.processed, args.results, fig_dir)
    print("confusion matrices ...")
    mats = fig_confusion(args.processed, args.results, thr, fig_dir)
    print("galleries ...")
    gal = {s: fig_gallery(args.processed, args.results, s, thr, fig_dir) for s in TEST_SPLITS}
    print("tables ...")
    counts = class_counts(args.processed)
    tables(
        metrics,
        counts,
        train_cfg,
        run_cfg,
        sessions,
        train_info,
        {"thr": thr, "matrices": mats},
        gal,
        args.out,
        gen,
    )
    print("done ->", args.out)


if __name__ == "__main__":
    main()
