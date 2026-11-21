"""Shared COCO evaluator: every reported number comes from here (CLAUDE.md, Section 6).

    python -m uatd.eval_coco --gt data/processed/coco/test1.json \\
        --pred runs/rfdetr/<exp>/predictions/test1.json --out runs/rfdetr/<exp>/metrics/test1.json

Predictions: a COCO results list ``[{"image_id", "category_id", "bbox": [x, y, w, h],
"score"}]`` in original-image pixels, with our image IDs and category IDs 0-9.

Reports mAP@[.5:.95], mAP50, mAP75, AP_s/m/l, AR@1/10/100, per-class AP50 and
AP50:95, and AP on UATD-relative size bins when ``configs/eval.yaml`` defines them
(frozen in roadmap step 3). Also P, R and F1 with the Ultralytics definition used for
the CLAUDE.md Section 5 baseline (IoU 0.5, at the confidence of maximum mean F1; see
``uatd.pr_f1``), overall and per class.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from uatd.pr_f1 import precision_recall_f1

STAT_NAMES = (
    "mAP50_95",
    "mAP50",
    "mAP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)
DEFAULT_EVAL_CONFIG = Path("configs/eval.yaml")


def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def _run(gt: COCO, dt: COCO, area_rngs: list[list[float]] | None = None, labels=None) -> COCOeval:
    ev = COCOeval(gt, dt, iouType="bbox")
    if area_rngs is not None:
        ev.params.areaRng = area_rngs
        ev.params.areaRngLbl = labels
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        if area_rngs is None:
            ev.summarize()
    return ev


def load_size_bins(path: Path = DEFAULT_EVAL_CONFIG) -> dict[str, list[float]] | None:
    """``size_bins: {tiny: [0, A], ...}`` in pixel area, or None if not frozen yet."""
    if not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg.get("size_bins")


def evaluate(gt_path: Path, predictions: list[dict], size_bins: dict | None = None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
    img_ids = set(gt.getImgIds())
    preds = [p for p in predictions if p["image_id"] in img_ids]
    if len(preds) != len(predictions):
        raise ValueError(f"{len(predictions) - len(preds)} predictions have unknown image_id")
    result: dict = {
        "gt": str(gt_path),
        "num_images": len(img_ids),
        "num_gt": len(gt.getAnnIds()),
        "num_predictions": len(preds),
    }
    if not preds:
        result.update({k: 0.0 for k in STAT_NAMES})
        result.update(P=0.0, R=0.0, F1=0.0, conf_at_max_F1=0.0)
        return result

    prf = precision_recall_f1(gt.dataset, preds)
    result.update({k: prf[k] for k in ("P", "R", "F1", "conf_at_max_F1")})

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(preds))
    ev = _run(gt, dt)
    result.update({k: float(v) for k, v in zip(STAT_NAMES, ev.stats, strict=True)})

    # precision: [T iou, R recall, K class, A area, M maxdets]; area 0 = all, maxdets -1 = 100
    prec = ev.eval["precision"]
    iou50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        name = gt.cats[cat_id]["name"]
        per_class[name] = {
            "AP50": _mean_valid(prec[iou50, :, k, 0, -1]),
            "AP50_95": _mean_valid(prec[:, :, k, 0, -1]),
            "num_gt": len(gt.getAnnIds(catIds=[cat_id])),
            **{k: prf["per_class"].get(cat_id, {}).get(k, 0.0) for k in ("P", "R", "F1")},
        }
    result["per_class"] = per_class

    if size_bins:
        labels = ["all", *size_bins]
        rngs = [[0.0, 1e10], *[list(map(float, v)) for v in size_bins.values()]]
        ev_s = _run(gt, dt, rngs, labels)
        p = ev_s.eval["precision"]
        result["size_bins"] = {
            lbl: {
                "range": rngs[a],
                "AP50_95": _mean_valid(p[:, :, :, a, -1]),
                "AP50": _mean_valid(p[iou50, :, :, a, -1]),
                "num_gt": int(
                    sum(
                        1
                        for an in gt.dataset["annotations"]
                        if rngs[a][0] <= an["area"] <= rngs[a][1]  # inclusive, like pycocotools
                    )
                ),
            }
            for a, lbl in enumerate(labels)
            if a > 0
        }
    return result


def format_summary(m: dict) -> str:
    lines = [
        f"images {m['num_images']}  gt {m['num_gt']}  preds {m['num_predictions']}",
        "  ".join(
            f"{k} {m[k]:.4f}" for k in ("P", "R", "F1", "mAP50", "mAP75", "mAP50_95", "AR100")
        ),
    ]
    for name, v in m.get("per_class", {}).items():
        lines.append(
            f"  {name:13s} P {v['P']:.4f}  R {v['R']:.4f}  "
            f"AP50 {v['AP50']:.4f}  AP50:95 {v['AP50_95']:.4f}"
        )
    for name, v in m.get("size_bins", {}).items():
        lines.append(
            f"  [{name}] AP50 {v['AP50']:.4f}  AP50:95 {v['AP50_95']:.4f}  n={v['num_gt']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    args = ap.parse_args(argv)
    preds = json.loads(args.pred.read_text(encoding="utf-8"))
    m = evaluate(args.gt, preds, load_size_bins(args.eval_config))
    print(format_summary(m))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
