"""Precision, recall and F1 exactly as Ultralytics reports them (CLAUDE.md, Section 5 baseline).

The YOLO26n baseline's P and R come from Ultralytics' validator (8.4.215). To compare
every model on the same footing, this module re-implements that definition and applies it
to any model's predictions:

1. Matching at IoU 0.5 (``BaseValidator.match_predictions``): per image, detections are taken
   in descending confidence order; each claims the unclaimed ground-truth box of the same
   class with the highest IoU, and is a true positive if that IoU >= 0.5.
2. Curves (``ap_per_class``): for every class present in the ground truth, precision and
   recall as a function of confidence, interpolated on 1000 points in [0, 1].
3. Operating point: F1 averaged over classes, smoothed with a box filter of width 0.1;
   P, R and F1 are read at its maximum and averaged over classes.

At most ``max_det`` (300, Ultralytics' default) detections per image are used.
Cross-checked against Ultralytics in ``tests/test_eval.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-16  # ap_per_class
IOU_EPS = 1e-7  # box_iou


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes, shape [len(a), len(b)], computed in float32 like Ultralytics."""
    a = a.astype(np.float32)[:, None, :]
    b = b.astype(np.float32)[None, :, :]
    wh = np.clip(np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + np.float32(IOU_EPS))


def match_image(
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    iou_thr: float = 0.5,
) -> np.ndarray:
    """True-positive flags for one image; predictions must be sorted by descending score."""
    correct = np.zeros(len(pred_xyxy), dtype=bool)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return correct
    iou = box_iou(gt_xyxy, pred_xyxy) * (gt_cls[:, None] == pred_cls[None, :])  # [G, D]
    matched = np.zeros(len(gt_xyxy), dtype=bool)
    for j in np.flatnonzero((iou >= iou_thr).any(0)):
        available = np.where(matched, 0, iou[:, j])
        k = int(available.argmax())
        if available[k] >= iou_thr:
            correct[j] = True
            matched[k] = True
    return correct


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def pr_at_max_f1(
    tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray
) -> dict:
    """Ultralytics ``ap_per_class`` P/R/F1 part (IoU 0.5 column only)."""
    i = np.argsort(-conf, kind="stable")
    tp, conf, pred_cls = tp[i].astype(float), conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    p_curve, r_curve = np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        sel = pred_cls == c
        n_l, n_p = nt[ci], sel.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (n_l + EPS)
        r_curve[ci] = np.interp(-x, -conf[sel], recall, left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[sel], precision, left=1)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
    k = int(smooth(f1_curve.mean(0), 0.1).argmax()) if nc else 0
    p, r, f1 = p_curve[:, k], r_curve[:, k], f1_curve[:, k]
    return {
        "P": float(p.mean()) if nc else 0.0,
        "R": float(r.mean()) if nc else 0.0,
        "F1": float(f1.mean()) if nc else 0.0,
        "conf_at_max_F1": float(x[k]),
        "per_class": {
            int(c): {"P": float(p[ci]), "R": float(r[ci]), "F1": float(f1[ci])}
            for ci, c in enumerate(unique_classes)
        },
    }


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def precision_recall_f1(
    gt_coco: dict, predictions: list[dict], iou_thr: float = 0.5, max_det: int = 300
) -> dict:
    """P/R/F1 for COCO-format ground truth and predictions (our category IDs)."""
    gts: dict[int, list] = defaultdict(list)
    for a in gt_coco["annotations"]:
        if not a.get("iscrowd", 0):
            gts[a["image_id"]].append((a["category_id"], _xywh_to_xyxy(a["bbox"])))
    preds: dict[int, list] = defaultdict(list)
    for p in predictions:
        preds[p["image_id"]].append((p["score"], p["category_id"], _xywh_to_xyxy(p["bbox"])))

    tp_all, conf_all, cls_all, target_cls = [], [], [], []
    for img in gt_coco["images"]:
        g = gts.get(img["id"], [])
        target_cls += [c for c, _ in g]
        d = sorted(preds.get(img["id"], []), key=lambda t: -t[0])[:max_det]
        if not d:
            continue
        d_xyxy = np.array([t[2] for t in d], dtype=float)
        d_cls = np.array([t[1] for t in d])
        g_xyxy = np.array([b for _, b in g], dtype=float).reshape(-1, 4)
        g_cls = np.array([c for c, _ in g])
        tp_all.append(match_image(d_xyxy, d_cls, g_xyxy, g_cls, iou_thr))
        conf_all.append(np.array([t[0] for t in d], dtype=float))
        cls_all.append(d_cls)
    if not tp_all:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "conf_at_max_F1": 0.0, "per_class": {}}
    return pr_at_max_f1(
        np.concatenate(tp_all),
        np.concatenate(conf_all),
        np.concatenate(cls_all),
        np.array(target_cls),
    )
