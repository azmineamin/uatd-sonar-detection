"""Shared evaluator sanity checks on a tiny hand-made COCO set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from uatd.eval_coco import evaluate


@pytest.fixture()
def gt(tmp_path: Path) -> Path:
    data = {
        "images": [{"id": i, "file_name": f"{i}.png", "width": 100, "height": 100} for i in (1, 2)],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 40, 40],
                "area": 1600,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 0, "name": "ball"}, {"id": 1, "name": "circle cage"}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(data))
    return p


def test_perfect_predictions_score_one(gt: Path):
    preds = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [50, 50, 40, 40], "score": 0.8},
    ]
    m = evaluate(gt, preds, size_bins={"tiny": [0, 500], "rest": [500, 1e10]})
    assert m["mAP50"] == pytest.approx(1.0) and m["mAP50_95"] == pytest.approx(1.0)
    assert m["per_class"]["ball"]["AP50"] == pytest.approx(1.0)
    assert m["size_bins"]["tiny"]["num_gt"] == 1
    assert m["size_bins"]["tiny"]["AP50"] == pytest.approx(1.0)


def test_wrong_class_scores_zero_for_that_class(gt: Path):
    preds = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 2, "category_id": 0, "bbox": [50, 50, 40, 40], "score": 0.8},
    ]
    m = evaluate(gt, preds)
    assert m["per_class"]["ball"]["AP50"] == pytest.approx(1.0)
    assert m["per_class"]["circle cage"]["AP50"] == pytest.approx(0.0)
    assert m["mAP50"] == pytest.approx(0.5)


def test_half_iou_box_counts_at_50_not_75(gt: Path):
    # 20x20 GT vs 20x30 pred sharing the top: IoU = 400/600 = 0.667
    preds = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 30], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [50, 50, 40, 40], "score": 0.8},
    ]
    m = evaluate(gt, preds)
    assert m["per_class"]["ball"]["AP50"] == pytest.approx(1.0)
    assert m["mAP75"] == pytest.approx(0.5)


def test_empty_and_unknown_image(gt: Path):
    assert evaluate(gt, [])["mAP50"] == 0.0
    with pytest.raises(ValueError):
        evaluate(gt, [{"image_id": 99, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1.0}])
    assert not math.isnan(evaluate(gt, [])["mAP50_95"])


def _random_case(seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    images, annotations, preds = [], [], []
    for img_id in range(1, 41):
        images.append({"id": img_id, "file_name": f"{img_id}.png", "width": 640, "height": 640})
        for _ in range(rng.integers(0, 4)):
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(10, 120, 2)
            c = int(rng.integers(0, 4))
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": img_id,
                    "category_id": c,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            if rng.random() < 0.85:  # detected, with jitter and sometimes the wrong class
                j = rng.normal(0, 0.15 * min(w, h), 4)
                cls = c if rng.random() < 0.85 else int(rng.integers(0, 4))
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x + j[0], y + j[1], max(1, w + j[2]), max(1, h + j[3])],
                        "score": float(rng.uniform(0.2, 1.0)),
                    }
                )
        for _ in range(rng.integers(0, 4)):  # false positives
            x, y = rng.uniform(0, 500, 2)
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": int(rng.integers(0, 4)),
                    "bbox": [x, y, *rng.uniform(10, 120, 2)],
                    "score": float(rng.uniform(0, 0.7)),
                }
            )
    cats = [{"id": i, "name": f"c{i}"} for i in range(4)]
    return {"images": images, "annotations": annotations, "categories": cats}, preds


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pr_f1_matches_ultralytics(seed: int):
    pytest.importorskip("ultralytics")
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    from uatd.pr_f1 import precision_recall_f1

    gt, preds = _random_case(seed)
    ours = precision_recall_f1(gt, preds)

    v = DetectionValidator()
    v.iouv = torch.linspace(0.5, 0.95, 10)
    tps, confs, clss, tcls = [], [], [], []
    for img in gt["images"]:
        g = [a for a in gt["annotations"] if a["image_id"] == img["id"]]
        d = sorted((p for p in preds if p["image_id"] == img["id"]), key=lambda p: -p["score"])
        tcls += [a["category_id"] for a in g]
        if not d:
            continue
        xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
        gb = torch.tensor([xyxy(a["bbox"]) for a in g], dtype=torch.float32).reshape(-1, 4)
        db = torch.tensor([xyxy(p["bbox"]) for p in d], dtype=torch.float32)
        gc = torch.tensor([a["category_id"] for a in g])
        dc = torch.tensor([p["category_id"] for p in d])
        tps.append(v.match_predictions(dc, gc, box_iou(gb, db)).numpy())
        confs.append(np.array([p["score"] for p in d]))
        clss.append(dc.numpy())
    _, _, p, r, f1, *_ = ap_per_class(
        np.concatenate(tps), np.concatenate(confs), np.concatenate(clss), np.array(tcls)
    )
    assert ours["P"] == pytest.approx(p.mean(), abs=1e-9)
    assert ours["R"] == pytest.approx(r.mean(), abs=1e-9)
    assert ours["F1"] == pytest.approx(f1.mean(), abs=1e-9)
    assert 0.3 < ours["P"] < 1.0 and 0.3 < ours["R"] < 1.0  # a non-trivial case
