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
