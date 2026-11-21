"""Dataset statistics and the frozen size-bin config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from uatd import stats
from uatd.eval_coco import evaluate, load_size_bins


def _coco(areas_side: list[int], width=100, height=200) -> dict:
    anns = [
        {
            "id": k + 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 0, s, s],
            "area": s * s,
            "iscrowd": 0,
        }
        for k, s in enumerate(areas_side)
    ]
    return {
        "images": [{"id": 1, "file_name": "1.png", "width": width, "height": height}],
        "annotations": anns,
        "categories": [{"id": 0, "name": "ball"}],
    }


def test_tiny_threshold_is_train_p25():
    coco = _coco([1, 2, 3, 4, 5])  # areas 1, 4, 9, 16, 25
    assert stats.tiny_threshold(coco) == pytest.approx(4.0)


def test_resized_box_size():
    s = stats.split_stats(_coco([10], width=100, height=200))
    r = s["after_square_resize"]["512"]
    assert r["w"]["p50"] == pytest.approx(10 * 512 / 100)
    assert r["h"]["p50"] == pytest.approx(10 * 512 / 200)


def test_eval_config_is_frozen_and_bins_are_disjoint(tmp_path: Path):
    cfg = tmp_path / "eval.yaml"
    st = stats.split_stats(_coco([1, 2, 3, 4, 5]))
    stats.write_eval_config(cfg, 16.0, st)
    bins = yaml.safe_load(cfg.read_text())["size_bins"]
    assert bins["tiny"] == [0.0, 16.0] and bins["non_tiny"][0] > 16.0
    with pytest.raises(FileExistsError):
        stats.write_eval_config(cfg, 9.0, st)

    # evaluator: a box exactly at the threshold is tiny, counted once
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(_coco([4, 5])))  # areas 16 (tiny) and 25 (non-tiny)
    preds = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 4, 4], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 5, 5], "score": 0.8},
    ]
    m = evaluate(gt, preds, load_size_bins(cfg))
    assert m["size_bins"]["tiny"]["num_gt"] == 1
    assert m["size_bins"]["non_tiny"]["num_gt"] == 1
