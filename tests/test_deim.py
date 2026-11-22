"""DEIM-D-FINE wiring: config rendering, schedule, log parsing."""

from __future__ import annotations

import json
from pathlib import Path

from uatd.deim_support import epochs_logged, render_config

TEMPLATE = Path("configs/deim/uatd_deim_dfine_s.yml")


def test_template_paths_and_settings(tmp_path: Path):
    cfg = render_config(TEMPLATE, tmp_path / "DEIM", tmp_path / "data", tmp_path / "run", False)
    assert cfg["__include__"] == [
        (tmp_path / "DEIM").resolve().as_posix() + "/configs/deim_dfine/deim_hgnetv2_s_coco.yml"
    ]
    assert "${" not in json.dumps(cfg)
    tr, va = cfg["train_dataloader"], cfg["val_dataloader"]
    assert tr["dataset"]["ann_file"].endswith("/coco/train.json")
    assert va["dataset"]["ann_file"].endswith("/coco/val.json")
    files = json.dumps(cfg).replace(tmp_path.resolve().as_posix(), "")
    assert "test1" not in files and "test2" not in files  # never in training (rule 3)
    assert cfg["num_classes"] == 10 and cfg["remap_mscoco_category"] is False
    assert tr["total_batch_size"] == 16  # same effective batch as RF-DETR
    # the authors' 60-epoch recipe, internally consistent
    assert cfg["epoches"] == 60
    assert tr["collate_fn"]["stop_epoch"] == tr["dataset"]["transforms"]["policy"]["epoch"][-1]
    assert cfg["epoches"] - tr["collate_fn"]["stop_epoch"] == cfg["no_aug_epoch"]
    # linear batch scaling from the official S config (batch 32, lr 4e-4, backbone 2e-4)
    assert cfg["optimizer"]["lr"] == 0.0002
    assert cfg["optimizer"]["params"][0]["lr"] == 0.0001
    # EMA horizon ~2.6 UATD epochs (427 iters/epoch), like 2.7 COCO epochs officially
    horizon_epochs = 1 / (1 - cfg["ema"]["decay"]) / (6835 // 16)
    assert 2.0 < horizon_epochs < 3.5
    assert cfg["train_dataloader"]["collate_fn"]["ema_restart_decay"] == cfg["ema"]["decay"]


def test_smoke_visits_every_phase(tmp_path: Path):
    cfg = render_config(TEMPLATE, tmp_path, tmp_path, tmp_path, True)
    policy = cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"]
    stop = cfg["train_dataloader"]["collate_fn"]["stop_epoch"]
    assert policy == [1, 2, 3] and stop == 3 and cfg["epoches"] == 4
    # smoke overrides only touch the schedule and batch, not the data paths
    assert cfg["train_dataloader"]["dataset"]["ann_file"].endswith("/coco/train.json")


def test_epochs_logged(tmp_path: Path):
    assert epochs_logged(tmp_path) == []
    (tmp_path / "log.txt").write_text(
        '{"epoch": 0, "test_coco_eval_bbox": [0.1]}\n{"epoch": 1}\n', encoding="utf-8"
    )
    assert epochs_logged(tmp_path) == [0, 1]
