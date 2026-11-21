"""Build the folder layout RF-DETR expects from our processed COCO view.

RF-DETR (``dataset_file="roboflow"``) reads::

    <dataset_dir>/train/_annotations.coco.json  + images
    <dataset_dir>/valid/_annotations.coco.json  + images
    <dataset_dir>/test/_annotations.coco.json   + images

We map train -> train, val -> valid, test1 -> test. RF-DETR selects its best
checkpoint on ``valid`` (our val split), never on test (CLAUDE.md rule 3).
``run_test`` stays False; test1/test2 are scored only by our own evaluator.

Images are hard-linked from ``<processed>/images`` (no extra disk), with a copy
fallback. COCO image IDs, category IDs (0-9) and ``uatd_id`` are kept unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

RF_SPLITS: dict[str, str] = {"train": "train", "valid": "val", "test": "test1"}


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_rfdetr_dataset(processed: Path, out: Path | None = None) -> Path:
    """Create (or refresh) the RF-DETR dataset folder and return its path."""
    out = out or processed / "rfdetr"
    for rf_split, split in RF_SPLITS.items():
        coco = json.loads((processed / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        d = out / rf_split
        d.mkdir(parents=True, exist_ok=True)
        for img in coco["images"]:
            src = processed / "images" / img["file_name"]
            name = Path(img["file_name"]).name
            _link(src, d / name)
            img["file_name"] = name
        (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return out
