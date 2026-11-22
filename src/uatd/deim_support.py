"""Shared helpers for the DEIM-D-FINE comparison row (CVPR 2025 / ICLR 2025).

DEIM is not a pip package, so the official repo is cloned at a pinned commit
(``DEIM_COMMIT``) and used as-is; nothing in it is modified. Our side only provides the
config (``configs/deim/``), the COCO-pretrained weights and the fixed split.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

DEIM_REPO = "https://github.com/Intellindust-AI-Lab/DEIM.git"
DEIM_COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"  # 2026-03-24
DEFAULT_DEIM_DIR = Path("third_party/DEIM")

# Official COCO checkpoint of DEIM-D-FINE-S (README, "Model Zoo"); fine-tuned from, like
# RF-DETR is fine-tuned from its COCO weights.
PRETRAINED = {
    "s": {
        "file": "deim_dfine_hgnetv2_s_coco_120e.pth",
        "gdrive_id": "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC",
    }
}

# Overrides for --smoke: 4 tiny epochs that still pass through every phase of the
# schedule (no-aug, Mosaic + MixUp, aug without Mosaic, final no-aug stage 2).
SMOKE_OVERRIDES = {
    "epoches": 4,
    "flat_epoch": 1,
    "no_aug_epoch": 1,
    "warmup_iter": 5,
    "checkpoint_freq": 1000,
    "ema": {"warmups": 5},
    "train_dataloader": {
        "dataset": {"transforms": {"policy": {"epoch": [1, 2, 3]}}},
        "collate_fn": {"mixup_epochs": [1, 2], "stop_epoch": 3},
        "total_batch_size": 2,
        "num_workers": 0,
    },
    "val_dataloader": {"total_batch_size": 2, "num_workers": 0},
}


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure_deim(deim_dir: Path = DEFAULT_DEIM_DIR) -> Path:
    """Clone the official repo at the pinned commit (once) and check it is unmodified."""
    deim_dir = deim_dir.resolve()
    if not (deim_dir / "train.py").exists():
        deim_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", DEIM_REPO, str(deim_dir)], check=True)
    head = _git("rev-parse", "HEAD", cwd=deim_dir)
    if head != DEIM_COMMIT:
        _git("fetch", "-q", "origin", cwd=deim_dir)
        _git("checkout", "-q", DEIM_COMMIT, cwd=deim_dir)
    if _git("status", "--porcelain", "--untracked-files=no", cwd=deim_dir):
        raise RuntimeError(f"{deim_dir} has local modifications; DEIM must be used unmodified")
    return deim_dir


def ensure_pretrained(dest_dir: Path, size: str = "s") -> Path:
    """Download the official COCO checkpoint from Google Drive (via gdown) once."""
    info = PRETRAINED[size]
    dest = dest_dir / info["file"]
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    import gdown

    gdown.download(id=info["gdrive_id"], output=str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"could not download {info['file']} from Google Drive (quota?). Download it "
            f"manually from the DEIM README and place it at {dest}"
        )
    return dest


def _merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def render_config(template: Path, deim_dir: Path, data: Path, out: Path, smoke: bool) -> dict:
    """Fill the template's ${DEIM}/${DATA}/${OUT} with absolute POSIX paths."""
    text = template.read_text(encoding="utf-8")
    for key, path in (("DEIM", deim_dir), ("DATA", data), ("OUT", out)):
        text = text.replace("${" + key + "}", Path(path).resolve().as_posix())
    cfg = yaml.safe_load(text)
    if smoke:
        _merge(cfg, SMOKE_OVERRIDES)
    return cfg


def write_config(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def epochs_logged(run_dir: Path) -> list[int]:
    """Epochs DEIM finished (one JSON line per epoch in log.txt)."""
    log = run_dir / "log.txt"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(int(json.loads(line)["epoch"]))
    return out


def import_deim(deim_dir: Path):
    """Make the official ``engine`` package importable (for prediction)."""
    p = str(deim_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    from engine.core import YAMLConfig  # type: ignore[import-not-found]

    return YAMLConfig
