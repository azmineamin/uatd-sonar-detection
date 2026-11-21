"""Provenance for every run: git hash, environment, resolved config (CLAUDE.md rule 7)."""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_info(repo: Path | None = None) -> dict:
    def run(*cmd: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *cmd], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def env_info(packages: tuple[str, ...] = ("torch", "torchvision", "rfdetr", "ultralytics")) -> dict:
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for p in packages:
        with contextlib.suppress(md.PackageNotFoundError):
            info["packages"][p] = md.version(p)
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
