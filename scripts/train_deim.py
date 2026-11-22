"""Train DEIM-D-FINE-S (CVPR 2025 / ICLR 2025) on the fixed UATD split, resumably.

Comparison row only: DEIM/D-FINE descend from the RT-DETR codebase (CLAUDE.md rule 6).

    python scripts/train_deim.py --data data/processed --seed 0 --time-budget-hours 10.5

Runs the official ``train.py`` (repo pinned in ``uatd.deim_support``) as a subprocess
with ``configs/deim/uatd_deim_dfine_s.yml``. Run directory: ``<runs-dir>/<exp>/``
(default ``runs/deim/dfine_s_s<seed>``).

Checkpointing / resume:
* DEIM writes ``last.pth`` after every epoch of the augmentation phase, and
  ``best_stg1.pth`` / ``best_stg2.pth`` when val AP improves (phase 1 / final phase).
* Re-running the same command resumes from the newest loadable checkpoint (``-r``).
  A disconnect during the short final phase (2 epochs) repeats that phase, which is how
  DEIM itself starts it (from ``best_stg1.pth``).
* Before resuming, existing ``best_stg*.pth`` are copied to ``best_stg*_s<k>.pth``:
  DEIM re-bases its "best so far" on resume, so a later epoch could overwrite a better
  file. ``scripts/predict_deim.py`` scores all of them on val and keeps the best.
* ``--time-budget-hours`` stops training before a Kaggle session ends.
* ``status.json``: ``running`` / ``interrupted`` / ``done`` / ``failed``. A ``done`` run
  is never retrained or overwritten (rule 7).

Selection uses val only; test1/test2 are never seen during training (rule 3).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from uatd.deim_support import (
    DEFAULT_DEIM_DIR,
    DEIM_COMMIT,
    ensure_deim,
    ensure_pretrained,
    epochs_logged,
    render_config,
    write_config,
)
from uatd.runinfo import env_info, git_info, now, write_json


def find_resume_checkpoint(out: Path) -> Path | None:
    """Newest checkpoint that loads (a kill mid-save can corrupt last.pth)."""
    import torch

    def epoch_of(p: Path) -> int:
        m = re.search(r"checkpoint(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    candidates = [out / "last.pth"] + sorted(
        out.glob("checkpoint[0-9]*.pth"), key=epoch_of, reverse=True
    )
    for p in candidates:
        if not p.exists():
            continue
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
            if "last_epoch" in state:
                return p
        except Exception as exc:  # noqa: BLE001 - any load failure means "unusable"
            print(f"[resume] {p.name} is not loadable ({exc!r}); trying an older one")
    return None


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop the trainer and its children (a Windows venv python.exe is a launcher that
    starts the real interpreter as a child; DataLoader workers are children too)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


def snapshot_best(out: Path, session_idx: int) -> None:
    for name in ("best_stg1", "best_stg2"):
        src = out / f"{name}.pth"
        if src.exists():
            shutil.copy2(src, out / f"{name}_s{session_idx}.pth")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--config", type=Path, default=Path("configs/deim/uatd_deim_dfine_s.yml"))
    ap.add_argument("--deim-dir", type=Path, default=DEFAULT_DEIM_DIR)
    ap.add_argument("--pretrained", type=Path, help="default: download the official COCO ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true", help="AMP is on by default (as upstream)")
    ap.add_argument("--time-budget-hours", type=float, help="stop cleanly after this long")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/deim"))
    ap.add_argument("--exp", help="run name (default dfine_s_s<seed>)")
    ap.add_argument("--smoke", action="store_true", help="4 tiny epochs, batch 2")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp = args.exp or f"dfine_s_s{args.seed}" + ("_smoke" if args.smoke else "")
    out = (args.runs_dir / exp).resolve()
    status_path = out / "status.json"

    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "done":
            print(f"[train] {out} is already done; not retraining. Use a new --exp to train again.")
            return 0

    out.mkdir(parents=True, exist_ok=True)
    deim_dir = ensure_deim(args.deim_dir)
    cfg = render_config(args.config, deim_dir, args.data, out, args.smoke)
    cfg_path = out / "config.yml"
    write_config(cfg, cfg_path)

    sessions_path = out / "sessions.json"
    sessions = (
        json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    )
    resume = find_resume_checkpoint(out)
    if resume is not None:
        snapshot_best(out, len(sessions))
        init = ["-r", str(resume)]
    else:
        pretrained = args.pretrained or ensure_pretrained(deim_dir / "weights")
        init = ["-t", str(Path(pretrained).resolve())]

    # official train.py, via a torchvision-compatibility shim (see uatd.deim_compat)
    cmd = [sys.executable, "-m", "uatd.deim_compat", str(deim_dir), "-c", str(cfg_path),
           "--seed", str(args.seed), "-d", args.device, *init]  # fmt: skip
    if not args.no_amp:
        cmd.append("--use-amp")

    config_json = out / "run_config.json"
    if not config_json.exists():
        write_json(
            config_json,
            {
                "pipeline": "DEIM-D-FINE-S (Huang et al., CVPR 2025; Peng et al., ICLR 2025)",
                "role": "comparison row (RT-DETR lineage, CLAUDE.md rule 6)",
                "deim_repo_commit": DEIM_COMMIT,
                "template": str(args.config),
                "resolved_config": cfg,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "command": cmd,
                "git": git_info(),
                "env": env_info(),
                "created": now(),
            },
        )
    session = {"start": now(), "resume_from": str(resume) if resume else None,
               "command": cmd, "git": git_info(), "env": env_info()}  # fmt: skip
    sessions.append(session)
    write_json(sessions_path, sessions)
    print(f"[train] run dir {out}")
    print(f"[train] resume from {resume}" if resume else f"[train] fine-tune from {init[1]}")
    print("$", " ".join(cmd), flush=True)

    write_json(status_path, {"state": "running", "updated": now()})
    t0 = time.time()
    budget_hit = threading.Event()
    with (out / "train_stdout.log").open("a", encoding="utf-8") as log:
        # unbuffered child, so its log streams live (also into Kaggle's cell output)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, cwd=deim_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, encoding="utf-8", errors="replace", env=env,
        )  # fmt: skip

        def stop() -> None:
            print(f"\n[train] time budget of {args.time_budget_hours} h reached; stopping")
            budget_hit.set()
            kill_tree(proc)

        timer = None
        if args.time_budget_hours:
            timer = threading.Timer(args.time_budget_hours * 3600, stop)
            timer.daemon = True
            timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            kill_tree(proc)
            rc = proc.wait()
            budget_hit.set()
        finally:
            if timer is not None:
                timer.cancel()

    hours = (time.time() - t0) / 3600
    total = int(cfg["epoches"])
    logged = epochs_logged(out)
    finished = rc == 0 and logged and max(logged) == total - 1
    if finished:
        session.update(end=now(), hours=hours, result="finished", epochs_done=len(set(logged)))
        write_json(sessions_path, sessions)
        write_json(status_path, {"state": "done", "updated": now()})
        print(f"[train] done ({total} epochs). Next: scripts/predict_deim.py --run {out}")
        return 0
    if budget_hit.is_set():
        session.update(end=now(), hours=hours, result="interrupted")
        write_json(sessions_path, sessions)
        write_json(status_path, {"state": "interrupted", "updated": now()})
        print("[train] stopped by time budget; re-run the same command to resume")
        return 0
    session.update(end=now(), hours=hours, result="failed", returncode=rc)
    write_json(sessions_path, sessions)
    write_json(status_path, {"state": "failed", "updated": now(), "returncode": rc})
    print(f"[train] DEIM exited with code {rc}; see {out / 'train_stdout.log'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
