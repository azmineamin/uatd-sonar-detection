"""Train RF-DETR (ICLR 2026) on the fixed UATD split, resumably.

    python scripts/train_rfdetr.py --data data/processed --model small --epochs 100 \\
        --batch 8 --grad-accum 2 --seed 0 --time-budget-hours 10.5

Run directory: ``<runs-dir>/<exp>/`` (default ``runs/rfdetr/<model>_r<res>_s<seed>``).

Checkpointing / resume:
* RF-DETR writes ``last.ckpt`` (full optimizer + scheduler state) after every epoch,
  plus ``checkpoint_<epoch>.ckpt`` every ``--checkpoint-interval`` epochs.
* Re-running the same command resumes from the newest loadable checkpoint.
* ``--time-budget-hours`` stops training cleanly before a Kaggle session ends; the
  last finished epoch is kept and the next run continues from it.
* ``status.json`` records ``running`` / ``interrupted`` / ``done`` / ``failed``.
  A ``done`` run is never retrained or overwritten (CLAUDE.md rule 7).

Model selection uses our val split (RF-DETR's ``valid``), with early stopping.
test1/test2 are never seen during training (rule 3).
"""

from __future__ import annotations

import _thread
import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

from uatd.rfdetr_data import build_rfdetr_dataset
from uatd.runinfo import env_info, git_info, now, write_json

MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}
_INTERRUPTED_BY_BUDGET = threading.Event()


def find_resume_checkpoint(out: Path) -> Path | None:
    """Newest checkpoint that actually loads (a kill mid-save can corrupt last.ckpt)."""
    import torch

    def epoch_of(p: Path) -> int:
        m = re.search(r"checkpoint_(\d+)\.ckpt$", p.name)
        return int(m.group(1)) if m else -1

    candidates = [out / "last.ckpt"] + sorted(
        out.glob("checkpoint_*.ckpt"), key=epoch_of, reverse=True
    )
    for p in candidates:
        if not p.exists():
            continue
        try:
            torch.load(p, map_location="cpu", weights_only=False)
            return p
        except Exception as exc:  # noqa: BLE001 - any load failure means "unusable"
            print(f"[resume] {p.name} is not loadable ({exc!r}); trying an older one")
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", choices=MODELS, default="small")
    ap.add_argument("--resolution", type=int, help="default: the model's native resolution")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--patience", type=int, default=15, help="early stopping on val mAP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--time-budget-hours", type=float, help="stop cleanly after this long")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/rfdetr"))
    ap.add_argument("--exp", help="run name (default <model>_r<res>_s<seed>)")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, batch 2, no workers")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.batch, args.grad_accum, args.num_workers = 1, 2, 1, 0
        args.checkpoint_interval = 1_000
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import rfdetr

    model_cls = getattr(rfdetr, MODELS[args.model])
    model = model_cls()
    resolution = args.resolution or model.model_config.resolution
    exp = args.exp or f"{args.model}_r{resolution}_s{args.seed}" + ("_smoke" if args.smoke else "")
    out = args.runs_dir / exp
    status_path = out / "status.json"

    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "done":
            print(f"[train] {out} is already done; not retraining. Use a new --exp to train again.")
            return 0

    out.mkdir(parents=True, exist_ok=True)
    resume = find_resume_checkpoint(out)
    dataset_dir = build_rfdetr_dataset(args.data)

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(out),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=True,
        early_stopping_patience=args.patience,
        tensorboard=args.tensorboard,
        progress_bar="tqdm",
        run_test=False,
    )
    if args.resolution:
        train_kwargs["resolution"] = args.resolution
    if resume is not None:
        train_kwargs["resume"] = str(resume)

    config_path = out / "run_config.json"
    if not config_path.exists():
        write_json(
            config_path,
            {
                "pipeline": "RF-DETR (Robinson et al., ICLR 2026)",
                "model": MODELS[args.model],
                "resolution": resolution,
                "args": vars(args),
                "train_kwargs": train_kwargs,
                "git": git_info(),
                "env": env_info(),
                "created": now(),
            },
        )
    sessions_path = out / "sessions.json"
    sessions = (
        json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    )
    session = {
        "start": now(),
        "resume_from": str(resume) if resume else None,
        "git": git_info(),
        "env": env_info(),
    }
    sessions.append(session)
    write_json(sessions_path, sessions)
    print(f"[train] run dir {out}")
    print(f"[train] resume from {resume}" if resume else "[train] fresh start")

    timer = None
    if args.time_budget_hours:

        def stop() -> None:
            print(f"\n[train] time budget of {args.time_budget_hours} h reached; stopping cleanly")
            _INTERRUPTED_BY_BUDGET.set()
            _thread.interrupt_main()

        timer = threading.Timer(args.time_budget_hours * 3600, stop)
        timer.daemon = True
        timer.start()

    write_json(status_path, {"state": "running", "updated": now()})
    t0 = time.time()
    try:
        model.train(**train_kwargs)
    except (KeyboardInterrupt, SystemExit) as exc:
        if _INTERRUPTED_BY_BUDGET.is_set():
            session.update(end=now(), hours=(time.time() - t0) / 3600, result="interrupted")
            write_json(sessions_path, sessions)
            write_json(status_path, {"state": "interrupted", "updated": now()})
            print("[train] stopped by time budget; re-run the same command to resume")
            return 0
        write_json(status_path, {"state": "failed", "updated": now(), "error": repr(exc)})
        raise
    except Exception as exc:
        write_json(status_path, {"state": "failed", "updated": now(), "error": repr(exc)})
        raise
    finally:
        if timer is not None:
            timer.cancel()

    session.update(end=now(), hours=(time.time() - t0) / 3600, result="finished")
    write_json(sessions_path, sessions)
    write_json(status_path, {"state": "done", "updated": now()})
    print(f"[train] done. Best checkpoint (val mAP): {out / 'checkpoint_best_total.pth'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
