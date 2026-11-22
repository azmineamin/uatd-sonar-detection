"""Predict with a trained DEIM-D-FINE run and score it with the shared evaluator.

    python scripts/predict_deim.py --run runs/deim/dfine_s_s0 --data data/processed

1. Checkpoint selection on **val only**: every ``best_stg*.pth`` in the run (including the
   copies kept at each resume, see ``train_deim.py``) is scored on val with the shared
   evaluator; the one with the highest val mAP50:95 is used. Scores go to
   ``<run>/selection.json``.
2. That checkpoint predicts each split (default val, test1, test2) once:
   ``<run>/predictions/<split>.json`` (COCO results, our IDs) and
   ``<run>/metrics/<split>.json`` (``uatd.eval_coco``).

Inference follows the official ``tools/inference/torch_inf.py``: EMA weights, deploy mode,
RGB image resized to 640x640, boxes returned in original-image pixels, top 300 per image.
Splits whose prediction file exists are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from uatd.deim_support import DEFAULT_DEIM_DIR, ensure_deim, import_deim
from uatd.eval_coco import DEFAULT_EVAL_CONFIG, evaluate, format_summary, load_size_bins
from uatd.runinfo import env_info, git_info, now, write_json


def load_model(deim_dir: Path, cfg_path: Path, ckpt: Path, device: str):
    YAMLConfig = import_deim(deim_dir)
    cfg = YAMLConfig(str(cfg_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # weights come from the checkpoint
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    model = cfg.model
    model.load_state_dict(weights)
    size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    return model.deploy().to(device), cfg.postprocessor.deploy().to(device), tuple(size)


@torch.no_grad()
def predict_split(model, post, size, processed: Path, split: str, threshold: float,
                  batch: int, device: str) -> list[dict]:  # fmt: skip
    coco = json.loads((processed / "coco" / f"{split}.json").read_text(encoding="utf-8"))
    valid_ids = {c["id"] for c in coco["categories"]}
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    images = coco["images"]
    results: list[dict] = []
    for i in tqdm(range(0, len(images), batch), desc=f"predict {split}", unit="batch"):
        chunk = images[i : i + batch]
        tensors, sizes = [], []
        for im in chunk:
            with Image.open(processed / "images" / im["file_name"]) as pil:
                pil = pil.convert("RGB")
                sizes.append([pil.width, pil.height])
                tensors.append(tf(pil))
        x = torch.stack(tensors).to(device)
        orig = torch.tensor(sizes, device=device)
        labels, boxes, scores = post(model(x), orig)
        for im, lab, box, sco in zip(chunk, labels, boxes, scores, strict=True):
            keep = sco >= threshold
            for c, (x1, y1, x2, y2), s in zip(
                lab[keep].tolist(), box[keep].tolist(), sco[keep].tolist(), strict=True
            ):
                if c not in valid_ids:
                    raise ValueError(f"model predicted unknown class id {c}")
                results.append(
                    {
                        "image_id": im["id"],
                        "category_id": int(c),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": s,
                    }
                )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--deim-dir", type=Path, default=DEFAULT_DEIM_DIR)
    ap.add_argument("--splits", nargs="+", default=["val", "test1", "test2"])
    ap.add_argument("--checkpoint", type=Path, help="skip selection and use this checkpoint")
    ap.add_argument(
        "--threshold", type=float, default=0.001, help="low, so mAP sees the full PR curve"
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    deim_dir = ensure_deim(args.deim_dir)
    cfg_path = args.run / "config.yml"
    size_bins = load_size_bins(args.eval_config)
    val_gt = args.data / "coco" / "val.json"
    sel_path = args.run / "selection.json"

    # ---- 1. choose the checkpoint on val ----
    if args.checkpoint:
        ckpt, val_preds = args.checkpoint, None
    elif sel_path.exists() and not args.force:
        ckpt, val_preds = Path(json.loads(sel_path.read_text())["selected"]), None
        print(f"[select] using earlier selection {ckpt.name}")
    else:
        candidates = sorted(args.run.glob("best_stg*.pth"))
        if not candidates:
            raise FileNotFoundError(f"no best_stg*.pth in {args.run}")
        scores, best = [], None
        for c in candidates:
            model, post, size = load_model(deim_dir, cfg_path, c, args.device)
            preds = predict_split(model, post, size, args.data, "val", args.threshold,
                                  args.batch, args.device)  # fmt: skip
            m = evaluate(val_gt, preds)
            scores.append({"checkpoint": c.name, "val_mAP50_95": m["mAP50_95"],
                           "val_mAP50": m["mAP50"]})  # fmt: skip
            print(f"[select] {c.name}: val mAP50:95 {m['mAP50_95']:.4f}  mAP50 {m['mAP50']:.4f}")
            key = (m["mAP50_95"], m["mAP50"])
            if best is None or key > best[0]:
                best = (key, c, preds)
            del model, post
        _, ckpt, val_preds = best
        selection = {
            "rule": "highest val mAP50:95 (shared evaluator), tie: mAP50",
            "selected": str(ckpt),
            "candidates": scores,
            "time": now(),
        }
        write_json(sel_path, selection)
        print(f"[select] -> {ckpt.name}")

    # ---- 2. predict and score each split ----
    model = None
    for split in args.splits:
        pred_path = args.run / "predictions" / f"{split}.json"
        if split == "val" and val_preds is not None:
            preds = val_preds
            write_json(pred_path, preds)  # type: ignore[arg-type]
        elif pred_path.exists() and not args.force:
            print(f"[predict] {split}: using existing {pred_path}")
            preds = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            if model is None:
                model, post, size = load_model(deim_dir, cfg_path, ckpt, args.device)
            preds = predict_split(model, post, size, args.data, split, args.threshold,
                                  args.batch, args.device)  # fmt: skip
            write_json(pred_path, preds)  # type: ignore[arg-type]
        metrics = evaluate(args.data / "coco" / f"{split}.json", preds, size_bins)
        metrics.update(checkpoint=str(ckpt), threshold=args.threshold, git=git_info(),
                       env=env_info(), time=now())  # fmt: skip
        write_json(args.run / "metrics" / f"{split}.json", metrics)
        print(f"[{split}]\n{format_summary(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
