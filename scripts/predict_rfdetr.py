"""Predict with a trained RF-DETR run and score it with the shared evaluator.

    python scripts/predict_rfdetr.py --run runs/rfdetr/small_r512_s0 --data data/processed

For each split (default val, test1, test2) writes
``<run>/predictions/<split>.json`` (COCO results, our image/category IDs) and
``<run>/metrics/<split>.json`` (from ``uatd.eval_coco``). Splits whose prediction
file already exists are skipped (resumable); use ``--force`` to redo them.
The checkpoint is the one RF-DETR selected on val (``checkpoint_best_total.pth``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from uatd.eval_coco import DEFAULT_EVAL_CONFIG, evaluate, format_summary, load_size_bins
from uatd.runinfo import env_info, git_info, now, write_json


def predict_split(model, processed: Path, split: str, threshold: float, batch: int) -> list[dict]:
    coco = json.loads((processed / "coco" / f"{split}.json").read_text(encoding="utf-8"))
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    images = coco["images"]
    results: list[dict] = []
    for i in tqdm(range(0, len(images), batch), desc=f"predict {split}", unit="batch"):
        chunk = images[i : i + batch]
        pil = [Image.open(processed / "images" / im["file_name"]).convert("RGB") for im in chunk]
        dets = model.predict(pil, threshold=threshold, include_source_image=False)
        if not isinstance(dets, list):
            dets = [dets]
        for im, d in zip(chunk, dets, strict=True):
            for (x1, y1, x2, y2), score, name in zip(
                d.xyxy, d.confidence, d.data["class_name"], strict=True
            ):
                if name == "__background__":  # RF-DETR's no-object slot, not a detection
                    continue
                if name not in name_to_id:
                    raise ValueError(f"model predicted unknown class {name!r}")
                results.append(
                    {
                        "image_id": im["id"],
                        "category_id": name_to_id[name],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        for p in pil:
            p.close()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", nargs="+", default=["val", "test1", "test2"])
    ap.add_argument("--checkpoint", type=Path, help="default <run>/checkpoint_best_total.pth")
    ap.add_argument(
        "--threshold", type=float, default=0.001, help="low, so mAP sees the full PR curve"
    )
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or args.run / "checkpoint_best_total.pth"
    model = None
    size_bins = load_size_bins(args.eval_config)
    for split in args.splits:
        pred_path = args.run / "predictions" / f"{split}.json"
        if pred_path.exists() and not args.force:
            print(f"[predict] {split}: using existing {pred_path}")
            preds = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            if model is None:
                from rfdetr import RFDETR

                model = RFDETR.from_checkpoint(ckpt, trust_checkpoint=True)
            preds = predict_split(model, args.data, split, args.threshold, args.batch)
            write_json(pred_path, preds)  # type: ignore[arg-type]

        metrics = evaluate(args.data / "coco" / f"{split}.json", preds, size_bins)
        metrics.update(
            checkpoint=str(ckpt),
            threshold=args.threshold,
            git=git_info(),
            env=env_info(),
            time=now(),
        )
        write_json(args.run / "metrics" / f"{split}.json", metrics)
        print(f"[{split}]\n{format_summary(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
