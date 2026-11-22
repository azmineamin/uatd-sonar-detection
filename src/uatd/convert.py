"""Convert raw UATD to YOLO and COCO views, driven by the committed manifests.

    python -m uatd.convert --raw-root /path/to/uatd-dataset --out data/processed

Layout written under ``--out``::

    images/{split}/{split}_{stem}.bmp     # symlink / hardlink / copy / lossless PNG
    labels/{split}/{split}_{stem}.txt     # YOLO: cls xc yc w h (normalised)
    coco/{split}.json                     # COCO: file_name relative to images/
    uatd_yolo.yaml                        # Ultralytics data file
    convert_report.json

File names and COCO ``uatd_id`` carry the split prefix (``test1_00247``) because
stems collide across raw folders. COCO ``image_id`` is an integer:
``FOLDER_CODE[folder] * 100000 + int(stem)``. COCO ``category_id`` equals the
YOLO class ID (0-9).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from uatd.constants import CLASS_NAMES, FOLDER_CODE, SPLITS
from uatd.rawindex import locate_pairs
from uatd.splits import DEFAULT_SPLITS_DIR, ManifestEntry, read_manifest
from uatd.voc import Box, clean_annotation

LINK_MODES = ("symlink", "hardlink", "copy", "png")


def coco_image_id(entry: ManifestEntry) -> int:
    return FOLDER_CODE[entry.folder] * 100_000 + int(entry.stem)


def yolo_lines(boxes: list[Box], width: int, height: int) -> list[str]:
    lines = []
    for b in boxes:
        xc = (b.x1 + b.x2) / 2 / width
        yc = (b.y1 + b.y2) / 2 / height
        lines.append(
            f"{b.class_id} {xc:.8f} {yc:.8f} {b.width / width:.8f} {b.height / height:.8f}"
        )
    return lines


def place_image(src: Path, dst_stem: Path, mode: str) -> Path:
    """Put the image at ``dst_stem`` + suffix. Pixel data is never altered."""
    dst = dst_stem.with_suffix(".png" if mode == "png" else src.suffix.lower())
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError:  # Windows without Developer Mode: fall back to a copy
            shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "png":
        with Image.open(src) as im:
            im.save(dst, format="PNG", compress_level=1)  # lossless; fast to write
    else:
        raise ValueError(f"unknown link mode {mode!r}")
    return dst


def empty_coco() -> dict:
    return {
        "info": {"description": "UATD (fixed split, cleaned)", "version": "1"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": n, "supercategory": "sonar_target"} for i, n in enumerate(CLASS_NAMES)
        ],
    }


def convert_split(
    split: str,
    raw_root: Path,
    out: Path,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    mode: str = "symlink",
    limit: int | None = None,
    workers: int = 16,
) -> dict:
    entries = read_manifest(split, splits_dir)
    if limit is not None:
        entries = entries[:limit]
    pairs = locate_pairs(raw_root, entries[0].folder) if entries else {}

    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for e in entries:
        if e.stem not in pairs:
            raise FileNotFoundError(
                f"{e.line} listed in {split}.txt but not found under {raw_root}"
            )

    def process(e: ManifestEntry):
        img_path, xml_path = pairs[e.stem]
        with Image.open(img_path) as im:
            w, h = im.size
        res = clean_annotation(xml_path.read_bytes(), w, h)
        if not res.valid:
            raise ValueError(
                f"{e.line} is in the manifest but fails cleaning ({res.drop_reason}); "
                "the raw data does not match the committed split"
            )
        dst = place_image(img_path, img_dir / e.uid, mode)
        (lbl_dir / f"{e.uid}.txt").write_text(
            "\n".join(yolo_lines(res.boxes, w, h)) + ("\n" if res.boxes else ""),
            encoding="utf-8",
            newline="\n",
        )
        return e, w, h, res, dst

    # Threads: reads from network-mounted storage (Kaggle input) are latency-bound and
    # PNG compression releases the GIL. map() keeps manifest order, so IDs are unchanged.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        processed = list(pool.map(process, entries))

    coco = empty_coco()
    per_class: Counter = Counter()
    ann_id = 1
    for e, w, h, res, dst in processed:
        image_id = coco_image_id(e)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{split}/{dst.name}",
                "width": w,
                "height": h,
                "uatd_id": e.uid,
                "source": e.line,
            }
        )
        for b in res.boxes:
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": b.class_id,
                    "bbox": [b.x1, b.y1, b.width, b.height],
                    "area": b.area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            per_class[b.class_name] += 1

    (out / "coco").mkdir(parents=True, exist_ok=True)
    (out / "coco" / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    return {
        "images": len(entries),
        "objects": sum(per_class.values()),
        "per_class": {c: per_class[c] for c in CLASS_NAMES},
    }


def write_yolo_yaml(out: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    text = (
        "# Generated by uatd.convert. `test` points at test1; use test2 explicitly.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test1\n"
        "test1: images/test1\n"
        "test2: images/test2\n"
        f"names:\n{names}\n"
    )
    (out / "uatd_yolo.yaml").write_text(text, encoding="utf-8", newline="\n")


def check_against_expected(report: dict, splits_dir: Path) -> None:
    expected = json.loads((splits_dir / "expected_counts.json").read_text(encoding="utf-8"))
    for split, got in report.items():
        exp = expected["splits"][split]
        for key in ("images", "objects", "per_class"):
            if got[key] != exp[key]:
                raise AssertionError(f"{split} {key}: got {got[key]}, expected {exp[key]}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--link-mode", default="symlink", choices=LINK_MODES)
    ap.add_argument("--limit", type=int, help="only the first N images per split")
    ap.add_argument("--smoke", action="store_true", help="same as --limit 6")
    ap.add_argument("--workers", type=int, default=16, help="parallel file reads/writes")
    args = ap.parse_args(argv)

    limit = 6 if args.smoke else args.limit
    report = {}
    for split in args.splits:
        report[split] = convert_split(
            split, args.raw_root, args.out, args.splits_dir, args.link_mode, limit, args.workers
        )
        print(
            f"{split:6s} images {report[split]['images']:5d} objects {report[split]['objects']:6d}"
        )
    write_yolo_yaml(args.out)
    (args.out / "convert_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if limit is None:
        check_against_expected(report, args.splits_dir)
        print("counts match data/splits/expected_counts.json")


if __name__ == "__main__":
    main()
