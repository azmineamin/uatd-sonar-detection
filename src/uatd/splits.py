"""Build the fixed train/val/test1/test2 manifests (CLAUDE.md, Section 4).

Run once, commit the output, never re-derive the split elsewhere::

    # from an extracted copy (Kaggle mirror or figshare unzip)
    python -m uatd.splits --raw-root /path/to/uatd-dataset

    # or straight from figshare, without downloading the images
    python -m uatd.splits --remote --cache data/raw/uatd_index.json.gz

Outputs in ``data/splits/``:

* ``{train,val,test1,test2}.txt`` - one ``<raw folder>/<stem>`` per line, in the
  notebook's order (train/val in shuffled order, tests sorted). Only images that
  survive cleaning are listed.
* ``expected_counts.json`` - image/object counts, per-class counts, dropped
  images with the reason, and cleaning event totals.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from uatd.constants import (
    CLASS_NAMES,
    SEED,
    SPLIT_SOURCE,
    SPLITS,
    TEST1_FOLDER,
    TEST2_FOLDER,
    TRAINING_FOLDER,
    VAL_FRACTION,
)
from uatd.rawindex import Index, index_local, index_remote, load_index, save_index
from uatd.voc import clean_annotation

DEFAULT_SPLITS_DIR = Path("data/splits")


@dataclass(frozen=True)
class ManifestEntry:
    """One image in a split. ``uid`` is the split-prefixed ID, e.g. ``test1_00247``."""

    split: str
    folder: str
    stem: str

    @property
    def uid(self) -> str:
        return f"{self.split}_{self.stem}"

    @property
    def line(self) -> str:
        return f"{self.folder}/{self.stem}"


def assign_splits(
    training_stems: list[str], test1_stems: list[str], test2_stems: list[str]
) -> dict[str, list[str]]:
    """Reproduce the notebook split (before cleaning). Inputs must be sorted."""
    for stems in (training_stems, test1_stems, test2_stems):
        if stems != sorted(stems):
            raise ValueError("stems must be sorted as strings")
    shuffled = list(training_stems)
    random.Random(SEED).shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_FRACTION)
    return {
        "train": shuffled[n_val:],
        "val": shuffled[:n_val],
        "test1": list(test1_stems),
        "test2": list(test2_stems),
    }


def build(index: Index) -> tuple[dict[str, list[ManifestEntry]], dict]:
    """Assign splits, clean every annotation, and return manifests + report."""
    raw = assign_splits(
        sorted(index[TRAINING_FOLDER]), sorted(index[TEST1_FOLDER]), sorted(index[TEST2_FOLDER])
    )
    manifests: dict[str, list[ManifestEntry]] = {}
    report: dict = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "class_names": list(CLASS_NAMES),
        "splits": {},
        "cleaning_events": {},
    }
    total_events: Counter = Counter()

    for split in SPLITS:
        folder = SPLIT_SOURCE[split]
        kept: list[ManifestEntry] = []
        dropped: list[dict] = []
        per_class: Counter = Counter()
        events: Counter = Counter()
        for stem in raw[split]:
            rec = index[folder][stem]
            res = clean_annotation(rec["xml"], rec["width"], rec["height"])
            events.update(res.events)
            if not res.valid:
                dropped.append({"stem": stem, "reason": res.drop_reason})
                continue
            kept.append(ManifestEntry(split, folder, stem))
            per_class.update(b.class_name for b in res.boxes)
        manifests[split] = kept
        total_events.update(events)
        report["splits"][split] = {
            "source_folder": folder,
            "images_before_cleaning": len(raw[split]),
            "images": len(kept),
            "objects": sum(per_class.values()),
            "per_class": {c: per_class[c] for c in CLASS_NAMES},
            "dropped": dropped,
            "cleaning_events": dict(sorted(events.items())),
        }
    report["cleaning_events"] = dict(sorted(total_events.items()))
    return manifests, report


def write(manifests: dict[str, list[ManifestEntry]], report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in manifests.items():
        text = "".join(e.line + "\n" for e in entries)
        (out_dir / f"{split}.txt").write_text(text, encoding="utf-8", newline="\n")
    (out_dir / "expected_counts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def read_manifest(split: str, splits_dir: Path = DEFAULT_SPLITS_DIR) -> list[ManifestEntry]:
    """Read a committed manifest. Every converter must use this."""
    if split not in SPLIT_SOURCE:
        raise ValueError(f"unknown split {split!r}")
    entries = []
    for line in (splits_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        folder, stem = line.split("/")
        if folder != SPLIT_SOURCE[split]:
            raise ValueError(f"{split}.txt lists {line!r}, expected folder {SPLIT_SOURCE[split]}")
        entries.append(ManifestEntry(split, folder, stem))
    return entries


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw-root", type=Path, help="folder containing UATD_Training/, ...")
    src.add_argument("--remote", action="store_true", help="read the figshare zips over HTTP")
    src.add_argument("--index", type=Path, help="use a cached index (.json.gz)")
    ap.add_argument("--cache", type=Path, help="save the built index here (.json.gz)")
    ap.add_argument("--out", type=Path, default=DEFAULT_SPLITS_DIR)
    args = ap.parse_args(argv)

    if args.index:
        index = load_index(args.index)
    elif args.remote:
        index = index_remote()
    else:
        index = index_local(args.raw_root)
    if args.cache:
        save_index(index, args.cache)

    manifests, report = build(index)
    write(manifests, report, args.out)
    for split in SPLITS:
        s = report["splits"][split]
        print(
            f"{split:6s} images {s['images']:5d} (dropped {len(s['dropped'])}) "
            f"objects {s['objects']:6d}"
        )
    print("cleaning events:", report["cleaning_events"])
    print(f"wrote manifests to {args.out}")


if __name__ == "__main__":
    main()
