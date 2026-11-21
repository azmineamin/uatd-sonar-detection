"""The committed split must match CLAUDE.md Section 4.3 exactly.

The numbers below are typed in from CLAUDE.md on purpose: they are the
specification, not values read back from the generated files.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from uatd.constants import CLASS_NAMES, SEED, SPLIT_SOURCE, SPLITS
from uatd.splits import assign_splits, build, read_manifest

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"

EXPECTED_IMAGES = {"train": 6835, "val": 758, "test1": 798, "test2": 798}
EXPECTED_OBJECTS = {"train": 11059, "val": 1237, "test1": 1169, "test2": 1154}
EXPECTED_BEFORE = {"train": 6840, "val": 760, "test1": 800, "test2": 800}
EXPECTED_DROPPED = {
    "train": {"02630", "00707", "00402", "02477", "06975"},
    "val": {"02589", "00438"},
    "test1": {"00247", "00614"},
    "test2": {"00065", "00754"},
}
# class -> (train, val, test1, test2)
EXPECTED_PER_CLASS = {
    "ball": (2733, 337, 197, 193),
    "circle cage": (606, 55, 98, 99),
    "cube": (2390, 254, 172, 166),
    "cylinder": (512, 52, 48, 45),
    "human body": (1163, 120, 76, 73),
    "metal bucket": (426, 50, 6, 5),
    "plane": (730, 65, 134, 135),
    "rov": (629, 71, 150, 150),
    "square cage": (868, 111, 167, 168),
    "tyre": (1002, 122, 121, 120),
}
EXPECTED_EVENTS = {"reversed_box_swapped": 8, "clipped_box": 2, "zero_area_box": 12}


@pytest.fixture(scope="module")
def manifests():
    return {s: read_manifest(s, SPLITS_DIR) for s in SPLITS}


@pytest.fixture(scope="module")
def counts():
    return json.loads((SPLITS_DIR / "expected_counts.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------- spec self-consistency


def test_spec_tables_are_consistent():
    for i, s in enumerate(SPLITS):
        assert sum(v[i] for v in EXPECTED_PER_CLASS.values()) == EXPECTED_OBJECTS[s]
        assert EXPECTED_BEFORE[s] - len(EXPECTED_DROPPED[s]) == EXPECTED_IMAGES[s]
    assert tuple(EXPECTED_PER_CLASS) == CLASS_NAMES


# --------------------------------------------------------------- committed manifests


@pytest.mark.parametrize("split", SPLITS)
def test_manifest_image_counts(manifests, split):
    stems = [e.stem for e in manifests[split]]
    assert len(stems) == EXPECTED_IMAGES[split]
    assert len(set(stems)) == len(stems)
    assert all(e.folder == SPLIT_SOURCE[split] for e in manifests[split])
    assert not set(stems) & EXPECTED_DROPPED[split]


def test_train_val_disjoint_and_complete(manifests):
    train = {e.stem for e in manifests["train"]}
    val = {e.stem for e in manifests["val"]}
    assert not train & val
    everything = train | val | EXPECTED_DROPPED["train"] | EXPECTED_DROPPED["val"]
    assert len(everything) == EXPECTED_BEFORE["train"] + EXPECTED_BEFORE["val"] == 7600


def test_manifest_order_reproduces_notebook_shuffle(manifests):
    """Re-run the seeded shuffle on all 7600 training stems and compare order."""
    all_training = sorted(
        {e.stem for e in manifests["train"]}
        | {e.stem for e in manifests["val"]}
        | EXPECTED_DROPPED["train"]
        | EXPECTED_DROPPED["val"]
    )
    test1 = sorted({e.stem for e in manifests["test1"]} | EXPECTED_DROPPED["test1"])
    test2 = sorted({e.stem for e in manifests["test2"]} | EXPECTED_DROPPED["test2"])
    raw = assign_splits(all_training, test1, test2)
    for split in SPLITS:
        assert len(raw[split]) == EXPECTED_BEFORE[split]
        assert set(raw[split]) & EXPECTED_DROPPED[split] == EXPECTED_DROPPED[split]
        kept = [s for s in raw[split] if s not in EXPECTED_DROPPED[split]]
        assert kept == [e.stem for e in manifests[split]]


def test_split_prefixed_ids_are_unique(manifests):
    uids = [e.uid for s in SPLITS for e in manifests[s]]
    assert len(uids) == len(set(uids))
    assert manifests["test1"][0].uid.startswith("test1_")


# ---------------------------------------------------------------- expected_counts.json


@pytest.mark.parametrize("split", SPLITS)
def test_expected_counts_json(counts, split):
    s = counts["splits"][split]
    i = SPLITS.index(split)
    assert s["images_before_cleaning"] == EXPECTED_BEFORE[split]
    assert s["images"] == EXPECTED_IMAGES[split]
    assert s["objects"] == EXPECTED_OBJECTS[split]
    assert s["per_class"] == {c: v[i] for c, v in EXPECTED_PER_CLASS.items()}
    assert {d["stem"] for d in s["dropped"]} == EXPECTED_DROPPED[split]


def test_cleaning_event_totals(counts):
    for k, v in EXPECTED_EVENTS.items():
        assert counts["cleaning_events"].get(k, 0) == v
    for k in ("missing_box", "malformed_coordinates", "non_finite_box"):
        assert counts["cleaning_events"].get(k, 0) == 0
    assert counts["seed"] == SEED == 42


# -------------------------------------------------------------------- unit behaviour


def test_assign_splits_matches_notebook_code():
    stems = [f"{i:05d}" for i in range(1, 101)]
    pairs = list(stems)
    random.Random(42).shuffle(pairs)
    out = assign_splits(stems, ["00001"], ["00001"])
    assert out["val"] == pairs[:10] and out["train"] == pairs[10:]


def test_assign_splits_requires_sorted_input():
    with pytest.raises(ValueError):
        assign_splits(["00002", "00001"], [], [])


# ------------------------------------------------- full rebuild (only if data present)


def _raw_index():
    cache = ROOT / "data" / "raw" / "uatd_index.json.gz"
    raw_root = os.environ.get("UATD_RAW_ROOT")
    if raw_root:
        from uatd.rawindex import index_local

        return index_local(Path(raw_root))
    if cache.exists():
        from uatd.rawindex import load_index

        return load_index(cache)
    return None


@pytest.mark.slow
def test_rebuild_from_raw_matches_committed(manifests, counts):
    index = _raw_index()
    if index is None:
        pytest.skip("set UATD_RAW_ROOT or build data/raw/uatd_index.json.gz")
    rebuilt, report = build(index)
    for split in SPLITS:
        assert [e.line for e in rebuilt[split]] == [e.line for e in manifests[split]]
    assert report == counts
