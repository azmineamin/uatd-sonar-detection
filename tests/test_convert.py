"""End-to-end check of splits + convert on a tiny synthetic UATD-shaped dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uatd import convert, splits
from uatd.constants import CLASS_NAMES, RAW_FOLDERS
from uatd.rawindex import index_local

W, H = 64, 48


def _voc(boxes: list[tuple[str, int, int, int, int]]) -> str:
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in boxes
    )
    return f"<annotation>{objs}</annotation>"


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"
    for folder, n in zip(RAW_FOLDERS, (20, 5, 5), strict=True):
        (root / folder / "images").mkdir(parents=True)
        (root / folder / "annotations").mkdir(parents=True)
        for i in range(1, n + 1):
            stem = f"{i:05d}"
            arr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / folder / "images" / f"{stem}.bmp")
            boxes = [(CLASS_NAMES[i % 10], 4, 4, 20, 16)]
            if folder == RAW_FOLDERS[1] and i == 3:
                boxes.append(("cube", 70, 4, 80, 10))  # zero area after clip -> drop image
            if i == 2:
                boxes.append(("tyre", 70, 30, 30, 40))  # reversed + clipped, still valid
            (root / folder / "annotations" / f"{stem}.xml").write_text(_voc(boxes))
        # an image without XML must be ignored
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / folder / "images" / "99999.bmp")
    return root


@pytest.fixture()
def built(raw_root: Path, tmp_path: Path) -> Path:
    manifests, report = splits.build(index_local(raw_root))
    out = tmp_path / "splits"
    splits.write(manifests, report, out)
    return out


def test_build_drops_and_counts(built: Path):
    rep = json.loads((built / "expected_counts.json").read_text())
    assert rep["splits"]["val"]["images"] == 2 and rep["splits"]["train"]["images"] == 18
    assert rep["splits"]["test1"]["images"] == 4
    assert [d["stem"] for d in rep["splits"]["test1"]["dropped"]] == ["00003"]
    assert rep["cleaning_events"]["reversed_box_swapped"] == 3
    assert rep["cleaning_events"]["clipped_box"] == 4
    assert "99999" not in (built / "test2.txt").read_text()


@pytest.mark.parametrize("mode", ["copy", "png"])
def test_convert_views(raw_root: Path, built: Path, tmp_path: Path, mode: str):
    out = tmp_path / "processed"
    report = {s: convert.convert_split(s, raw_root, out, built, mode=mode) for s in splits.SPLITS}
    convert.check_against_expected(report, built)

    # split-prefixed names, no collisions between folders
    t1 = sorted(p.name for p in (out / "images" / "test1").iterdir())
    tr = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert all(n.startswith("test1_") for n in t1) and all(n.startswith("train_") for n in tr)

    # pixel data is identical to the source
    src = raw_root / "UATD_Test_2" / "images" / "00001.bmp"
    dst = next((out / "images" / "test2").glob("test2_00001.*"))
    assert np.array_equal(np.asarray(Image.open(src)), np.asarray(Image.open(dst)))

    # YOLO label for the reversed+clipped box: x 30..64, y 30..40 on 64x48
    lbl = (out / "labels" / "test2" / "test2_00002.txt").read_text().splitlines()
    cls, xc, yc, w, h = lbl[1].split()
    assert int(cls) == 9
    assert np.allclose(
        [float(xc), float(yc), float(w), float(h)], [47 / 64, 35 / 48, 34 / 64, 10 / 48]
    )

    # COCO json loads in pycocotools; IDs unique across splits
    from pycocotools.coco import COCO

    ids = []
    for s in splits.SPLITS:
        c = COCO(str(out / "coco" / f"{s}.json"))
        ids += c.getImgIds()
        for img in c.dataset["images"]:
            assert (out / "images" / img["file_name"]).exists()
            assert img["uatd_id"].startswith(f"{s}_")
    assert len(ids) == len(set(ids))


def test_manifest_mismatch_is_an_error(raw_root: Path, built: Path, tmp_path: Path):
    # corrupt a listed image's XML so it no longer passes cleaning
    first = (built / "test2.txt").read_text().split()[0].split("/")[1]
    (raw_root / "UATD_Test_2" / "annotations" / f"{first}.xml").write_text(
        _voc([("ball", 5, 5, 5, 5)])
    )
    with pytest.raises(ValueError, match="fails cleaning"):
        convert.convert_split("test2", raw_root, tmp_path / "p", built)


def test_nested_kaggle_layout(raw_root: Path, tmp_path: Path):
    """Kaggle unzips to <root>/UATD_Test_1/UATD_Test_1/{images,annotations}; must still work."""
    import shutil

    nested = tmp_path / "kaggle_input"
    for folder in RAW_FOLDERS:
        shutil.copytree(raw_root / folder, nested / folder / folder)
    flat_manifests, flat_report = splits.build(index_local(raw_root))
    nested_manifests, nested_report = splits.build(index_local(nested))
    assert nested_report == flat_report
    assert {s: [e.line for e in v] for s, v in nested_manifests.items()} == {
        s: [e.line for e in v] for s, v in flat_manifests.items()
    }
    out = tmp_path / "splits_nested"
    splits.write(nested_manifests, nested_report, out)
    rep = convert.convert_split("test1", nested, tmp_path / "proc", out, mode="copy")
    assert rep["images"] == nested_report["splits"]["test1"]["images"]
