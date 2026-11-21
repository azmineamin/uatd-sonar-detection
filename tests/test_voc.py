"""Cleaning rules from CLAUDE.md Section 4.2."""

from __future__ import annotations

import pytest

from uatd.voc import UnknownClassError, clean_annotation


def xml(*objects: str) -> str:
    size = "<size><width>100</width><height>50</height></size>"
    return f"<annotation>{size}{''.join(objects)}</annotation>"


def obj(name="ball", xmin="10", ymin="10", xmax="20", ymax="20", bndbox=True) -> str:
    coords = f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax>"
    box = f"<bndbox>{coords}</bndbox>" if bndbox else ""
    return f"<object><name>{name}</name>{box}</object>"


def test_valid_box():
    r = clean_annotation(xml(obj()), 100, 50)
    assert r.valid and len(r.boxes) == 1
    b = r.boxes[0]
    assert (b.class_id, b.x1, b.y1, b.x2, b.y2) == (0, 10, 10, 20, 20)


def test_name_is_stripped_and_lowercased():
    r = clean_annotation(xml(obj(name="  Square Cage ")), 100, 50)
    assert r.boxes[0].class_name == "square cage" and r.boxes[0].class_id == 8


def test_unknown_class_raises():
    with pytest.raises(UnknownClassError):
        clean_annotation(xml(obj(name="submarine")), 100, 50)


def test_reversed_coordinates_are_swapped():
    r = clean_annotation(xml(obj(xmin="30", xmax="10", ymin="40", ymax="5")), 100, 50)
    assert r.valid and r.events["reversed_box_swapped"] == 1
    b = r.boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 5, 30, 40)


def test_clipping_uses_real_image_size():
    r = clean_annotation(xml(obj(xmin="-5", ymin="10", xmax="120", ymax="60")), 100, 50)
    assert r.valid and r.events["clipped_box"] == 1
    b = r.boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 10, 100, 50)


@pytest.mark.parametrize(
    "o",
    [
        obj(bndbox=False),
        obj(xmin="abc"),
        obj(xmin=""),
        obj(xmax="nan"),
        obj(xmax="inf"),
        obj(xmin="10", xmax="10"),  # zero width
        obj(xmin="150", xmax="160"),  # zero width after clipping
    ],
)
def test_one_invalid_object_drops_whole_image(o):
    r = clean_annotation(xml(obj(), o), 100, 50)
    assert not r.valid
    assert r.drop_reason.startswith("invalid_object")


def test_parse_error_drops_image():
    r = clean_annotation("<annotation><object>", 100, 50)
    assert not r.valid and r.drop_reason.startswith("xml_parse_error")


def test_no_objects_is_valid_background():
    r = clean_annotation(xml(), 100, 50)
    assert r.valid and r.boxes == []
