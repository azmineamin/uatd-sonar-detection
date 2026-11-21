"""Pascal-VOC XML parsing and the fixed cleaning rules (CLAUDE.md, Section 4.2).

This is the only place the cleaning rules live. Both the split builder and the
converters call :func:`clean_annotation`, so the rules cannot drift apart.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from uatd.constants import CLASS_NAMES, CLASS_TO_ID


class UnknownClassError(ValueError):
    """Raised when an XML <name> is not one of the 10 UATD classes."""


@dataclass(frozen=True)
class Box:
    """A cleaned box in absolute pixel coordinates (x1 < x2, y1 < y2)."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass
class CleanResult:
    """Outcome of cleaning one annotation file."""

    boxes: list[Box]
    n_objects: int
    events: Counter = field(default_factory=Counter)
    parse_error: str | None = None

    @property
    def valid(self) -> bool:
        """An image is kept only if every object survived cleaning."""
        return self.parse_error is None and len(self.boxes) == self.n_objects

    @property
    def drop_reason(self) -> str | None:
        if self.valid:
            return None
        if self.parse_error is not None:
            return f"xml_parse_error: {self.parse_error}"
        bad = [
            k
            for k in ("missing_box", "malformed_coordinates", "non_finite_box", "zero_area_box")
            if self.events[k]
        ]
        return "invalid_object: " + ",".join(bad)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def clean_annotation(xml_text: str | bytes, width: int, height: int) -> CleanResult:
    """Parse a VOC XML string and apply the fixed cleaning rules.

    ``width``/``height`` must be the real image size read with PIL, not the
    size written inside the XML.

    Rules, in order, for every <object> (same order as the notebook):
      1. name is stripped and lower-cased; an unknown name raises.
      2. missing <bndbox>, unparseable or non-finite coordinates -> invalid.
      3. xmin > xmax or ymin > ymax -> swap.
      4. clip to [0, width] x [0, height].
      5. zero area after clipping -> invalid.
    The caller drops the whole image if ``result.valid`` is False.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return CleanResult(boxes=[], n_objects=0, parse_error=str(exc))

    events: Counter = Counter()
    boxes: list[Box] = []
    objects = root.findall("object")

    for obj in objects:
        events["objects"] += 1
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_ID:
            raise UnknownClassError(f"Unknown class name {name!r}")
        cid = CLASS_TO_ID[name]

        bnd = obj.find("bndbox")
        if bnd is None:
            events["missing_box"] += 1
            continue
        try:
            x1 = float(bnd.findtext("xmin"))  # type: ignore[arg-type]
            y1 = float(bnd.findtext("ymin"))  # type: ignore[arg-type]
            x2 = float(bnd.findtext("xmax"))  # type: ignore[arg-type]
            y2 = float(bnd.findtext("ymax"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            events["malformed_coordinates"] += 1
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            events["non_finite_box"] += 1
            continue

        if x1 > x2 or y1 > y2:
            events["reversed_box_swapped"] += 1
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

        original = (x1, y1, x2, y2)
        x1, x2 = _clip(x1, 0, width), _clip(x2, 0, width)
        y1, y2 = _clip(y1, 0, height), _clip(y2, 0, height)
        if (x1, y1, x2, y2) != original:
            events["clipped_box"] += 1

        if x2 <= x1 or y2 <= y1:
            events["zero_area_box"] += 1
            continue

        boxes.append(Box(cid, x1, y1, x2, y2))

    return CleanResult(boxes=boxes, n_objects=len(objects), events=events)
