"""Index the raw UATD data: for each folder, the paired stems, XML text and image size.

Two sources produce the same index:

* :func:`index_local` reads an extracted copy (Kaggle mirror or figshare unzip).
* :func:`index_remote` reads the figshare zips over HTTP range requests, fetching
  only the XML files and the first few KB of each BMP (enough for its size).
  This avoids downloading ~4.7 GB just to rebuild or check the manifests.

Index format::

    {folder: {stem: {"width": int, "height": int, "xml": str,
                     "image": str, "annotation": str}}}

``image``/``annotation`` are paths (local) or zip member names (remote).
"""

from __future__ import annotations

import gzip
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

from PIL import Image

from uatd.constants import (
    ANNOTATION_DIR_NAMES,
    FIGSHARE_URLS,
    IMAGE_DIR_NAMES,
    IMAGE_EXTS,
    RAW_FOLDERS,
)

Index = dict[str, dict[str, dict]]


# --------------------------------------------------------------------------- local


def find_data_dirs(folder: Path) -> tuple[Path, Path]:
    """Locate the image and annotation directories like the notebook does.

    The first directory (in ``rglob`` order) whose lower-cased name is in
    IMAGE_DIR_NAMES / ANNOTATION_DIR_NAMES wins. Falls back to ``folder`` itself.
    """
    img_dir: Path | None = None
    ann_dir: Path | None = None
    for p in folder.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name in IMAGE_DIR_NAMES and img_dir is None:
            img_dir = p
        elif name in ANNOTATION_DIR_NAMES and ann_dir is None:
            ann_dir = p
    if img_dir is None or ann_dir is None:
        return folder, folder
    return img_dir, ann_dir


def _index_files(directory: Path, exts: Iterable[str]) -> dict[str, Path]:
    exts = {e.lower() for e in exts}
    return {p.stem: p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts}


def locate_pairs(raw_root: Path, folder: str) -> dict[str, tuple[Path, Path]]:
    """Return ``{stem: (image_path, xml_path)}`` for stems present in both."""
    split_path = raw_root / folder
    if not split_path.is_dir():
        raise FileNotFoundError(f"Could not find {folder} in {raw_root}")
    img_dir, ann_dir = find_data_dirs(split_path)
    imgs = _index_files(img_dir, IMAGE_EXTS)
    xmls = _index_files(ann_dir, {".xml"})
    return {s: (imgs[s], xmls[s]) for s in sorted(imgs.keys() & xmls.keys())}


def _local_entry(pair: tuple[Path, Path]) -> dict:
    img, xml = pair
    with Image.open(img) as im:  # reads the header only
        w, h = im.size
    return {
        "width": w,
        "height": h,
        "xml": xml.read_text(encoding="utf-8", errors="replace"),
        "image": str(img),
        "annotation": str(xml),
    }


def index_local(raw_root: Path, folders: Iterable[str] = RAW_FOLDERS, workers: int = 16) -> Index:
    """Threads, because on network-mounted storage (Kaggle input) each small read waits
    on latency; results keep the sorted stem order of ``locate_pairs``."""
    index: Index = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for folder in folders:
            pairs = locate_pairs(raw_root, folder)
            index[folder] = dict(zip(pairs, pool.map(_local_entry, pairs.values()), strict=True))
    return index


# -------------------------------------------------------------------------- remote


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (follows redirects)."""

    def __init__(self, url: str, retries: int = 5, timeout: float = 60.0) -> None:
        self.url, self.retries, self.timeout = url, retries, timeout
        self.pos = 0
        with self._get(0, 0) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _get(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        for attempt in range(self.retries):
            try:
                with self._get(self.pos, end) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        n = len(data)
        buf[:n] = data
        self.pos += n
        return n


def _open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 16))


def _zip_pairs(names: Iterable[str]) -> dict[str, tuple[str, str]]:
    imgs: dict[str, str] = {}
    xmls: dict[str, str] = {}
    for n in names:
        p = PurePosixPath(n)
        if n.endswith("/") or len(p.parts) < 2:
            continue
        parent = p.parent.name.lower()
        suffix = p.suffix.lower()
        if parent in IMAGE_DIR_NAMES and suffix in IMAGE_EXTS:
            imgs[p.stem] = n
        elif parent in ANNOTATION_DIR_NAMES and suffix == ".xml":
            xmls[p.stem] = n
    return {s: (imgs[s], xmls[s]) for s in sorted(imgs.keys() & xmls.keys())}


def _read_chunk(url: str, chunk: list[tuple[str, str, str]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with _open_remote_zip(url) as zf:
        for stem, img_name, xml_name in chunk:
            xml = zf.read(xml_name).decode("utf-8", errors="replace")
            with zf.open(img_name) as fh:
                head = fh.read(8192)
            with Image.open(io.BytesIO(head)) as im:
                w, h = im.size
            out[stem] = {
                "width": w,
                "height": h,
                "xml": xml,
                "image": img_name,
                "annotation": xml_name,
            }
    return out


def index_remote(
    folders: Iterable[str] = RAW_FOLDERS,
    workers: int = 16,
    log: Callable[[str], None] = print,
) -> Index:
    index: Index = {}
    for folder in folders:
        url = FIGSHARE_URLS[folder]
        with _open_remote_zip(url) as zf:
            pairs = _zip_pairs(zf.namelist())
        items = [(s, i, x) for s, (i, x) in pairs.items()]
        chunks = [items[k::workers] for k in range(workers)]
        log(f"{folder}: {len(items)} pairs, fetching XML + image headers ...")
        t0 = time.time()
        entries: dict[str, dict] = {}
        with ThreadPoolExecutor(workers) as ex:
            for part in ex.map(partial(_read_chunk, url), chunks):
                entries.update(part)
        index[folder] = {s: entries[s] for s in sorted(entries)}
        log(f"{folder}: done in {time.time() - t0:.0f}s")
    return index


# --------------------------------------------------------------------------- cache


def save_index(index: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(index, f)


def load_index(path: Path) -> Index:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
