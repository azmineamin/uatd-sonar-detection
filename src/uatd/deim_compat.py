"""Run the official DEIM ``train.py`` with a torchvision compatibility shim.

    python -m uatd.deim_compat <deim_dir> <train.py arguments...>

DEIM's custom transforms (``engine/data/transforms/_transforms.py``) implement the
torchvision v2 hooks under their old private names ``_transform`` / ``_get_params``.
torchvision >= 0.21 calls the public names ``transform`` / ``make_params`` instead, so
the custom transforms raise ``NotImplementedError`` (Kaggle ships a newer torchvision).

The shim maps the public names to DEIM's own methods, for DEIM classes only, which is
exactly what older torchvision did. No DEIM file is modified and nothing else changes.
"""

from __future__ import annotations

import inspect
import runpy
import sys
from pathlib import Path


def patch_transforms() -> list[str]:
    import torchvision.transforms.v2 as T
    from engine.data.transforms import _transforms  # type: ignore[import-not-found]

    patched = []
    for name, cls in inspect.getmembers(_transforms, inspect.isclass):
        if not issubclass(cls, T.Transform) or cls.__module__ != _transforms.__name__:
            continue
        if "_transform" in cls.__dict__ and "transform" not in cls.__dict__:
            cls.transform = cls.__dict__["_transform"]
            patched.append(f"{name}.transform")
        if "_get_params" in cls.__dict__ and "make_params" not in cls.__dict__:
            cls.make_params = cls.__dict__["_get_params"]
            patched.append(f"{name}.make_params")
    return patched


def main() -> None:
    deim_dir = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(deim_dir))
    print(f"[deim_compat] mapped torchvision hooks: {', '.join(patch_transforms()) or 'none'}")
    sys.argv = [str(deim_dir / "train.py"), *sys.argv[2:]]
    runpy.run_path(str(deim_dir / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
