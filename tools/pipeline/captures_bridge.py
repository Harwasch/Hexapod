"""The seam to `tools/captures`, which already knows how to package a splat.

`tools/captures/splat_tiles.convert()` writes the SPZ/`KHR_gaussian_splatting` tileset the
console renders, byte-stably, and this sprint automates around that code rather than
replacing it. The two directories are separate uv projects, and the sibling declares
`package = false` -- it is a set of scripts, not a distribution -- so it cannot be a path
dependency and there is nothing to `pip install`. Copying the packer here would fork the
one piece of code with a byte-identity gate on it, which is the worst of the options.

So the import is a `sys.path` insertion, in this one module, computed from this file's
location. Two things keep it honest rather than hidden:

  * `mypy_path = ["../captures"]` in pyproject.toml, so mypy resolves `splat_tiles` to the
    real file and type-checks every call into it (with `follow_imports = "silent"`, since
    the sibling project does not run mypy and its errors are not this project's to fix);
  * `tests/test_captures_bridge.py`, which runs the real `convert()` on a generated PLY and
    asserts the files it writes are exactly the ones the `splat/` artifact declares -- so a
    stubbed `package` stage and the real one cannot drift apart unnoticed.

A8 replaces the stub with a call to `splat_tiles_convert()`. Nothing else changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["CAPTURES_DIR", "splat_tiles_convert"]

CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"


def _ensure_importable() -> None:
    path = str(CAPTURES_DIR)
    if not CAPTURES_DIR.is_dir():
        raise ModuleNotFoundError(
            f"tools/captures is not next to tools/pipeline (looked in {CAPTURES_DIR})"
        )
    if path not in sys.path:
        sys.path.insert(0, path)


def splat_tiles_convert(
    ply: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    height: float,
    max_gaussians: int = 400_000,
    opacity_min: float = 0.02,
    geometric_error: float = 2.0,
) -> dict[str, float | int]:
    """Call the sibling project's packer, unchanged, and return its statistics."""
    _ensure_importable()
    from splat_tiles import convert

    return convert(ply, out_dir, lat, lon, height, max_gaussians, opacity_min, geometric_error)
