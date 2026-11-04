"""Immutable NumPy prior arrays; mmap avoids per-step ZIP decompression and bulk RAM copies."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from prior_maths import CACHE_FORMAT

FIELDS = ("depth", "normal", "mask")


def load(marker: Path) -> dict[str, Any]:
    info = json.loads(marker.read_text())
    if info["format"] != CACHE_FORMAT:
        raise ValueError("prior cache format changed")
    arrays = {}
    for name in FIELDS:
        # Copy-on-write mappings expose writable NumPy buffers to torch without
        # allowing a consumer to alter the immutable on-disk predictions.
        array = np.load(marker.with_suffix(f".{name}.npy"), mmap_mode="c", allow_pickle=False)
        expected = info["arrays"][name]
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError(f"prior cache {name} header changed")
        arrays[name] = array
    return arrays


def complete(marker: Path) -> bool:
    try:
        load(marker)
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return True


def write(root: Path, key: str, values: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f"{key}.json"
    info: dict[str, Any] = {"format": CACHE_FORMAT, "arrays": {}}
    for name in FIELDS:
        array = np.asarray(values[name], dtype=bool if name == "mask" else np.float32)
        destination = root / f"{key}.{name}.npy"
        with tempfile.NamedTemporaryFile(dir=root, prefix=".prior-", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        info["arrays"][name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    # Commit marker last: interrupted entries are never accepted as cache hits.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".prior-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(info, handle)
    try:
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
    return marker
