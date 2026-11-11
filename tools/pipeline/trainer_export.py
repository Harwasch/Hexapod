"""Bound the pinned trainer's PLY export; its training loop ignores returned bytes.

Delegate row encoding/filtering/SH order to gsplat, at most 65,536 rows at a time.
Only intercept file-writing PLY calls; other export consumers retain upstream behavior.
Publish atomically so the live watcher cannot mistake a paused write for a complete PLY.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def write(export: Callable[..., bytes], kwargs: dict[str, Any], *, chunk: int = 65536) -> None:
    target = Path(kwargs["save_to"])
    partial = target.with_suffix(target.suffix + ".part")
    count = 0
    offset = 0
    try:
        with partial.open("wb") as output:
            for start in range(0, len(kwargs["means"]), chunk):
                arguments = {**kwargs, "save_to": None}
                arguments.update({key: kwargs[key][start : start + chunk] for key in FIELDS})
                encoded = export(**arguments)
                header, marker, rows = encoded.partition(b"end_header\n")
                if not marker:
                    raise ValueError("gsplat PLY export header changed")
                lines = header.splitlines(keepends=True)
                vertex = next(line for line in lines if line.startswith(b"element vertex "))
                count += int(vertex.split()[-1])
                if start == 0:
                    offset = header.index(vertex) + len(b"element vertex ")
                    output.write(header.replace(vertex, b"element vertex " + b"0" * 20 + b"\n"))
                    output.write(marker)
                output.write(rows)
            if not offset:
                raise ValueError("cannot export an empty training model")
            output.seek(offset)
            output.write(f"{count:020d}".encode())
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
