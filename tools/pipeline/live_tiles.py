"""Bounded child of the live watcher: package with the scan's existing LOD writer.

The watcher can terminate this process at the next snapshot deadline. A Python thread
cannot interrupt a slow numpy call safely. Input is read-only; output is staged outside
the checkpoint tree until the whole conversion succeeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from captures_bridge import splat_tiles_convert


def package(ply: Path, directory: Path) -> dict[str, float | int]:
    # Only the tileset root is georeferenced; GLB positions and SH stay solve-local.
    # The console drops that root transform, retaining all child translations.
    stats = splat_tiles_convert(ply, directory, lat=0, lon=0, height=0)
    return {
        "gaussians": stats["gaussians"],
        "tiles": stats["tiles"],
        "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(package(Path(sys.argv[1]), Path(sys.argv[2]))))
