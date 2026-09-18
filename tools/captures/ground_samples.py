"""Sample the model's ground height on a grid, to compare with the globe's terrain.

Drone EXIF altitudes are rarely on the WGS84 ellipsoid the globe uses, so a reconstruction
lands a few metres above or below the terrain. This prints grid cells with the model's
ground height (a low percentile of the points in the cell) as JSON; the console's terrain
is sampled at the same longitude/latitude and the median difference becomes the site's
height offset (see build_site.py --height-offset).

Usage:
    python ground_samples.py model.laz [--cell 8] [--max-cells 24]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("laz", type=Path)
    parser.add_argument("--cell", type=float, default=8.0)
    parser.add_argument("--max-cells", type=int, default=24)
    args = parser.parse_args()
    cloud = laspy.read(str(args.laz))
    x, y, z = np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)
    ix = np.floor((x - x.min()) / args.cell).astype(int)
    iy = np.floor((y - y.min()) / args.cell).astype(int)
    key = ix * 100000 + iy
    order = np.argsort(key)
    _keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    rich = np.argsort(-counts)[: args.max_cells * 4]
    rng = np.random.default_rng(3)
    chosen = rng.choice(rich, min(args.max_cells, rich.size), replace=False)
    transformer = Transformer.from_crs(
        cloud.header.parse_crs(), CRS.from_epsg(4326), always_xy=True
    )
    samples = []
    for cell in chosen:
        sel = order[starts[cell] : starts[cell] + counts[cell]]
        ground = float(np.percentile(z[sel], 5))
        lon, lat = transformer.transform(float(x[sel].mean()), float(y[sel].mean()))
        samples.append({"lon": lon, "lat": lat, "z": round(ground, 2), "n": int(counts[cell])})
    print(json.dumps(samples))


if __name__ == "__main__":
    main()
