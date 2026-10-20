"""A synthetic site 10.4 km across, so per-tile origins have a fixture to be held to.

SPZ stores positions as 24-bit fixed point: +/-2047.99976 m from a tile's frame at 12
fractional bits. Every scan this repo had packed fitted, so nothing exercised what happens
past it -- which, until per-tile origins, was a silent wrap (docs/DECISIONS/
0011-splat-tile-origins.md). This scene is the smallest one that does: a track 10.4 km
east-west, a spur 1.8 km north from its east end, and three small dense plots, one at the
site origin, one 2.6 km west, one 2.6 km east and 1.8 km north, which a viewer can look for.

Deterministic with no RNG: every gaussian is placed by formula (the plots on golden-angle
discs) and snapped to the 1/4096 m grid, and every coordinate is under 4096 m or a whole
number, so float32 holds each one exactly.

Usage:
    python synthetic_far.py OUT_DIR             # the fixture: OUT_DIR/splat, OUT_DIR/source
    python synthetic_far.py report TILES_DIR    # per tile: origin, bits, decode error
"""

from __future__ import annotations

import math

import numpy as np

from splat_render import Splats

LAT, LON, HEIGHT = 28.0389, -82.6966, 0.0
#: Gaussians a tile: small, so a 10.4 km site is a tree several levels deep.
TILE_GAUSSIANS = 1000
#: Where the plots stand (east, north, up), metres in the site frame.
PLOTS: dict[str, tuple[float, float, float]] = {
    "origin": (0.0, 0.0, 1.0),
    "west": (-2600.0, 0.0, 1.0),
    "north-east": (2600.0, 1800.0, 1.0),
}
PLOT_GAUSSIANS = 600
PLOT_RADIUS_M = 3.0
PLOT_COLOURS: dict[str, tuple[float, float, float]] = {
    "origin": (0.95, 0.15, 0.15),
    "west": (0.15, 0.85, 0.2),
    "north-east": (0.2, 0.4, 0.95),
}
TRACK_COLOUR = (0.5, 0.5, 0.5)
SCALE_M = 0.4
OPACITY = 0.9
#: SPZ's grid at 12 fractional bits.
QUANTUM = 1.0 / 4096


def _disc(centre: tuple[float, float, float]) -> np.ndarray:
    """`PLOT_GAUSSIANS` points on a golden-angle (Vogel) disc of `PLOT_RADIUS_M`."""
    k = np.arange(PLOT_GAUSSIANS, dtype=np.float64)
    radius = PLOT_RADIUS_M * np.sqrt((k + 0.5) / PLOT_GAUSSIANS)
    angle = k * math.pi * (3.0 - math.sqrt(5.0))
    x, y, z = centre
    return np.stack(
        [x + radius * np.cos(angle), y + radius * np.sin(angle), np.full_like(k, z)], axis=1
    )


def scene() -> Splats:
    """The track, the spur and the plots, in that order."""
    east = np.arange(-5200.0, 5200.0 + 1.0, 2.0)
    track = np.stack([east, np.zeros_like(east), np.zeros_like(east)], axis=1)
    north = np.arange(2.0, 1800.0 + 1.0, 2.0)
    spur = np.stack([np.full_like(north, 2600.0), north, np.zeros_like(north)], axis=1)
    discs = [_disc(centre) for centre in PLOTS.values()]
    positions = np.concatenate([track, spur, *discs])
    positions = np.round(positions / QUANTUM) * QUANTUM
    colours = np.concatenate(
        [
            np.tile(TRACK_COLOUR, (track.shape[0] + spur.shape[0], 1)),
            *(np.tile(PLOT_COLOURS[name], (PLOT_GAUSSIANS, 1)) for name in PLOTS),
        ]
    )
    n = positions.shape[0]
    return Splats(
        positions.astype(np.float32),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        np.full((n, 3), SCALE_M, np.float32),
        colours.astype(np.float32),
        np.full(n, OPACITY, np.float32),
    )
