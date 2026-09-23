"""What Lane 1 does to gaussians: read them, normalise them, and derive from them.

Everything here is pure: a path or an array in, an array or a plain document out. The
stages in `stages.py` are the thin layer that resolves artifacts and writes files, so
that what is worth testing directly can be.

Three shapes arrive at this project and one leaves it:

* a **3DGS PLY** -- Scaniverse, Polycam, Postshot, Luma, OpenSplat and gsplat all write
  one, with `f_dc_*` spherical-harmonic DC terms and a logit `opacity`;
* a **`.spz`** -- what Scaniverse exports natively, and what `splat_tiles.pack_spz` already
  writes, so ingesting it is `unpack_spz` and nothing else;
* a PLY that uses **`red`/`green`/`blue`/`alpha`** instead, which some exporters do;

and out of all three comes `canonical.ply`: binary little-endian, exactly the fourteen
properties `splat_tiles.convert` reads, in one fixed order. Both lanes converge on that
file, which is why one `package` implementation serves Lane 1 and Lane 2 alike.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from captures_bridge import SplatFormatError, read_ply, sigmoid, unpack_spz

__all__ = [
    "CANONICAL_PROPERTIES",
    "GroundSample",
    "Splat",
    "ground_samples",
    "read_splat",
    "render_thumbnail",
    "write_ply",
]

F32 = npt.NDArray[np.float32]

#: Exactly what `splat_tiles.convert` reads, in the order `canonical.ply` writes them.
CANONICAL_PROPERTIES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)

#: The zeroth spherical-harmonic coefficient: colour = SH_C0 * f_dc + 0.5.
SH_C0 = 0.28209479177387814

#: Vertex-colour property names, and the SH DC term each one becomes.
_COLOUR_ALIASES: Mapping[str, str] = {
    "red": "f_dc_0",
    "green": "f_dc_1",
    "blue": "f_dc_2",
}

#: The clamp on a 0..1 alpha before it is turned into a logit. Same reasoning as
#: `splat_tiles.SPZ_ALPHA_EPS`: a quarter of a byte step, so 0 and 255 survive the round
#: trip back to bytes.
ALPHA_EPS = 0.25 / 255.0

SourceFormat = Literal["ply", "spz"]

#: What `read_splat` will open, in the order a directory of several files is preferred in.
SPLAT_SUFFIXES: tuple[str, ...] = (".spz", ".ply")


@dataclass(frozen=True)
class Splat:
    """A normalised splat, and what the file it came from turned out to be."""

    columns: dict[str, F32]
    source_format: SourceFormat
    source_name: str
    source_bytes: int
    source_checksum: str
    #: Every property the file declared, in file order -- including the ones dropped.
    properties_in: tuple[str, ...]
    #: Properties the file had that `canonical.ply` does not carry (`f_rest_*`, normals).
    dropped: tuple[str, ...]
    #: Gaussians whose position or opacity was not finite. Kept, not removed: `convert`
    #: already neutralises them, and removing them here would make two stages disagree
    #: about how many gaussians the capture has.
    non_finite: int

    @property
    def count(self) -> int:
        return int(self.columns["x"].shape[0])

    @property
    def xyz(self) -> F32:
        return np.stack([self.columns["x"], self.columns["y"], self.columns["z"]], axis=1)

    @property
    def alpha(self) -> F32:
        return _f32(sigmoid(self.columns["opacity"]))

    def bbox(self) -> tuple[list[float], list[float]]:
        """Local-frame (east, north, up) minimum and maximum, over the finite gaussians."""
        xyz = self.xyz
        finite = np.isfinite(xyz).all(axis=1)
        if not bool(finite.any()):
            raise SplatFormatError(
                f"{self.source_name} has {self.count} gaussians and not one of them has a "
                f"finite position"
            )
        kept = xyz[finite]
        return (
            [float(v) for v in kept.min(axis=0)],
            [float(v) for v in kept.max(axis=0)],
        )


def _f32(values: Any) -> F32:
    return np.ascontiguousarray(values, dtype=np.float32)


def _logit(alpha: F32) -> F32:
    clamped = np.clip(alpha, ALPHA_EPS, 1.0 - ALPHA_EPS)
    return _f32(np.log(clamped / (1.0 - clamped)))


def pick_splat_file(directory: Path) -> Path:
    """The one file in an upload directory this lane can read.

    Sorted, so a directory with two `.ply` files in it picks the same one on every run
    rather than whichever the filesystem happened to hand over first.
    """
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    for suffix in SPLAT_SUFFIXES:
        for candidate in files:
            if candidate.suffix.lower() == suffix:
                return candidate
    found = ", ".join(sorted({p.suffix.lower() or "(no extension)" for p in files})) or "nothing"
    raise SplatFormatError(
        f"no splat file in the upload: this lane reads {', '.join(SPLAT_SUFFIXES)} and the "
        f"upload holds {len(files)} file(s) with extensions: {found}. A capture that needs "
        f"reconstructing belongs in the photo-reconstruct recipe"
    )


def read_splat(path: Path) -> Splat:
    """Read a `.ply` or `.spz` and normalise it to the canonical property set."""
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    checksum = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if suffix == ".spz":
        source: SourceFormat = "spz"
        data: dict[str, F32] = {k: _f32(v) for k, v in unpack_spz(raw).items()}
    elif suffix == ".ply":
        source = "ply"
        data = {k: _f32(v) for k, v in read_ply(path).items()}
    else:
        what = suffix or "a file with no extension"
        raise SplatFormatError(
            f"{path.name}: this lane reads {', '.join(SPLAT_SUFFIXES)}, not {what}"
        )
    properties_in = tuple(data)
    columns = _normalise(path.name, data)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    non_finite = int((~(np.isfinite(xyz).all(axis=1) & np.isfinite(columns["opacity"]))).sum())
    return Splat(
        columns=columns,
        source_format=source,
        source_name=path.name,
        source_bytes=len(raw),
        source_checksum=checksum,
        properties_in=properties_in,
        dropped=tuple(name for name in properties_in if name not in columns),
        non_finite=non_finite,
    )


def _normalise(name: str, data: dict[str, F32]) -> dict[str, F32]:
    """The canonical fourteen, or a refusal that names the properties that are missing."""
    columns = dict(data)
    if "f_dc_0" not in columns and all(alias in columns for alias in _COLOUR_ALIASES):
        # Vertex colours are 0..255 bytes; the DC term is the inverse of
        # `colour = SH_C0 * f_dc + 0.5`, which is what every 3DGS renderer assumes.
        for alias, target in _COLOUR_ALIASES.items():
            columns[target] = _f32((columns[alias] / 255.0 - 0.5) / SH_C0)
    if "opacity" not in columns and "alpha" in columns:
        columns["opacity"] = _logit(_f32(columns["alpha"] / 255.0))
    missing = [prop for prop in CANONICAL_PROPERTIES if prop not in columns]
    if missing:
        have = ", ".join(sorted(data)[:16]) or "none"
        extra = "" if len(data) <= 16 else f" (and {len(data) - 16} more)"
        raise SplatFormatError(
            f"{name} is not a gaussian splat this lane can read: it has no "
            f"{', '.join(missing)}. Its properties are: {have}{extra}. A PlayCanvas or "
            f"SuperSplat compressed export (packed_position, packed_rotation, ...) needs "
            f"the `ply_compressed` normalize impl, which is not built yet"
        )
    counts = {prop: int(columns[prop].shape[0]) for prop in CANONICAL_PROPERTIES}
    if len(set(counts.values())) != 1:
        raise SplatFormatError(f"{name}: its properties have different lengths: {counts}")
    return {prop: columns[prop] for prop in CANONICAL_PROPERTIES}


def write_ply(path: Path, columns: Mapping[str, F32]) -> int:
    """Write `canonical.ply`: binary little-endian, the fourteen properties, in order.

    Byte-identical for identical arrays -- there is no timestamp, no generator string and
    no dictionary iteration order in it.
    """
    count = int(columns["x"].shape[0])
    dtype = np.dtype([(prop, "<f4") for prop in CANONICAL_PROPERTIES])
    record = np.empty(count, dtype=dtype)
    for prop in CANONICAL_PROPERTIES:
        record[prop] = columns[prop]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {prop}" for prop in CANONICAL_PROPERTIES]
    header.append("end_header")
    payload = ("\n".join(header) + "\n").encode("ascii") + record.tobytes()
    path.write_bytes(payload)
    return len(payload)


# ---------------------------------------------------------------------------------------
# Derived products: the thumbnail and the ground samples
# ---------------------------------------------------------------------------------------

#: Metres per degree of latitude. Good to about 0.5 % anywhere, which is well inside the
#: uncertainty of a hand-placed capture and is the same constant the worker registers with.
METRES_PER_DEGREE = 111_320.0


def render_thumbnail(
    splat: Splat,
    path: Path,
    *,
    size: int = 512,
    alpha_min: float = 0.1,
    quality: int = 82,
    thicken: int = 3,
) -> dict[str, int]:
    """An elevation view of the splat, as a JPEG: east across, up the page.

    Deliberately the simplest thing that is recognisably the capture: the nearest
    sufficiently opaque gaussian wins each pixel (a z-buffer, resolved with a lexsort so
    it does not depend on numpy's undefined ordering for duplicate fancy-index writes),
    and its SH DC colour is written flat. No splatting, no blending, no sorting by
    opacity -- a thumbnail in a list of sites, not a render. A `thicken`-wide maximum
    filter then grows each point into a dot, which is what stops a 12,000-gaussian
    capture looking like a dusting of single pixels; supersampling instead would average
    every point against the background and come out darker than the capture is.
    """
    xyz = splat.xyz
    alpha = splat.alpha
    visible = np.isfinite(xyz).all(axis=1) & np.isfinite(alpha) & (alpha >= alpha_min)
    drawn = size
    image = np.zeros((drawn, drawn, 3), dtype=np.uint8)
    kept = int(visible.sum())
    if kept:
        points = xyz[visible]
        colour = np.clip(
            SH_C0
            * np.stack(
                [splat.columns[f"f_dc_{i}"][visible] for i in range(3)],
                axis=1,
            )
            + 0.5,
            0.0,
            1.0,
        )
        across, up, depth = points[:, 0], points[:, 2], points[:, 1]
        px = _to_pixels(across, drawn, flip=False)
        py = _to_pixels(up, drawn, flip=True)
        flat = py * drawn + px
        # Primary key the pixel, secondary key the depth: the first row of each pixel's
        # run is its nearest gaussian.
        order = np.lexsort((depth, flat))
        first = np.unique(flat[order], return_index=True)[1]
        chosen = order[first]
        image.reshape(-1, 3)[flat[chosen]] = np.round(colour[chosen] * 255.0).astype(np.uint8)
    picture = Image.fromarray(image, mode="RGB")
    if thicken > 1:
        picture = picture.filter(ImageFilter.MaxFilter(thicken))
    picture.save(path, format="JPEG", quality=quality, optimize=True)
    return {"gaussians": kept, "size": size, "bytes": path.stat().st_size}


def _to_pixels(values: F32, size: int, *, flip: bool) -> npt.NDArray[np.intp]:
    low, high = float(values.min()), float(values.max())
    span = high - low
    # A capture with no extent along an axis (a single gaussian, a flat plane) projects to
    # the middle of the image rather than dividing by zero.
    scaled = np.full(values.shape, 0.5) if span <= 0 else (values - low) / span
    if flip:
        scaled = 1.0 - scaled
    margin = 0.04
    pixels = np.floor((margin + scaled * (1 - 2 * margin)) * size)
    return np.clip(pixels, 0, size - 1).astype(np.intp)


@dataclass(frozen=True)
class GroundSample:
    """One grid cell's ground height, in the capture's own frame and on the globe."""

    lon: float
    lat: float
    z: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {"lon": self.lon, "lat": self.lat, "z": round(self.z, 3), "n": self.n}


def ground_samples(
    splat: Splat,
    *,
    lat: float,
    lon: float,
    cell_m: float = 2.0,
    percentile: float = 5.0,
    min_points: int = 8,
    max_cells: int = 64,
) -> tuple[GroundSample, ...]:
    """The capture's own ground height per grid cell, richest cells first.

    The height is a low percentile of the gaussians in the cell rather than the minimum,
    because one stray gaussian under the floor would otherwise become the ground. Cells
    are ranked by how many gaussians they hold and tie-broken by position, so the set is
    the same on every run -- unlike `tools/captures/ground_samples.py`, which samples
    cells with a seeded RNG.
    """
    xyz = splat.xyz
    finite = np.isfinite(xyz).all(axis=1)
    points = xyz[finite]
    if points.shape[0] == 0 or cell_m <= 0:
        return ()
    east, north, up = points[:, 0], points[:, 1], points[:, 2]
    ix = np.floor((east - east.min()) / cell_m).astype(np.int64)
    iy = np.floor((north - north.min()) / cell_m).astype(np.int64)
    key = ix * (int(iy.max()) + 1) + iy
    order = np.argsort(key, kind="stable")
    keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    # -count first, then the cell key: deterministic, and the densest cells are the ones
    # with a ground in them rather than a canopy edge.
    ranked = np.lexsort((keys, -counts))
    samples: list[GroundSample] = []
    for index in ranked:
        if len(samples) >= max_cells:
            break
        if int(counts[index]) < min_points:
            continue
        rows = order[starts[index] : starts[index] + counts[index]]
        samples.append(
            GroundSample(
                lon=_offset_lon(lon, lat, float(east[rows].mean())),
                lat=_offset_lat(lat, float(north[rows].mean())),
                z=float(np.percentile(up[rows], percentile)),
                n=int(counts[index]),
            )
        )
    return tuple(samples)


def _offset_lat(lat: float, north_m: float) -> float:
    return lat + north_m / METRES_PER_DEGREE


def _offset_lon(lon: float, lat: float, east_m: float) -> float:
    scale = max(math.cos(math.radians(lat)), 1e-6)
    return lon + east_m / (METRES_PER_DEGREE * scale)


def median_of(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None
