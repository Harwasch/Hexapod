"""The stages after training, on a splat that is never held whole: place, thumbnail, ground.

`gaussians.py` says what these do to a splat in memory -- `orient`, `transform`,
`render_thumbnail`, `ground_samples` -- and stays the definition: the tests hold this
module to it. This module does the same to a splat on disk, a chunk at a time
(`splat_io`), so that the 2 GB worker ingests, places, thumbnails and samples a splat of
any size in the same memory: those stages measured ~0.75 GB a million gaussians when they
loaded the file (apps/api/app/worker/README.md), which is what held training to 2M. (A
`.spz` upload is still unpacked whole -- see `open_splat` -- and then streamed.)

**The same answer, not a similar one.** Every per-gaussian operation is element-wise
(`gaussians.transform` rounds each row identically however many rows it is given), so
it is applied chunk by chunk. Every whole-splat statistic -- the 2nd/98th percentile
footprint `orient` recentres on, each ground cell's 5th-percentile height, the 5th/95th
percentile frame of the thumbnail -- is numpy's own `np.percentile`, computed exactly in
passes over the chunks (`outofcore.py`). The thumbnail's z-buffer keeps the nearest
gaussian per pixel with the earliest row winning a tie, which is what the whole-splat
lexsort chooses. So `canonical.ply`, `thumbnail.jpg` and the ground samples' heights are
byte-identical to the in-memory path; the one exception is a ground sample's longitude
and latitude, whose cell mean is accumulated in float64 here where `np.mean` sums float32
pairwise (a difference of order 1e-7 m on a cell a few metres across; see
`ground_samples`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

import gaussians
import outofcore
import splat_io
from captures_bridge import SplatFormatError, sigmoid

__all__ = [
    "ArraySource",
    "PlySource",
    "SplatSource",
    "Step",
    "Written",
    "ground_samples",
    "open_splat",
    "orient_to",
    "thumbnail",
]

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Columns = dict[str, F32]

CANONICAL = gaussians.CANONICAL_PROPERTIES
#: The file properties a canonical column can come from (`gaussians._normalise`).
_ALIASES = ("red", "green", "blue", "alpha")


# ---------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------


class SplatSource:
    """A splat read a chunk at a time, as the canonical columns `gaussians.read_splat`
    would give for the same rows."""

    count: int
    source_format: gaussians.SourceFormat
    source_name: str
    source_bytes: int
    properties_in: tuple[str, ...]
    dropped: tuple[str, ...]
    chunk: int = splat_io.CHUNK

    def read(self, start: int, stop: int) -> Columns:
        raise NotImplementedError

    def checksum(self) -> str:
        raise NotImplementedError

    def chunks(self, chunk: int | None = None) -> Iterator[tuple[int, Columns]]:
        for start, stop in splat_io.ranges(self.count, chunk or self.chunk):
            yield start, self.read(start, stop)


class PlySource(SplatSource):
    """A PLY on disk: rows read by range (`splat_io.SplatReader`) and normalised."""

    def __init__(self, path: Path, *, chunk: int = splat_io.CHUNK) -> None:
        self.reader = splat_io.SplatReader(path, chunk=chunk)
        self.chunk = self.reader.chunk
        self.count = self.reader.count
        self.source_format = "ply"
        self.source_name = path.name
        self.source_bytes = path.stat().st_size
        self.properties_in = self.reader.properties
        # Refuse now, by name, what `read_splat` would refuse: a PLY with no way to the
        # canonical fourteen.
        empty = {name: np.zeros(0, np.float32) for name in self.properties_in}
        kept = gaussians.normalise(path.name, empty)
        self.dropped = tuple(name for name in self.properties_in if name not in kept)
        self._raw = tuple(
            name for name in self.properties_in if name in CANONICAL or name in _ALIASES
        )

    def read(self, start: int, stop: int) -> Columns:
        return gaussians.normalise(self.source_name, self.reader.read(start, stop, self._raw))

    def checksum(self) -> str:
        return self.reader.checksum()


class ArraySource(SplatSource):
    """A splat already in memory (an unpacked `.spz`, a test's arrays), read by range."""

    def __init__(self, splat: gaussians.Splat, *, chunk: int = splat_io.CHUNK) -> None:
        self.splat = splat
        self.chunk = chunk
        self.count = splat.count
        self.source_format = splat.source_format
        self.source_name = splat.source_name
        self.source_bytes = splat.source_bytes
        self.properties_in = splat.properties_in
        self.dropped = splat.dropped

    def read(self, start: int, stop: int) -> Columns:
        return {name: self.splat.columns[name][start:stop] for name in CANONICAL}

    def checksum(self) -> str:
        return self.splat.source_checksum


def open_splat(path: Path, *, chunk: int = splat_io.CHUNK) -> SplatSource:
    """A `.ply` streamed from disk; a `.spz` unpacked whole, since SPZ is one gzip stream
    of column blocks (all positions, then all alphas, ...) and reading one row range means
    inflating everything before it. A phone's `.spz` is small: its PLY equivalent is
    what grows with a trained scene."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return PlySource(path, chunk=chunk)
    return ArraySource(gaussians.read_splat(path), chunk=chunk)


# ---------------------------------------------------------------------------------------
# Transforms, a chunk at a time
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One `gaussians.transform`: `x' = scale * rotation @ x + translation`."""

    rotation: F64
    translation: F64 | None = None
    scale: float = 1.0

    def apply(self, columns: Mapping[str, F32]) -> Columns:
        return gaussians.transform(columns, self.rotation, self.translation, self.scale)


def _apply(columns: Columns, steps: Sequence[Step]) -> Columns:
    for step in steps:
        columns = step.apply(columns)
    return columns


def _xyz(columns: Mapping[str, F32]) -> tuple[F32, F32, F32, npt.NDArray[np.bool_]]:
    x, y, z = columns["x"], columns["y"], columns["z"]
    return x, y, z, np.isfinite(x) & np.isfinite(y) & np.isfinite(z)


def _moved(source: SplatSource, steps: Sequence[Step]) -> Iterator[Columns]:
    for _, columns in source.chunks():
        yield _apply(columns, steps)


# ---------------------------------------------------------------------------------------
# Ground samples
# ---------------------------------------------------------------------------------------


def ground_samples(
    source: SplatSource,
    steps: Sequence[Step] = (),
    *,
    lat: float,
    lon: float,
    cell_m: float = 2.0,
    percentile: float = 5.0,
    min_points: int = 8,
    max_cells: int = 64,
    positions: bool = True,
) -> tuple[gaussians.GroundSample, ...]:
    """`gaussians.ground_samples` of the splat after `steps`, in passes over its chunks.

    The same cells, ranked the same way, with the same heights (`np.percentile` of each
    cell's up-coordinates, exactly). A cell's longitude and latitude come from the mean of
    its gaussians' east and north, which `gaussians.ground_samples` takes with `np.mean`
    over float32 -- pairwise float32 sums in an order only the whole array has. Here it is
    a float64 sum rounded to float32 at the end, which differs from that in the last bit
    or two of a float32: under a micrometre on a metres-wide cell. `positions=False` skips
    that pass for a caller that wants only the heights (`orient_to`'s recentring).
    """
    count = 0
    east_low = north_low = None
    for columns in _moved(source, steps):
        x, y, _, finite = _xyz(columns)
        if not bool(finite.any()):
            continue
        count += int(finite.sum())
        east_chunk, north_chunk = x[finite].min(), y[finite].min()
        east_low = east_chunk if east_low is None else min(east_low, east_chunk)
        north_low = north_chunk if north_low is None else min(north_low, north_chunk)
    if count == 0 or cell_m <= 0 or east_low is None or north_low is None:
        return ()

    def cells(columns: Columns) -> tuple[I64, I64, npt.NDArray[np.bool_]]:
        x, y, _, finite = _xyz(columns)
        east, north = x[finite], y[finite]
        ix = np.floor((east - east_low) / cell_m).astype(np.int64)
        iy = np.floor((north - north_low) / cell_m).astype(np.int64)
        return ix, iy, finite

    rows = 1
    for columns in _moved(source, steps):
        _, iy, _ = cells(columns)
        if iy.size:
            rows = max(rows, int(iy.max()) + 1)

    def keyed() -> Iterator[tuple[I64, None]]:
        for columns in _moved(source, steps):
            ix, iy, _ = cells(columns)
            yield ix * rows + iy, None

    # The densest cells, ties to the lower key: `np.lexsort((keys, -counts))` kept to the
    # first `max_cells` of those with `min_points`, partition by partition and merged.
    best_keys = np.zeros(0, dtype=np.int64)
    best_counts = np.zeros(0, dtype=np.int64)
    for keys, counts, _ in outofcore.group_counts(keyed, rows=count):
        enough = counts >= min_points
        best_keys = np.concatenate([best_keys, keys[enough]])
        best_counts = np.concatenate([best_counts, counts[enough]])
        ranked = np.lexsort((best_keys, -best_counts))[:max_cells]
        best_keys, best_counts = best_keys[ranked], best_counts[ranked]
    if best_keys.size == 0:
        return ()
    wanted = [int(key) for key in best_keys]

    def heights() -> Iterator[tuple[F32, I64]]:
        for columns in _moved(source, steps):
            ix, iy, finite = cells(columns)
            yield columns["z"][finite], ix * rows + iy

    found = outofcore.grouped_percentile(heights, wanted, [percentile])
    sums: dict[int, tuple[float, float]] = {}
    if positions:
        lookup = np.asarray(sorted(wanted), dtype=np.int64)
        east_sum = np.zeros(lookup.shape[0], dtype=np.float64)
        north_sum = np.zeros(lookup.shape[0], dtype=np.float64)
        for columns in _moved(source, steps):
            ix, iy, finite = cells(columns)
            key = ix * rows + iy
            where = np.minimum(np.searchsorted(lookup, key), lookup.shape[0] - 1)
            hit = lookup[where] == key
            east_sum += np.bincount(
                where[hit], weights=columns["x"][finite][hit], minlength=lookup.shape[0]
            )
            north_sum += np.bincount(
                where[hit], weights=columns["y"][finite][hit], minlength=lookup.shape[0]
            )
        sums = {
            int(k): (float(e), float(n))
            for k, e, n in zip(lookup, east_sum, north_sum, strict=True)
        }
    samples: list[gaussians.GroundSample] = []
    for cell, n in zip(wanted, best_counts, strict=True):
        east_mean = north_mean = 0.0
        if positions:
            east_total, north_total = sums[cell]
            east_mean = float(np.float32(east_total / int(n)))
            north_mean = float(np.float32(north_total / int(n)))
        samples.append(
            gaussians.GroundSample(
                lon=gaussians._offset_lon(lon, lat, east_mean),
                lat=gaussians._offset_lat(lat, north_mean),
                z=float(found[cell][0]),
                n=int(n),
            )
        )
    return tuple(samples)


# ---------------------------------------------------------------------------------------
# Orient and write: `place`, and Lane 1's ingest of a PLY
# ---------------------------------------------------------------------------------------


def _recentre(source: SplatSource, steps: Sequence[Step], cell_m: float) -> F64 | None:
    """`gaussians.orient`'s recentring translation for the splat after `steps`; None when
    no gaussian has a finite position (then `orient` moves nothing)."""

    def footprint() -> Iterator[tuple[F32, I64]]:
        for columns in _moved(source, steps):
            x, y, _, finite = _xyz(columns)
            count = int(finite.sum())
            yield (
                np.concatenate([x[finite], y[finite]]),
                np.repeat(np.arange(2, dtype=np.int64), count),
            )

    found = outofcore.grouped_percentile(footprint, [0, 1], [2.0, 98.0])
    if 0 not in found:
        return None
    translation = np.zeros(3)
    # As `orient`: np.percentile(xyz[finite, :2], q, axis=0), then the mid-point in float32.
    low = np.array([found[0][0], found[1][0]])
    high = np.array([found[0][1], found[1][1]])
    centre = (low + high) / 2.0
    translation[:2] = -centre
    cells = ground_samples(source, steps, lat=0.0, lon=0.0, cell_m=cell_m, positions=False)
    if cells:
        translation[2] = -float(
            np.percentile([cell.z for cell in cells], gaussians.BASE_CELL_PERCENTILE)
        )
    return translation


@dataclass(frozen=True)
class Written:
    """What was written: rows, bytes, the finite bounding box, the rows whose position or
    opacity was not finite as read, and the frame `orient_to` applied (None for a plain
    `transform_to`)."""

    count: int
    bytes: int
    low: list[float]
    high: list[float]
    non_finite: int
    frame: gaussians.Frame | None = None


def _write(
    source: SplatSource,
    out: Path,
    steps: Sequence[Step],
    on_chunk: Callable[[Columns], None] | None,
) -> Written:
    count = non_finite = 0
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    with splat_io.PlyWriter(out, CANONICAL, count=source.count) as writer:
        for _, read in source.chunks():
            # Counted as read, as `read_splat` counts `non_finite`: before any transform.
            _, _, _, finite_in = _xyz(read)
            non_finite += int((~(finite_in & np.isfinite(read["opacity"]))).sum())
            columns = _apply(read, steps)
            writer.append(columns)
            x, y, z, finite = _xyz(columns)
            count += int(x.shape[0])
            if bool(finite.any()):
                for index, values in enumerate((x, y, z)):
                    low[index] = min(low[index], float(values[finite].min()))
                    high[index] = max(high[index], float(values[finite].max()))
            if on_chunk is not None:
                on_chunk(columns)
    if not np.isfinite(low).all():
        # `Splat.bbox`'s refusal, after the file is written as it was before.
        raise SplatFormatError(
            f"{source.source_name} has {count} gaussians and not one of them has a finite position"
        )
    return Written(
        count=count,
        bytes=out.stat().st_size,
        low=[float(v) for v in low],
        high=[float(v) for v in high],
        non_finite=non_finite,
    )


def transform_to(
    source: SplatSource,
    out: Path,
    steps: Sequence[Step],
    *,
    on_chunk: Callable[[Columns], None] | None = None,
) -> Written:
    """`write_ply(transform(...))` of every step in turn, a chunk at a time."""
    return _write(source, out, steps, on_chunk)


def orient_to(
    source: SplatSource,
    out: Path,
    *,
    before: Sequence[Step] = (),
    up_axis: str | None = None,
    heading_deg: float = 0.0,
    recentre: bool = True,
    cell_m: float = 2.0,
    on_chunk: Callable[[Columns], None] | None = None,
) -> Written:
    """`gaussians.orient` of the splat after `before`, written to `out` as canonical.ply.

    The same validation, rotation, recentring and `Frame` as `orient`; the rows are those
    `write_ply(orient(...)[0].columns)` writes, byte for byte. `on_chunk` sees each
    written chunk.
    """
    axis = up_axis if up_axis is not None else gaussians.default_up_axis(source.source_format)
    if axis not in gaussians.UP_AXES:
        raise ValueError(
            f"upAxis {axis!r} is not one of {', '.join(gaussians.UP_AXES)}. It names the axis "
            f"of the uploaded file that points up: y for Scaniverse/SPZ, -y for 3DGS/COLMAP .ply"
        )
    origin = "capture" if up_axis is not None else f"format-default ({source.source_format})"
    heading = float(heading_deg)
    if not math.isfinite(heading):
        raise ValueError(f"headingDeg must be a finite number of degrees, not {heading_deg!r}")
    rotation = gaussians.heading_rotation(heading) @ gaussians.UP_AXES[axis]
    steps = [*before, Step(rotation)]
    translation = np.zeros(3)
    if recentre:
        moved = _recentre(source, steps, cell_m)
        if moved is not None:
            translation = moved
            # `orient` applies the translation as a second transform, and only when there
            # was a finite gaussian to measure it from: an identity step is not a no-op on
            # a non-finite row (0 * inf), so it is not added where `orient` would not.
            steps.append(Step(np.eye(3), translation))
    written = _write(source, out, steps, on_chunk)
    frame = gaussians.Frame(
        up_axis=axis,
        up_axis_source=origin,
        heading_deg=heading,
        rotation=rotation,
        translation=translation,
        recentred=recentre,
    )
    return replace(written, frame=frame)


def median_gaussian_m(source: SplatSource) -> float:
    """The median gaussian radius in metres -- a splat's answer to "how fine is this?",
    `source_meta.json`'s `medianGaussianM`: `np.median(np.exp(scales[finite]))` over every
    axis of every gaussian whose three log-scales are finite, rounded to 6 places."""

    def radii() -> Iterator[F32]:
        for _, columns in source.chunks():
            scales = np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1)
            finite = np.isfinite(scales).all(axis=1)
            yield np.exp(scales[finite]).reshape(-1)

    value = outofcore.median(radii)
    return 0.0 if value is None else round(float(value), 6)


# ---------------------------------------------------------------------------------------
# The thumbnail
# ---------------------------------------------------------------------------------------


def thumbnail(
    source: SplatSource,
    path: Path,
    *,
    size: int = 512,
    alpha_min: float = 0.1,
    quality: int = 82,
    thicken: int = 3,
) -> dict[str, int]:
    """`gaussians.render_thumbnail`, drawn a chunk at a time into a running z-buffer.

    The frame is the 5th-95th percentile of the visible gaussians' east and up, measured
    over the whole splat first (exactly); then each chunk's nearest gaussian per pixel
    replaces the one drawn so far only when strictly nearer, so a tie goes to the earlier
    row -- the one the whole-splat lexsort puts first.
    """

    def visible_of(columns: Columns) -> tuple[npt.NDArray[np.bool_], F32]:
        *_, finite = _xyz(columns)
        alpha = np.ascontiguousarray(sigmoid(columns["opacity"]), dtype=np.float32)
        return finite & np.isfinite(alpha) & (alpha >= alpha_min), alpha

    def axes() -> Iterator[tuple[F32, I64]]:
        for _, columns in source.chunks():
            visible, _ = visible_of(columns)
            count = int(visible.sum())
            yield (
                np.concatenate([columns["x"][visible], columns["z"][visible]]),
                np.repeat(np.arange(2, dtype=np.int64), count),
            )

    clip = gaussians.THUMBNAIL_CLIP_PERCENT
    found = outofcore.grouped_percentile(axes, [0, 1], [clip, 100.0 - clip])
    image = np.zeros((size, size, 3), dtype=np.uint8)
    kept = 0
    if 0 in found:
        low = np.array([found[0][0], found[1][0]])
        high = np.array([found[0][1], found[1][1]])
        nearest = np.full(size * size, np.inf, dtype=np.float32)
        flat_image = image.reshape(-1, 3)
        for _, columns in source.chunks():
            visible, _ = visible_of(columns)
            kept += int(visible.sum())
            if not bool(visible.any()):
                continue
            colour = np.clip(
                gaussians.SH_C0
                * np.stack([columns[f"f_dc_{i}"][visible] for i in range(3)], axis=1)
                + 0.5,
                0.0,
                1.0,
            )
            across, up, depth = (
                columns["x"][visible],
                columns["z"][visible],
                columns["y"][visible],
            )
            framed, px, py = gaussians.frame_pixels(across, up, low, high, size)
            colour, depth = colour[framed], depth[framed]
            flat = py * size + px
            order = np.lexsort((depth, flat))
            first = np.unique(flat[order], return_index=True)[1]
            chosen = order[first]
            pixels = flat[chosen]
            nearer = depth[chosen] < nearest[pixels]
            pixels, chosen = pixels[nearer], chosen[nearer]
            nearest[pixels] = depth[chosen]
            flat_image[pixels] = np.round(colour[chosen] * 255.0).astype(np.uint8)
    picture = Image.fromarray(image, mode="RGB")
    if thicken > 1:
        picture = picture.filter(ImageFilter.MaxFilter(thicken))
    picture.save(path, format="JPEG", quality=quality, optimize=True)
    return {"gaussians": kept, "size": size, "bytes": path.stat().st_size}


def as_source(splat: gaussians.Splat, chunk: int = splat_io.CHUNK) -> ArraySource:
    """A splat in memory as a source -- how the tests run this module and `gaussians`
    side by side on the same rows."""
    return ArraySource(splat, chunk=chunk)
