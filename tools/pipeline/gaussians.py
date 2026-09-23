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

**`canonical.ply` is east/north/up, z up, and that is a conversion, not an assumption.**
Everything downstream -- the tileset's node matrix, the thumbnail, the ground samples --
reads z as up, and until this module converted axes, nothing did: a Scaniverse `.spz`
(y up) and an Inria/COLMAP `.ply` (y down) both landed on the globe tipped 90 degrees,
and `splat_ground` measured "ground" along the capture's depth. `orient` is the
conversion. See `UP_AXIS_EVIDENCE` for what each format's default rests on.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from captures_bridge import SplatFormatError, read_ply, sigmoid, unpack_spz

__all__ = [
    "CANONICAL_PROPERTIES",
    "UP_AXES",
    "Frame",
    "GroundSample",
    "Splat",
    "default_up_axis",
    "ground_samples",
    "orient",
    "read_splat",
    "render_thumbnail",
    "transform",
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
# The frame: which way is up, which way is north, where the origin is
# ---------------------------------------------------------------------------------------

F64 = npt.NDArray[np.float64]

#: The proper rotation that takes each named axis of a file onto +z (east/north/up's up).
#:
#: Each one is chosen so that the file's *forward* lands on north as well: for a y-up
#: file (right/up/back, OpenGL's and SPZ's convention) forward is -z, and for a y-down
#: file (right/down/forward, OpenCV's and COLMAP's) forward is +z; both come out facing
#: +y. So a capture with no heading correction faces north rather than somewhere
#: arbitrary, and a heading is a single rotation about z on top.
UP_AXES: Mapping[str, F64] = {
    "z": np.eye(3),
    "-z": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "-x": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
}

#: What a file's up axis is taken to be when nobody said, by format -- and why.
#:
#: Not a guess, and not uniform: the two formats disagree, and the evidence for each is
#: recorded here because the next person to see a capture land upside down needs to
#: know which of these facts to doubt.
#:
#: * **`.spz` is y up.** nianticlabs/spz's README: "By default, SPZ stores data in an RUB
#:   coordinate system following the OpenGL and three.js convention", and its loader
#:   converts to that on write. Measured on real files 2026-09-23: two public Scaniverse
#:   share-page scans (`scaniverse.com/api/media/<id>/gaussians.spz`) render upright
#:   with y up and upside down with y down. **Counter-evidence, also measured:** six of
#:   Spark's sample `.spz` files (sparkjs.dev/assets/splats) are y *down* -- Spark's own
#:   quick start rotates `butterfly.spz` 180 degrees about x -- because a file written
#:   without declaring its coordinate system is stored unconverted. So `.spz` defaults to
#:   the specification and to the phone app that writes the format, and a y-down one needs
#:   `upAxis: "-y"` on the capture.
#: * **`.ply` is y down.** The same README: PLY "typically uses RDF", and the library's
#:   own `saveSplatToPly` converts to right/down/forward. Inria's 3DGS, and gsplat and
#:   nerfstudio run without world normalisation, write the COLMAP world frame, which is
#:   the first camera's -- y down, *tilted by however that camera was held* (nerfstudio's
#:   COLMAP parser: "Colmap optimized world often have y direction of the first camera
#:   pointing towards down direction"). Measured: Inria's Tanks-and-Temples `train`
#:   (Voxel51/gaussian_splatting on Hugging Face) is upright with y down. Polycam, Luma,
#:   KIRI and Postshot `.ply` exports were **not** measured -- no public sample was
#:   downloadable without an account -- so for them this default is the PLY convention,
#:   not an observation, and the capture override is the remedy.
#:
#: There is deliberately no `auto`. Estimating up from the geometry (the dominant plane's
#: normal) has a sign ambiguity nothing in a splat resolves, fails on an object with no
#: ground under it, and was not validated on enough real captures here to ship.
DEFAULT_UP_AXIS: Mapping[str, str] = {"spz": "y", "ply": "-y"}

UP_AXIS_EVIDENCE: Mapping[str, str] = {
    "spz": (
        "SPZ specification (right/up/back) and two measured Scaniverse scans; Spark's "
        "sample files are y-down counter-examples"
    ),
    "ply": (
        "the 3DGS/COLMAP convention (right/down/forward), measured on Inria's `train`; "
        "Polycam, Luma, KIRI and Postshot exports not measured"
    ),
}


#: Which of the per-cell ground heights `orient` calls the capture's base. See `orient`.
BASE_CELL_PERCENTILE = 25.0


def default_up_axis(source_format: str) -> str:
    return DEFAULT_UP_AXIS.get(source_format, "z")


@dataclass(frozen=True)
class Frame:
    """The similarity `orient` applied: `enu = scale * rotation @ file + translation`.

    Recorded in `source_meta.json` so that a capture that lands wrong can be diagnosed
    from its own run -- which axis was called up, on whose say-so, and how far the origin
    moved -- rather than by re-running it.
    """

    up_axis: str
    up_axis_source: str
    heading_deg: float
    rotation: F64
    translation: F64
    scale: float = 1.0
    recentred: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": "enu",
            "upAxis": self.up_axis,
            "upAxisSource": self.up_axis_source,
            "headingDeg": self.heading_deg,
            "scale": self.scale,
            "rotation": [[round(float(v), 9) for v in row] for row in self.rotation],
            "translationM": [round(float(v), 6) for v in self.translation],
            "recentred": self.recentred,
        }


def heading_rotation(heading_deg: float) -> F64:
    """Turn the capture so its forward (north, after `UP_AXES`) faces `heading_deg`.

    A compass bearing: clockwise from north, seen from above. So it is a rotation about
    +z by *minus* the angle.
    """
    angle = -math.radians(heading_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_to_quat(rotation: F64) -> F64:
    """A proper rotation matrix to a w-first unit quaternion (Shepperd's method)."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out = np.asarray(q, dtype=np.float64)
    return out / np.linalg.norm(out)


def transform(
    columns: Mapping[str, F32],
    rotation: F64,
    translation: F64 | None = None,
    scale: float = 1.0,
) -> dict[str, F32]:
    """Apply `x' = scale * R @ x + t` to every gaussian, not just to its centre.

    Three things move and one deliberately does not:

    * **positions** by the whole similarity;
    * **orientations**: a gaussian's `rot_*` is the rotation from its own axes to the
      file's, so the new one is `q_R * q` -- the Hamilton product with R on the left. A
      splat whose centres were rotated and whose quaternions were not keeps every
      ellipsoid pointing the old way, which is visible as a capture made of needles;
    * **log-scales** by `ln(scale)`, since a uniform scale multiplies every axis length;
    * **colour does not**, and that is exact rather than approximate: `canonical.ply`
      carries only the DC spherical-harmonic term, which is view-independent and so
      invariant under rotation. Bands above DC are dropped on read (`Splat.dropped`),
      so there is nothing here that would need a Wigner rotation -- and if they are ever
      carried, this function is where that rotation has to be added.
    """
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.allclose(r @ r.T, np.eye(3), atol=1e-6):
        raise ValueError("transform: rotation must be a 3x3 orthonormal matrix")
    if np.linalg.det(r) < 0:
        raise ValueError(
            "transform: rotation has determinant -1, which is a mirror. A mirrored splat "
            "is not a rotated one -- its quaternions cannot follow it -- so it is refused"
        )
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"transform: scale must be positive and finite, not {scale!r}")
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    out = {name: np.array(values, dtype=np.float32, copy=True) for name, values in columns.items()}
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    moved = scale * (xyz @ r.T) + t
    for index, axis in enumerate(("x", "y", "z")):
        out[axis] = _f32(moved[:, index])
    qw, qx, qy, qz = matrix_to_quat(r)
    w, x, y, z = (np.asarray(columns[f"rot_{i}"], dtype=np.float64) for i in range(4))
    out["rot_0"] = _f32(qw * w - qx * x - qy * y - qz * z)
    out["rot_1"] = _f32(qw * x + qx * w + qy * z - qz * y)
    out["rot_2"] = _f32(qw * y - qx * z + qy * w + qz * x)
    out["rot_3"] = _f32(qw * z + qx * y - qy * x + qz * w)
    if scale != 1.0:
        shift = math.log(scale)
        for i in range(3):
            out[f"scale_{i}"] = _f32(np.asarray(columns[f"scale_{i}"], dtype=np.float64) + shift)
    return out


def orient(
    splat: Splat,
    *,
    up_axis: str | None = None,
    heading_deg: float = 0.0,
    recentre: bool = True,
    cell_m: float = 2.0,
) -> tuple[Splat, Frame]:
    """Turn a splat into east/north/up about its own footprint, and say what was done.

    `up_axis` None means the format's default (`DEFAULT_UP_AXIS`); anything else is an
    override from the capture and is recorded as one.

    **Recentring** moves the origin to the centre of the capture's footprint, horizontally,
    and to its own measured ground, vertically. The placed coordinate is where somebody
    put the capture -- the console sends where its camera was looking -- and that point
    should be the capture's middle, not wherever the exporter's origin happened to be
    (for a phone app, where the phone was when tracking started). Both statistics are
    robust on purpose:

    * the footprint is the middle of the 2nd-98th percentile range of each horizontal
      axis, so a few floaters or a background shell cannot drag it;
    * the base is the lower quartile of `ground_samples`' per-cell heights -- the same
      per-cell statistic the viewer's clamp rests on the terrain. Not the median: under a
      tree most cells' lowest surface is canopy (the committed fixture's median cell is
      3.9 m up a 6.5 m tree), and the quartile reaches the cells that have ground in them.
      With no ground cells (a capture smaller than a cell), the vertical origin is left
      alone.

    It is sound with the clamp either way: the clamp compares each sample's `origin.height
    + z` against the terrain at the same longitude and latitude, and a constant vertical
    shift of the model moves every `z` by the same amount, so the offset it computes is
    unchanged. The horizontal shift does change where the samples are, which is the point.
    """
    axis = up_axis if up_axis is not None else default_up_axis(splat.source_format)
    if axis not in UP_AXES:
        raise ValueError(
            f"upAxis {axis!r} is not one of {', '.join(UP_AXES)}. It names the axis of the "
            f"uploaded file that points up: y for Scaniverse/SPZ, -y for 3DGS/COLMAP .ply"
        )
    source = "capture" if up_axis is not None else f"format-default ({splat.source_format})"
    heading = float(heading_deg)
    if not math.isfinite(heading):
        raise ValueError(f"headingDeg must be a finite number of degrees, not {heading_deg!r}")
    rotation = heading_rotation(heading) @ UP_AXES[axis]
    turned = replace(splat, columns=transform(splat.columns, rotation))
    translation = np.zeros(3)
    if recentre:
        xyz = turned.xyz
        finite = np.isfinite(xyz).all(axis=1)
        if bool(finite.any()):
            low = np.percentile(xyz[finite, :2], 2.0, axis=0)
            high = np.percentile(xyz[finite, :2], 98.0, axis=0)
            centre = (low + high) / 2.0
            translation[:2] = -centre
            cells = ground_samples(turned, lat=0.0, lon=0.0, cell_m=cell_m)
            if cells:
                translation[2] = -float(
                    np.percentile([cell.z for cell in cells], BASE_CELL_PERCENTILE)
                )
            turned = replace(turned, columns=transform(turned.columns, np.eye(3), translation))
    frame = Frame(
        up_axis=axis,
        up_axis_source=source,
        heading_deg=heading,
        rotation=rotation,
        translation=translation,
        recentred=recentre,
    )
    return turned, frame


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
        # Framed on the 1st-99th percentile of each axis, at one scale for both, so a sky
        # dome, a background shell or a handful of floaters cannot shrink the capture to a
        # speck, and a tall tree is not squashed into a square. Points outside the frame
        # are simply not drawn.
        framed, px, py = _frame(across, up, drawn)
        colour, depth = colour[framed], depth[framed]
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


#: The share of points on each side of each axis a thumbnail leaves out of its frame.
THUMBNAIL_CLIP_PERCENT = 5.0


def _frame(
    across: F32, up: F32, size: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Which points are inside the frame, and their pixel columns and rows."""
    low = np.array(
        [np.percentile(across, THUMBNAIL_CLIP_PERCENT), np.percentile(up, THUMBNAIL_CLIP_PERCENT)]
    )
    high = np.array(
        [
            np.percentile(across, 100.0 - THUMBNAIL_CLIP_PERCENT),
            np.percentile(up, 100.0 - THUMBNAIL_CLIP_PERCENT),
        ]
    )
    centre = (low + high) / 2.0
    span = float(max(high - low))
    # A capture with no extent (a single gaussian) projects to the middle of the image
    # rather than dividing by zero.
    half = span / 2.0 if span > 0 else 1.0
    inside = (np.abs(across - centre[0]) <= half) & (np.abs(up - centre[1]) <= half)
    margin = 0.04
    scale = (1 - 2 * margin) * size / (2.0 * half)
    px = np.floor(size / 2.0 + (across[inside] - centre[0]) * scale)
    py = np.floor(size / 2.0 - (up[inside] - centre[1]) * scale)
    return (
        inside,
        np.clip(px, 0, size - 1).astype(np.intp),
        np.clip(py, 0, size - 1).astype(np.intp),
    )


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
