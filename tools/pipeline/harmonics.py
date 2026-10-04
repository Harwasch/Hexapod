"""Spherical harmonics past the DC term: which bands a splat ships, and how they turn.

**Why this exists.** gsplat trains spherical harmonics to degree 3 -- 45 `f_rest_*` of a
gaussian's 59 floats -- and until this module every stage after training threw them away:
`canonical.ply` was exactly `gaussians.CANONICAL_PROPERTIES`, so a capture shipped its
degree-0 colour only, the same from every side. The packer (`tools/captures/
splat_tiles.py`) and both web renderers already carry and draw SH when a PLY has it, so
the colour detail the GPU paid for was lost in the middle of the pipeline, not at either
end. `ship_sh_degree` (the `train` stage's for Lane 2, `normalize`'s for Lane 1; 0 by default,
so every capture is what it was until somebody chooses otherwise) is how many bands now
ride through `trained.ply` -> `gated.ply` -> `canonical.ply` -> the tiles.

**Carrying a band means turning it.** A gaussian's SH colour is a function of the
direction it is seen from, *in the frame its coefficients were fitted in*. `place` turns
the trained splat from COLMAP's frame into east/north/up, and Lane 1's `normalize` turns
an upload by its up axis and heading (`gaussians.transform`, the only place either
rotates). A splat whose centres and quaternions were turned and whose SH was not shows
each gaussian's shine from the wrong side: a highlight that faced the camera's path now
faces the ground. So `rotate` is applied wherever `transform` is, and it is exact:

* **What it computes.** For a proper rotation R, the coefficients k' such that the turned
  gaussian seen from R d has the colour the original had from d, for every direction d:
  `sum k'_m Y_m(d) = sum k_m Y_m(R^T d)`. Each band l is closed under rotation (the 2l+1
  functions of one degree span a space every rotation maps onto itself), so per band it
  is one (2l+1) x (2l+1) matrix -- the real Wigner D-matrix of R, in this basis's
  ordering and signs -- applied to each colour channel.
* **How.** Not by the Ivanic-Ruedenberg recurrence, whose sign and ordering conventions
  differ between every paper and library that prints one: by fitting it to the very basis
  the trainer evaluates (`basis`, transcribed from gsplat's / Inria's `eval_sh`). Over
  `_SAMPLES` well-spread directions, `A[s] = Y(d_s)` and `B[s] = Y(R^T d_s)`, and
  `D = lstsq(A, B)` is the matrix with `A D = B`. Because the band is closed under R that
  system has an exact solution, so the fit is the D-matrix to rounding (~1e-15), for
  degree 1, 2 and 3 alike, with no convention to get wrong. Degree 1 also has a closed
  form -- the band is `C1 * (-y, z, -x)`, a signed permutation P of the direction, so
  `D1 = P R P^T` -- and the tests hold the fit to it, and every degree to the defining
  property itself (turn the splat and the view: the same colour).
* **What it does not handle**, and refuses rather than gets wrong: a mirror (determinant
  -1; `gaussians.transform` refuses those already, since a quaternion cannot follow one,
  and odd bands would also change sign under it), and a PLY whose `f_rest_*` is not a
  whole degree (`degree_of`'s `exact`): a fourth band cannot be turned here, so readers
  truncate to at most degree 3 first (`truncate`), as the packer does.

Uniform scale and translation do not touch SH: a direction is a direction however far
away or however big the gaussian is.

**Layout.** 3DGS trainers (Inria, gsplat's `export_splats`, OpenSplat) write the bands
above DC as `f_rest_0 .. f_rest_{3K-1}`, *channel-major*: all K coefficients of red,
then green, then blue (`shN (N, K, 3) -> permute(0, 2, 1) -> reshape(N, 3K)`), with K the
coefficients a channel has at the file's degree (3, 8, 15). Truncating to a lower degree
is therefore not a prefix of the names: degree 1 of a degree-3 file is
`f_rest_{c * 15 + j}` for j < 3, renumbered `f_rest_{c * 3 + j}` (`sources`).
`canonical.ply` puts them where the trainers do, after `f_dc_*` and before `opacity`
(`gaussians.ply_properties`), and degree 0 is exactly the fourteen it always was.

Pure numpy, no import of the rest of the pipeline, so `gaussians`, `splat_stream`,
`quality` and the tests can all lean on it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEGREES",
    "MAX_DEGREE",
    "SH_C0",
    "SH_DIMS",
    "band_rotation",
    "basis",
    "check_degree",
    "degree_of",
    "evaluate",
    "rest_names",
    "rotate",
    "sources",
    "stride_of",
    "truncate",
]

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

#: The DC term: colour = SH_C0 * f_dc + 0.5 (+ the bands below).
SH_C0 = 0.28209479177387814
#: Inria's `utils/sh_utils.py` and gsplat's `_eval_sh_bases_fast` constants, bands 1-3.
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

#: Coefficients a colour channel has above DC, by degree 0..3 (nianticlabs/spz's
#: `dimForDegree`; `tools/captures/splat_tiles.SH_DIMS`, kept equal by the tests).
SH_DIMS: tuple[int, ...] = (0, 3, 8, 15)
#: The most any renderer here draws: CesiumJS 1.145 and Spark evaluate bands 1-3, and the
#: packer writes no more (a fourth band in an SPZ v2 file is unreadable by their loaders).
MAX_DEGREE = 3
#: What a recipe may ship. 0 is the default everywhere: the capture as it always was.
DEGREES: tuple[int, ...] = (0, 1, 2, 3)

_REST = "f_rest_"


def check_degree(value: object, *, name: str = "ship_sh_degree") -> int:
    """A shipped SH degree from a recipe or a run's params, or a refusal that says why.

    None (the parameter absent) is 0. A bool is refused even though Python calls it an
    int: `ship_sh_degree: true` is a recipe that meant something and said something else.
    """
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{name} must be one of {', '.join(map(str, DEGREES))}, not {value!r}")
    try:
        degree = int(value)
    except ValueError:
        raise ValueError(
            f"{name} must be one of {', '.join(map(str, DEGREES))}, not {value!r}"
        ) from None
    if degree not in DEGREES:
        raise ValueError(
            f"{name} must be one of {', '.join(map(str, DEGREES))} (SH bands above the DC "
            f"colour; the renderers draw at most {MAX_DEGREE}), not {value!r}"
        )
    return degree


def rest_names(degree: int) -> tuple[str, ...]:
    """`f_rest_*` of a splat carrying exactly `degree`, channel-major, in name order."""
    return tuple(f"{_REST}{i}" for i in range(3 * SH_DIMS[degree]))


def stride_of(names: Iterable[str], *, source: str = "the splat") -> int:
    """Coefficients a channel has in `names`' `f_rest_*` (K of `f_rest_0..f_rest_{3K-1}`),
    0 without any; refused when they are not one unbroken run of a multiple of three."""
    rest = [name for name in names if name.startswith(_REST)]
    if not rest:
        return 0
    expected = {f"{_REST}{i}" for i in range(len(rest))}
    if set(rest) != expected or len(rest) % 3:
        raise ValueError(
            f"{source}: its {len(rest)} f_rest_* properties are not f_rest_0..f_rest_{{3K-1}} "
            f"(K coefficients for each of red, green and blue)"
        )
    return len(rest) // 3


def degree_of(names: Iterable[str], *, source: str = "the splat", exact: bool = False) -> int:
    """The SH degree `names` carry: the highest whose bands fit their `f_rest_*`, at most
    `MAX_DEGREE` (a fourth band is ignored, as nianticlabs/spz's PLY loader ignores it).
    `exact` refuses a run of coefficients that is not a whole degree 0-3 -- what a caller
    that is about to rotate them needs, since a coefficient it does not know the band of
    cannot be turned."""
    stride = stride_of(names, source=source)
    degree = max(d for d, need in enumerate(SH_DIMS) if need <= stride)
    if exact and SH_DIMS[degree] != stride:
        raise ValueError(
            f"{source} carries {stride} SH coefficients a channel, which is not a whole "
            f"degree 0-{MAX_DEGREE} ({', '.join(map(str, SH_DIMS))}); truncate it first"
        )
    return degree


def sources(degree: int, stride: int) -> dict[str, str]:
    """Each `f_rest_*` of a splat truncated to `degree`, and the name it comes from in a
    file with `stride` coefficients a channel: `f_rest_{c*K'+j}` <- `f_rest_{c*K+j}`."""
    want = SH_DIMS[degree]
    if want > stride:
        raise ValueError(f"degree {degree} needs {want} coefficients a channel; there are {stride}")
    return {
        f"{_REST}{c * want + j}": f"{_REST}{c * stride + j}" for c in range(3) for j in range(want)
    }


def truncate(columns: Mapping[str, npt.ArrayLike], degree: int) -> dict[str, F32]:
    """The `f_rest_*` of bands 1..`degree` out of columns that hold at least that many,
    renumbered channel-major at their own stride (`sources`). Empty for degree 0."""
    if degree == 0:
        return {}
    mapping = sources(degree, stride_of(columns))
    return {
        out: np.ascontiguousarray(columns[name], dtype=np.float32) for out, name in mapping.items()
    }


# ---------------------------------------------------------------------------------------
# The basis, and the colour it gives
# ---------------------------------------------------------------------------------------


def basis(degree: int, dirs: npt.ArrayLike) -> F64:
    """The real SH basis functions of bands 1..`degree` at unit directions `dirs` (n, 3),
    as (n, SH_DIMS[degree]) -- the DC term excluded, which no rotation touches.

    Transcribed from Inria's `eval_sh` (gaussian-splatting `utils/sh_utils.py`), which
    gsplat's `_eval_sh_bases_fast` and CUDA kernels, CesiumJS's `evaluateSH` and
    nianticlabs/spz all agree with: the order, the signs and the constants a trained PLY's
    coefficients mean anything in.
    """
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    columns: list[F64] = []
    if degree >= 1:
        columns += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        columns += [
            SH_C2[0] * x * y,
            SH_C2[1] * y * z,
            SH_C2[2] * (2.0 * zz - xx - yy),
            SH_C2[3] * x * z,
            SH_C2[4] * (xx - yy),
        ]
    if degree >= 3:
        columns += [
            SH_C3[0] * y * (3.0 * xx - yy),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4.0 * zz - xx - yy),
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3[4] * x * (4.0 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3.0 * yy),
        ]
    if not columns:
        return np.zeros((d.shape[0], 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def evaluate(f_dc: npt.ArrayLike, rest: npt.ArrayLike | None, dirs: npt.ArrayLike) -> F64:
    """The colour (before the renderer's clamp) a gaussian with DC term `f_dc` (n, 3) and
    higher bands `rest` (n, K, 3) shows from unit directions `dirs` (n, 3) -- the
    direction from the camera to the gaussian, in the coefficients' own frame."""
    colour = SH_C0 * np.asarray(f_dc, dtype=np.float64).reshape(-1, 3) + 0.5
    if rest is None:
        return colour
    coefficients = np.asarray(rest, dtype=np.float64)
    count = coefficients.shape[1]
    degree = SH_DIMS.index(count)
    values = basis(degree, dirs)
    shaded: F64 = colour + (values[:, :, None] * coefficients).sum(axis=1)
    return shaded


# ---------------------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------------------


def _fibonacci(count: int) -> F64:
    """`count` near-uniform unit directions: a fixed, well-conditioned set to fit D on."""
    i = np.arange(count, dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / count)
    azimuth = math.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)],
        axis=1,
    )


#: Directions D is fitted on: 64, against at most 7 unknowns a band, spread evenly so
#: every band's system is well conditioned for any rotation.
_SAMPLES = _fibonacci(64)

#: Where each band's coefficients start, per channel: band l is `[_BAND_START[l],
#: _BAND_START[l] + 2l + 1)` of the channel's K.
_BAND_START = (0, 0, 3, 8)

#: How far from 0 or +-1 a fitted D entry is taken to be exactly that (`band_rotation`).
_SNAP = 1e-12


def _band(band: int, dirs: F64) -> F64:
    start = _BAND_START[band]
    return basis(band, dirs)[:, start : start + 2 * band + 1]


def _proper(rotation: npt.ArrayLike) -> F64:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.allclose(r @ r.T, np.eye(3), atol=1e-6):
        raise ValueError("SH rotation: rotation must be a 3x3 orthonormal matrix")
    if np.linalg.det(r) < 0:
        raise ValueError(
            "SH rotation: a determinant of -1 is a mirror, not a rotation; the odd bands "
            "would change sign under it, and the splat's quaternions cannot follow it either"
        )
    return r


def band_rotation(rotation: npt.ArrayLike, band: int) -> F64:
    """The (2l+1) x (2l+1) matrix D that turns band `band`'s coefficients k (one colour
    channel, in `basis` order) into `D @ k` when the splat is turned by `rotation`, so that
    the turned splat seen from `R d` shows the colour the original showed from `d`.

    Fitted, exactly, on `_SAMPLES` (see the module): `A D = B` with `A = Y(d)` and
    `B = Y(R^T d)` -- `R^T d` is `d @ R` for row vectors.
    """
    if not 1 <= band <= MAX_DEGREE:
        raise ValueError(f"SH rotation: band must be 1-{MAX_DEGREE}, not {band}")
    r = _proper(rotation)
    a = _band(band, _SAMPLES)
    b = _band(band, _SAMPLES @ r)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    out: F64 = np.ascontiguousarray(solution, dtype=np.float64)
    # The fit's rounding, ~1e-16, snapped away: entries of a signed permutation (every
    # `UP_AXES` turn, the identity of a recentring step) come out exactly 0 and +-1, so a
    # coefficient that was 0 stays 0 rather than becoming 1e-17. An entry this small that
    # is genuinely there moves a coefficient by under a millionth of a float32 ulp.
    out[np.abs(out) < _SNAP] = 0.0
    unit = np.abs(np.abs(out) - 1.0) < _SNAP
    out[unit] = np.sign(out[unit])
    return out


def rotate(columns: Mapping[str, npt.ArrayLike], rotation: npt.ArrayLike) -> dict[str, F32]:
    """Every `f_rest_*` in `columns`, turned by `rotation` (`band_rotation` per band and
    channel); empty when there are none.

    Each output coefficient is accumulated element-wise, in float64, in a fixed order --
    not through a matrix product, which may take a different BLAS kernel and so a
    different rounding for a different number of rows. So a splat turned a chunk at a
    time (`splat_stream`) is bit-identical to one turned whole, as `gaussians.transform`
    already guarantees for positions and quaternions. Terms whose D entry is exactly 0
    are skipped, so a signed permutation (an `UP_AXES` turn, the identity) moves each
    coefficient bit for bit, -0.0 included, and a non-finite one does not spread.
    """
    names = [name for name in columns if name.startswith(_REST)]
    if not names:
        return {}
    degree = degree_of(names, exact=True)
    stride = SH_DIMS[degree]
    r = _proper(rotation)
    out: dict[str, F32] = {}
    with np.errstate(invalid="ignore", over="ignore"):
        for band in range(1, degree + 1):
            d = band_rotation(r, band)
            start = _BAND_START[band]
            width = 2 * band + 1
            for c in range(3):
                inputs = [
                    np.asarray(columns[f"{_REST}{c * stride + start + j}"], dtype=np.float64)
                    for j in range(width)
                ]
                for i in range(width):
                    total: F64 | None = None
                    for j in range(width):
                        if d[i, j] == 0.0:
                            continue
                        term = d[i, j] * inputs[j]
                        total = term if total is None else total + term
                    if total is None:  # a D row of zeros: not a rotation, but say so safely
                        total = np.zeros_like(inputs[0])
                    out[f"{_REST}{c * stride + start + i}"] = np.ascontiguousarray(
                        total, dtype=np.float32
                    )
    return out
