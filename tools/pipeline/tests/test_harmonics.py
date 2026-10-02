"""Spherical harmonics past DC: carried through the pipeline, and turned with the splat.

The property everything here holds `harmonics.rotate` to is the definition of turning a
splat correctly: **turn the splat by R and look at it from R d, and every gaussian shows
the colour it showed from d.** Checked numerically for degrees 1, 2 and 3 on random
rotations, coefficients and directions -- against a basis transcribed independently below
from Inria's `eval_sh`, the convention every trainer and renderer here agrees with -- and
then through the code that applies it: `gaussians.transform`, the chunked writer `place`
uses, and the `train -> place -> package` stages on a stand-in trainer's SH-3 export.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import gaussians
import harmonics
import splat_io
import splat_stream
import splat_tiles
from conftest import make_recipe
from executor import execute
from recipe import load_recipe
from runners import LocalRunner, RunnerSet
from workdir import Workdir

rng = np.random.default_rng(20261002)


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation (a normalised random quaternion)."""
    q = generator.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def unit(n: int) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def inria_eval_sh(deg: int, sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """gaussian-splatting's `utils/sh_utils.py` `eval_sh`, verbatim but for numpy: `sh` is
    (..., 3, (deg + 1)^2) -- channel first, coefficient last -- and `dirs` unit vectors."""
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    c3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    result = c0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - c1 * y * sh[..., 1] + c1 * z * sh[..., 2] - c1 * x * sh[..., 3]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + c2[0] * xy * sh[..., 4]
                + c2[1] * yz * sh[..., 5]
                + c2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + c2[3] * xz * sh[..., 7]
                + c2[4] * (xx - yy) * sh[..., 8]
            )
            if deg > 2:
                result = (
                    result
                    + c3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + c3[1] * xy * z * sh[..., 10]
                    + c3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + c3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + c3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + c3[5] * z * (xx - yy) * sh[..., 14]
                    + c3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
    return result


def columns_with_sh(count: int, degree: int, seed: int = 3) -> dict[str, np.ndarray]:
    """A splat's canonical fourteen and `degree` bands of f_rest_*, channel-major."""
    g = np.random.default_rng(seed)
    q = g.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    columns = {
        "x": g.normal(size=count) * 3,
        "y": g.normal(size=count) * 3,
        "z": g.normal(size=count) * 3,
        **{f"f_dc_{i}": g.normal(scale=0.5, size=count) for i in range(3)},
        "opacity": g.normal(size=count),
        **{f"scale_{i}": g.normal(-3, 0.5, size=count) for i in range(3)},
        **{f"rot_{i}": q[:, i] for i in range(4)},
        **{name: g.normal(scale=0.3, size=count) for name in harmonics.rest_names(degree)},
    }
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


def rest_of(columns: dict[str, np.ndarray], degree: int) -> np.ndarray:
    """f_rest_* as (n, K, 3): channel-major in the PLY, [:, j, c] = f_rest_{c * K + j}."""
    k = harmonics.SH_DIMS[degree]
    return np.stack(
        [np.stack([columns[f"f_rest_{c * k + j}"] for c in range(3)], axis=1) for j in range(k)],
        axis=1,
    ).astype(np.float64)


def dc_of(columns: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)


# --- the basis ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_basis_is_inrias_eval_sh(degree: int) -> None:
    """The order, signs and constants a trained PLY's coefficients mean anything in."""
    n = 500
    dirs = unit(n)
    dc = rng.normal(size=(n, 3))
    rest = rng.normal(size=(n, harmonics.SH_DIMS[degree], 3))
    ours = harmonics.evaluate(dc, rest, dirs) - 0.5
    stacked = np.concatenate([dc[:, None, :], rest], axis=1).transpose(0, 2, 1)
    theirs = inria_eval_sh(degree, stacked, dirs)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-12)


def test_the_basis_is_orthonormal_on_the_sphere() -> None:
    """Real SH with these constants are orthonormal: (1/4pi) integral Y_i Y_j = delta_ij
    / (4pi), here by a 40,000-point Fibonacci quadrature. A wrong constant or a swapped
    sign in `basis` shows up as an off-diagonal or a diagonal away from 1."""
    count = 40_000
    i = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * i / count)
    azimuth = math.pi * (1 + 5**0.5) * i
    dirs = np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1
    )
    values = harmonics.basis(3, dirs)
    gram = 4 * math.pi * values.T @ values / count
    np.testing.assert_allclose(gram, np.eye(15), atol=2e-3)
    assert 4 * math.pi * harmonics.SH_C0**2 == pytest.approx(1.0)


def test_the_sh_dims_are_the_packers() -> None:
    assert tuple(splat_tiles.SH_DIMS) == harmonics.SH_DIMS
    assert splat_tiles.SH_MAX_DEGREE == harmonics.MAX_DEGREE


# --- rotation ----------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_turning_the_splat_and_the_view_together_keeps_every_colour(degree: int) -> None:
    """The definition: the turned splat seen from R d is the original seen from d."""
    n = 2_000
    columns = columns_with_sh(n, degree)
    dirs = unit(n)
    for _ in range(5):
        r = random_rotation(rng)
        turned = {**columns, **harmonics.rotate(columns, r)}
        before = harmonics.evaluate(dc_of(columns), rest_of(columns, degree), dirs)
        after = harmonics.evaluate(dc_of(turned), rest_of(turned, degree), dirs @ r.T)
        # float32 coefficients out: a few ulps of a value around 1.
        np.testing.assert_allclose(after, before, rtol=0, atol=2e-6)
        # And turning it is not the identity: from the old direction the colour moved.
        unmoved = harmonics.evaluate(dc_of(turned), rest_of(turned, degree), dirs)
        assert np.abs(unmoved - before).max() > 1e-2


def test_without_the_rotation_the_shine_faces_the_wrong_way() -> None:
    """The failure this exists to prevent, measured: the same turned splat with its SH
    left alone shows a different colour from R d than the original did from d."""
    n = 2_000
    columns = columns_with_sh(n, 3)
    dirs = unit(n)
    r = random_rotation(rng)
    before = harmonics.evaluate(dc_of(columns), rest_of(columns, 3), dirs)
    stale = harmonics.evaluate(dc_of(columns), rest_of(columns, 3), dirs @ r.T)
    assert np.abs(stale - before).mean() > 0.05


def test_degree_one_is_the_closed_form() -> None:
    """Band 1 is C1 * (-y, z, -x) = C1 * P d, so its D-matrix is P R P^T exactly."""
    p = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    for _ in range(10):
        r = random_rotation(rng)
        np.testing.assert_allclose(harmonics.band_rotation(r, 1), p @ r @ p.T, atol=1e-12)


@pytest.mark.parametrize("band", [1, 2, 3])
def test_each_band_turns_by_an_orthogonal_representation_of_the_rotation(band: int) -> None:
    """D(R) is orthogonal, D(I) = I, and D(R1 R2) = D(R1) D(R2): it is the Wigner D-matrix
    of R in this basis, not just some fit."""
    width = 2 * band + 1
    np.testing.assert_allclose(harmonics.band_rotation(np.eye(3), band), np.eye(width), atol=1e-12)
    for _ in range(5):
        r1, r2 = random_rotation(rng), random_rotation(rng)
        d1, d2 = harmonics.band_rotation(r1, band), harmonics.band_rotation(r2, band)
        np.testing.assert_allclose(d1 @ d1.T, np.eye(width), atol=1e-12)
        np.testing.assert_allclose(harmonics.band_rotation(r1 @ r2, band), d1 @ d2, atol=1e-12)


def test_a_mirror_or_a_partial_degree_is_refused() -> None:
    columns = columns_with_sh(10, 2)
    with pytest.raises(ValueError, match="mirror"):
        harmonics.rotate(columns, np.diag([1.0, 1.0, -1.0]))
    partial = {k: v for k, v in columns.items() if k != "f_rest_23"}
    with pytest.raises(ValueError, match="f_rest"):
        harmonics.rotate(partial, np.eye(3))
    four_a_channel = {f"f_rest_{i}": np.zeros(1, np.float32) for i in range(12)}
    with pytest.raises(ValueError, match="whole degree"):
        harmonics.rotate(four_a_channel, np.eye(3))


# --- degrees, names, truncation ----------------------------------------------------------


def test_the_degree_a_recipe_may_ship() -> None:
    assert [harmonics.check_degree(v) for v in (None, 0, 1, 2, 3, "3")] == [0, 0, 1, 2, 3, 3]
    for bad in (4, -1, True, 1.5, "one", [1]):
        with pytest.raises(ValueError, match="sh_degree"):
            harmonics.check_degree(bad)


def test_truncating_keeps_the_low_bands_of_each_channel_not_a_prefix_of_the_names() -> None:
    """Degree 1 of a degree-3 PLY: red's first three, green's, blue's -- f_rest_0-2, 15-17
    and 30-32 -- renumbered 0-8, channel-major still."""
    assert harmonics.sources(1, 15) == {
        "f_rest_0": "f_rest_0",
        "f_rest_1": "f_rest_1",
        "f_rest_2": "f_rest_2",
        "f_rest_3": "f_rest_15",
        "f_rest_4": "f_rest_16",
        "f_rest_5": "f_rest_17",
        "f_rest_6": "f_rest_30",
        "f_rest_7": "f_rest_31",
        "f_rest_8": "f_rest_32",
    }
    columns = columns_with_sh(50, 3)
    low = harmonics.truncate(columns, 1)
    np.testing.assert_array_equal(rest_of(low, 1), rest_of(columns, 3)[:, :3, :])
    assert harmonics.truncate(columns, 0) == {}
    with pytest.raises(ValueError, match="needs 8"):
        harmonics.truncate(columns_with_sh(5, 1), 2)


def test_truncation_commutes_with_rotation() -> None:
    """Each band turns on its own, so turning then truncating is truncating then turning
    -- which is why `train` may truncate before `place` turns."""
    columns = columns_with_sh(200, 3)
    r = random_rotation(rng)
    turned_then_cut = harmonics.truncate({**columns, **harmonics.rotate(columns, r)}, 2)
    cut_then_turned = harmonics.rotate(harmonics.truncate(columns, 2), r)
    assert set(turned_then_cut) == set(cut_then_turned) == set(harmonics.rest_names(2))
    for name, values in turned_then_cut.items():
        np.testing.assert_array_equal(values, cut_then_turned[name])


def test_turning_by_the_identity_changes_no_bit() -> None:
    """The fit's rounding noise is snapped away, so the identity -- `orient`'s recentring
    step, the `z` up axis -- leaves every float32 coefficient exactly as it was."""
    columns = columns_with_sh(300, 3)
    for name, values in harmonics.rotate(columns, np.eye(3)).items():
        np.testing.assert_array_equal(values, columns[name])


# --- through the code that applies it ----------------------------------------------------


def test_transform_turns_the_sh_and_leaves_it_alone_under_scale_and_translation() -> None:
    columns = columns_with_sh(500, 3)
    r = random_rotation(rng)
    moved = gaussians.transform(columns, r, np.array([5.0, -2.0, 1.0]), 2.5)
    expected = harmonics.rotate(columns, r)
    for name, values in expected.items():
        np.testing.assert_array_equal(moved[name], values)
    # Pure translation and scale: not a coefficient changes.
    shifted = gaussians.transform(columns, np.eye(3), np.array([1.0, 2.0, 3.0]), 3.0)
    for name in harmonics.rest_names(3):
        np.testing.assert_allclose(shifted[name], columns[name], atol=1e-7)


def test_a_chunked_transform_is_bit_identical_to_a_whole_one() -> None:
    """`place` turns a splat a chunk at a time; its SH must come out as if whole."""
    columns = columns_with_sh(1_001, 3)
    r = random_rotation(rng)
    whole = gaussians.transform(columns, r)
    pieces = [
        gaussians.transform({k: v[a:b] for k, v in columns.items()}, r)
        for a, b in ((0, 7), (7, 640), (640, 1_001))
    ]
    for name in harmonics.rest_names(3):
        np.testing.assert_array_equal(whole[name], np.concatenate([p[name] for p in pieces]))


def write_raw_ply(path: Path, columns: dict[str, np.ndarray]) -> Path:
    """A trainer-shaped PLY: x y z, f_dc, f_rest, opacity, scale, rot (gsplat's order)."""
    degree = harmonics.degree_of(columns)
    names = gaussians.ply_properties(degree)
    count = columns["x"].shape[0]
    rows = np.zeros(count, dtype=[(name, "<f4") for name in names])
    for name in names:
        rows[name] = columns[name]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in names] + ["end_header"]
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    return path


def test_canonical_ply_puts_the_bands_where_the_trainers_do_and_degree_0_is_unchanged(
    tmp_path: Path,
) -> None:
    assert gaussians.ply_properties(0) == gaussians.CANONICAL_PROPERTIES
    names = gaussians.ply_properties(1)
    assert names[:6] == gaussians.CANONICAL_PROPERTIES[:6]
    assert names[6:15] == tuple(f"f_rest_{i}" for i in range(9))
    assert names[15:] == gaussians.CANONICAL_PROPERTIES[6:]
    columns = columns_with_sh(64, 3)
    ply = write_raw_ply(tmp_path / "raw.ply", columns)
    dc_only = gaussians.read_splat(ply)
    assert dc_only.sh_degree == 0 and "f_rest_0" in dc_only.dropped
    plain = {k: v for k, v in columns.items() if not k.startswith("f_rest_")}
    gaussians.write_ply(tmp_path / "a.ply", dc_only.columns)
    gaussians.write_ply(tmp_path / "b.ply", plain)
    # A raw dict, every band in it, still writes the fourteen unless a degree is asked for.
    gaussians.write_ply(tmp_path / "c.ply", columns)
    assert (tmp_path / "a.ply").read_bytes() == (tmp_path / "b.ply").read_bytes()
    assert (tmp_path / "c.ply").read_bytes() == (tmp_path / "b.ply").read_bytes()
    with pytest.raises(ValueError, match="truncate first"):
        gaussians.write_ply(tmp_path / "d.ply", columns, sh_degree=1)
    one = gaussians.read_splat(ply, sh_degree=1)
    assert one.sh_degree == 1
    assert set(one.columns) == set(gaussians.ply_properties(1))
    np.testing.assert_array_equal(rest_of(one.columns, 1), rest_of(columns, 3)[:, :3, :])
    # Read from red's, green's and blue's first three; the rest of every channel dropped.
    used = {f"f_rest_{c * 15 + j}" for c in range(3) for j in range(3)}
    assert used.isdisjoint(one.dropped)
    assert {f"f_rest_{i}" for i in range(45)} - used <= set(one.dropped)
    # A file with fewer bands than asked ships what it has.
    fewer = write_raw_ply(tmp_path / "one.ply", one.columns)
    assert gaussians.read_splat(fewer, sh_degree=3).sh_degree == 1


def test_the_streamed_placement_turns_the_sh_exactly_as_the_whole_one(tmp_path: Path) -> None:
    """What `place` runs (`splat_stream.orient_to`, prime-sized chunks) against what
    `gaussians.orient` does whole: the same canonical.ply, SH bands included and turned."""
    columns = columns_with_sh(3_001, 2)
    ply = write_raw_ply(tmp_path / "trained.ply", columns)
    source = splat_stream.open_splat(ply, chunk=997, sh_degree=None)
    assert source.sh_degree == 2 and source.properties == gaussians.ply_properties(2)
    step = splat_stream.Step(random_rotation(rng), np.array([3.0, 1.0, -2.0]), 1.7)
    placed = splat_stream.orient_to(source, tmp_path / "canonical.ply", before=[step], up_axis="z")
    whole = gaussians.read_splat(ply, sh_degree=2)
    moved = gaussians.transform(whole.columns, step.rotation, step.translation, step.scale)
    turned, frame = gaussians.orient(dataclasses.replace(whole, columns=moved), up_axis="z")
    gaussians.write_ply(tmp_path / "whole.ply", turned.columns, sh_degree=2)
    assert placed.frame is not None
    np.testing.assert_array_equal(placed.frame.translation, frame.translation)
    assert (tmp_path / "canonical.ply").read_bytes() == (tmp_path / "whole.ply").read_bytes()
    layout = splat_io.read_layout(tmp_path / "canonical.ply")
    assert layout.properties == gaussians.ply_properties(2)
    # And the colour property holds on what was written: seen from R d, as before from d.
    out = gaussians.read_splat(tmp_path / "canonical.ply", sh_degree=3)
    dirs = unit(out.count)
    r = frame.rotation @ step.rotation
    before = harmonics.evaluate(dc_of(columns), rest_of(columns, 2), dirs)
    after = harmonics.evaluate(dc_of(out.columns), rest_of(out.columns, 2), dirs @ r.T)
    np.testing.assert_allclose(after, before, atol=3e-6)


def test_measuring_passes_leave_the_sh_out(tmp_path: Path) -> None:
    ply = write_raw_ply(tmp_path / "s.ply", columns_with_sh(100, 3))
    source = splat_stream.open_splat(ply, sh_degree=1)
    _, first = next(source.chunks(with_sh=False))
    assert tuple(first) == gaussians.CANONICAL_PROPERTIES
    _, full = next(source.chunks())
    assert set(full) == set(gaussians.ply_properties(1))


# --- the stages: train -> place -> package, on a stand-in trainer ------------------------

STAND_IN = Path(__file__).resolve().parent / "gsplat_stand_in.py"
HOLDOUT_STAND_IN = Path(__file__).resolve().parent / "holdout_stand_in.py"


def _seed(workdir: Workdir) -> None:
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        (frames / f"frame_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg")
    poses = workdir.input_path("poses")
    poses.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).write_bytes(b"colmap")
    (poses / "poses.json").write_text(json.dumps({"registered": 4}), encoding="utf-8")
    angle = 0.7
    rotation = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [0.0, 0.0, -1.0],
        [math.sin(angle), math.cos(angle), 0.0],
    ]
    georef = {
        "lat": 45.0,
        "lon": -93.0,
        "height": 250.0,
        "frame": {"source": "test", "rotation": rotation, "scale": 1.0, "recentre": True},
    }
    workdir.input_path("georef.json").write_text(json.dumps(georef), encoding="utf-8")


def _lane(sh_degree: int | None, *, holdout_error: bool = True) -> list[dict[str, object]]:
    train: dict[str, object] = {
        "iterations": 300,
        "trainer": str(STAND_IN),
        "python": sys.executable,
        # gsplat's own flag: the stand-in exports degree 3, as the real trainer always does.
        "extra_args": ["--ckpt-every", "100", "--gaussians", "200", "--sh_degree", "3"],
        "holdout_error": holdout_error,
        "holdout_script": str(HOLDOUT_STAND_IN),
    }
    if sh_degree is not None:
        train["ship_sh_degree"] = sh_degree
    return [
        {"id": "train", "impl": "gsplat", "params": train},
        {"id": "place", "impl": "place_splat", "params": {}},
        {"id": "package", "impl": "splat_tiles", "params": {}},
    ]


def _run(tmp_path: Path, name: str, sh_degree: int | None) -> Workdir:
    workdir = Workdir.create(tmp_path / name)
    _seed(workdir)
    recipe = make_recipe(_lane(sh_degree), inputs=["frames", "poses", "georef.json"])
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    return workdir


def test_by_default_nothing_past_dc_ships(tmp_path: Path) -> None:
    """sh_degree absent: trained.ply and canonical.ply are the fourteen, the tiles SH 0 --
    every capture before this existed -- even from a trainer that exported SH 3."""
    workdir = _run(tmp_path, "dc", None)
    trained = workdir.artifact_path("train", "trained.ply")
    assert splat_io.read_layout(trained).properties == gaussians.CANONICAL_PROPERTIES
    canonical = workdir.artifact_path("place", "canonical.ply")
    assert splat_io.read_layout(canonical).properties == gaussians.CANONICAL_PROPERTIES
    step = json.loads(workdir.step_path("package").read_text())
    assert step["metrics"]["sh_degree"] == 0
    log = workdir.log_path("train").read_text()
    assert "--sh-degree 0" in log


@pytest.mark.parametrize("degree", [1, 3])
def test_a_shipped_degree_rides_through_and_is_turned_into_east_north_up(
    tmp_path: Path, degree: int
) -> None:
    workdir = _run(tmp_path, f"sh{degree}", degree)
    raw = next((workdir.work_dir("train") / "gsplat" / "ply").glob("point_cloud_*.ply"))
    exported = gaussians.read_splat(raw, sh_degree=3)
    assert exported.sh_degree == 3
    trained = gaussians.read_splat(workdir.artifact_path("train", "trained.ply"), sh_degree=3)
    assert trained.sh_degree == degree
    np.testing.assert_array_equal(
        rest_of(trained.columns, degree),
        rest_of(exported.columns, 3)[:, : harmonics.SH_DIMS[degree], :],
    )
    metrics = json.loads(workdir.artifact_path("train", "train_metrics.json").read_text())
    assert metrics["settings"]["shDegree"] == degree
    assert metrics["settings"]["shDegreeTrained"] == 3
    # The held-out error is asked for at the degree that ships.
    assert f"--sh-degree {degree}" in workdir.log_path("train").read_text()
    assert metrics["holdout"]["shDegree"] == degree
    # place: the same gaussians, each band turned by the frame's rotation.
    canonical = gaussians.read_splat(workdir.artifact_path("place", "canonical.ply"), sh_degree=3)
    assert canonical.sh_degree == degree and canonical.count == trained.count
    georef = json.loads(workdir.input_path("georef.json").read_text())
    rotation = np.asarray(georef["frame"]["rotation"])
    dirs = unit(trained.count)
    before = harmonics.evaluate(dc_of(trained.columns), rest_of(trained.columns, degree), dirs)
    after = harmonics.evaluate(
        dc_of(canonical.columns), rest_of(canonical.columns, degree), dirs @ rotation.T
    )
    np.testing.assert_allclose(after, before, atol=3e-6)
    place = json.loads(workdir.step_path("place").read_text())
    assert place["metrics"]["shDegree"] == degree
    # package: every tile carries it, as CesiumJS counts it.
    package = json.loads(workdir.step_path("package").read_text())
    assert package["metrics"]["sh_degree"] == degree


# --- Lane 1: an upload's own SH, turned by its up axis ----------------------------------


def _ingest(tmp_path: Path, upload: Path, normalize: dict[str, object]) -> Workdir:
    workdir = Workdir.create(tmp_path / "run")
    folder = workdir.input_path("upload")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / upload.name).write_bytes(upload.read_bytes())
    placed: dict[str, dict[str, object]] = {
        "georeference": {"lat": 28.0, "lon": -82.7, "height": 20.0},
        "normalize": normalize,
        "register": {"slug": "shiny", "title": "Shiny"},
    }
    execute(load_recipe("splat-ingest").with_params(placed), workdir, RunnerSet.local())
    return workdir


@pytest.mark.parametrize("suffix", [".ply", ".spz"])
def test_lane_1_ships_an_uploads_own_sh_turned_upright(tmp_path: Path, suffix: str) -> None:
    """A y-down `.ply` (or a y-up `.spz`) with SH 3: `normalize` at `ship_sh_degree: 3` turns
    it into east/north/up -- positions, quaternions and every band -- and the tiles carry
    degree 3; the colour from every direction is the file's from the turned-back one."""
    columns = columns_with_sh(3_000, 3, seed=11)
    if suffix == ".ply":
        upload = write_raw_ply(tmp_path / "scan.ply", columns)
        up_axis = "-y"
    else:
        upload = tmp_path / "scan.spz"
        upload.write_bytes(
            splat_tiles.pack_spz(
                np.stack([columns[a] for a in "xyz"], axis=1),
                dc_of(columns).astype(np.float32),
                columns["opacity"],
                np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1),
                np.stack([columns[f"rot_{i}"] for i in (1, 2, 3, 0)], axis=1),
                rest_of(columns, 3).astype(np.float32),
            )
        )
        up_axis = "y"
    original = gaussians.read_splat(upload, sh_degree=3)
    assert original.sh_degree == 3
    workdir = _ingest(tmp_path, upload, {"ship_sh_degree": 3, "heading_deg": 30.0})

    canonical_ply = workdir.artifact_path("normalize", "canonical.ply")
    canonical = gaussians.read_splat(canonical_ply, sh_degree=3)
    assert canonical.sh_degree == 3
    meta = json.loads(workdir.artifact_path("normalize", "source_meta.json").read_text())
    assert meta["shDegree"] == 3
    assert not any(name.startswith("f_rest") for name in meta["dropped"])
    r = np.asarray(meta["frame"]["rotation"])
    np.testing.assert_allclose(
        r, gaussians.heading_rotation(30.0) @ gaussians.UP_AXES[up_axis], atol=1e-8
    )
    dirs = unit(canonical.count)
    before = harmonics.evaluate(dc_of(original.columns), rest_of(original.columns, 3), dirs)
    after = harmonics.evaluate(dc_of(canonical.columns), rest_of(canonical.columns, 3), dirs @ r.T)
    np.testing.assert_allclose(after, before, atol=3e-6)
    package = json.loads(workdir.step_path("package").read_text())
    assert package["metrics"]["sh_degree"] == 3
    manifest = json.loads(workdir.artifact_path("manifest", "manifest.json").read_text())
    assert manifest["splat"]["shDegree"] == 3


def test_lane_1_by_default_drops_the_uploads_sh_as_it_always_did(tmp_path: Path) -> None:
    upload = write_raw_ply(tmp_path / "scan.ply", columns_with_sh(500, 3, seed=12))
    workdir = _ingest(tmp_path, upload, {"up_axis": "z"})
    canonical = workdir.artifact_path("normalize", "canonical.ply")
    assert splat_io.read_layout(canonical).properties == gaussians.CANONICAL_PROPERTIES
    meta = json.loads(workdir.artifact_path("normalize", "source_meta.json").read_text())
    assert meta["shDegree"] == 0 and "f_rest_44" in meta["dropped"]
    manifest = json.loads(workdir.artifact_path("manifest", "manifest.json").read_text())
    assert manifest["splat"]["shDegree"] == 0


def test_a_malformed_f_rest_run_is_dropped_unread_unless_its_bands_are_asked_for(
    tmp_path: Path,
) -> None:
    """Ten f_rest_* (not three a channel): a DC-only read never looks at them, as before
    SH could ship; asking for bands is what refuses them, by name."""
    columns = columns_with_sh(50, 0)
    names = [*gaussians.CANONICAL_PROPERTIES[:6], *(f"f_rest_{i}" for i in range(10))]
    names += list(gaussians.CANONICAL_PROPERTIES[6:])
    rows = np.zeros(50, dtype=[(name, "<f4") for name in names])
    for name in gaussians.CANONICAL_PROPERTIES:
        rows[name] = columns[name]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 50"]
    header += [f"property float {name}" for name in names] + ["end_header"]
    ply = tmp_path / "odd.ply"
    ply.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    assert gaussians.read_splat(ply).sh_degree == 0
    assert splat_stream.open_splat(ply).sh_degree == 0
    with pytest.raises(ValueError, match="f_rest"):
        gaussians.read_splat(ply, sh_degree=1)
    with pytest.raises(ValueError, match="f_rest"):
        splat_stream.open_splat(ply, sh_degree=None)
