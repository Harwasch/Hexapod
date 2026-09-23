"""The up axis: a capture's own frame into east/north/up, gaussians and all.

Every splat here is synthetic and of known orientation -- a trunk of needle-shaped
gaussians standing along the file's up axis, on a disc of ground -- so "did it land
upright" is a number rather than a picture. The pictures were checked too, on real
downloaded splats; `gaussians.DEFAULT_UP_AXIS` records what they showed.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

import gaussians
from captures_bridge import SplatFormatError, unpack_spz

rng = np.random.default_rng(7)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    return gaussians.matrix_to_quat(m)


def _tree(up: str, *, ground: int = 3000, trunk: int = 600) -> dict[str, np.ndarray]:
    """A trunk 4 m tall standing on a 6 m disc, with `up` the file's up axis.

    The trunk's gaussians are needles: long along their own local x, and each one's
    quaternion turns local x onto the file's up. So after a correct conversion both the
    positions and the needles point along +z.
    """
    radius = 3.0 * np.sqrt(rng.random(ground))
    angle = rng.random(ground) * 2 * math.pi
    enu_ground = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), rng.normal(0, 0.02, ground)], axis=1
    )
    enu_trunk = np.stack(
        [rng.normal(0, 0.05, trunk), rng.normal(0, 0.05, trunk), rng.random(trunk) * 4.0], axis=1
    )
    enu = np.concatenate([enu_ground, enu_trunk]) + np.array([40.0, -25.0, 7.0])
    # The file's frame is ENU turned by the inverse of the conversion under test.
    to_file = gaussians.UP_AXES[up].T
    xyz = enu @ to_file.T
    needle = _matrix_to_quat(to_file @ _rotation_x_to_z())
    count = xyz.shape[0]
    quats = np.tile(needle, (count, 1))
    columns = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "f_dc_0": np.zeros(count),
        "f_dc_1": np.full(count, 0.5),
        "f_dc_2": np.zeros(count),
        "opacity": np.full(count, 3.0),
        "scale_0": np.full(count, math.log(0.2)),
        "scale_1": np.full(count, math.log(0.01)),
        "scale_2": np.full(count, math.log(0.01)),
        "rot_0": quats[:, 0],
        "rot_1": quats[:, 1],
        "rot_2": quats[:, 2],
        "rot_3": quats[:, 3],
    }
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


def _rotation_x_to_z() -> np.ndarray:
    """Local needle axis x onto +z: a -90 degree turn about y."""
    return np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def _splat(columns: dict[str, np.ndarray], fmt: str = "ply") -> gaussians.Splat:
    return gaussians.Splat(
        columns=columns,
        source_format=fmt,  # type: ignore[arg-type]
        source_name=f"tree.{fmt}",
        source_bytes=0,
        source_checksum="sha256:0",
        properties_in=tuple(columns),
        dropped=(),
        non_finite=0,
    )


def _needle_directions(columns: dict[str, np.ndarray]) -> np.ndarray:
    quats = np.stack([columns[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    return np.stack([_quat_to_matrix(q) @ np.array([1.0, 0.0, 0.0]) for q in quats])


@pytest.mark.parametrize("up", ["y", "-y", "z", "-z", "x", "-x"])
def test_a_tree_stands_up_whatever_axis_its_file_called_up(up: str) -> None:
    splat = _splat(_tree(up))

    turned, frame = gaussians.orient(splat, up_axis=up, recentre=False)

    extent = np.ptp(turned.xyz, axis=0)
    # The disc is 6 m across and the trunk 4 m tall: up is the short axis of the whole,
    # and the trunk rises from the ground rather than hanging from it.
    assert extent[2] == pytest.approx(4.0, abs=0.2)
    assert extent[0] == pytest.approx(6.0, abs=0.3) and extent[1] == pytest.approx(6.0, abs=0.3)
    trunk = turned.xyz[-600:]
    assert float(trunk[:, 2].max() - trunk[:, 2].min()) == pytest.approx(4.0, abs=0.1)
    assert float(np.median(trunk[:, 2])) > float(np.median(turned.xyz[:-600, 2])) + 1.0
    # The needles turned with their centres: every one points along +z.
    directions = _needle_directions(turned.columns)
    assert np.abs(directions[:, 2]).min() > 0.999
    assert frame.up_axis == up and frame.up_axis_source == "capture"


def test_the_wrong_axis_leaves_it_lying_down() -> None:
    """The bug this module fixes, reproduced: a y-up file read as z-up."""
    splat = _splat(_tree("y"))

    turned, _ = gaussians.orient(splat, up_axis="z", recentre=False)

    extent = np.ptp(turned.xyz, axis=0)
    assert extent[2] == pytest.approx(6.0, abs=0.3), "the disc is standing on its edge"


def test_the_format_decides_when_nobody_said() -> None:
    spz, frame_spz = gaussians.orient(_splat(_tree("y"), "spz"), recentre=False)
    ply, frame_ply = gaussians.orient(_splat(_tree("-y"), "ply"), recentre=False)

    assert frame_spz.up_axis == "y" and frame_spz.up_axis_source == "format-default (spz)"
    assert frame_ply.up_axis == "-y" and frame_ply.up_axis_source == "format-default (ply)"
    assert np.ptp(spz.xyz, axis=0)[2] == pytest.approx(4.0, abs=0.2)
    assert np.ptp(ply.xyz, axis=0)[2] == pytest.approx(4.0, abs=0.2)


def test_forward_faces_north_for_both_phone_conventions() -> None:
    """A y-up file's forward is -z and a y-down file's is +z; both should face north."""
    assert gaussians.UP_AXES["y"] @ np.array([0.0, 0.0, -1.0]) == pytest.approx([0, 1, 0])
    assert gaussians.UP_AXES["-y"] @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0, 1, 0])
    for rotation in gaussians.UP_AXES.values():
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_heading_is_a_compass_bearing() -> None:
    """Heading 90: the capture's forward (north) now faces east."""
    rotation = gaussians.heading_rotation(90.0)

    assert rotation @ np.array([0.0, 1.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_covariances_transform_exactly_as_a_rotation_says() -> None:
    """Sigma' = A Sigma A^T for every gaussian, for a rotation that is not an axis swap."""
    count = 200
    quats = rng.normal(size=(count, 4))
    scales = rng.normal(-3.0, 0.5, size=(count, 3))
    columns = {
        "x": rng.normal(size=count),
        "y": rng.normal(size=count),
        "z": rng.normal(size=count),
        "scale_0": scales[:, 0],
        "scale_1": scales[:, 1],
        "scale_2": scales[:, 2],
        **{f"rot_{i}": quats[:, i] for i in range(4)},
    }
    typed = {k: np.asarray(v, dtype=np.float32) for k, v in columns.items()}
    rotation = gaussians.heading_rotation(37.0) @ gaussians.UP_AXES["-y"]

    out = gaussians.transform(typed, rotation, np.array([1.0, 2.0, 3.0]), scale=2.5)

    for i in range(count):
        before = _quat_to_matrix(quats[i].astype(np.float64))
        s = np.diag(np.exp(2 * scales[i].astype(np.float32).astype(np.float64)))
        expected = 2.5**2 * rotation @ (before @ s @ before.T) @ rotation.T
        after = _quat_to_matrix(np.array([out[f"rot_{k}"][i] for k in range(4)], np.float64))
        s2 = np.diag(np.exp(2 * np.array([out[f"scale_{k}"][i] for k in range(3)], np.float64)))
        assert after @ s2 @ after.T == pytest.approx(expected, rel=1e-4, abs=1e-9)
    moved = 2.5 * np.stack([typed[a] for a in "xyz"], axis=1) @ rotation.T + [1, 2, 3]
    assert np.stack([out[a] for a in "xyz"], axis=1) == pytest.approx(moved, abs=1e-5)


def test_a_mirror_is_refused() -> None:
    columns = _tree("z")
    with pytest.raises(ValueError, match="mirror"):
        gaussians.transform(columns, np.diag([1.0, 1.0, -1.0]))


def test_an_unknown_axis_is_refused_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match="-y"):
        gaussians.orient(_splat(_tree("z")), up_axis="up")


def test_recentring_puts_the_origin_at_the_footprint_and_the_ground() -> None:
    splat = _splat(_tree("-y"))

    turned, frame = gaussians.orient(splat, up_axis="-y")

    ground = turned.xyz[:-600]
    # The file's origin was 47 m away; the placed coordinate is now the disc's centre,
    # and the disc sits at z = 0 to within its own 2 cm roughness.
    assert np.median(ground[:, 0]) == pytest.approx(0.0, abs=0.3)
    assert np.median(ground[:, 1]) == pytest.approx(0.0, abs=0.3)
    assert np.median(ground[:, 2]) == pytest.approx(0.0, abs=0.1)
    assert frame.recentred
    assert frame.translation == pytest.approx([-40.0, 25.0, -7.0], abs=0.3)


def test_recentring_leaves_the_ground_samples_clamp_unchanged_but_for_position() -> None:
    """The viewer rests `origin.height + z` of each sample on the terrain there. A
    vertical shift moves every z by the same amount, so the offset it computes is the
    same; only where the samples are changes, which is the point."""
    splat = _splat(_tree("-y"))
    near, _ = gaussians.orient(splat, up_axis="-y")
    far, far_frame = gaussians.orient(splat, up_axis="-y", recentre=False)

    zs_near = sorted(s.z for s in gaussians.ground_samples(near, lat=0.0, lon=0.0))
    zs_far = sorted(s.z for s in gaussians.ground_samples(far, lat=0.0, lon=0.0))
    assert len(zs_near) == len(zs_far) > 3
    assert not far_frame.recentred
    assert float(np.median(zs_far)) - float(np.median(zs_near)) == pytest.approx(7.0, abs=0.1)


def _pack_v3(columns: dict[str, np.ndarray]) -> bytes:
    """SPZ version 3, written by a transcription of nianticlabs/spz's packer.

    Independent of the reader under test: `packQuaternionSmallestThree` from
    `src/cc/load-spz.cc`, line by line.
    """
    count = columns["x"].shape[0]
    header = struct.pack("<IIIBBBB", 0x5053474E, 3, count, 0, 12, 0, 0)
    xyz = np.stack([columns[a] for a in "xyz"], axis=1)
    fixed = np.round(xyz * 4096).astype(np.int32).astype(np.uint32).reshape(-1)
    pos = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], axis=1)
    alpha = np.full(count, 200, np.uint8)
    colours = np.full(count * 3, 128, np.uint8)
    scales = np.full(count * 3, 100, np.uint8)
    rotations = bytearray()
    for i in range(count):
        q = np.array([columns[f"rot_{k}"][i] for k in (1, 2, 3, 0)], np.float64)  # xyzw
        q = q / np.linalg.norm(q)
        largest = int(np.argmax(np.abs(q)))
        negate = q[largest] < 0
        comp = largest
        for k in range(4):
            if k != largest:
                negbit = int((q[k] < 0) ^ negate)
                mag = int(float((1 << 9) - 1) * (abs(q[k]) / math.sqrt(0.5)) + 0.5)
                comp = (comp << 10) | (negbit << 9) | mag
        rotations += struct.pack("<I", comp)
    raw = (
        header
        + pos.astype(np.uint8).tobytes()
        + alpha.tobytes()
        + colours.tobytes()
        + scales.tobytes()
        + bytes(rotations)
    )
    return gzip.compress(raw, mtime=0)


def test_spz_version_3_quaternions_read_back_as_the_same_rotations() -> None:
    columns = _tree("y", ground=40, trunk=10)
    quats = rng.normal(size=(50, 4))
    for k in range(4):
        columns[f"rot_{k}"] = quats[:, k].astype(np.float32)

    back = unpack_spz(_pack_v3(columns))

    for i in range(50):
        want = _quat_to_matrix(quats[i])
        got = _quat_to_matrix(np.array([back[f"rot_{k}"][i] for k in range(4)], np.float64))
        # Nine bits of magnitude: better than a degree, which is what v3 exists for.
        angle = math.degrees(math.acos(min(1.0, (np.trace(want.T @ got) - 1) / 2)))
        assert angle < 0.5
    assert back["x"] == pytest.approx(np.round(columns["x"] * 4096) / 4096, abs=1e-6)


def test_spz_version_4_is_refused_by_name() -> None:
    with pytest.raises(SplatFormatError, match="version 4"):
        unpack_spz(b"NGSP" + bytes(28))


def test_the_normalize_stage_converts_and_records_the_frame(tmp_path: Path) -> None:
    from conftest import make_recipe
    from executor import execute
    from runners import RunnerSet
    from workdir import Workdir

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    gaussians.write_ply(upload_dir / "tree.ply", _tree("y"))
    recipe = make_recipe(
        [{"id": "normalize", "impl": "ingest_splat", "params": {"up_axis": "y"}}],
        inputs=["upload"],
    )
    workdir = Workdir.create(tmp_path / "run")
    workdir.input_path("upload").mkdir(parents=True)
    (workdir.input_path("upload") / "tree.ply").write_bytes((upload_dir / "tree.ply").read_bytes())

    execute(recipe, workdir, RunnerSet.local())

    canonical = gaussians.read_splat(workdir.out_dir("normalize") / "canonical.ply")
    assert np.ptp(canonical.xyz, axis=0)[2] == pytest.approx(4.0, abs=0.2)
    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frame"]["upAxis"] == "y"
    assert meta["frame"]["upAxisSource"] == "capture"
    assert meta["extentM"]["up"] == pytest.approx(4.0, abs=0.2)
