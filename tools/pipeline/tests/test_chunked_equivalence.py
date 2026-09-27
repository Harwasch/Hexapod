"""The chunked stages after training give the whole-splat stages' answers, byte for byte.

`quality`, `place`, `thumbnail`, `ground_samples` and Lane 1's `normalize` now read a splat
a chunk at a time (`splat_io`, `splat_stream`, `outofcore`), so that the 2 GB worker and the
CPU box run them in memory that does not grow with the splat. Each is run here beside the
whole-splat path it replaced, on a few hundred thousand gaussians cut into many small
chunks, and the files must match:

* `quality` against `quality_in_memory.py`, a frozen copy of the stage before it was
  chunked -- gated.ply and coverage.ply byte for byte, quality.json value for value;
* `place`, `thumbnail`, `normalize` against `gaussians.orient`/`transform`/`write_ply` and
  `render_thumbnail` -- canonical.ply, coverage_enu.ply and thumbnail.jpg byte for byte;
* `ground_samples` against `gaussians.ground_samples`: the same cells, counts and heights,
  and positions to float32 rounding (the one documented difference: a cell's mean is a
  float64 sum here, a float32 pairwise one there).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import gaussians
import holdout_maths
import outofcore
import quality
import quality_in_memory
import sfm
import splat_io
import splat_stream
import stages
import support_mask
from conftest import make_recipe
from executor import execute
from runners import RunnerSet
from synthetic_scene import TARGET, look_at, ring, scene, write_model
from workdir import Workdir

#: Small, and prime, so every stage crosses many chunk boundaries at odd places -- and
#: still above `quality.CULL_MIN_ROWS`, so the cameras are culled per cell.
CHUNK = 37_003


class _Context:
    """Enough of `StageContext` to run the frozen whole-splat quality stage."""

    def __init__(self, inputs: dict[str, Path], out: Path, params: dict[str, Any]) -> None:
        self._inputs = inputs
        self._out = out
        self.params = params
        self.lines: list[str] = []
        out.mkdir(parents=True, exist_ok=True)

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def input(self, name: str) -> Path:
        return self._inputs[name]

    def has_input(self, name: str) -> bool:
        return name in self._inputs

    def output(self, name: str) -> Path:
        return self._out / name

    def log(self, message: str) -> None:
        self.lines.append(message)


def _splat(n: int, *, seed: int, shuffle: bool) -> dict[str, np.ndarray]:
    """The synthetic orbit's gaussians, made less tidy: random colours, opacities (a
    tenth of them too faint to occlude), shapes and orientations, a far wall no ring
    camera faces, and a few rows with no finite position."""
    rng = np.random.default_rng(seed)
    columns, _ = scene(n, seed=seed)
    wall = rng.random(n) < 0.08
    columns["x"] = np.where(wall, rng.uniform(-6, 6, n), columns["x"]).astype(np.float32)
    columns["y"] = np.where(wall, 9.0 + rng.normal(0, 0.05, n), columns["y"]).astype(np.float32)
    columns["z"] = np.where(wall, rng.uniform(0, 3, n), columns["z"]).astype(np.float32)
    for index in range(3):
        columns[f"f_dc_{index}"] = rng.normal(0, 0.8, n).astype(np.float32)
        columns[f"scale_{index}"] = rng.normal(-5.0, 0.7, n).astype(np.float32)
    opacity = rng.normal(2.0, 1.0, n)
    opacity[rng.random(n) < 0.1] = -3.0
    columns["opacity"] = opacity.astype(np.float32)
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    for index in range(4):
        columns[f"rot_{index}"] = quats[:, index].astype(np.float32)
    broken = rng.choice(n, size=5, replace=False)
    columns["x"][broken[:3]] = np.nan
    columns["z"][broken[3:]] = np.inf
    if shuffle:
        order = rng.permutation(n)
        columns = {name: values[order] for name, values in columns.items()}
    return columns


def _walk(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cameras along a line, all facing the same way: an ROI the optical axes cannot
    fix, so `estimate_roi` falls back to each camera's median depth."""
    out = []
    for index in range(count):
        centre = np.array([-1.5 + 3.0 * index / max(1, count - 1), -2.5, 0.8])
        out.append((look_at(centre, centre + np.array([0.0, 1.0, -0.2])), centre))
    return out


def _inputs(
    root: Path, *, n: int, cameras: list[tuple[np.ndarray, np.ndarray]], held: bool, metric: bool
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    columns = _splat(n, seed=n, shuffle=held)
    trained = root / "trained.ply"
    gaussians.write_ply(trained, columns)
    write_model(root / "poses", cameras)
    (root / "train_metrics.json").write_text(json.dumps({"psnr": 21.7}))
    frame = {
        "source": "exif-gps-similarity" if metric else "camera-up",
        "scale": 1.7 if metric else 1.0,
        "rotation": np.eye(3).tolist(),
        "translationM": [0.4, -0.2, 0.1] if metric else None,
        "recentre": not metric,
    }
    georef: dict[str, Any] = {"lat": 46.8, "lon": -91.9, "height": 180.0, "frame": frame}
    if metric:
        georef["scaleSource"] = "exif-gps"
    (root / "georef.json").write_text(json.dumps(georef))
    inputs = {
        "trained.ply": trained,
        "poses": root / "poses",
        "georef.json": root / "georef.json",
        "train_metrics.json": root / "train_metrics.json",
    }
    if held:
        rng = np.random.default_rng(7)
        error = rng.gamma(2.0, 0.03, n).astype(np.float32)
        error[rng.random(n) < 0.2] = np.nan
        weight = rng.exponential(4.0, n).astype(np.float32)
        summary = {
            "status": "ok",
            "views": 3,
            "meanPsnr": 22.1,
            "perView": [{"name": "a.jpg", "psnr": 22.0, "bias": 0.01, "sharpness": 1.0}],
        }
        holdout_maths.write(root / "holdout", error, weight, None, summary)
        inputs["holdout"] = root / "holdout"
    return inputs


QUALITY_PARAMS: dict[str, Any] = {
    "mode": "refine",
    "bar": "balanced",
    "keep_max_holdout_error": 0.25,
    "coverage_points": 40_000,
}


def _run_chunked(
    root: Path, inputs: dict[str, Path], params: dict[str, Any], chunk: int
) -> Workdir:
    workdir = Workdir.create(root)
    for name, path in inputs.items():
        target = workdir.input_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for child in path.iterdir():
                (target / child.name).write_bytes(child.read_bytes())
        else:
            target.write_bytes(path.read_bytes())
    chunked = {"chunk_gaussians": chunk}
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": {**params, **chunked}},
                {"id": "place", "impl": "place_splat", "params": chunked},
                {"id": "thumbnail", "impl": "splat_thumbnail", "params": chunked},
                {"id": "ground", "impl": "splat_ground", "params": {**chunked, "cell_m": 0.5}},
            ],
            inputs=sorted(inputs),
        ),
        workdir,
        RunnerSet.local(),
    )
    return workdir


def _place_in_memory(gated: Path, georef: dict[str, Any], out: Path) -> gaussians.Frame | None:
    """`place_splat` as it was: read whole, transform, orient, write."""
    trained = gaussians.read_splat(gated)
    frame = georef["frame"]
    rotation = np.asarray(frame.get("rotation") or np.eye(3), dtype=np.float64)
    scale = float(frame.get("scale") or 1.0)
    offset = frame.get("translationM")
    translation = None if offset is None else np.asarray(offset, dtype=np.float64)
    columns = gaussians.transform(trained.columns, rotation, translation, scale)
    placed = gaussians.Splat(
        columns=columns,
        source_format=trained.source_format,
        source_name=trained.source_name,
        source_bytes=trained.source_bytes,
        source_checksum=trained.source_checksum,
        properties_in=trained.properties_in,
        dropped=trained.dropped,
        non_finite=trained.non_finite,
    )
    recentred = None
    if frame.get("recentre"):
        placed, recentred = gaussians.orient(placed, up_axis="z")
    gaussians.write_ply(out, placed.columns)
    return recentred


def _same_json(chunked: Any, whole: Any, where: str = "") -> None:
    assert type(chunked) is type(whole), f"{where}: {chunked!r} != {whole!r}"
    if isinstance(whole, dict):
        assert set(chunked) == set(whole), f"{where}: keys {sorted(chunked)} != {sorted(whole)}"
        for key in whole:
            _same_json(chunked[key], whole[key], f"{where}.{key}")
    elif isinstance(whole, list):
        assert len(chunked) == len(whole), f"{where}: {len(chunked)} != {len(whole)} items"
        for index, (left, right) in enumerate(zip(chunked, whole, strict=True)):
            _same_json(left, right, f"{where}[{index}]")
    else:
        assert chunked == whole, f"{where}: {chunked!r} != {whole!r}"


@pytest.mark.parametrize(
    ("label", "n", "held", "metric", "chunk"),
    [
        ("orbit, held-out, shuffled", 300_000, True, False, CHUNK),
        ("orbit, held-out, default chunks", 300_000, True, False, splat_io.CHUNK),
        ("walk, metric", 120_000, False, True, CHUNK),
    ],
)
def test_the_chunked_stages_write_what_the_whole_splat_stages_wrote(
    tmp_path: Path, label: str, n: int, held: bool, metric: bool, chunk: int
) -> None:
    cameras = ring(24, radius=1.5, height=1.0) if held else _walk(20)
    inputs = _inputs(tmp_path / "inputs", n=n, cameras=cameras, held=held, metric=metric)
    params = dict(QUALITY_PARAMS)

    whole = tmp_path / "whole"
    context = _Context(inputs, whole / "quality", params)
    quality_in_memory.support_gate(cast(Any, context))
    workdir = _run_chunked(tmp_path / "chunked", inputs, params, chunk)
    chunked = workdir.out_dir("quality")

    # quality: the gated splat and the coverage cloud, byte for byte; the report, value
    # for value (so a difference names the field).
    for name in ("gated.ply", "coverage.ply"):
        assert (chunked / name).read_bytes() == (whole / "quality" / name).read_bytes(), name
    document = json.loads((chunked / "quality.json").read_text())
    reference = json.loads((whole / "quality" / "quality.json").read_text())
    _same_json(document, reference)
    assert document["gaussians"]["in"] == n, label
    if not held:
        # The walk's ROI really did come from the cameras' median depths.
        assert document["pointedRoi"]["method"] == "median-depth"
    else:
        assert document["heldOut"]["status"] == "ok"

    # place: canonical.ply and the coverage overlay, byte for byte.
    georef = json.loads(inputs["georef.json"].read_text())
    canonical = whole / "canonical.ply"
    recentred = _place_in_memory(whole / "quality" / "gated.ply", georef, canonical)
    placed_dir = workdir.out_dir("place")
    assert (placed_dir / "canonical.ply").read_bytes() == canonical.read_bytes()
    frame = georef["frame"]
    xyz, tiers = quality.read_coverage(whole / "quality" / "coverage.ply")

    moved = stages.place_points(
        xyz,
        np.asarray(frame["rotation"], dtype=np.float64),
        None if frame["translationM"] is None else np.asarray(frame["translationM"]),
        float(frame["scale"]),
        recentred,
    )
    quality.write_coverage(whole / "coverage_enu.ply", moved, tiers)
    assert (placed_dir / "coverage_enu.ply").read_bytes() == (
        whole / "coverage_enu.ply"
    ).read_bytes()

    # thumbnail: the same JPEG.
    gaussians.render_thumbnail(gaussians.read_splat(canonical), whole / "thumbnail.jpg")
    assert (workdir.out_dir("thumbnail") / "thumbnail.jpg").read_bytes() == (
        whole / "thumbnail.jpg"
    ).read_bytes()

    # ground samples: the same cells and heights; positions to float32 rounding.
    samples = gaussians.ground_samples(
        gaussians.read_splat(canonical), lat=46.8, lon=-91.9, cell_m=0.5
    )
    written = json.loads((workdir.out_dir("ground") / "ground_samples.json").read_text())
    assert len(written["samples"]) == len(samples) > 0
    for got, want in zip(written["samples"], samples, strict=True):
        assert (got["n"], got["z"]) == (want.n, round(want.z, 3))
        # A micrometre on the ground is ~1e-11 degrees.
        assert got["lon"] == pytest.approx(want.lon, abs=1e-10)
        assert got["lat"] == pytest.approx(want.lat, abs=1e-10)


def test_measure_support_is_the_whole_splat_measurement_with_cameras_culled(tmp_path: Path) -> None:
    """The chunked support pass (occluders regrouped into blocks, cameras culled per cell)
    against the frozen whole-array one, on points that straddle every frustum edge."""
    rng = np.random.default_rng(3)
    n = 60_000
    xyz = rng.uniform([-4, -4, -1], [4, 4, 3], size=(n, 3)).astype(np.float32)
    xyz[:500] = TARGET + rng.normal(0, 1e-3, size=(500, 3))  # a dense clump
    xyz[500:520] = np.nan
    alpha = rng.random(n).astype(np.float32)
    footprint = np.exp(rng.normal(-4, 1, n)).astype(np.float32)
    rigs = [ring(12, radius=2.0, height=1.2), _walk(9)]
    for index, rig in enumerate(rigs):
        directory = tmp_path / f"support-{index}"
        model = sfm.read_model(write_model(directory / "poses", rig))
        cameras = quality.Cameras.from_model(model)
        frozen = quality_in_memory.Cameras.from_model(model)
        for chunk in (7_001, 50_000):
            chunked = quality.measure_support(xyz, alpha, cameras, footprint=footprint, chunk=chunk)
            whole = quality_in_memory.measure_support(
                xyz, alpha, frozen, footprint=footprint, chunk=chunk
            )
            np.testing.assert_array_equal(chunked.views, whole.views)
            np.testing.assert_array_equal(chunked.spread_deg, whole.spread_deg)
            np.testing.assert_array_equal(chunked.gsd, whole.gsd)
            np.testing.assert_array_equal(chunked.camera_depth, whole.camera_depth)
            assert int((chunked.views > 0).sum()) > n // 4


@pytest.mark.parametrize("budget", [outofcore.BUDGET, 3_000])
def test_a_streamed_mask_is_the_mask_built_in_memory(budget: int) -> None:
    rng = np.random.default_rng(9)
    points = np.concatenate(
        [
            rng.uniform([0, 0, 0], [2, 1, 0.01], size=(40_000, 3)),
            rng.uniform([3, 0, 0], [3.01, 1, 1], size=(20_000, 3)),
            [[np.nan, 0, 0]],
        ]
    )
    prints = np.full(points.shape[0], 0.002)

    def chunks() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, points.shape[0], 9_999):
            yield points[start : start + 9_999], prints[start : start + 9_999]

    original = outofcore.BUDGET
    outofcore.BUDGET = budget
    try:
        for kwargs in ({}, {"max_voxels": 5_000}):
            streamed = support_mask.build_from(
                support_mask.StreamedPoints(chunks),
                **kwargs,
            )
            whole = support_mask.build(points, prints, **kwargs)
            assert streamed is not None and whole is not None
            assert streamed.to_dict() == whole.to_dict()
    finally:
        outofcore.BUDGET = original


def test_lane_1_ingest_of_a_ply_is_the_whole_file_orient(tmp_path: Path) -> None:
    """`normalize` of a phone PLY with vertex colours and SH bands, chunked, against
    `read_splat` + `orient` + `write_ply` of the whole file."""
    rng = np.random.default_rng(12)
    n = 90_000
    columns = _splat(n, seed=5, shuffle=True)
    names = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "alpha"]
    names += [f"f_rest_{i}" for i in range(9)]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    dtype = np.dtype(
        [(name, "u1" if name in ("red", "green", "blue", "alpha") else "<f4") for name in names]
    )
    rows = np.zeros(n, dtype=dtype)
    for name in names:
        if name in columns:
            rows[name] = columns[name]
        elif name in ("red", "green", "blue", "alpha"):
            rows[name] = rng.integers(0, 256, n)
        else:
            rows[name] = rng.normal(size=n)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(
        f"property {'uchar' if dtype[name].kind == 'u' else 'float'} {name}\n" for name in names
    )
    header += "end_header\n"
    upload = tmp_path / "run" / "inputs" / "upload"
    workdir = Workdir.create(tmp_path / "run")
    upload = workdir.input_path("upload")
    upload.mkdir(parents=True, exist_ok=True)
    (upload / "phone.ply").write_bytes(header.encode("ascii") + rows.tobytes())
    params = {"chunk_gaussians": CHUNK, "up_axis": "y", "heading_deg": 30.0}
    execute(
        make_recipe(
            [{"id": "normalize", "impl": "ingest_splat", "params": params}], inputs=["upload"]
        ),
        workdir,
        RunnerSet.local(),
    )
    read = gaussians.read_splat(upload / "phone.ply")
    splat, frame = gaussians.orient(read, up_axis="y", heading_deg=30.0)
    gaussians.write_ply(tmp_path / "canonical.ply", splat.columns)
    out = workdir.out_dir("normalize")
    assert (out / "canonical.ply").read_bytes() == (tmp_path / "canonical.ply").read_bytes()
    meta = json.loads((out / "source_meta.json").read_text())
    low, high = splat.bbox()
    scales = np.stack([splat.columns[f"scale_{i}"] for i in range(3)], axis=1)
    finite = np.isfinite(scales).all(axis=1)
    assert meta["bboxLocalM"] == {"min": low, "max": high}
    assert meta["nonFinite"] == read.non_finite == 5
    assert meta["medianGaussianM"] == round(float(np.median(np.exp(scales[finite]))), 6)
    assert meta["checksum"] == read.source_checksum
    assert meta["properties"] == list(read.properties_in)
    assert meta["dropped"] == list(read.dropped)
    assert meta["frame"]["translationM"] == frame.to_dict()["translationM"]
    assert not math.isnan(meta["frame"]["headingDeg"])


def test_a_chunked_transform_rounds_every_row_as_a_whole_one() -> None:
    """`gaussians.transform` per axis, so a row's result does not depend on how many rows
    it was transformed with (a BLAS product may change kernels with the row count)."""
    columns = _splat(20_011, seed=2, shuffle=False)
    rotation = gaussians.heading_rotation(37.0) @ gaussians.UP_AXES["-y"]
    whole = gaussians.transform(columns, rotation, np.array([1.0, -2.0, 0.5]), 3.3)
    step = splat_stream.Step(rotation, np.array([1.0, -2.0, 0.5]), 3.3)
    for size in (1, 7, 4096):
        parts = [
            step.apply({k: v[s : s + size] for k, v in columns.items()})
            for s in range(0, 20_011, size)
        ]
        for name in gaussians.CANONICAL_PROPERTIES:
            joined = np.concatenate([part[name] for part in parts])
            assert joined.tobytes() == whole[name].tobytes(), (size, name)
