from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import arkit_ingest
import capture_bundle as bundle
import sfm
from contracts import StageContext


def frame(index: int = 0) -> dict[str, Any]:
    return {
        "index": index,
        "timestamp": index / 30,
        "transform": np.eye(4).ravel().tolist(),
        "intrinsics": [500, 0, 160, 0, 520, 120, 0, 0, 1],
        "imageWidth": 320,
        "imageHeight": 240,
        "trackingState": "normal",
        "depth": {
            "file": f"depth_{index}.f16",
            "confidence": f"depth_{index}.u8",
            "width": 2,
            "height": 1,
        },
    }


def make_bundle(
    root: Path,
    frames: list[dict[str, Any]] | None = None,
    locations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root.mkdir()
    frames = frames or [frame()]
    (root / "rgb.mov").write_bytes(b"fake video; parser does not decode")
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    for record in frames:
        (root / record["depth"]["file"]).write_bytes(np.array([1, 2], dtype="<f2").tobytes())
        (root / record["depth"]["confidence"]).write_bytes(bytes([1, 2]))
    if locations is not None:
        (root / "location.jsonl").write_text("".join(json.dumps(f) + "\n" for f in locations))
    manifest = {
        "version": 1,
        "format": "hex-capture",
        "device": "synthetic",
        "startedAt": "2026-01-01T00:00:00Z",
        "video": "rgb.mov",
        "frameLog": "frames.jsonl",
        "frameCount": len(frames),
        "depthFrameCount": len(frames),
        "smoothedDepthFrameCount": 0,
        "files": [{"path": p.name, "bytes": p.stat().st_size} for p in sorted(root.iterdir())],
    }
    if locations is not None:
        manifest["locationLog"] = "location.jsonl"
        valid = [fix for fix in locations if fix["horizontalAccuracyM"] > 0]
        manifest["location"] = (
            min(valid, key=lambda fix: fix["horizontalAccuracyM"]) if valid else None
        )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def archive(root: Path, target: Path) -> None:
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(root.iterdir()):
            tar.add(path, arcname=path.name)


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, -0.7])
def test_pose_projection_inverse_and_quaternion(angle: float) -> None:
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    transform[:3, 3] = [2, -3, 4]
    r, t = bundle.world_to_camera(transform)
    np.testing.assert_allclose(bundle.camera_to_world(r, t), transform, atol=1e-12)
    np.testing.assert_allclose(sfm.quat_to_matrix(bundle.quaternion(r).tolist()), r, atol=1e-12)
    camera_point = np.array([0.4, 0.3, -2])
    world_point = transform[:3, :3] @ camera_point + transform[:3, 3]
    colmap_point = r @ world_point + t
    np.testing.assert_allclose(colmap_point, [0.4, -0.3, 2], atol=1e-12)
    ar_pixel = [
        500 * camera_point[0] / -camera_point[2] + 160,
        520 * -camera_point[1] / -camera_point[2] + 120,
    ]
    colmap_pixel = [
        500 * colmap_point[0] / colmap_point[2] + 160,
        520 * colmap_point[1] / colmap_point[2] + 120,
    ]
    np.testing.assert_allclose(ar_pixel, colmap_pixel, atol=1e-10)
    np.testing.assert_allclose(-r.T @ t, transform[:3, 3], atol=1e-12)


def test_axis_flip_and_reject_reflection() -> None:
    r, t = bundle.world_to_camera(np.eye(4))
    np.testing.assert_array_equal(r, np.diag([1, -1, -1]))
    np.testing.assert_array_equal(t, np.zeros(3))
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        bundle.world_to_camera(reflection)


def test_bundle_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = make_bundle(source, [frame(0), frame(1)])
    tar = tmp_path / "capture.hexcap"
    archive(source, tar)
    parsed = bundle.unpack(tar, tmp_path / "parsed")
    assert parsed.manifest == manifest
    assert parsed.frames == [frame(0), frame(1)]
    assert (parsed.directory / "depth_0.f16").read_bytes() == np.array(
        [1, 2], dtype="<f2"
    ).tobytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "count",
        "size",
        "path",
        "intrinsics",
        "index",
        "timestamp",
        "extra",
        "depth",
        "nan",
    ],
)
def test_malformed_bundle(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source"
    manifest = make_bundle(source)
    if mutation == "version":
        manifest["version"] = 2
    elif mutation == "count":
        manifest["frameCount"] = 2
    elif mutation == "size":
        manifest["files"][0]["bytes"] += 1
    elif mutation == "path":
        manifest["files"][0]["path"] = "../outside"
    elif mutation == "extra":
        (source / "undeclared").write_bytes(b"extra")
    else:
        record = frame()
        if mutation == "intrinsics":
            record["intrinsics"] = [0] * 9
        elif mutation == "index":
            record["index"] = 1
        elif mutation == "timestamp":
            record["timestamp"] = 1
        elif mutation == "nan":
            record["transform"][0] = float("nan")
        elif mutation == "depth":
            record["depth"]["width"] = 3
        (source / "frames.jsonl").write_text(json.dumps(record) + "\n")
        for entry in manifest["files"]:
            entry["bytes"] = (source / entry["path"]).stat().st_size
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.read_bundle(source)


@pytest.mark.parametrize(
    "kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "directory"]
)
def test_archive_rejects_unsafe_members(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "bad.hexcap"
    with tarfile.open(path, "w") as tar:
        entry = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape"}.get(kind, "file"))
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../escape"
        elif kind == "hardlink":
            entry.type = tarfile.LNKTYPE
            entry.linkname = "../escape"
        elif kind == "directory":
            entry.type = tarfile.DIRTYPE
        tar.addfile(entry, io.BytesIO())
        if kind == "duplicate":
            tar.addfile(entry, io.BytesIO())
    with pytest.raises(ValueError):
        bundle.unpack(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_seed_uses_database_image_ids_and_per_frame_calibration(tmp_path: Path) -> None:
    db_path = tmp_path / "colmap.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        db.execute(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, "
            "width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)"
        )
        db.execute("INSERT INTO images VALUES (17, 'frame_00000000.png', 1)")
        db.execute("INSERT INTO images VALUES (3, 'frame_00000001.png', 1)")
    records = [frame(0), frame(1)]
    records[1]["intrinsics"][0] = 600
    records[1]["transform"][3] = 2
    model = tmp_path / "model"
    arkit_ingest.seed_model(db_path, records, model)
    lines = (model / "images.txt").read_text().splitlines()
    assert lines[0].startswith("17 ") and lines[2].startswith("3 ")
    assert lines[1] == lines[3] == ""
    assert lines[2].split()[5:8] == ["-2.0", "0.0", "0.0"]
    with sqlite3.connect(db_path) as db:
        params = db.execute("SELECT params FROM cameras WHERE camera_id=3").fetchone()[0]
    np.testing.assert_array_equal(np.frombuffer(params, dtype="<f8"), [600, 520, 160, 120])


def normalized_bundle(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    from test_normalize import make_clip

    from conftest import make_recipe
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    source = tmp_path / "source"
    make_bundle(source, records, locations=location_fixes())
    (source / "rgb.mov").unlink()
    make_clip(source / "rgb.mov", frames=len(records), fps=30)
    # make_clip creates stills beside its output; only archive the declared payload.
    import shutil

    shutil.rmtree(source / "stills")
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        entry["bytes"] = (source / entry["path"]).stat().st_size
    (source / "manifest.json").write_text(json.dumps(manifest))
    work = Workdir.create(tmp_path / "run")
    upload = work.input_path("upload")
    upload.mkdir()
    archive(source, upload / "capture.hexcap")
    recipe = make_recipe(
        [{"id": "normalize", "impl": "ffmpeg_frames", "params": {"capture_bundle": True}}],
        inputs=["upload"],
    )
    execute(recipe, work, RunnerSet(cpu=LocalRunner()))
    return work.stage_dir("normalize") / "out"


def test_normalize_keeps_every_frame_at_native_size(tmp_path: Path) -> None:
    records = [frame(i) for i in range(4)]
    records[0]["trackingState"] = "limited:initializing"
    records[2]["trackingState"] = "limited:excessiveMotion"
    out = normalized_bundle(tmp_path, records)
    assert len(list((out / "frames").glob("*.png"))) == 4
    source_meta = json.loads((out / "source_meta.json").read_text())
    frames_meta = json.loads((out / "frames_meta.json").read_text())
    assert source_meta["location"] == {
        "lat": 37.002,
        "lon": -122,
        "altitudeM": 102,
        "accuracyM": 5,
        "source": "capture-bundle-corelocation",
    }
    assert frames_meta["arkit"] == records
    assert frames_meta["locations"] == location_fixes()
    assert len(frames_meta["frames"]) == 4
    # Exercise the actual georeference readers, including the historical altitude key.
    from types import SimpleNamespace

    import stages

    ctx = cast(
        StageContext, SimpleNamespace(has_input=lambda _: True, input=lambda name: out / name)
    )
    assert stages._sole_location(ctx, ()) == (37.002, -122, 102)
    assert stages._video_location_accuracy(ctx) == 5


def test_placement_keeps_arkit_metres_and_gravity(tmp_path: Path) -> None:
    from typing import cast

    import stages
    from contracts import StageContext

    class Defaults:
        def param(self, name: str, default: object = None) -> object:
            return default

    (tmp_path / "poses.json").write_text(json.dumps({"source": "hex-capture-v1"}))
    placed = stages._frame_from_poses(cast(StageContext, Defaults()), tmp_path, estimate_scale=True)
    assert placed["scale"] == 1.0
    assert placed["scaleSource"] == "arkit"
    assert placed["scaleEstimate"] is None
    np.testing.assert_allclose(np.asarray(placed["rotation"]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def pose_context(tmp_path: Path, records: list[dict[str, Any]], **params: Any) -> Any:
    from types import SimpleNamespace

    (tmp_path / "frames_meta.json").write_text(
        json.dumps({"format": "hex-capture-v1", "arkit": records})
    )
    (tmp_path / "frames").mkdir()
    (tmp_path / "poses").mkdir()
    return SimpleNamespace(
        work_dir=tmp_path,
        input=lambda name: tmp_path / name,
        output=lambda name: tmp_path / name,
        param=lambda name, default=None: params.get(name, default),
    )


@pytest.mark.parametrize("minimum,normal,needed", [(10, 3, 10), (0, 1, 2)])
def test_pose_requires_enough_normal_frames(
    tmp_path: Path, minimum: int, normal: int, needed: int
) -> None:
    records = [frame(i) for i in range(normal + 2)]
    for record in records[:2]:
        record["trackingState"] = "limited:initializing"
    ctx = pose_context(tmp_path, records, min_tracked_frames=minimum)
    with pytest.raises(
        ValueError,
        match=(
            f"capture bundle has {normal} of {normal + 2} frames with normal "
            f"ARKit tracking; at least {needed} are needed"
        ),
    ):
        arkit_ingest.pose(ctx)


@pytest.mark.parametrize("phone_like", [False, True])
def test_pose_excludes_limited_frames_from_database_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phone_like: bool
) -> None:
    from types import SimpleNamespace

    records = [frame(i) for i in range(13)]
    records[0]["trackingState"] = records[1]["trackingState"] = "limited:initializing"
    records[7]["trackingState"] = "limited:excessiveMotion"
    if phone_like:
        records = phone_frames()
        out = normalized_bundle(tmp_path, records)
        assert len(list((out / "frames").glob("*.png"))) == len(records)
        # Preserve original observations, including degenerate untracked transforms.
        assert json.loads((out / "frames_meta.json").read_text())["arkit"] == records
        records = json.loads((out / "frames_meta.json").read_text())["arkit"]
    normal = [f for f in records if f["trackingState"] == "normal"]
    names = [f"frame_{f['index']:08d}.png" for f in normal]
    ctx = pose_context(tmp_path, records)
    if phone_like:
        ctx.input = lambda name: out / name
    commands = []

    def run(argv: list[str]) -> None:
        commands.append(argv)
        if argv[1] == "feature_extractor":
            selected = Path(argv[argv.index("--image_list_path") + 1]).read_text().splitlines()
            assert selected == names
            with sqlite3.connect(tmp_path / "arkit.db") as db:
                db.execute(
                    "CREATE TABLE images (image_id INTEGER PRIMARY KEY, "
                    "name TEXT, camera_id INTEGER)"
                )
                db.execute(
                    "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, "
                    "width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)"
                )
                db.executemany(
                    "INSERT INTO images VALUES (?, ?, 1)",
                    [(100 + i, name) for i, name in enumerate(selected)],
                )
        elif argv[1] == "point_triangulator":
            seed = (tmp_path / "arkit-seed" / "images.txt").read_text().splitlines()[::2]
            assert [line.split()[-1] for line in seed] == names

    ctx.run = run
    monkeypatch.setattr(sfm, "colmap_exe", lambda: "colmap-stand-in")
    images = []
    for record in normal:
        transform = np.array(record["transform"]).reshape(4, 4)
        rotation = transform[:3, :3]
        if phone_like:
            error = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            assert 1e-6 < error < 1.1e-4
        u, _, vt = np.linalg.svd(rotation)
        corrected = u @ vt
        assert np.linalg.det(corrected) > 0
        r = np.diag([1, -1, -1]) @ corrected.T
        t = -r @ transform[:3, 3]
        images.append(SimpleNamespace(rotation=r, tvec=t, to_dict=lambda: {}))
    model = SimpleNamespace(
        registered=10,
        points3d=1,
        cameras=[],
        images=images,
        by_name=lambda: dict(zip(names, images, strict=True)),
    )
    monkeypatch.setattr(sfm, "read_model", lambda _: model)
    outcome = arkit_ingest.pose(ctx)
    seed_lines = (tmp_path / "arkit-seed" / "images.txt").read_text().splitlines()[::2]
    for line, expected in zip(seed_lines, images, strict=True):
        values = [float(v) for v in line.split()[1:8]]
        np.testing.assert_allclose(sfm.quat_to_matrix(values[:4]), expected.rotation, atol=1e-12)
        np.testing.assert_allclose(values[4:], expected.tvec, atol=1e-12)
    report = json.loads((tmp_path / "poses" / "poses.json").read_text())
    assert report["tracking"] == {
        "total": len(records),
        "normal": 10,
        "excluded": 4 if phone_like else 3,
        "excludedByState": (
            {"notAvailable": 3, "limited:initializing": 1}
            if phone_like
            else {"limited:initializing": 2, "limited:excessiveMotion": 1}
        ),
    }
    assert outcome.metrics["excluded"] == (4 if phone_like else 3)
    excluded_counts = outcome.metrics["excludedByState"]
    assert isinstance(excluded_counts, str)
    assert json.loads(excluded_counts) == report["tracking"]["excludedByState"]
    assert len(commands) == 3


def phone_frames() -> list[dict[str, Any]]:
    """Synthetic float32 poses with prescribed drift, plus tracking startup records."""
    records = [frame(i) for i in range(14)]
    for i, record in enumerate(records):
        c, s = np.cos(i * 0.13), np.sin(i * 0.13)
        transform = np.eye(4)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        # Model accumulated float32 drift, not just one cast of an exact rotation.
        drift = np.geomspace(1e-6, 5e-5, len(records))[i]
        rotation[0] *= 1 + drift
        rotation[1, 2] += drift / 2
        transform[:3, :3] = rotation
        transform[:3, 3] = [i * 0.03, 1.7, -2]
        transform[3, 0] = 2e-7
        record["transform"] = transform.astype(np.float32).astype(np.float64).ravel().tolist()
    for i, transform in enumerate((np.eye(4), np.zeros((4, 4)), np.diag([0, 1, 0, 1]))):
        records[i]["trackingState"] = "notAvailable"
        records[i]["transform"] = transform.ravel().tolist()
    records[3]["trackingState"] = "limited:initializing"
    records[3]["transform"] = np.zeros((4, 4)).ravel().tolist()
    return records


@pytest.mark.parametrize("kind", ["scale", "shear", "reflection", "last-row", "nan"])
def test_normal_pose_rejection_names_frame_and_measurement(tmp_path: Path, kind: str) -> None:
    records = [frame(i) for i in range(3)]
    transform = np.eye(4)
    if kind == "scale":
        transform[0, 0] = 1.1
    elif kind == "shear":
        transform[0, 1] = 0.02
    elif kind == "reflection":
        transform[0, 0] = -1
    elif kind == "last-row":
        transform[3, 0] = 2e-6
    else:
        transform[0, 0] = np.nan
    records[2]["transform"] = transform.ravel().tolist()
    source = tmp_path / "source"
    make_bundle(source, records)
    with pytest.raises(ValueError, match=r"proper rigid transform.*frame 2.*error="):
        bundle.read_bundle(source)


@pytest.mark.parametrize("field,bound", [("rotation", 1e-3), ("last-row", 1e-6)])
@pytest.mark.parametrize("factor,accepted", [(0.999, True), (1.001, False)])
def test_pose_precision_policy_boundary(
    field: str, bound: float, factor: float, accepted: bool
) -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1, -2, 3]
    if field == "rotation":
        transform[0, 0] = np.sqrt(1 + factor * bound)
    else:
        transform[3, 0] = factor * bound
    if not accepted:
        with pytest.raises(ValueError, match="proper rigid"):
            bundle.world_to_camera(transform)
        return
    r, t = bundle.world_to_camera(transform)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1)
    np.testing.assert_allclose(-r.T @ t, transform[:3, 3], atol=1e-12)


@pytest.mark.parametrize("state", ["notAvailable", "limited:initializing"])
@pytest.mark.parametrize("transform", [[0.0] * 15, [float("inf")] * 16])
def test_untracked_pose_still_requires_finite_4x4(
    tmp_path: Path, state: str, transform: list[float]
) -> None:
    record = frame()
    record.update(trackingState=state, transform=transform)
    source = tmp_path / "source"
    make_bundle(source, [record])
    with pytest.raises(ValueError):
        bundle.read_bundle(source)


def location_fixes() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": 1800000000 + i,
            "lat": 37 + i / 1000,
            "lon": -122,
            "altitudeM": 100 + i,
            "horizontalAccuracyM": accuracy,
            "verticalAccuracyM": 10,
        }
        for i, accuracy in enumerate([-1, 20, 5, 5])
    ]


@pytest.mark.parametrize("vertical", [10, -1])
def test_bundle_location_round_trip(tmp_path: Path, vertical: int) -> None:
    fixes = location_fixes()
    fixes[2]["verticalAccuracyM"] = vertical
    source = tmp_path / "source"
    make_bundle(source, locations=fixes)
    tar = tmp_path / "capture.hexcap"
    archive(source, tar)
    parsed = bundle.unpack(tar, tmp_path / "out")
    assert parsed.locations == fixes
    assert parsed.location == {
        "lat": 37.002,
        "lon": -122,
        "altitudeM": 102 if vertical >= 0 else None,
        "accuracyM": 5,
        "source": "capture-bundle-corelocation",
    }


@pytest.mark.parametrize("mutation", ["latitude", "nan", "summary", "missing-log", "log-path"])
def test_bundle_rejects_invalid_locations(tmp_path: Path, mutation: str) -> None:
    fixes = location_fixes()
    if mutation == "latitude":
        fixes[0]["lat"] = 91
    elif mutation == "nan":
        fixes[0]["timestamp"] = float("nan")
    source = tmp_path / "source"
    manifest = make_bundle(source, locations=fixes)
    if mutation == "summary":
        manifest["location"] = fixes[1]
    elif mutation == "missing-log":
        del manifest["locationLog"]
    elif mutation == "log-path":
        manifest["locationLog"] = "../location.jsonl"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.read_bundle(source)


def test_no_usable_bundle_location(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_bundle(source, locations=location_fixes()[:1])
    parsed = bundle.read_bundle(source)
    assert parsed.locations == location_fixes()[:1]
    assert parsed.location is None


def test_optional_live_splat_is_inventoried_validated_and_not_a_frame(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    manifest = make_bundle(root)
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
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
    ]
    ply = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        + "".join("property float " + name + "\n" for name in names)
        + "end_header\n"
    ).encode()
    (root / "livesplat.ply").write_bytes(ply)
    manifest["files"].append({"path": "livesplat.ply", "bytes": len(ply)})
    manifest["liveSplat"] = "livesplat.ply"
    (root / "manifest.json").write_text(json.dumps(manifest))
    parsed = bundle.read_bundle(root)
    assert len(parsed.frames) == 1 and parsed.manifest["liveSplat"] == "livesplat.ply"
    # An inventoried arbitrary payload or an unreferenced PLY still fails closed.
    (root / "livesplat.ply").write_bytes(b"x" * len(ply))
    with pytest.raises(ValueError, match="header"):
        bundle.read_bundle(root)
    (root / "livesplat.ply").write_bytes(ply)
    del manifest["liveSplat"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="referenced"):
        bundle.read_bundle(root)
