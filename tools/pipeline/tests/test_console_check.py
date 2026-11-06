from __future__ import annotations

import base64
import contextlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from captures_bridge import pack_spz
from experiments.bench import __main__ as cli
from experiments.bench import console_check, console_gpu, ledger
from experiments.bench.console_maths import (
    PNG_NAMES,
    artifact_info,
    live_inputs,
    orbit_camera,
    psnr,
)
from experiments.bench.scene import read_json, write_json

RUN = "4769ee65e33c44f2aacd87a45dfa3811"


def payload() -> dict[str, Any]:
    points = np.zeros(4, dtype=[("xyz", "<i2", (3,)), ("rgb", "u1", (3,))])
    points["xyz"][:, 0] = [0, 1, 2, 100]
    return {
        "origin": [10, 20, 30],
        "scale": 32767,
        "up": [0, 1, 0],
        "cameras": "",
        "cameraCount": 0,
        "pointCount": 4,
        "points": base64.b64encode(points.tobytes()).decode(),
    }


def test_orbit_uses_upper_median_rank_radius_and_world_to_opencv() -> None:
    spec = orbit_camera(payload(), [1600, 900])
    assert spec["target"] == [12, 20, 30]  # upper median, not average of middle pair
    assert spec["radius"] == 98  # floor(N*.9), not interpolated percentile
    distance = 98 / math.tan(math.radians(25)) * 1.1
    assert spec["distance"] == pytest.approx(distance)
    view = np.array(spec["viewmat"])
    np.testing.assert_allclose(view @ [*spec["eye"], 1], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(view @ [*spec["target"], 1], [0, 0, distance, 1], atol=1e-12)
    assert np.linalg.det(view[:3, :3]) == pytest.approx(1)
    assert spec["K"][0][0] == pytest.approx(250 / math.tan(math.radians(25)))
    # Mobile fit uses the narrower initial horizontal FOV, then a 4:3 panel.
    mobile = orbit_camera(payload(), [400, 900])
    assert mobile["distance"] == pytest.approx(distance * 900 / 400)
    assert (mobile["width"], mobile["height"]) == (800, 600)


@pytest.mark.parametrize("up", [[0, -1, 0], [1, 2, 3], [0, 0, 0]])
def test_orbit_rotation_matches_three_unit_vector_and_antiparallel_cases(up: list[int]) -> None:
    spec = orbit_camera({**payload(), "up": up}, [1600, 900])
    orient = np.array(spec["sceneToDisplayRotation"])
    np.testing.assert_allclose(orient @ spec["up"], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(orient @ orient.T, np.eye(3), atol=1e-12)


def test_spz_reader_reports_stored_sh_without_axis_changes(tmp_path: Path) -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], np.float32)
    path = tmp_path / "snapshot.spz"
    path.write_bytes(
        pack_spz(
            xyz,
            np.zeros((2, 3)),
            np.zeros(2),
            np.zeros((2, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            np.zeros((2, 3, 3)),
        )
    )
    info = artifact_info(path)
    assert info["count"] == 2 and info["shDegree"] == 1
    assert info["bytes"] == path.stat().st_size
    assert info["bytesPerSplat"] == path.stat().st_size / 2
    assert psnr(np.zeros((2, 3)), np.full((2, 3), 0.1)) == pytest.approx(20)
    assert psnr(xyz, xyz) is None


def test_pool_selection_uses_time_and_excludes_test_neighbours_and_internal_val(
    tmp_path: Path,
) -> None:
    names = [f"f{i}" for i in range(12)]
    write_json(
        tmp_path / "frames_meta.json",
        {"frames": [{"name": name, "offsetS": 12 - i} for i, name in enumerate(names)]},
    )
    split = {"pool": names[:11], "test": [names[11]]}
    row = {"frameSelection": {"excluded": ["f10"]}, "trainParams": {"test_every": 2}}
    # Lexically sorted training pool minus internal val = f1,f3,f5,f7,f9; reverse in time.
    assert console_check.selected_frames(tmp_path, split, row) == ["f9", "f5", "f1"]


def test_snapshot_selection_prefers_logged_last_and_falls_back_to_checkpoint(
    tmp_path: Path,
) -> None:
    import live

    folder = tmp_path / "live"
    folder.mkdir()
    (folder / "train.log").write_text(
        live.CAMERAS_TAG
        + json.dumps(
            {
                **payload(),
                "v": 1,
                "seq": 1,
                "final": True,
                "registered": 4,
                "frames": 4,
                "pointsTotal": 4,
            }
        )
        + "\n"
        + live.SPLAT_TAG
        + json.dumps(
            {
                "v": 1,
                "step": 30,
                "total": 30,
                "count": 1,
                "of": 2,
                "bytes": 1,
                "key": "runs/test/train/checkpoint/live/splat_000030_99.spz",
                "up": [0, 1, 0],
            }
        )
        + "\n"
    )
    fallback = tmp_path / "train/checkpoint/live"
    fallback.mkdir(parents=True)
    chosen = fallback / "splat_000030_99.spz"
    chosen.write_bytes(b"x")
    assert live_inputs(tmp_path)[1] == chosen
    chosen.unlink()
    (folder / "splat_000020_98.spz").write_bytes(b"x")
    with pytest.raises(ValueError, match="last console SPZ missing"):
        live_inputs(tmp_path)


def test_cli_budget_and_missing_ply_fail_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parser().parse_args(["console-check", "--scene", "beach-full", "--run", RUN])
    plan = cli.plans_for(args)[0]
    priced = ledger.price_plan([], [plan], args.cap, args.total_cap)
    assert plan["compareRun"] == console_check.BASELINE
    assert priced["reservedUsd"] == pytest.approx(1.021392 * 1440 / 3600)
    assert priced["reservedUsd"] < 0.49
    assert cli.train_options(plan) == {"memory": (16384, 16384)}
    module = SimpleNamespace(
        app=SimpleNamespace(run=contextlib.nullcontext),
        volume=SimpleNamespace(
            read_file=lambda _: [b'{"digest":"frozen","splitDigest":"split"}'],
            iterdir=lambda *_args, **_kw: [],
        ),
    )
    monkeypatch.setattr(cli, "load_modal", lambda: module)
    with pytest.raises(ValueError, match=r"trained\.ply missing"):
        cli.remote_batch([plan], tmp_path, tmp_path / "ledger.jsonl", 0.49, 0.49)
    assert not (tmp_path / "ledger.jsonl").exists()
    args.total_cap = 1
    with pytest.raises(ValueError, match="at most"):
        cli.plans_for(args)


def test_console_check_rehearsal_transports_all_labelled_views_and_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("rehearsal cannot load Modal"))
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    common = ["--scene", "beach-full", "--state-dir", str(tmp_path), "--rehearse"]
    assert cli.main(["run", *common, "--seeds", "42,43", "--live", "--keep-splat"]) == 0
    root = tmp_path / "rehearsal"
    runs = sorted(p.name for p in (root / "runs").iterdir())
    assert cli.main(["console-check", *common, "--run", runs[0], "--compare-run", runs[1]]) == 0
    output = next((root / "runs").glob("console-check-*"))
    result = read_json(output / "console-check.json")
    assert result["status"] == "complete" and result["costUsd"] == 0
    assert result["sources"]["snapshot"]["shDegree"] == 0
    assert (output / "cameras.json").is_file()
    assert {p.name for p in output.glob("*.png")} == set(PNG_NAMES)
    assert result["cameras"]["cameras"]["orbit"]["viewport"] == [1600, 900]
    # Missing full PLY is explicit, never silently substituted with a preview.
    (root / "runs" / runs[0] / "trained.ply").unlink()
    assert cli.main(["console-check", *common, "--run", runs[0], "--compare-run", runs[1]]) == 1
    assert "missing trained.ply" in capsys.readouterr().out


def test_transport_rejects_path_traversal(tmp_path: Path) -> None:
    result = {
        "runId": "console-check-" + "a" * 32,
        "status": "failed",
        "pngs": {"../escape.png": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()},
    }
    with pytest.raises(ValueError, match="invalid/oversized"):
        console_check.save_local(tmp_path, result)


def test_gpu_loader_streams_full_sh_and_keeps_scales_opacity_and_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import splat_io

    class Tensor(np.ndarray):
        def copy_(self, other: np.ndarray) -> None:
            np.copyto(self, other)

        def exp_(self) -> None:
            np.exp(self, out=self)

        def sigmoid_(self) -> None:
            self[:] = 1 / (1 + np.exp(-self))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32=np.float32,
            from_numpy=np.asarray,
            empty=lambda shape, **_kw: np.empty(shape, np.float32).view(Tensor),
        ),
    )
    count = 11
    columns = {k: np.arange(count, dtype=np.float32) for k in ("x", "y", "z")}
    columns.update({f"rot_{i}": np.full(count, float(i == 0), np.float32) for i in range(4)})
    columns.update({f"scale_{i}": np.full(count, -i, np.float32) for i in range(3)})
    columns.update({f"f_dc_{i}": np.full(count, i, np.float32) for i in range(3)})
    columns.update({f"f_rest_{i}": np.full(count, i, np.float32) for i in range(45)})
    columns["opacity"] = np.zeros(count, np.float32)
    path = tmp_path / "trained.ply"
    with splat_io.PlyWriter(path, tuple(columns), count=count) as writer:
        writer.append(columns)
    reads: list[int] = []

    class Bounded(splat_io.SplatReader):
        def __init__(self, path: Path) -> None:
            super().__init__(path, chunk=3)

        def read(self, start: int, stop: int, names: Any = None) -> Any:
            reads.append(stop - start)
            return super().read(start, stop, names)

    monkeypatch.setattr(splat_io, "SplatReader", Bounded)
    model = console_gpu.load_model(artifact_info(path))
    assert reads == [3, 3, 3, 2]
    assert model["colors"].shape == (11, 16, 3)
    np.testing.assert_array_equal(model["colors"][0, 0], [0, 1, 2])
    np.testing.assert_array_equal(model["colors"][0, 1], [0, 15, 30])
    np.testing.assert_array_equal(model["colors"][0, 15], [14, 29, 44])
    np.testing.assert_array_equal(model["means"][:, 1], columns["y"])
    np.testing.assert_array_equal(model["quats"][0], [1, 0, 0, 0])
    np.testing.assert_array_equal(model["opacities"], np.full(count, 0.5))
    np.testing.assert_allclose(model["scales"][0], np.exp([0, -1, -2]))


def test_renderer_uses_classic_stored_degree_and_given_world_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor(np.ndarray):
        def clamp(self, low: float, high: float) -> Tensor:
            return np.clip(self, low, high).view(Tensor)

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray(self)

    seen: dict[str, Any] = {}

    def rasterization(**kwargs: Any) -> tuple[Any, None, None]:
        seen.update(kwargs)
        return np.array([[[[1.2, 0.4, -0.1]]]], np.float32).view(Tensor), None, None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=contextlib.nullcontext,
            float32=np.float32,
            tensor=lambda value, **_kw: np.array(value),
        ),
    )
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=rasterization))
    camera = orbit_camera(payload(), [1600, 900])
    pixels = console_gpu.render_model({}, camera, 3)
    assert seen["rasterize_mode"] == "classic" and seen["sh_degree"] == 3
    assert seen["near_plane"] == camera["near"] and seen["far_plane"] == camera["far"]
    np.testing.assert_array_equal(seen["viewmats"][0], camera["viewmat"])
    np.testing.assert_allclose(pixels[0, 0], [1, 0.4, 0])
