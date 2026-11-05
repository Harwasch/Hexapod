"""Camera-path yardstick: exact segments, assumed scale, streaming and local-only CLI."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sfm
from experiments import distance
from experiments.bench import __main__ as cli
from experiments.bench import ledger
from splat_io import PlyWriter, SplatReader


def geometry(scale: float = 1.0) -> dict[str, Any]:
    return {
        "scale": {"status": "estimated", "metresPerUnit": scale},
        "path": {"centres": [[0, 0, 0], [10, 0, 0]]},
    }


def test_exact_segments_endpoints_repeated_camera_and_single_camera() -> None:
    path = [[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 10, 0]]
    points = [[5, 0, 0], [-3, 4, 0], [5, 4, 3], [10, 15, 0], [9, 5, 0]]
    np.testing.assert_allclose(distance.distance_to_path(points, path), [0, 5, 5, 5, 1])
    np.testing.assert_allclose(distance.distance_to_path([[1, 2, 2]], [[0, 0, 0]]), [3])
    with pytest.raises(ValueError, match="finite"):
        distance.distance_to_path([[np.nan, 0, 0]], path)
    with pytest.raises(ValueError, match="Nx3"):
        distance.distance_to_path(points, [])


def test_distance_is_rigid_transform_invariant_and_scales_linearly() -> None:
    rng = np.random.default_rng(9)
    path, points = rng.normal(size=(9, 3)), rng.normal(size=(100, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    expected = distance.distance_to_path(points, path)
    actual = distance.distance_to_path(7 * points @ rotation + 100, 7 * path @ rotation + 100)
    np.testing.assert_allclose(actual, 7 * expected, atol=1e-12)


def test_robust_ground_scale_recovers_sloping_plane_with_wall_outliers() -> None:
    rng = np.random.default_rng(17)
    ground = np.column_stack([rng.uniform(-4, 4, (1200, 2)), np.zeros(1200)])
    walls = np.column_stack([rng.uniform(-4, 4, 300), np.zeros(300), rng.uniform(0.2, 4, 300)])
    cameras = np.column_stack([np.linspace(-3, 3, 50), np.zeros(50), np.full(50, 0.7)])
    # 0.7 model units corresponds to the assumed 1.4m. Rotate the scene off-axis.
    angle = 0.2
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    points = np.concatenate([ground, walls]) @ rotation.T + [8, 3, -5]
    cameras = cameras @ rotation.T + [8, 3, -5]
    result = distance.ground_scale(points, cameras, [0, 0, 1])
    assert result["metresPerUnit"] == pytest.approx(2, abs=1e-10)
    assert result["cameraHeightUnits"]["median"] == pytest.approx(0.7)
    assert result["inlierShare"] > 0.5
    assert distance.ground_scale(points, cameras, [0, 0, 1], 1.6)["metresPerUnit"] == pytest.approx(
        1.6 / 0.7
    )
    with pytest.raises(ValueError, match="positive"):
        distance.ground_scale(points, cameras, [0, 0, 1], -1)
    with pytest.raises(ValueError, match="ground"):
        distance.ground_scale(np.zeros((3, 3)), cameras, [0, 0, 1])


def write_ply(path: Path, far: bool = True) -> None:
    distances = np.array([0, 2, 5, 10, 20, 50, 51], dtype=float)
    columns = {
        "x": np.full(7, 5),
        "y": distances / 2,
        "z": np.zeros(7),
        "opacity": np.array([0, 0, 0, 0, 0, 0, np.log(3)]),
    }
    if far:
        columns["far_field"] = np.array([0, 0, 0, 0, 0, 1, 1])
    with PlyWriter(path, tuple(columns), count=7) as writer:
        writer.append(columns)


def test_streamed_bands_opacity_denominators_far_subsets_and_strict_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ply = tmp_path / "splat.ply"
    write_ply(ply)
    monkeypatch.setattr(distance, "CHUNK", 2)
    original = SplatReader.read
    reads = []

    def bounded(
        self: SplatReader, start: int, stop: int, names: Sequence[str] | None = None
    ) -> Any:
        assert stop - start <= 2
        reads.append((start, stop))
        return original(self, start, stop, names)

    monkeypatch.setattr(SplatReader, "read", bounded)
    result = distance.measure_ply(ply, geometry(2))
    assert len(reads) == 4
    assert [b["count"] for b in result["bands"]] == [1, 1, 1, 1, 1, 2]
    assert result["bands"][-1]["opacityShare"] == pytest.approx(1.25 / 3.75)
    assert result["bands"][-1]["countShare"] == pytest.approx(2 / 7)
    assert result["farField"]["countShare"] == pytest.approx(2 / 7)
    assert result["beyond"]["10"]["count"] == 3  # Strictly >10, not >=10.
    assert result["beyond"]["20"]["count"] == 2
    assert sum(b["opacityShare"] for b in result["bands"]) == pytest.approx(1)
    assert sum(b["countShare"] for b in result["bands"]) == pytest.approx(1)
    write_ply(ply, far=False)
    assert distance.measure_ply(ply, geometry())["farField"] is None


def test_empty_ply_and_unavailable_scale_do_not_fabricate_shares(tmp_path: Path) -> None:
    ply = tmp_path / "empty.ply"
    with PlyWriter(ply, ("x", "y", "z", "opacity"), count=0):
        pass
    result = distance.measure_ply(ply, geometry())
    assert result["bands"][0]["countShare"] is None
    assert result["bands"][0]["opacityShare"] is None
    g = {**geometry(), "scale": {"status": "unavailable", "reason": "no ground"}}
    result = distance.measure_ply(ply, g)
    assert result["status"] == "unavailable" and result["bands"] is None


def test_capture_order_uses_metadata_then_numeric_filename_never_registration_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tuple(
        sfm.Image(i, name, (1, 0, 0, 0), (-x, 0, 0), 1, 0, 0)
        for i, name, x in ((9, "f_10.jpg", 10), (2, "f_2.jpg", 2), (20, "f_1.jpg", 1))
    )
    model = type("Model", (), {"images": images})()
    monkeypatch.setattr(sfm, "read_model", lambda _: model)
    monkeypatch.setattr(sfm, "camera_up", lambda _: None)
    result = distance.scene_geometry(tmp_path)
    assert result["path"]["names"] == ["f_1.jpg", "f_2.jpg", "f_10.jpg"]
    assert result["scale"]["status"] == "unavailable"
    (tmp_path / "frames_meta.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"name": name, "candidateIndex": index, "original": "clip.mov"}
                    for index, name in enumerate(["f_10.jpg", "f_1.jpg", "f_2.jpg"])
                ]
            }
        )
    )
    result = distance.scene_geometry(tmp_path)
    assert result["path"]["names"] == ["f_10.jpg", "f_1.jpg", "f_2.jpg"]
    assert result["path"]["includesHeldout"] is True


def test_cli_is_local_without_ledger_modal_or_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ply = tmp_path / "splat.ply"
    write_ply(ply)
    g = geometry()
    g["scale"].update(assumedCameraHeightM=1.4, inlierPoints=100, localPoints=150)
    monkeypatch.setattr(distance, "scene_geometry", lambda *a: g)
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("local command imported Modal"))
    monkeypatch.setattr(ledger, "read", lambda *a: pytest.fail("local command read ledger"))
    monkeypatch.setattr(cli, "git_commit", lambda: pytest.fail("local command read git"))
    output = tmp_path / "report"
    assert (
        cli.main(
            [
                "distance",
                "--scene-dir",
                str(tmp_path),
                "--ply",
                str(ply),
                "--ply",
                str(ply),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["rows"]) == 2
    assert json.loads((output / "distance.json").read_text()) == result
    assert "Beyond 20 m" in (output / "distance.md").read_text()


def test_nonfinite_opacity_is_rejected(tmp_path: Path) -> None:
    ply = tmp_path / "invalid.ply"
    with PlyWriter(ply, ("x", "y", "z", "opacity"), count=1) as writer:
        writer.append({"x": [0], "y": [0], "z": [0], "opacity": [np.nan]})
    with pytest.raises(ValueError, match="nonfinite opacity"):
        distance.measure_ply(ply, geometry())


def test_cannot_infer_capture_order_from_arbitrary_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = (sfm.Image(1, "arbitrary.jpg", (1, 0, 0, 0), (0, 0, 0), 1, 0, 0),)
    model = type("Model", (), {"images": images})()
    monkeypatch.setattr(sfm, "read_model", lambda _: model)
    with pytest.raises(ValueError, match="capture order"):
        distance.scene_geometry(tmp_path)
