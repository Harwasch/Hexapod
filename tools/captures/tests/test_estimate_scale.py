"""estimate_scale.py: a registered phone scan's size, from the pose model its run left.

The model here is a phone orbit written the way COLMAP leaves one -- in a frame of its own
choosing, turned, moved and in units that are not metres -- with a known truth: 0.4 m to
the unit, the phone held 1.5 m over a floor. The estimate must find that whatever frame
COLMAP chose, the request it builds must be relative to the model as it was registered, and
nothing may reach the API but a dry run's reads unless `--apply` is given an API and a token
by hand -- and then only for a scan whose scale nobody measured.
"""

from __future__ import annotations

import io
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import estimate_scale

#: The orbit's own truth: metres per model unit, how high the phone was held, cameras.
METRES_PER_UNIT = 0.4
HELD_AT_M = 1.5
CAMERAS = 36
#: COLMAP's frame for the model: an arbitrary turn and offset (y down is its usual first
#: camera's; any rotation will do, which is the point).
COLMAP_TURN = Rotation.from_euler("xyz", [97.0, -18.0, 35.0], degrees=True).as_matrix()
COLMAP_OFFSET = np.array([3.0, -1.0, 2.0])

CAPTURE, SITE, ASSET, JOB = (
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
)


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera rotation of a camera at `centre` looking at `target`, z up: COLMAP's
    camera looks down +z with +y down the image."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def write_orbit(
    directory: Path,
    *,
    floor: bool = True,
    turn: np.ndarray = COLMAP_TURN,
    ring_m: float = 2.0,
    table: tuple[float, float] | None = None,
) -> tuple[Path, np.ndarray]:
    """`cameras.bin`, `images.bin` and `points3D.bin` of a phone orbit, `ring_m` round a
    table-top subject (0.7-1.0 m up) over a floor out to 4 m -- or, `floor=False`, of the
    subject alone; with `table` (radius, height), on a table top that wide and that high --
    in COLMAP's frame `turn`, `COLMAP_OFFSET` and `METRES_PER_UNIT`.

    Returns the directory and the model's camera-up (the world's +z in its frame)."""
    rng = np.random.default_rng(3)
    target = np.array([0.0, 0.0, 0.6])
    angles = 2 * np.pi * np.arange(CAMERAS) / CAMERAS
    centres = np.stack(
        [ring_m * np.cos(angles), ring_m * np.sin(angles), np.full(CAMERAS, HELD_AT_M)], axis=1
    )
    radius, angle = 4.0 * np.sqrt(rng.random(20_000)), 2 * np.pi * rng.random(20_000)
    ground = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(20_000)], 1)
    subject = np.stack(
        [rng.normal(0, 0.2, 8_000), rng.normal(0, 0.2, 8_000), rng.uniform(0.7, 1.0, 8_000)], 1
    )
    if table is not None:
        # The floor the table hides is not reconstructed; its top is.
        ground = ground[np.hypot(ground[:, 0], ground[:, 1]) > table[0]]
        radius, angle = table[0] * np.sqrt(rng.random(20_000)), 2 * np.pi * rng.random(20_000)
        top = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.full(20_000, table[1])], 1
        )
        subject = np.concatenate([top, subject + [0.0, 0.0, table[1]]])
    world = np.concatenate([ground, subject]) if floor else subject

    def to_model(x: np.ndarray) -> np.ndarray:
        return x @ turn.T / METRES_PER_UNIT + COLMAP_OFFSET

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cameras.bin").open("wb") as out:
        out.write(struct.pack("<Q", 1))
        out.write(struct.pack("<IiQQ", 1, 1, 32, 24))  # PINHOLE
        out.write(struct.pack("<4d", 30.0, 30.0, 16.0, 12.0))
    with (directory / "images.bin").open("wb") as out:
        out.write(struct.pack("<Q", CAMERAS))
        for index, centre in enumerate(centres):
            rotation = _look_at(centre, target) @ turn.T
            x, y, z, w = Rotation.from_matrix(rotation).as_quat()
            t = -rotation @ to_model(centre)
            out.write(struct.pack("<IdddddddI", index + 1, w, x, y, z, *t, 1))
            out.write(f"frame_{index:04d}.jpg".encode() + b"\0")
            out.write(struct.pack("<Q", 0))
    points = to_model(world)
    with (directory / "points3D.bin").open("wb") as out:
        out.write(struct.pack("<Q", points.shape[0]))
        for index, (x, y, z) in enumerate(points):
            out.write(struct.pack("<Q3d3Bd", index + 1, x, y, z, 128, 128, 128, 0.5))
            out.write(struct.pack("<Q", 0))
    return directory, turn @ np.array([0.0, 0.0, 1.0])


# --- the estimate ----------------------------------------------------------------------


def test_the_estimate_is_the_scans_metres_per_unit_whatever_frame_colmap_chose(
    tmp_path: Path,
) -> None:
    poses, up = write_orbit(tmp_path / "a")
    other, _ = write_orbit(
        tmp_path / "b",
        turn=Rotation.from_euler("zyx", [-60.0, 170.0, 12.0], degrees=True).as_matrix(),
    )

    found = estimate_scale.estimate(poses)
    again = estimate_scale.estimate(other)

    assert found.estimate is not None and again.estimate is not None
    assert found.estimate.scale == pytest.approx(METRES_PER_UNIT, rel=0.03)
    assert found.estimate.n_cameras == CAMERAS
    assert found.estimate.camera_height_units == pytest.approx(
        HELD_AT_M / METRES_PER_UNIT, rel=0.03
    )
    assert 19.0 < found.estimate.uncertainty_pct < 25.0
    # Levelled by how the phone was held, so COLMAP's choice of frame is no part of it.
    assert np.asarray(found.up) == pytest.approx(up, abs=1e-9)
    assert again.estimate.scale == pytest.approx(found.estimate.scale, rel=1e-9)


def test_the_ground_check_is_quiet_when_the_cameras_stood_over_the_floor(
    tmp_path: Path,
) -> None:
    found = estimate_scale.estimate(write_orbit(tmp_path / "a")[0])

    assert found.estimate is not None and found.ground is not None
    assert abs(found.ground.gap_units * found.estimate.scale) < 0.05


def test_a_table_top_read_as_the_floor_is_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cameras held 1.5 m up, leaning over a 2 m-wide table 0.75 m high: the low surface
    within reach is the table top, the cameras are 0.75 m over it, and the estimate comes
    out twice the truth. It is not refused -- nothing in the geometry says it is a table --
    but the dry run says how far that surface is above the floor round it."""
    poses, _ = write_orbit(tmp_path / "table", ring_m=1.2, table=(2.0, 0.75))

    found = estimate_scale.estimate(poses)
    status = estimate_scale.main([str(poses)])

    assert found.estimate is not None and found.ground is not None
    assert found.estimate.scale == pytest.approx(2 * METRES_PER_UNIT, rel=0.05)
    assert found.ground.gap_units * METRES_PER_UNIT == pytest.approx(0.75, rel=0.05)
    assert status == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["groundCheck"]["gapM"] == pytest.approx(1.5, rel=0.05)
    assert "a table top or a step read as the floor" in captured.err


def test_a_model_that_shows_no_ground_gets_no_estimate(tmp_path: Path) -> None:
    poses, _ = write_orbit(tmp_path / "subject", floor=False)

    assert estimate_scale.estimate(poses).estimate is None


def test_the_request_is_relative_to_the_model_as_registered(tmp_path: Path) -> None:
    """An unresolved scan was registered at one unit to the metre, so the factor is the
    estimate itself; a model registered at 0.5 m per unit is drawn at `estimate / 0.5`."""
    found = estimate_scale.estimate(write_orbit(tmp_path / "a")[0]).estimate
    assert found is not None

    unresolved = estimate_scale.request_body(found)
    halved = estimate_scale.request_body(found, frame_scale=0.5)

    assert unresolved["scale"] == found.scale
    assert halved["scale"] == pytest.approx(2.0 * found.scale)
    evidence = unresolved["evidence"]
    assert evidence["method"] == "camera-height-estimate"
    assert evidence["metresPerUnit"] == found.scale
    assert evidence["cameraHeightM"] == HELD_AT_M
    assert evidence["cameraHeightUnits"] == found.camera_height_units
    assert evidence["cameras"] == CAMERAS
    assert evidence["uncertaintyPct"] == round(found.uncertainty_pct, 1)
    assert "estimate, not a measurement" in evidence["note"]
    assert "as registered at 0.5 m per unit" in halved["evidence"]["note"]


# --- the API -----------------------------------------------------------------------------


@dataclass
class FakeApi:
    """The API's public reads for one capture, and every write sent to it."""

    scale_source: str = "unresolved"
    frame_scale: float = 1.0
    up: list[float] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    reads: list[str] = field(default_factory=list)
    writes: list[tuple[str, dict[str, str], dict[str, Any]]] = field(default_factory=list)

    def urlopen(self, request: Any, timeout: float = 0) -> io.BytesIO:
        url, method = request.full_url, request.get_method()
        if method == "PUT":
            body = json.loads(request.data.decode("utf-8"))
            self.writes.append((url, dict(request.header_items()), body))
            return io.BytesIO(json.dumps({"id": ASSET, "renderConfig": body}).encode())
        self.reads.append(url)
        if url.endswith(f"/api/v1/captures/{CAPTURE}"):
            return io.BytesIO(json.dumps({"id": CAPTURE, "siteId": SITE}).encode())
        if url.endswith(f"/api/v1/sites/{SITE}"):
            return io.BytesIO(json.dumps(self._site()).encode())
        raise AssertionError(f"unexpected read {url}")

    def _site(self) -> dict[str, Any]:
        frame: dict[str, Any] = {"source": "camera-up", "scale": self.frame_scale}
        if self.up:
            frame["up"] = {"up": self.up}
        render: dict[str, Any] = {"scale": 1.0, "scaleEvidence": self.evidence}
        return {
            "id": SITE,
            "metadata": {
                "jobId": JOB,
                "registration": {"georef": {"scaleSource": self.scale_source, "frame": frame}},
            },
            "assets": [
                {
                    "id": "55555555-5555-5555-5555-555555555555",
                    "representation": "mesh",
                    "createdAt": "2026-09-01T00:00:00Z",
                },
                {
                    "id": ASSET,
                    "representation": "gaussian-splat",
                    "createdAt": "2026-09-28T16:55:00Z",
                    "renderConfig": render,
                    "provenance": {"scaleSource": self.scale_source},
                },
            ],
        }


@pytest.fixture
def poses(tmp_path: Path) -> tuple[Path, list[float]]:
    directory, up = write_orbit(tmp_path / "poses")
    return directory, [float(v) for v in up]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, poses: tuple[Path, list[float]]) -> Iterator[FakeApi]:
    fake = FakeApi(up=poses[1])
    monkeypatch.setattr(estimate_scale.urllib.request, "urlopen", fake.urlopen)
    yield fake


@pytest.mark.parametrize("flags", [[], ["--dry-run"]])
def test_a_dry_run_prints_the_request_and_sends_nothing(
    poses: tuple[Path, list[float]],
    api: FakeApi,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    status = estimate_scale.main([str(poses[0]), "--capture", CAPTURE, *flags])

    assert status == 0
    assert api.writes == []
    report = json.loads(capsys.readouterr().out)
    assert report["capture"]["assetId"] == ASSET
    assert report["capture"]["jobId"] == JOB
    assert report["capture"]["registeredScaleSource"] == "unresolved"
    assert report["upDisagreementDeg"] == pytest.approx(0.0, abs=1e-6)
    assert report["request"]["scale"] == pytest.approx(METRES_PER_UNIT, rel=0.03)
    assert report["estimate"]["method"] == "camera-height"
    # The reads went to the public API, as `fetch_capture` reads it.
    assert all(url.startswith(estimate_scale.DEFAULT_API) for url in api.reads)


def test_apply_puts_the_estimate_with_the_write_token(
    poses: tuple[Path, list[float]], api: FakeApi
) -> None:
    api.frame_scale = 0.5

    status = estimate_scale.main(
        [
            str(poses[0]),
            "--capture",
            CAPTURE,
            "--apply",
            "--api-url",
            "http://api.test/",
            "--token",
            "secret",
        ]
    )

    assert status == 0
    assert len(api.writes) == 1
    url, headers, body = api.writes[0]
    assert url == f"http://api.test/api/v1/assets/{ASSET}/scale"
    assert headers["Authorization"] == "Bearer secret"
    assert body["scale"] == pytest.approx(METRES_PER_UNIT / 0.5, rel=0.03)
    assert body["evidence"]["method"] == "camera-height-estimate"
    assert all(url.startswith("http://api.test/") for url in api.reads)


@pytest.mark.parametrize(
    "flags",
    [
        ["--apply", "--token", "secret"],
        ["--apply", "--api-url", "http://api.test"],
        ["--apply", "--api-url", "http://api.test", "--token", "secret", "--no-capture"],
    ],
)
def test_apply_needs_the_capture_the_api_and_the_token_spelled_out(
    poses: tuple[Path, list[float]], api: FakeApi, flags: list[str]
) -> None:
    capture = [] if "--no-capture" in flags else ["--capture", CAPTURE]
    flags = [flag for flag in flags if flag != "--no-capture"]

    with pytest.raises(SystemExit) as stopped:
        estimate_scale.main([str(poses[0]), *capture, *flags])

    assert stopped.value.code == 2
    assert api.writes == []


@pytest.mark.parametrize(
    ("scale_source", "evidence"),
    [
        ("exif-gps", None),
        ("manual", None),
        ("camera-height-estimate", None),
        ("unresolved", {"method": "measured-length", "registeredScaleSource": "unresolved"}),
    ],
)
def test_a_scan_whose_scale_somebody_knows_is_not_estimated_over(
    poses: tuple[Path, list[float]],
    api: FakeApi,
    scale_source: str,
    evidence: dict[str, Any] | None,
) -> None:
    api.scale_source, api.evidence = scale_source, evidence
    flags = ["--capture", CAPTURE, "--apply", "--api-url", "http://api.test", "--token", "t"]

    refused = estimate_scale.main([str(poses[0]), *flags])
    forced = estimate_scale.main([str(poses[0]), *flags, "--force"])

    assert refused == 1
    assert forced == 0
    assert len(api.writes) == 1


def test_poses_from_another_run_are_refused(
    poses: tuple[Path, list[float]], api: FakeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registered run recorded the camera-up it levelled by; a model whose own is
    elsewhere is not that run's, and its estimate is of other tiles."""
    api.up = [0.0, 0.0, 1.0] if abs(poses[1][2]) < 0.9 else [1.0, 0.0, 0.0]

    status = estimate_scale.main(
        [str(poses[0]), "--capture", CAPTURE, "--apply", "--api-url", "http://a", "--token", "t"]
    )

    assert status == 1
    assert api.writes == []
    assert "not that run's model" in capsys.readouterr().err


def test_no_estimate_is_an_error_and_nothing_is_sent(
    tmp_path: Path, api: FakeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    subject, _ = write_orbit(tmp_path / "subject", floor=False)

    status = estimate_scale.main(
        [str(subject), "--capture", CAPTURE, "--apply", "--api-url", "http://a", "--token", "t"]
    )

    assert status == 1
    assert api.writes == []
    assert json.loads(capsys.readouterr().out)["estimate"] is None
