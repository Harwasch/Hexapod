"""`georeference` / `exif_gps`, scored against GPS the fixture knows the answer to.

The fixture is the same 40 frames of the committed synthetic tree that
`test_pose_colmap.py` reconstructs, with one addition: each frame carries an EXIF GPS tag
written at the camera's *true* position, put on the globe about a chosen origin. So the
alignment has a right answer, and what is asserted is how far from it the stage lands --
not that a JSON file appeared.

Measured on this development VM while this was written, on exact GPS (no noise):

    frames with a fix          40 / 40
    aligned, inliers           40, 40
    residual, median           0.0088 m
    residual, rms              0.0107 m
    residual, max              0.0275 m
    recovered metric scale     1.3e-5 relative to `umeyama` on the same points
    reported uncertainty       5.0 m (the floor, not the residual)

and with 1 m of seeded noise per axis: residual 1.20 m median, 1.60 m rms, 4.10 m max,
scale still within 0.2% of `umeyama`, uncertainty still 5.0 m.

The residual on exact GPS is COLMAP's own pose error and nothing else, which is why it is
millimetres. The thresholds below are set where a real regression would show: the failure
this is really guarding against is reading `model_aligner`'s transform wrongly, and that
lands at about 9 m on this fixture -- three orders of magnitude away from what passes.

There is a second run with 1 m of seeded GPS noise. Its job is not accuracy: it is that
the residual is *reported* rather than hidden, and that the reported uncertainty does not
improve just because the residual happened to.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import gps_frames
import numpy as np
import pytest

import exif
import gaussians
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from stages import EXIF_GPS_UNCERTAINTY_FLOOR_M, UNALIGNED_UNCERTAINTY_M
from workdir import Workdir

#: The same count `test_pose_colmap.py` uses, so the reconstruction is the same one.
FRAMES = 40

#: Where the orbit is put on the globe: the synthetic tree's own catalog coordinate.
ORIGIN = (28.0389, -82.6966, 10.8)

requires_colmap = pytest.mark.skipif(
    not sfm.colmap_available(),
    reason="colmap is not on this machine (CI installs it; see .github/workflows/ci.yml)",
)


def test_colmap_is_installed_in_ci() -> None:
    """The one test in this file that never skips; see `test_pose_colmap.py`."""
    if os.environ.get("CI", "").lower() == "true":
        assert sfm.colmap_available(), (
            "colmap is missing under CI, so every alignment test here would skip and "
            "`georeference: exif_gps` would be unverified"
        )


# --- the fixture holds itself honest -------------------------------------------------


def test_the_fixture_inverts_the_projection_the_stage_uses() -> None:
    """A fixture built with the same approximation as the code would agree wrongly.

    `gps_frames.enu_to_geodetic` is Ferrari's closed form; `exif.enu_offsets` is the
    forward WGS84 map the stage itself calls. Round-tripping one through the other has to
    come back to where it started, or every residual below is measuring the fixture.
    """
    for east, north, up in [(0.0, 0.0, 0.0), (9.0, 0.0, 3.6), (-120.0, 240.0, -15.0)]:
        lat, lon, alt = gps_frames.enu_to_geodetic(east, north, up, *ORIGIN)
        fix = exif.Fix(name="f", lat=lat, lon=lon, alt=alt)
        back = exif.enu_offsets([fix], ORIGIN)["f"]
        # Micrometres. Not a claim about the ellipsoid: both directions are closed-form
        # double precision over a few hundred metres, and this is what that is worth.
        assert np.allclose(back, (east, north, up), atol=1e-6), f"{back} != {(east, north, up)}"


def test_a_gps_tag_survives_the_jpeg_it_is_spliced_into(tmp_path: Path) -> None:
    """The EXIF goes in without touching the scan, and comes back out as it went in."""
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=2)
    before = (frames / "frame_0000.jpg").read_bytes()

    gps_frames.write_gps(frames / "frame_0000.jpg", 28.0389, -82.6966, 10.8)
    fix = exif.read_fix(frames / "frame_0000.jpg")

    assert fix is not None
    # 1e-7 degrees is about 1 cm, and the tag is written to 1e-4 arcsec (about 3 mm).
    assert fix.lat == pytest.approx(28.0389, abs=1e-7)
    assert fix.lon == pytest.approx(-82.6966, abs=1e-7)
    assert fix.alt == pytest.approx(10.8, abs=1e-3)
    # The compressed scan is untouched: the whole original file is still in there.
    after = (frames / "frame_0000.jpg").read_bytes()
    assert after.endswith(before[2:]) and len(after) > len(before)
    # And a frame nobody tagged has no fix rather than a zero one.
    assert exif.read_fix(frames / "frame_0001.jpg") is None


def test_a_southern_western_below_sea_level_fix_keeps_its_signs(tmp_path: Path) -> None:
    """Hemisphere refs and the altitude ref byte, which are four separate ways to be wrong."""
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=1)
    gps_frames.write_gps(frames / "frame_0000.jpg", -33.8688, 151.2093, -4.25)

    fix = exif.read_fix(frames / "frame_0000.jpg")

    assert fix is not None
    assert fix.lat == pytest.approx(-33.8688, abs=1e-7)
    assert fix.lon == pytest.approx(151.2093, abs=1e-7)
    assert fix.alt == pytest.approx(-4.25, abs=1e-3)


# --- argv and readers, which need no COLMAP ------------------------------------------


def test_a_zero_alignment_error_is_refused_here_rather_than_by_colmap() -> None:
    """COLMAP 3.9.1's own refusal: "You must provide a maximum alignment error > 0"."""
    with pytest.raises(ValueError, match="must be > 0"):
        sfm.model_aligner_argv(Path("m"), Path("o"), Path("r"), Path("t"), max_error_m=0.0)


@requires_colmap
def test_the_aligner_is_asked_for_a_frame_this_project_chose() -> None:
    argv = sfm.model_aligner_argv(Path("m"), Path("o"), Path("r"), Path("t"), max_error_m=5.0)

    assert argv[1] == "model_aligner"
    assert argv[argv.index("--ref_is_gps") + 1] == "0"
    assert argv[argv.index("--alignment_type") + 1] == "custom"
    assert argv[argv.index("--alignment_max_error") + 1] == "5"


def test_a_transform_that_is_not_eight_numbers_is_refused(tmp_path: Path) -> None:
    """A COLMAP that writes a 4x4 matrix instead must not be read as a quaternion."""
    path = tmp_path / "transform.txt"
    path.write_text(" ".join("1" for _ in range(16)), encoding="utf-8")

    with pytest.raises(ValueError, match="not the 8"):
        sfm.read_similarity(path)


def test_the_residual_is_computed_from_the_transform_not_from_colmaps_output() -> None:
    """A known similarity, applied to known centres, must come back as zero residual."""
    rotation = sfm.quat_to_matrix((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    similarity = sfm.Similarity(scale=2.0, rotation=rotation, translation=np.array([1.0, 2.0, 3.0]))
    centres = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0), "c": (0.0, 0.0, 1.0)}
    reference = {
        name: tuple(similarity.apply(np.asarray([value]))[0]) for name, value in centres.items()
    }

    residuals = sfm.alignment_residuals(similarity, centres, reference)

    assert set(residuals) == {"a", "b", "c"}
    assert max(residuals.values()) < 1e-9


# --- the located branch, which needs no COLMAP either --------------------------------


def _georeference_only(
    root: Path,
    frames: Path,
    *,
    source_meta: dict[str, object] | None,
    params: dict[str, object] | None = None,
) -> Path:
    workdir = Workdir.create(root)
    target = workdir.input_path("frames")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(frames.iterdir()):
        (target / path.name).write_bytes(path.read_bytes())
    inputs = ["frames"]
    if source_meta is not None:
        workdir.input_path("source_meta.json").write_text(json.dumps(source_meta), "utf-8")
        inputs.append("source_meta.json")
    execute(
        make_recipe(
            [{"id": "georeference", "impl": "exif_gps", "params": params or {}}], inputs=inputs
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    return workdir.artifact_path("georeference", "georef.json")


def test_a_video_location_places_the_capture_and_claims_nothing_else(tmp_path: Path) -> None:
    """The iPhone case: one coordinate off the container, no per-frame EXIF, no poses.

    A0's sizing note for this stage was to read both iPhone location keys. Both are read,
    in `video.parse_probe`, and the answer is already in `source_meta.json` -- so what is
    asserted here is that the stage uses it and does not overclaim from it.
    """
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=3)

    path = _georeference_only(
        tmp_path / "run",
        frames,
        source_meta={"location": {"lat": 37.8, "lon": -122.4, "alt": 12.5, "source": "iso6709"}},
    )
    georef = json.loads(path.read_text())

    assert (georef["lat"], georef["lon"], georef["height"]) == (37.8, -122.4, 12.5)
    assert georef["georefMethod"] == "exif-gps"
    assert georef["scaleSource"] == "unresolved"
    assert georef["uncertaintyM"] == UNALIGNED_UNCERTAINTY_M
    assert georef["alignment"] is None
    assert "not aligned" in georef["note"]


def test_frames_with_no_exif_no_location_and_no_coordinate_say_what_to_do(
    tmp_path: Path,
) -> None:
    """Refused, rather than placed at (0, 0): the Gulf of Guinea is not a default."""
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=3)

    with pytest.raises(Exception, match="lat/lon"):
        _georeference_only(tmp_path / "run", frames, source_meta=None)


def test_a_video_with_no_location_falls_back_to_the_captures_own_coordinate(
    tmp_path: Path,
) -> None:
    """The console sends where its camera was looking; the worker hands it over as
    `lat`/`lon`. It is used, and recorded as the hand placement it is."""
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=3)

    georef = json.loads(
        _georeference_only(
            tmp_path / "run",
            frames,
            source_meta={"location": None},
            params={"lat": 51.5007, "lon": -0.1246, "heading_deg": 30.0},
        ).read_text()
    )

    assert (georef["lat"], georef["lon"], georef["height"]) == (51.5007, -0.1246, 0.0)
    assert georef["georefMethod"] == "manual"
    assert georef["scaleSource"] == "unresolved"
    assert georef["uncertaintyM"] == UNALIGNED_UNCERTAINTY_M
    # No poses were given, so nothing could level it -- and the frame says so rather than
    # claiming a rotation. `place` recentres it regardless.
    assert georef["frame"]["source"] == "none"
    assert georef["frame"]["recentre"] is True
    assert georef["frame"]["headingDeg"] == 30.0


def test_a_location_on_the_video_wins_over_the_captures_coordinate(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=3)

    georef = json.loads(
        _georeference_only(
            tmp_path / "run",
            frames,
            source_meta={"location": {"lat": 37.8, "lon": -122.4, "alt": 12.5}},
            params={"lat": 51.5007, "lon": -0.1246},
        ).read_text()
    )

    assert (georef["lat"], georef["georefMethod"]) == (37.8, "exif-gps")


def test_fixes_without_a_pose_model_locate_but_do_not_align(tmp_path: Path) -> None:
    """EXIF on every frame and no reconstruction: a coordinate, and no scale claim."""
    frames = tmp_path / "frames"
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=6)
    gps_frames.tag_orbit(frames, {p.name: p.centre for p in truth.poses}, ORIGIN)

    georef = json.loads(_georeference_only(tmp_path / "run", frames, source_meta=None).read_text())

    assert georef["scaleSource"] == "unresolved"
    assert georef["alignment"] is None
    assert georef["fixes"]["frames"] == 6
    # The median of a symmetric orbit is its centre, to well inside a metre.
    assert georef["lat"] == pytest.approx(ORIGIN[0], abs=1e-5)
    assert georef["lon"] == pytest.approx(ORIGIN[1], abs=1e-5)


# --- the measured integration test ---------------------------------------------------


@pytest.fixture(scope="module")
def orbit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, tree_frames.Truth]:
    """Render the orbit, tag it with exact GPS, and reconstruct it once for this module.

    The pose model is reused by every alignment below: only the GPS changes between them,
    and re-running COLMAP to change a JPEG header would add a minute per case.
    """
    root = tmp_path_factory.mktemp("exif-orbit")
    workdir = Workdir.create(root / "pose")
    frames = workdir.input_path("frames")
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=FRAMES)
    gps_frames.tag_orbit(frames, {p.name: p.centre for p in truth.poses}, ORIGIN)
    execute(
        make_recipe(
            [{"id": "pose", "impl": "colmap", "params": {"matcher": "exhaustive"}}],
            inputs=["frames"],
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    return frames, workdir.artifact_path("pose", "poses"), truth


def _align(
    root: Path, frames: Path, poses: Path, noise_m: float, truth: tree_frames.Truth
) -> dict[str, Any]:
    """Run the georeference stage over a copy of the frames retagged with `noise_m`."""
    workdir = Workdir.create(root)
    target = workdir.input_path("frames")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(frames.iterdir()):
        (target / path.name).write_bytes(path.read_bytes())
    if noise_m > 0.0:
        # Strip the exact tag first: `write_gps` prepends, and two APP1 segments would
        # leave the reader looking at whichever one it found first.
        for path in sorted(target.iterdir()):
            raw = path.read_bytes()
            length = int.from_bytes(raw[4:6], "big")
            path.write_bytes(raw[:2] + raw[4 + length :])
        gps_frames.tag_orbit(
            target, {p.name: p.centre for p in truth.poses}, ORIGIN, noise_m=noise_m
        )
    seeded = workdir.input_path("poses")
    seeded.mkdir(parents=True, exist_ok=True)
    for path in sorted(poses.iterdir()):
        (seeded / path.name).write_bytes(path.read_bytes())
    execute(
        make_recipe([{"id": "georeference", "impl": "exif_gps"}], inputs=["frames", "poses"]),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    document = json.loads(workdir.artifact_path("georeference", "georef.json").read_text())
    assert isinstance(document, dict)
    return document


@requires_colmap
def test_exact_gps_recovers_the_frame_it_was_written_in(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """The centrepiece: where the capture lands, and how big it says it is.

    Measured 0.0088 m median residual and a scale within 1.3e-5 of `umeyama`. The bounds
    are where a regression shows, not at those values: misreading `model_aligner`'s
    transform lands at about 9 m on this fixture, three orders of magnitude away.
    """
    frames, poses, truth = orbit
    georef = _align(tmp_path / "exact", frames, poses, 0.0, truth)

    assert georef["georefMethod"] == "exif-gps"
    assert georef["scaleSource"] == "exif-gps"
    assert georef["fixes"]["frames"] == FRAMES
    assert georef["alignment"]["images"] == FRAMES
    assert georef["alignment"]["inliers"] == FRAMES

    # Where: the median of a symmetric orbit is its centre, and the height is the camera
    # height above the origin because that is where the fixes actually were.
    assert georef["lat"] == pytest.approx(ORIGIN[0], abs=1e-5)
    assert georef["lon"] == pytest.approx(ORIGIN[1], abs=1e-5)
    assert georef["height"] == pytest.approx(ORIGIN[2] + 3.6, abs=0.1)

    # How well: the residual is provenance, and it is per image as well as summarised.
    residual = georef["alignment"]["residualM"]
    assert residual["median"] < 0.5, f"median residual {residual['median']} m"
    assert residual["max"] < 2.0
    assert len(residual["perImage"]) == FRAMES

    # How big: the similarity's scale is the reconstruction's metric scale, and the truth
    # for it is the similarity that takes the recovered centres onto the known ones.
    model = sfm.read_model(poses)
    known = truth.by_name()
    fit = sfm.umeyama(
        np.stack([image.centre for image in model.images]),
        np.stack([known[image.name].centre for image in model.images]),
    )
    assert georef["alignment"]["scale"] == pytest.approx(fit.scale, rel=0.01)

    # Applied now, by `place`, from the `frame` block that carries the same similarity.
    assert georef["alignment"]["applied"] is True
    frame = georef["frame"]
    assert frame["source"] == "exif-gps-similarity" and frame["recentre"] is False
    assert frame["scale"] == georef["alignment"]["scale"]


@requires_colmap
def test_placing_by_the_similarity_puts_the_reconstruction_on_the_scene_it_came_from(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """The alignment, applied -- and checked against the scene rather than against itself.

    COLMAP's sparse points are handed to `place` as if they were a trained splat. What
    comes out must sit on the committed tree the orbit was rendered from, upright, in the
    east/north/up frame about the georeferenced origin: each point within centimetres of
    a fixture gaussian or of the rendered ground. Placed without the similarity
    (identity), they are not, which is what `applied: false` used to mean.

    Measured here on 2026-09-23: 0.011 m median with the similarity, 0.406 m without it.
    The bounds are 0.1 m and a factor of ten, where a regression would show.
    """
    frames, poses, truth = orbit
    georef = _align(tmp_path / "exact", frames, poses, 0.0, truth)
    points = sfm.read_points(poses)
    fixture = gaussians.read_splat(FIXTURE_PLY)
    # The fixture is in ENU about ORIGIN; the georeference's origin is the median fix,
    # which on this orbit is ORIGIN raised by the camera height.
    offset = exif.enu_offsets(
        (exif.Fix(name="o", lat=georef["lat"], lon=georef["lon"], alt=georef["height"]),),
        ORIGIN,
    )["o"]
    scene = fixture.xyz.astype(np.float64) - np.asarray(offset)

    def placed_distance(frame: dict[str, Any]) -> float:
        workdir = Workdir.create(tmp_path / f"place-{frame['source']}")
        trained = workdir.input_path("trained.ply")
        trained.parent.mkdir(parents=True, exist_ok=True)
        gaussians.write_ply(trained, _points_as_gaussians(points))
        (workdir.input_path("georef.json")).write_text(json.dumps({**georef, "frame": frame}))
        execute(
            make_recipe(
                [{"id": "place", "impl": "place_splat"}], inputs=["trained.ply", "georef.json"]
            ),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )
        placed = gaussians.read_splat(workdir.artifact_path("place", "canonical.ply"))
        sample = placed.xyz[:: max(1, placed.count // 400)].astype(np.float64)
        nearest = np.sqrt(((sample[:, None, :] - scene[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        # The render also has a textured ground at the fixture's z = 0 (`tree_frames`),
        # and most of SIFT's points are on it: distance to that plane counts too.
        ground = np.abs(sample[:, 2] + offset[2])
        return float(np.median(np.minimum(nearest, ground)))

    applied = placed_distance(georef["frame"])
    identity = placed_distance(
        {
            "source": "none",
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translationM": None,
            "recentre": False,
        }
    )

    assert applied < 0.1, f"median {applied:.3f} m from the scene with the similarity applied"
    assert identity > 10 * applied, f"identity {identity:.3f} m vs applied {applied:.3f} m"


def _points_as_gaussians(points: np.ndarray) -> dict[str, np.ndarray]:
    count = points.shape[0]
    columns = {axis: points[:, i] for i, axis in enumerate("xyz")}
    columns.update({f"f_dc_{i}": np.zeros(count) for i in range(3)})
    columns.update({"opacity": np.full(count, 2.0)})
    columns.update({f"scale_{i}": np.full(count, -4.0) for i in range(3)})
    columns.update({"rot_0": np.ones(count), "rot_1": np.zeros(count)})
    columns.update({"rot_2": np.zeros(count), "rot_3": np.zeros(count)})
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


@requires_colmap
def test_the_uncertainty_is_floored_rather_than_quoting_the_residual(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """Millimetres of residual must not become a claim of millimetres of accuracy.

    Every fix in one capture shares the receiver's bias, and a bias common to all of them
    moves the whole reconstruction without changing any residual at all.
    """
    frames, poses, truth = orbit
    georef = _align(tmp_path / "floor", frames, poses, 0.0, truth)

    assert georef["alignment"]["residualM"]["rms"] < EXIF_GPS_UNCERTAINTY_FLOOR_M
    assert georef["uncertaintyM"] == EXIF_GPS_UNCERTAINTY_FLOOR_M


@requires_colmap
def test_noisy_gps_still_aligns_and_says_how_noisy_it_was(
    orbit: tuple[Path, Path, tree_frames.Truth], tmp_path: Path
) -> None:
    """One metre of seeded noise per axis. The point is that it shows up in the residual."""
    frames, poses, truth = orbit
    georef = _align(tmp_path / "noisy", frames, poses, 1.0, truth)
    exact = _align(tmp_path / "exact2", frames, poses, 0.0, truth)

    assert georef["alignment"]["images"] == FRAMES
    # The noise is visible, and the alignment is not: the residual grew by orders of
    # magnitude while the recovered scale did not move by a percent.
    assert georef["alignment"]["residualM"]["rms"] > 10 * exact["alignment"]["residualM"]["rms"]
    assert georef["alignment"]["scale"] == pytest.approx(exact["alignment"]["scale"], rel=0.01)
    # ...and the reported uncertainty did not improve because the residual happened to.
    assert georef["uncertaintyM"] >= EXIF_GPS_UNCERTAINTY_FLOOR_M
