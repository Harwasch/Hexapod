"""Lane 1 end to end under LocalRunner: a real `.ply` in, a real tileset out.

Everything here runs the shipped `splat-ingest` recipe with the shipped implementations
over the committed 12,000-gaussian fixture. Nothing is stubbed, nothing is mocked, and the
tileset that comes out is parsed rather than counted: a `splat.glb` that is not a GLB, or
whose accessor count disagrees with the manifest, fails here.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

import pytest

import gaussians
from conftest import FIXTURE_PLY, tree
from executor import execute
from recipe import load_recipe
from registry import known_impls
from runners import RunnerSet
from workdir import Workdir

#: Somewhere real, so the tileset's root transform is a place rather than the Gulf of Guinea.
LAT, LON, HEIGHT = 28.0389, -82.6966, 22.5

PLACEMENT: dict[str, dict[str, Any]] = {
    "georeference": {"lat": LAT, "lon": LON, "height": HEIGHT, "uncertainty_m": 10.0},
    "normalize": {"sensor": "Scaniverse", "captured_at": "2026-09-12"},
    "register": {"slug": "orchard-tree", "title": "Orchard tree"},
}


def _run(root: Path, source: Path = FIXTURE_PLY) -> Workdir:
    workdir = Workdir.create(root)
    upload = workdir.input_path("upload")
    upload.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, upload / source.name)
    recipe = load_recipe("splat-ingest").with_params(PLACEMENT)
    execute(recipe, workdir, RunnerSet.local())
    return workdir


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Workdir:
    return _run(tmp_path_factory.mktemp("lane1") / "run")


def _json(workdir: Workdir, stage: str, name: str) -> dict[str, Any]:
    loaded = json.loads((workdir.out_dir(stage) / name).read_text(encoding="utf-8"))
    return dict(loaded)


def _glb_json(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    assert blob[:4] == b"glTF"
    length, kind = struct.unpack_from("<II", blob, 12)
    assert kind == 0x4E4F534A
    return dict(json.loads(blob[20 : 20 + length]))


def test_the_recipe_runs_every_stage_for_real(run: Workdir) -> None:
    stages = [path.name for path in sorted(run.stages_dir.iterdir())]

    assert sorted(stages) == sorted(
        [
            "normalize",
            "georeference",
            "package",
            "thumbnail",
            "ground_samples",
            "manifest",
            "register",
        ]
    )
    steps = [json.loads(run.step_path(name).read_text()) for name in stages]
    assert {step["runner"] for step in steps} == {"local"}


def test_a_real_ply_becomes_a_real_tileset(run: Workdir) -> None:
    tiles = run.out_dir("package") / "splat"
    tileset = json.loads((tiles / "tileset.json").read_text(encoding="utf-8"))
    document = _glb_json(tiles / "splat.glb")

    assert sorted(entry.name for entry in tiles.iterdir()) == ["splat.glb", "tileset.json"]
    assert document["accessors"][0]["count"] == 12_000
    assert "KHR_gaussian_splatting" in document["extensionsRequired"]
    # The root transform is the ENU-to-ECEF frame at the placed coordinate: its
    # translation column is a point on the ellipsoid, ~6378 km from the centre of the earth.
    translation = tileset["root"]["transform"][12:15]
    radius = sum(value * value for value in translation) ** 0.5
    assert 6_350_000 < radius < 6_400_000
    assert tileset["root"]["content"]["uri"] == "splat.glb"


def test_the_canonical_ply_is_the_fixture_normalised(run: Workdir) -> None:
    canonical = gaussians.read_splat(run.out_dir("normalize") / "canonical.ply")
    source = gaussians.read_splat(FIXTURE_PLY)

    assert canonical.count == source.count == 12_000
    for name in gaussians.CANONICAL_PROPERTIES:
        assert canonical.columns[name].tolist() == source.columns[name].tolist()


def test_source_meta_says_what_the_uploaded_bytes_turned_out_to_be(run: Workdir) -> None:
    meta = _json(run, "normalize", "source_meta.json")

    assert meta["format"] == "ply"
    assert meta["gaussians"] == 12_000
    assert meta["bytes"] == FIXTURE_PLY.stat().st_size
    assert meta["checksum"].startswith("sha256:")
    assert meta["sensor"] == "Scaniverse"
    assert meta["nonFinite"] == 0
    assert meta["extentM"]["east"] == pytest.approx(4.438, abs=0.01)


def test_the_thumbnail_is_a_jpeg_of_the_capture(run: Workdir) -> None:
    path = run.out_dir("thumbnail") / "thumbnail.jpg"
    blob = path.read_bytes()

    assert blob[:3] == b"\xff\xd8\xff"  # JPEG SOI + marker
    assert blob[-2:] == b"\xff\xd9"
    assert 2_000 < len(blob) < 400_000


def test_ground_samples_are_the_captures_own_heights_on_the_globe(run: Workdir) -> None:
    ground = _json(run, "ground_samples", "ground_samples.json")

    assert ground["frame"] == "enu"
    assert ground["origin"] == {"lat": LAT, "lon": LON, "height": HEIGHT}
    assert ground["cellM"] == 2.0
    assert ground["samples"], "the fixture is 6 m across, so 2 m cells must produce some"
    for sample in ground["samples"]:
        # Every sample is within a few tens of metres of where the capture was placed:
        # these are the capture's own coordinates, not the globe's.
        assert abs(sample["lat"] - LAT) < 0.001
        assert abs(sample["lon"] - LON) < 0.001
        assert sample["n"] >= 8
    assert ground["medianZ"] == pytest.approx(
        sorted(sample["z"] for sample in ground["samples"])[len(ground["samples"]) // 2],
        abs=1.0,
    )


def test_the_manifest_is_honest_about_a_hand_placed_capture(run: Workdir) -> None:
    manifest = _json(run, "manifest", "manifest.json")

    assert manifest["georeference"]["georefMethod"] == "manual"
    # Not zero: a capture dropped on a globe by hand is not a surveyed one, and this is
    # the field the inspector reads when it decides how confidently to phrase itself.
    assert manifest["georeference"]["uncertaintyM"] == 10.0
    assert manifest["georeference"]["scaleSource"] == "unresolved"
    assert manifest["resolution"]["gsdM"] is None, "a splat has no pixels behind it"
    assert manifest["resolution"]["medianGaussianM"] > 0
    assert manifest["capture"]["sensor"] == "Scaniverse"
    assert manifest["capture"]["capturedAt"] == "2026-09-12"
    assert manifest["splat"]["gaussiansIn"] == 12_000
    assert manifest["splat"]["gaussiansPackaged"] == 12_000
    assert (
        manifest["splat"]["bytes"]
        == (run.out_dir("package") / "splat" / "splat.glb").stat().st_size
    )
    assert manifest["tools"]["splatTiles"].startswith("sha256:")
    assert manifest["run"]["recipe"] == "splat-ingest"
    assert "runId" not in manifest["run"], "a run id would make two identical runs differ"


def test_the_manifest_extent_is_the_packaged_splats_own_bounding_box(run: Workdir) -> None:
    manifest = _json(run, "manifest", "manifest.json")
    document = _glb_json(run.out_dir("package") / "splat" / "splat.glb")

    assert manifest["splat"]["bboxLocalM"]["min"] == document["accessors"][0]["min"]
    assert manifest["splat"]["bboxLocalM"]["max"] == document["accessors"][0]["max"]
    extent = manifest["splat"]["extentM"]
    assert extent["east"] == pytest.approx(4.438, abs=0.01)
    assert extent["north"] == pytest.approx(5.807, abs=0.01)
    assert extent["up"] == pytest.approx(6.460, abs=0.01)


def test_the_registration_carries_the_extent_and_the_manifest(run: Workdir) -> None:
    registration = _json(run, "register", "registration.json")

    assert registration["slug"] == "orchard-tree"
    assert registration["artifacts"] == ["splat.glb", "tileset.json"]
    assert registration["thumbnail"] == "thumbnail.jpg"
    assert registration["bboxLocalM"]["max"][2] == pytest.approx(6.46, abs=0.01)
    # The whole manifest rides along, so the site's metadata records how it was made
    # without the worker having to fetch a second file.
    assert registration["manifest"]["georeference"]["uncertaintyM"] == 10.0


def test_an_spz_upload_produces_the_same_tileset_shape(tmp_path: Path) -> None:
    from test_spz_ingest import _spz_of

    source = tmp_path / "scan.spz"
    source.write_bytes(_spz_of(FIXTURE_PLY))

    workdir = _run(tmp_path / "run", source)

    meta = _json(workdir, "normalize", "source_meta.json")
    assert meta["format"] == "spz"
    assert meta["gaussians"] == 12_000
    document = _glb_json(workdir.out_dir("package") / "splat" / "splat.glb")
    assert document["accessors"][0]["count"] == 12_000


def test_two_runs_over_the_same_input_are_byte_identical(tmp_path: Path) -> None:
    """The same rule the stubs keep, kept by the real implementations.

    `step.json` and `log.txt` are excluded because they carry wall time, which A6 already
    said is the one thing a byte-identity check must not cover.
    """
    first = tree(_run(tmp_path / "a").root)
    second = tree(_run(tmp_path / "b").root)

    def outputs(files: dict[str, bytes]) -> dict[str, bytes]:
        return {
            name: blob
            for name, blob in files.items()
            if not name.endswith(("step.json", "log.txt", "child.json"))
        }

    assert outputs(first).keys() == outputs(second).keys()
    assert outputs(first) == outputs(second)


def test_the_three_new_stages_were_a_recipe_edit_and_three_decorators() -> None:
    """A6's central claim, checked rather than asserted in prose.

    `thumbnail`, `ground_samples` and `manifest` are registered implementations named by
    the recipe. Nothing in `executor.py`, `runners.py`, `plan.py` or `workdir.py` knows
    any of them exists -- which is why this test can only look them up by name.
    """
    recipe = load_recipe("splat-ingest")
    impls = {stage.id: stage.impl for stage in recipe.stages}

    assert impls["thumbnail"] == "splat_thumbnail"
    assert impls["ground_samples"] == "splat_ground"
    assert impls["manifest"] == "capture_manifest"
    assert {"splat_thumbnail", "splat_ground", "capture_manifest"} <= set(known_impls())
