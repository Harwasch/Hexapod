"""Lane 1, the whole way: uploaded bytes in object storage to a site on the globe.

This is the A8 demonstration, run as a test. A capture with a real 12,000-gaussian `.ply`
in the bucket, a `splat-ingest` job, the shipped recipe and the shipped implementations
under the **local** runner -- which is now the default -- and at the end a site whose
asset points at a tileset that is really in the bucket, whose boundary is the capture's
measured extent rather than a placeholder square, and whose thumbnail is a JPEG.

Nothing here is stubbed. `moto` stands in for the bucket because no container image can be
pulled in this environment; every other moving part is the real one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.models import Artifact, Capture, CaptureFile, Epoch, Job, JobStep, Site
from app.models.enums import (
    ArtifactKind,
    CaptureKind,
    CaptureStatus,
    Representation,
    RunStatus,
    UploadStatus,
)
from app.storage import S3Storage
from app.worker.claim import claim_next
from app.worker.config import WorkerConfig
from app.worker.runner import JobSupervisor

BUCKET = "twin-lane1"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PLY = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"

#: Sheffield Park, the same corner of the world the other capture fixtures sit in.
LAT, LON, HEIGHT = 28.0389, -82.6966, 22.5


@pytest.fixture
def sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            public_base_url="https://cdn.example.com/twin-lane1",
        )


def _uploaded_capture(
    db: Session,
    storage: S3Storage,
    *,
    placed: bool = True,
    slug: str = "orchard-tree",
    captured_at: datetime = datetime(2026, 9, 12, tzinfo=UTC),
    lat: float = LAT,
) -> Capture:
    """A capture whose `.ply` really is in the bucket, as A4's upload would leave it."""
    capture = Capture(
        slug=slug,
        name=slug.replace("-", " ").capitalize(),
        kind=CaptureKind.GAUSSIAN_SPLAT,
        status=CaptureStatus.NOT_STARTED,
        sensor="Scaniverse",
        device="iPhone 15 Pro",
        captured_at=captured_at,
        # Lane 1 has no EXIF and no poses, so the coordinate is the operator's: this is
        # "the capture carries a coordinate", and the worker hands it to whichever stage
        # places captures by hand.
        #
        # `upAxis: "z"` because the committed fixture is written east/north/up, which is
        # not what a `.ply` is taken to be when nobody says (y down, the 3DGS
        # convention). It is also the path a real capture's override takes.
        metadata_=(
            {"lat": lat, "lon": LON, "height": HEIGHT, "upAxis": "z"} if placed else {"upAxis": "z"}
        ),
    )
    db.add(capture)
    db.flush()
    key = f"captures/{capture.id}/splat.ply"
    storage.put_object(key, FIXTURE_PLY.read_bytes(), "application/octet-stream")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="splat.ply",
            content_type="application/octet-stream",
            bytes=FIXTURE_PLY.stat().st_size,
            storage_key=key,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()
    return capture


def _run(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    capture: Capture,
    tmp_path: Path,
    params: dict[str, dict[str, object]] | None = None,
    *,
    expect: str = "complete",
) -> Job:
    job = Job(
        capture_id=capture.id,
        recipe="splat-ingest",
        recipe_version="0.1.0",
        params=params or {},
    )
    db.add(job)
    db.commit()
    session = sessions()
    assert claim_next(session, worker_id="worker-lane1", lease_s=60) is not None
    session.close()
    config = WorkerConfig(
        workdir_root=tmp_path / "runs",
        worker_id="worker-lane1",
        # The default since A8. Spelled out here because it is the thing under test.
        runner="local",
        lease_s=60.0,
        poll_s=0.2,
        retry_backoff_s=0.0,
    )
    assert JobSupervisor(sessions, storage, config).run(job.id) == expect
    db.expire_all()
    return job


def _glb_json(blob: bytes) -> dict[str, object]:
    assert blob[:4] == b"glTF"
    length, kind = struct.unpack_from("<II", blob, 12)
    assert kind == 0x4E4F534A
    return dict(json.loads(blob[20 : 20 + length]))


def test_a_dropped_ply_becomes_a_site_with_no_human_step(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = _uploaded_capture(db, storage)

    job = _run(db, sessions, storage, capture, tmp_path)

    finished = db.get(Job, job.id)
    assert finished is not None and finished.status is RunStatus.COMPLETE
    registered = db.get(Capture, capture.id)
    assert registered is not None
    assert registered.status is CaptureStatus.COMPLETE
    assert registered.site_id is not None

    site = db.get(Site, registered.site_id)
    assert site is not None
    assert site.slug == "orchard-tree"
    assert site.name == "Orchard tree"

    # The site's one asset points at a tileset that really is in the bucket, and that
    # tileset really is a KHR_gaussian_splatting GLB of 12,000 gaussians.
    assert len(site.assets) == 1
    url = str(site.assets[0].source["url"])
    key = f"runs/{job.id}/package/splat/tileset.json"
    assert url.endswith(key)
    tileset = json.loads(storage.get_object(key))
    assert tileset["root"]["content"]["uri"] == "splat.glb"
    glb = _glb_json(storage.get_object(f"runs/{job.id}/package/splat/splat.glb"))
    accessors = glb["accessors"]
    assert isinstance(accessors, list)
    assert accessors[0]["count"] == 12_000


def test_the_site_boundary_is_the_captures_measured_extent(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A7 drew a 60 m placeholder square and said A8's manifest would replace it.

    The fixture is a 4.4 m by 5.8 m tree, so its footprint must be metres across, not
    sixty, and it must be offset from the placed coordinate exactly as the gaussians are
    offset from the origin of their own frame.
    """
    capture = _uploaded_capture(db, storage)

    _run(db, sessions, storage, capture, tmp_path)

    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    site = db.get(Site, registered.site_id)
    assert site is not None
    encoded = db.scalar(select(Site.boundary.ST_AsGeoJSON()).where(Site.id == site.id))
    assert encoded is not None
    geometry = json.loads(encoded)
    # Stored as a MultiPolygon, as every site boundary is; one polygon, one outer ring.
    ring = geometry["coordinates"][0][0]
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    metres_per_degree = 111_320.0
    east_m = (max(lons) - min(lons)) * metres_per_degree * 0.8826  # cos(28.04 deg)
    north_m = (max(lats) - min(lats)) * metres_per_degree

    assert east_m == pytest.approx(4.44, abs=0.2)
    assert north_m == pytest.approx(5.81, abs=0.2)
    # And it is around where the capture was placed, not centred on it: the tree's
    # bounding box is off-centre in its own frame.
    assert min(lats) < LAT < max(lats)
    assert min(lons) < LON < max(lons)


def test_the_georeference_is_recorded_as_placed_not_surveyed(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = _uploaded_capture(db, storage)

    _run(db, sessions, storage, capture, tmp_path)

    registered = db.get(Capture, capture.id)
    assert registered is not None
    assert registered.georef_method is not None
    assert registered.georef_method.value == "manual"
    assert registered.scale_source is not None
    assert registered.scale_source.value == "unresolved"
    # Ten metres, not zero: the inspector must not read a hand placement as a survey.
    assert registered.uncertainty_m == pytest.approx(10.0)

    site = db.get(Site, registered.site_id) if registered.site_id else None
    assert site is not None
    manifest = site.metadata_["registration"]["manifest"]
    assert manifest["capture"]["sensor"] == "Scaniverse"
    assert manifest["capture"]["device"] == "iPhone 15 Pro"
    assert manifest["capture"]["capturedAt"].startswith("2026-09-12")
    assert manifest["resolution"]["gsdM"] is None
    assert manifest["splat"]["gaussiansPackaged"] == 12_000


def test_the_measured_ground_and_the_placement_reach_the_asset(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """`ground_samples.json` stops being an artifact nothing reads.

    The `splat_ground` stage has produced this file since A8 and nothing consumed it; a
    person ran `tools/captures/ground_samples.py`, read a second number out of a browser
    console, subtracted them and typed the difference into `heightOffsetM`. B4 is where
    the cells travel with the asset instead, as ellipsoid heights, so the viewer can do
    that subtraction against whatever terrain it actually has.

    The provenance rides with them for the same reason: the inspector is looking at an
    asset, not at the capture row that produced it.
    """
    capture = _uploaded_capture(db, storage)

    _run(db, sessions, storage, capture, tmp_path)

    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    site = db.get(Site, registered.site_id)
    assert site is not None
    splat = next(a for a in site.assets if a.representation == Representation.GAUSSIAN_SPLAT)

    render = splat.render_config
    assert render["clampToGround"] is True
    samples = render["groundSamples"]
    assert len(samples) > 0, "the tree is 4.4 m by 5.8 m, so a 2 m grid has cells in it"
    # Ellipsoid heights, not the local up-coordinate: the placed origin has been added.
    # A metre either side of the placement covers the tree's own ground, and what this is
    # really asserting is that nobody shipped a raw `z` of about 0 as a height of 22.5.
    for sample in samples:
        assert abs(sample["height"] - HEIGHT) < 5.0, sample
        assert abs(sample["lat"] - LAT) < 1e-3 and abs(sample["lon"] - LON) < 1e-3

    # ...and how it was placed, beside them, in the block the inspector already reads.
    provenance = render["provenance"]
    assert provenance["georefMethod"] == "manual"
    assert provenance["scaleSource"] == "unresolved"
    assert provenance["uncertaintyM"] == pytest.approx(10.0)


def test_a_finished_run_keeps_its_outputs_and_drops_its_scratch(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The upload copy and every stage's `work/` go; `out/` and `step.json` stay, which
    is everything retry-from-stage reads. A Lane 2 video run would otherwise leave the
    video and every candidate frame on the worker's volume."""
    capture = _uploaded_capture(db, storage)

    job = _run(db, sessions, storage, capture, tmp_path)

    root = tmp_path / "runs" / str(job.id)
    assert not (root / "inputs").exists()
    assert not any((root / "stages").glob("*/work"))
    assert (root / "stages" / "package" / "out" / "splat" / "tileset.json").is_file()
    assert (root / "stages" / "normalize" / "step.json").is_file()


def test_every_artifact_the_plan_asks_for_reaches_the_bucket_with_a_row(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = _uploaded_capture(db, storage)

    job = _run(db, sessions, storage, capture, tmp_path)

    rows = db.scalars(select(Artifact).join(JobStep).where(JobStep.job_id == job.id)).all()
    by_kind = {row.kind: row for row in rows}
    assert {
        ArtifactKind.SPLAT,
        ArtifactKind.TILES_3D,
        ArtifactKind.THUMBNAIL,
        ArtifactKind.GROUND_SAMPLES,
        ArtifactKind.MANIFEST,
    } <= set(by_kind)
    for row in rows:
        # A directory artifact is one row at a prefix; a file artifact is the object.
        key = (
            row.storage_key
            if row.kind is not ArtifactKind.TILES_3D
            else f"{row.storage_key}/tileset.json"
        )
        assert storage.get_object(key)
    thumbnail = storage.get_object(by_kind[ArtifactKind.THUMBNAIL].storage_key)
    assert thumbnail[:3] == b"\xff\xd8\xff"
    ground = json.loads(storage.get_object(by_kind[ArtifactKind.GROUND_SAMPLES].storage_key))
    assert ground["origin"]["lat"] == pytest.approx(LAT)
    assert ground["samples"]


def test_the_collision_grid_goes_up_beside_the_tiles_as_opaque_bytes(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """`collision.bin` (tools/captures splat_tiles `COLLISION_FORMAT`) is in the tileset's
    directory, so the whole-directory publish carries it; the root tile declares it; and it
    is served as `application/octet-stream`, the gzip being the client's to inflate."""
    capture = _uploaded_capture(db, storage)

    job = _run(db, sessions, storage, capture, tmp_path)

    tiles = f"runs/{job.id}/package/splat"
    tileset = json.loads(storage.get_object(f"{tiles}/tileset.json"))
    collision = tileset["root"]["extras"]["collision"]
    assert collision["format"] == "hexapod.collision" and collision["version"] == 1
    head = storage.head_object(f"{tiles}/{collision['uri']}")
    assert head is not None and head.content_type == "application/octet-stream"
    raw = gzip.decompress(storage.get_object(f"{tiles}/{collision['uri']}"))
    assert len(raw) == 76 * collision["bricks"] and collision["solidCells"] > 0


def test_the_thumbnail_endpoint_finally_has_something_to_point_at(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """`sites.thumbnail_url` existed and nothing set it. The thumbnail stage sets it."""
    capture = _uploaded_capture(db, storage)

    job = _run(db, sessions, storage, capture, tmp_path)

    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    site = db.get(Site, registered.site_id)
    assert site is not None
    assert site.thumbnail_url == (
        f"https://cdn.example.com/twin-lane1/runs/{job.id}/thumbnail/thumbnail.jpg"
    )
    assert storage.get_object(f"runs/{job.id}/thumbnail/thumbnail.jpg")[:3] == b"\xff\xd8\xff"


def test_a_capture_with_no_coordinate_falls_back_to_the_recipes_placement(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """Nothing is invented: a capture nobody placed gets the recipe's default, and the
    manifest still says the placement was manual with ten metres of uncertainty."""
    capture = _uploaded_capture(db, storage, placed=False)

    _run(db, sessions, storage, capture, tmp_path)

    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    site = db.get(Site, registered.site_id)
    assert site is not None
    georef = site.metadata_["registration"]["georef"]
    assert (georef["lat"], georef["lon"]) == (0.0, 0.0)
    assert registered.uncertainty_m == pytest.approx(10.0)


def test_job_params_override_the_captures_own_coordinate(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """`jobs.params` was recorded and never read until A8. A person asking for this run
    specifically beats both the recipe and the capture."""
    capture = _uploaded_capture(db, storage)

    _run(
        db,
        sessions,
        storage,
        capture,
        tmp_path,
        params={"georeference": {"lat": 51.5007, "lon": -0.1246, "uncertainty_m": 2.0}},
    )

    registered = db.get(Capture, capture.id)
    assert registered is not None and registered.site_id is not None
    site = db.get(Site, registered.site_id)
    assert site is not None
    georef = site.metadata_["registration"]["georef"]
    assert georef["lat"] == pytest.approx(51.5007)
    assert registered.uncertainty_m == pytest.approx(2.0)


def test_params_of_the_wrong_shape_dead_letter_with_a_message(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    capture = _uploaded_capture(db, storage)
    job = Job(
        capture_id=capture.id,
        recipe="splat-ingest",
        recipe_version="0.1.0",
        # Meant `{"georeference": {"lat": ...}}`. Silently dropping this would put the
        # site in the Gulf of Guinea and say nothing.
        params={"lat": 51.5},
    )
    db.add(job)
    db.commit()
    session = sessions()
    assert claim_next(session, worker_id="worker-lane1", lease_s=60) is not None
    session.close()

    outcome = JobSupervisor(
        sessions,
        storage,
        WorkerConfig(workdir_root=tmp_path / "runs", worker_id="worker-lane1", runner="local"),
    ).run(job.id)

    assert outcome == "error"
    db.expire_all()
    failed = db.get(Job, job.id)
    assert failed is not None
    assert "must be an object" in (failed.error or "")


def test_a_capture_that_is_not_a_splat_fails_with_a_message_naming_the_file(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """The refusal a person actually sees in the panel when they drop the wrong thing."""
    capture = Capture(
        slug="not-a-splat", name="Not a splat", kind=CaptureKind.GAUSSIAN_SPLAT, metadata_={}
    )
    db.add(capture)
    db.flush()
    key = f"captures/{capture.id}/scan.ply"
    storage.put_object(key, b"this is not a PLY file\n", "application/octet-stream")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="scan.ply",
            storage_key=key,
            bytes=23,
            status=UploadStatus.COMPLETE,
        )
    )
    job = Job(capture_id=capture.id, recipe="splat-ingest", recipe_version="0.1.0", params={})
    db.add(job)
    db.commit()
    session = sessions()
    assert claim_next(session, worker_id="worker-lane1", lease_s=60) is not None
    session.close()

    outcome = JobSupervisor(
        sessions,
        storage,
        WorkerConfig(
            workdir_root=tmp_path / "runs",
            worker_id="worker-lane1",
            runner="local",
            max_attempts=1,
            retry_backoff_s=0.0,
        ),
    ).run(job.id)

    assert outcome == "error"
    db.expire_all()
    failed = db.get(Job, job.id)
    assert failed is not None
    assert "scan.ply" in (failed.error or "")
    assert "end_header" in (failed.error or "")
    broken = db.get(Capture, capture.id)
    assert broken is not None and broken.status is CaptureStatus.ERROR
    assert broken.site_id is None


def test_the_run_is_repeatable_and_produces_the_same_tileset_bytes(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """Two runs over the same capture put identical bytes at two different keys."""
    first_capture = _uploaded_capture(db, storage)
    first = _run(db, sessions, storage, first_capture, tmp_path / "one")
    second_capture = Capture(
        slug="orchard-tree-again",
        name="Orchard tree again",
        kind=CaptureKind.GAUSSIAN_SPLAT,
        sensor="Scaniverse",
        device="iPhone 15 Pro",
        captured_at=datetime(2026, 9, 12, tzinfo=UTC),
        metadata_={"lat": LAT, "lon": LON, "height": HEIGHT, "upAxis": "z"},
    )
    db.add(second_capture)
    db.flush()
    key = f"captures/{second_capture.id}/splat.ply"
    storage.put_object(key, FIXTURE_PLY.read_bytes(), "application/octet-stream")
    db.add(
        CaptureFile(
            capture_id=second_capture.id,
            filename="splat.ply",
            storage_key=key,
            bytes=FIXTURE_PLY.stat().st_size,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()

    second = _run(db, sessions, storage, second_capture, tmp_path / "two")

    for name in ("splat.glb", "tileset.json"):
        assert storage.get_object(f"runs/{first.id}/package/splat/{name}") == storage.get_object(
            f"runs/{second.id}/package/splat/{name}"
        )
    assert isinstance(first.id, uuid.UUID) and first.id != second.id


def test_a_re_run_adds_a_revision_and_keeps_the_old_url(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    client: TestClient,
) -> None:
    """A10's reconciliation found that a re-run's reconstruction was invisible: the site
    kept run 1's geometry and took run 2's thumbnail. The globe must show the newest run.

    Changed with site epochs (decision D3): it used to do that by rewriting the asset's
    URL in place, so run 1's tiles stopped being referenced by anything. Now run 2 is a new
    *revision* of the capture's one epoch -- a second asset row, current -- and run 1's
    row is kept as it was, URL and render config included, with `validTo` set. The site
    still shows one splat (`GET /sites/{id}` lists only current revisions).
    """
    capture = _uploaded_capture(db, storage)

    first = _run(db, sessions, storage, capture, tmp_path / "one")
    site_id = db.get(Capture, capture.id).site_id  # type: ignore[union-attr]
    assert site_id is not None
    first_asset = db.get(Site, site_id).assets[0]  # type: ignore[union-attr]
    first_url = str(first_asset.source["url"])
    first_render = dict(first_asset.render_config)
    assert first_url.endswith(f"runs/{first.id}/package/splat/tileset.json")

    second = _run(db, sessions, storage, capture, tmp_path / "two")
    assert second.id != first.id

    db.expire_all()
    site = db.get(Site, site_id)
    assert site is not None
    splats = [a for a in site.assets if a.representation is Representation.GAUSSIAN_SPLAT]
    assert len(splats) == 2
    old = next(a for a in splats if a.id == first_asset.id)
    new = next(a for a in splats if a.id != first_asset.id)
    # Run 1's revision is history, untouched but for its validity.
    assert old.source["url"] == first_url
    assert old.valid_to is not None and old.default_visible is False
    assert old.render_config == first_render
    # Run 2's is what the epoch shows.
    assert str(new.source["url"]).endswith(f"runs/{second.id}/package/splat/tileset.json")
    assert new.valid_to is None and new.epoch_id == old.epoch_id is not None
    read = client.get(f"/api/v1/sites/{site_id}").json()
    assert [asset["id"] for asset in read["assets"]] == [str(new.id)]
    [epoch] = db.scalars(select(Epoch).where(Epoch.site_id == site_id)).all()
    assert epoch.job_id == second.id
    # And the picture and the geometry now come from the same run.
    assert site.thumbnail_url is not None
    assert f"runs/{second.id}/" in site.thumbnail_url
    assert str(site.metadata_["jobId"]) == str(second.id)


# --- a second capture of the same field (decision D3, the M2 evidence) -------------------


def _enu_north_m(
    lat: float, lon: float, height: float, origin: tuple[float, float, float]
) -> float:
    """North metres of a point about `origin`, by pyproj: an implementation independent of
    the pipeline's own WGS84 arithmetic (`exif.ecef_to_enu`)."""
    import math

    from pyproj import Transformer

    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = to_ecef.transform(lon, lat, height)
    ox, oy, oz = to_ecef.transform(origin[1], origin[0], origin[2])
    phi, lam = math.radians(origin[0]), math.radians(origin[1])
    dx, dy, dz = x - ox, y - oy, z - oz
    return float(
        -math.sin(phi) * math.cos(lam) * dx
        - math.sin(phi) * math.sin(lam) * dy
        + math.cos(phi) * dz
    )


def _row(db: Session, asset_id: uuid.UUID) -> dict[str, object]:
    row = db.execute(
        text("SELECT to_jsonb(a) FROM assets a WHERE id = :id"), {"id": asset_id}
    ).scalar_one()
    assert isinstance(row, dict)
    return row


def test_a_second_capture_of_the_same_field_joins_its_site_as_epoch_2(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    client: TestClient,
    record_property: Callable[[str, object], None],
) -> None:
    """The M2 demonstration: September's scan, then October's, about 10 m north.

    October is matched to September's site before its run starts (its point is within
    25 m of the site), so the pipeline places it in September's frame. It becomes epoch 2
    of the same site, its tiles hang off the same root transform, it sits where it was
    captured relative to epoch 1, and epoch 1's asset row is not touched -- not even its
    `updated_at`.
    """
    september = _uploaded_capture(db, storage)
    first = _run(db, sessions, storage, september, tmp_path / "september")
    site_id = db.get(Capture, september.id).site_id  # type: ignore[union-attr]
    assert site_id is not None
    [epoch_1_asset] = db.get(Site, site_id).assets  # type: ignore[union-attr]
    before = _row(db, epoch_1_asset.id)
    bbox_1 = db.get(Site, site_id).metadata_["registration"]["bboxLocalM"]  # type: ignore[union-attr]
    north_lat = LAT + 0.00009

    october = _uploaded_capture(
        db,
        storage,
        slug="orchard-tree-october",
        captured_at=datetime(2026, 10, 12, tzinfo=UTC),
        lat=north_lat,
    )
    second = _run(db, sessions, storage, october, tmp_path / "october")

    db.expire_all()
    assert db.get(Capture, october.id).site_id == site_id  # type: ignore[union-attr]
    site = db.get(Site, site_id)
    assert site is not None and site.frame is not None
    assert (site.frame.lat, site.frame.lon, site.frame.height) == (LAT, LON, HEIGHT)
    epochs = list(db.scalars(select(Epoch).where(Epoch.site_id == site_id).order_by(Epoch.ordinal)))
    assert [e.ordinal for e in epochs] == [1, 2]
    assert [e.join_method for e in epochs] == ["new-site", "spatial"]
    assert [e.job_id for e in epochs] == [first.id, second.id]
    after = _row(db, epoch_1_asset.id)
    assert after == before
    splats = sorted(site.assets, key=lambda a: a.created_at)
    urls = [str(a.source["url"]) for a in splats]
    assert len(set(urls)) == 2
    assert [a.epoch_id for a in splats] == [e.id for e in epochs]

    # One frame: both tilesets hang off the same root transform.
    def root_transform(job: Job) -> list[float]:
        tileset = json.loads(storage.get_object(f"runs/{job.id}/package/splat/tileset.json"))
        transform: list[float] = tileset["root"]["transform"]
        return transform

    assert root_transform(first) == root_transform(second)
    # And October sits where it was captured, relative to September: its north extreme
    # is the north offset of its coordinate further on. 1 mm, from first principles: the
    # SPZ position quantum is 2^-12 m (0.24 mm) on each bound, plus float32 at 10 m.
    # Read off October's own run: the site's metadata mirrors only the epoch it shows.
    bbox_2 = json.loads(storage.get_object(f"runs/{second.id}/register/registration.json"))[
        "bboxLocalM"
    ]
    assert site.metadata_["registration"]["bboxLocalM"] == bbox_1
    expected = _enu_north_m(north_lat, LON, HEIGHT, (LAT, LON, HEIGHT))
    assert abs((bbox_2["max"][1] - bbox_1["max"][1]) - expected) < 1e-3
    assert abs((bbox_2["min"][1] - bbox_1["min"][1]) - expected) < 1e-3
    assert abs(bbox_2["max"][0] - bbox_1["max"][0]) < 1e-3

    listed = client.get(f"/api/v1/sites/{site_id}/epochs").json()
    assert [item["observedAt"][:10] for item in listed] == ["2026-09-12", "2026-10-12"]
    assert len({item["currentAssetId"] for item in listed}) == 2
    read = client.get(f"/api/v1/sites/{site_id}").json()
    assert read["frame"] is not None and len(read["epochs"]) == 2 and len(read["assets"]) == 2

    record_property("epochs", json.dumps(listed, sort_keys=True))
    record_property("frame", json.dumps(read["frame"], sort_keys=True))
    record_property("north_offset_expected_m", expected)
    record_property("north_offset_measured_m", bbox_2["max"][1] - bbox_1["max"][1])
    record_property(
        "epoch1_row_sha256_before",
        hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    )
    record_property(
        "epoch1_row_sha256_after",
        hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
    )


def test_a_run_whose_frame_disagrees_with_its_site_dead_letters(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """A job told (through its params) to place an epoch in another frame is placed there
    consistently by the pipeline -- and refused at registration, never put on the site."""
    capture = _uploaded_capture(db, storage)
    _run(db, sessions, storage, capture, tmp_path / "one")
    site_id = db.get(Capture, capture.id).site_id  # type: ignore[union-attr]
    assert site_id is not None
    other = {"siteId": str(site_id), "lat": LAT - 0.0001, "lon": LON, "height": HEIGHT}
    assets_before = len(db.get(Site, site_id).assets)  # type: ignore[union-attr]

    failed = _run(
        db,
        sessions,
        storage,
        capture,
        tmp_path / "two",
        params={"normalize": {"site_frame": other}, "georeference": {"site_frame": other}},
        expect="error",
    )

    assert failed.status is RunStatus.ERROR
    assert "registration refused" in (failed.error or "")
    db.expire_all()
    assert len(db.get(Site, site_id).assets) == assets_before  # type: ignore[union-attr]
