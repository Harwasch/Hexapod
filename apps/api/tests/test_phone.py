"""The phone key: capture from a phone with a short shared key and nothing else."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models.capture import CaptureFile
from app.models.enums import ArtifactKind, RunStatus, UploadStatus
from app.models.job import Artifact, Job, JobStep
from app.services import phone_key
from app.storage import S3Storage, get_storage

KEY = "abcd-efgh-jkmn"
WRITE = "the-write-token"
PHONE = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    settings = Settings(
        api_write_token=WRITE,
        # A low iteration count keeps the suite fast; the stored format is the real one.
        api_phone_key_hash=phone_key.hash_key(KEY, salt=b"0123456789abcdef", iterations=1000),
        api_phone_daily_captures=3,
    )
    app = create_app(settings)

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as test_client:
        yield test_client


def uploaded(db: Session, capture_id: str) -> None:
    """A finished upload, written directly: the part PUTs go to storage, not this API."""
    db.add(
        CaptureFile(
            capture_id=uuid.UUID(capture_id),
            filename="walk.mov",
            content_type="video/quicktime",
            bytes=10,
            storage_key=f"captures/{capture_id}/source/walk.mov",
            status=UploadStatus.COMPLETE,
        )
    )
    db.flush()


def test_the_key_is_checked_and_typed_the_way_a_phone_types(client: TestClient) -> None:
    assert client.post("/api/v1/phone/check", headers=PHONE).status_code == 204
    loose = {"Authorization": f"Bearer  {KEY.upper()} "}
    assert client.post("/api/v1/phone/check", headers=loose).status_code == 204
    wrong = {"Authorization": "Bearer abcd-efgh-jkmm"}
    assert client.post("/api/v1/phone/check", headers=wrong).status_code == 401
    assert client.post("/api/v1/phone/check").status_code == 401


def test_a_phone_capture_is_placed_where_the_phone_was_and_can_upload(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/phone/captures",
        json={"lat": 44.9778, "lon": -93.265, "accuracyM": 8},
        headers=PHONE,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    capture = body["capture"]
    assert capture["metadata"] == {
        "origin": "phone-key",
        "lat": 44.9778,
        "lon": -93.265,
        "locationAccuracyM": 8.0,
    }
    # The upload token is an ordinary handoff token for this capture: the existing
    # upload routes take it, with no change to them.
    upload = {"Authorization": f"Bearer {body['uploadToken']}"}
    registered = client.post(
        f"/api/v1/captures/{capture['id']}/files",
        json={"filename": "walk.mov", "contentType": "video/quicktime", "bytes": 10},
        headers=upload,
    )
    assert registered.status_code == 201, registered.text


def test_the_key_starts_runs_only_on_its_own_captures(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    other = client.post(
        "/api/v1/captures",
        json={"name": "desktop", "kind": "video"},
        headers={"Authorization": f"Bearer {WRITE}"},
    ).json()
    uploaded(db, other["id"])

    run = {"recipe": "photo-reconstruct"}
    ok = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    assert ok.status_code == 202, ok.text
    assert ok.json()["recipe"] == "photo-reconstruct"
    refused = client.post(f"/api/v1/phone/captures/{other['id']}/process", json=run, headers=PHONE)
    assert refused.status_code == 401
    odd = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process", json={"recipe": "x"}, headers=PHONE
    )
    assert odd.status_code == 409


def test_the_key_stops_a_run_on_its_own_captures_only(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    other = client.post(
        "/api/v1/captures",
        json={"name": "desktop", "kind": "video"},
        headers={"Authorization": f"Bearer {WRITE}"},
    ).json()
    uploaded(db, other["id"])
    run = {"recipe": "photo-reconstruct"}
    started = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    client.post(
        f"/api/v1/captures/{other['id']}/jobs",
        json=run,
        headers={"Authorization": f"Bearer {WRITE}"},
    )

    assert client.post(f"/api/v1/phone/captures/{mine['id']}/stop").status_code == 401
    refused = client.post(f"/api/v1/phone/captures/{other['id']}/stop", headers=PHONE)
    assert refused.status_code == 401
    stopped = client.post(f"/api/v1/phone/captures/{mine['id']}/stop", headers=PHONE)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["id"] == started.json()["id"]
    assert stopped.json()["status"] == "cancelled"
    again = client.post(f"/api/v1/phone/captures/{mine['id']}/stop", headers=PHONE)
    assert again.status_code == 409
    # Stopped, it can be started again.
    retried = client.post(f"/api/v1/phone/captures/{mine['id']}/process", json=run, headers=PHONE)
    assert retried.status_code == 202, retried.text


def test_the_key_cannot_do_what_the_write_token_does(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/captures", json={"name": "n", "kind": "video"}, headers=PHONE
        ).status_code
        == 401
    )


def test_the_daily_limit_holds(client: TestClient) -> None:
    for _ in range(3):
        assert client.post("/api/v1/phone/captures", json={}, headers=PHONE).status_code == 201
    over = client.post("/api/v1/phone/captures", json={}, headers=PHONE)
    assert over.status_code == 409
    assert "limit" in over.json()["detail"]


def test_with_no_key_configured_the_phone_routes_are_closed(db: Session) -> None:
    app = create_app(Settings(api_write_token=WRITE))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        assert test_client.post("/api/v1/phone/check", headers=PHONE).status_code == 401


def test_a_phone_sets_only_the_options_it_is_allowed(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: Mapping[str, object], recipe: str = "photo-reconstruct") -> Response:
        return client.post(url, json={"recipe": recipe, "params": params}, headers=PHONE)

    for refused in (
        {"train": {"trainer": "/bin/sh"}},
        {"train": {"cap_max": 10}},
        {"train": {"cap_max": True}},
        {"pose": {"matcher": "sequential"}},
        {"normalize": "fast"},
        # The packaging cut that "Detail" used to be: every scan is now packed whole, and
        # Detail stays on the phone as its viewing budget.
        {"package": {"max_gaussians": 400_000}},
    ):
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)
    assert start({"normalize": {"up_axis": "sideways"}}, "splat-ingest").status_code == 409
    assert start({"package": {"max_gaussians": 800_000}}, "splat-ingest").status_code == 409
    # A splat file's place: a range each, and nothing else on the georeference stage.
    for refused_place in (
        {"lat": 91},
        {"lon": -181},
        {"lat": "north"},
        {"lat": True},
        {"uncertainty_m": 0},
    ):
        response = start({"georeference": refused_place}, "splat-ingest")
        assert response.status_code == 409, (refused_place, response.text)
    assert start({"georeference": {"lat": 46.134}}, "photo-reconstruct").status_code == 409

    chosen = {
        "normalize": {"max_side": 2400},
        "train": {"schedule_floor": 1.0, "cap_max": 1_000_000},
    }
    ok = start(chosen)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_splat_upload_can_be_placed_by_hand(client: TestClient, db: Session) -> None:
    """A .ply carries no location and a desktop browser often shares none: the phone key
    may send `georeference` lat/lon/height for splat-ingest, and the job keeps them."""
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    chosen = {
        "normalize": {"up_axis": "-y", "heading_deg": 0},
        "georeference": {"lat": 46.134, "lon": -123.881, "height": 5},
    }
    ok = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process",
        json={"recipe": "splat-ingest", "params": chosen},
        headers=PHONE,
    )
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_phone_may_turn_on_the_phone_capture_switches(client: TestClient, db: Session) -> None:
    """pose_opt, app_opt, bilateral_grid: JSON booleans only; more frames kept;
    `max_side` as a number or `auto`, `select` as one of the two frame rules; and the
    pose mapper and COLMAP version (`"3.9"` or `"4.2"`, strings) an A/B names."""
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: Mapping[str, object]) -> Response:
        return client.post(
            url, json={"recipe": "photo-reconstruct", "params": params}, headers=PHONE
        )

    for refused in (
        {"train": {"pose_opt": "true"}},
        {"train": {"bilateral_grid": 1}},
        {"normalize": {"keep": 1000}},
        {"normalize": {"max_side": "huge"}},
        {"normalize": {"max_side": 100}},
        {"normalize": {"max_side": True}},
        {"normalize": {"select": "blur_threshold"}},
        {"pose": {"mapper": "hierarchical"}},
        {"pose": {"colmap": "3.11"}},
        {"pose": {"colmap": 4.2}},
    ):
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)

    chosen = {
        "normalize": {"fps": 15, "keep": 180, "max_side": "auto", "select": "viewpoint"},
        "pose": {"mapper": "global", "colmap": "4.2"},
        "train": {"pose_opt": True, "app_opt": False, "bilateral_grid": True, "depth_loss": True},
    }
    ok = start(chosen)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_phone_may_ask_for_blocks_to_compare_a_capture_whole_and_in_blocks(
    client: TestClient, db: Session
) -> None:
    """`blocks` is `auto` or a count (tools/pipeline/blocks.py); the camera test's epsilon
    and the frozen ring are the two knobs the spool comparison calibrates."""
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: Mapping[str, object]) -> Response:
        return client.post(
            url, json={"recipe": "photo-reconstruct", "params": params}, headers=PHONE
        )

    for refused in (
        {"train": {"blocks": "many"}},
        {"train": {"blocks": 0}},
        {"train": {"blocks": 64}},
        {"train": {"block_epsilon": 0.9}},
        {"train": {"block_ring": "off"}},
    ):
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)

    chosen = {"train": {"blocks": 2, "block_epsilon": 0.08, "block_ring": False}}
    ok = start(chosen)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_phone_may_ask_how_blocks_train_and_how_many_images_a_step_sees(
    client: TestClient, db: Session
) -> None:
    """The efficiency experiments: blocks at once (`block_parallel`, 1 for the serial
    control), each block's schedule rule (`block_schedule`), and gsplat's `batch_size`."""
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: Mapping[str, object]) -> Response:
        return client.post(
            url, json={"recipe": "photo-reconstruct", "params": params}, headers=PHONE
        )

    for refused in (
        {"train": {"block_parallel": 0}},
        {"train": {"block_parallel": 32}},
        {"train": {"block_schedule": "half"}},
        {"train": {"batch_size": 0}},
        {"train": {"batch_size": 16}},
        {"train": {"batch_size": True}},
    ):
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)

    chosen = {
        "train": {"blocks": 2, "block_parallel": 1, "block_schedule": "share", "batch_size": 2}
    }
    ok = start(chosen)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == chosen


def test_a_quality_tier_scales_the_measured_budget_within_bounds(
    client: TestClient, db: Session
) -> None:
    """Quick and Best are multipliers on the budget the pipeline measures from the
    capture (`density_scale`), not gaussian counts; the pipeline's floor and ceilings
    still apply, and a multiplier outside a quarter to four times is refused."""
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    url = f"/api/v1/phone/captures/{mine['id']}/process"

    def start(params: Mapping[str, object]) -> Response:
        return client.post(
            url, json={"recipe": "photo-reconstruct", "params": params}, headers=PHONE
        )

    refusals: tuple[dict[str, object], ...] = (
        {"train": {"density_scale": 10}},
        {"train": {"density_scale": 0}},
        {"train": {"density_scale": "2"}},
        {"train": {"gaussian_density": 1.0}},  # the calibrated constant is the recipe's
    )
    for refused in refusals:
        response = start(refused)
        assert response.status_code == 409, (refused, response.text)

    best: dict[str, object] = {"train": {"schedule_floor": 1, "density_scale": 2}}
    ok = start(best)
    assert ok.status_code == 202, ok.text
    job = db.get(Job, uuid.UUID(ok.json()["id"]))
    assert job is not None and job.params == best


def test_a_finished_capture_downloads_its_placed_splat(client: TestClient, db: Session) -> None:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    capture_id = uuid.UUID(mine["id"])
    url = f"/api/v1/captures/{capture_id}/splat.ply"
    assert client.get(url, follow_redirects=False).status_code == 404

    job = Job(capture_id=capture_id, recipe="photo-reconstruct", recipe_version="7", params={})
    job.status = RunStatus.COMPLETE
    job.finished_at = datetime.now(tz=UTC)
    db.add(job)
    db.flush()
    for ordinal, stage, key in (
        (3, "train", "runs/x/train/trained.ply"),
        (6, "place", "runs/x/place/canonical.ply"),
    ):
        step = JobStep(job_id=job.id, stage_id=stage, ordinal=ordinal, impl=stage)
        step.status = RunStatus.COMPLETE
        db.add(step)
        db.flush()
        kind = ArtifactKind.SPLAT if stage == "place" else ArtifactKind.METADATA
        db.add(Artifact(job_step_id=step.id, kind=kind, storage_key=key, bytes=1))
    db.commit()

    response = client.get(url, follow_redirects=False)
    assert response.status_code == 307
    assert "runs/x/place/canonical.ply" in response.headers["location"]
