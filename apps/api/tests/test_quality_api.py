"""The quality bar, from the API's side: the phone's preview options, Refine, and the
verdict a finished run leaves on its capture."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Capture, Job, JobStep, Site
from app.models.enums import CaptureKind, RunStatus
from app.storage import S3Storage
from app.worker import registration
from app.worker.params import without_stale_roi
from app.worker.pipeline_bridge import load_recipe, plan_recipe
from tests.test_phone import PHONE, WRITE, client, uploaded

__all__ = ["client"]  # the fixture, re-used from the phone key's own tests

import_module("stages")  # registers the shipped implementations

#: photo-reconstruct v8, in order: `train` is ordinal 3.
STAGES = (
    "normalize",
    "pose",
    "mask",
    "train",
    "compensate",
    "georeference",
    "quality",
    "place",
    "package",
    "thumbnail",
    "ground_samples",
    "manifest",
    "register",
)

PREVIEW: dict[str, Any] = {
    "normalize": {"max_side": 1600, "fps": 4},
    "train": {"schedule_scale": 0.1, "cap_max": 200_000, "train_max_side": 800},
    "quality": {"mode": "preview", "bar": "balanced"},
    "package": {"max_gaussians": 400_000},
}

SUMMARY: dict[str, Any] = {
    "mode": "preview",
    "bar": "balanced",
    "barApplied": "balanced",
    "roi": {"center": [0.1, -0.2, 3.5], "radius": 0.8, "frame": "colmap"},
    "gaussians": {
        "in": 200_000,
        "out": 180_000,
        "keep": 120_000,
        "context": 60_000,
        "drop": 20_000,
    },
    "keepPct": 62.0,
    "contextPct": 90.5,
    "heldOutPsnr": 23.04,
    "views": {"medianRoi": 31},
    "gsd": {"metric": False, "medianRoiMm": None},
    "tips": [{"id": "from-above", "text": "Add frames from above."}],
}


def _process(client: TestClient, db: Session, params: dict[str, Any] | None = None) -> str:
    mine = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    uploaded(db, mine["id"])
    started = client.post(
        f"/api/v1/phone/captures/{mine['id']}/process",
        json={"recipe": "photo-reconstruct", "params": PREVIEW if params is None else params},
        headers=PHONE,
    )
    assert started.status_code == 202, started.text
    return str(mine["id"])


def _finish(db: Session, capture_id: str, *, quality: dict[str, Any] | None = SUMMARY) -> Job:
    """The run finished every stage, and its registration recorded the verdict."""
    capture = db.get(Capture, uuid.UUID(capture_id))
    assert capture is not None
    job = capture.jobs[-1]
    job.status = RunStatus.COMPLETE
    job.finished_at = datetime.now(tz=UTC)
    for ordinal, stage in enumerate(STAGES):
        step = JobStep(job_id=job.id, stage_id=stage, ordinal=ordinal, impl=stage, attempt=1)
        step.status = RunStatus.COMPLETE
        db.add(step)
    verdict = registration.capture_quality(quality, job.id, "https://cdn.example.com/c.ply")
    capture.quality = None if verdict is None else verdict.model_dump(mode="json", by_alias=True)
    db.commit()
    return job


# --- the phone's options ---------------------------------------------------------------


def test_the_preview_and_refine_options_are_whitelisted_and_checked(
    client: TestClient, db: Session
) -> None:
    capture_id = _process(client, db)
    job = db.get(Capture, uuid.UUID(capture_id)).jobs[-1]  # type: ignore[union-attr]
    assert job.params == PREVIEW
    client.post(f"/api/v1/phone/captures/{capture_id}/stop", headers=PHONE)
    url = f"/api/v1/phone/captures/{capture_id}/process"

    def start(params: dict[str, Any]) -> int:
        body = {"recipe": "photo-reconstruct", "params": params}
        return client.post(url, json=body, headers=PHONE).status_code

    for refused in (
        {"train": {"schedule_scale": 0.01}},
        {"train": {"train_max_side": 5000}},
        {"train": {"opacity_reg": 0.5}},
        {"train": {"antialiased": "yes"}},
        {"train": {"depth_loss": 1}},
        {"train": {"roi": [0, 0, 0]}},
        {"train": {"roi": {"center": [0, 0], "radius": 1}}},
        {"train": {"roi": {"center": [0, 0, "x"], "radius": 1}}},
        {"train": {"roi": {"center": [0, 0, 0], "radius": 0}}},
        {"train": {"roi": {"center": [0, 0, 0], "radius": 1, "frame": "enu"}}},
        {"quality": {"bar": "lenient"}},
        {"quality": {"mode": "draft"}},
        {"quality": {"keep_min_views": 1}},
    ):
        assert start(refused) == 409, refused

    chosen = {
        "train": {
            "antialiased": True,
            "depth_loss": False,
            "opacity_reg": 0.01,
            "roi": {"center": [1, 2.5, -3], "radius": 0.75},
        },
        "quality": {"bar": "strict", "mode": "refine"},
    }
    assert start(chosen) == 202


# --- Refine ----------------------------------------------------------------------------


def test_refine_resumes_the_preview_at_train_inside_its_region(
    client: TestClient, db: Session
) -> None:
    capture_id = _process(client, db)
    job = _finish(db, capture_id)

    refined = client.post(
        f"/api/v1/phone/captures/{capture_id}/refine",
        json={
            "params": {
                "normalize": {"max_side": 2400},
                "train": {"cap_max": 1_000_000},
                "quality": {"bar": "strict", "mode": "preview"},
                "package": {"max_gaussians": 800_000},
            }
        },
        headers=PHONE,
    )

    assert refined.status_code == 202, refined.text
    body = refined.json()
    assert body["id"] == str(job.id)  # the same run: its frames and poses are kept
    assert body["status"] == "not-started"
    assert body["params"] == {
        # What the kept frames were made with, not what was asked for now.
        "normalize": PREVIEW["normalize"],
        # The preview's short-schedule knobs are gone; the region is the preview's.
        "train": {"cap_max": 1_000_000, "roi": {"center": [0.1, -0.2, 3.5], "radius": 0.8}},
        "quality": {"bar": "strict", "mode": "refine"},
        "package": {"max_gaussians": 800_000},
    }
    states = {step["stageId"]: step["status"] for step in body["steps"]}
    assert [states[s] for s in ("normalize", "pose", "mask")] == ["complete"] * 3
    assert {states[s] for s in STAGES[3:]} == {"not-started"}
    # And it is running now, so a second Refine waits for it.
    again = client.post(f"/api/v1/phone/captures/{capture_id}/refine", json={}, headers=PHONE)
    assert again.status_code == 409


def test_refine_defaults_to_the_balanced_bar(client: TestClient, db: Session) -> None:
    capture_id = _process(client, db)
    _finish(db, capture_id)

    refined = client.post(f"/api/v1/phone/captures/{capture_id}/refine", json={}, headers=PHONE)

    assert refined.status_code == 202, refined.text
    assert refined.json()["params"]["quality"] == {"bar": "balanced", "mode": "refine"}


def test_refine_is_refused_where_there_is_nothing_to_refine(
    client: TestClient, db: Session
) -> None:
    capture_id = _process(client, db)
    url = f"/api/v1/phone/captures/{capture_id}/refine"
    # Still running.
    assert client.post(url, json={}, headers=PHONE).status_code == 409
    # Finished, but with no quality verdict (a run from before the quality stage).
    _finish(db, capture_id, quality=None)
    assert client.post(url, json={}, headers=PHONE).status_code == 409
    # Nobody else's capture, and not without the key.
    assert client.post(url, json={}).status_code == 401
    other = client.post(
        "/api/v1/captures",
        json={"name": "desktop", "kind": "video"},
        headers={"Authorization": f"Bearer {WRITE}"},
    ).json()
    refused = client.post(f"/api/v1/phone/captures/{other['id']}/refine", json={}, headers=PHONE)
    assert refused.status_code == 401
    # A capture never processed.
    fresh = client.post("/api/v1/phone/captures", json={}, headers=PHONE).json()["capture"]
    idle = client.post(f"/api/v1/phone/captures/{fresh['id']}/refine", json={}, headers=PHONE)
    assert idle.status_code == 409


def test_a_capture_reads_back_its_verdict(client: TestClient, db: Session) -> None:
    capture_id = _process(client, db)
    job = _finish(db, capture_id)

    quality = client.get(f"/api/v1/captures/{capture_id}").json()["quality"]

    assert quality == {
        "jobId": str(job.id),
        "mode": "preview",
        "bar": "balanced",
        "barApplied": "balanced",
        "keepPct": 62.0,
        "contextPct": 90.5,
        "heldOutPsnr": 23.04,
        "gaussians": {
            "total": 200_000,
            "kept": 180_000,
            "keep": 120_000,
            "context": 60_000,
            "drop": 20_000,
        },
        "roi": {"center": [0.1, -0.2, 3.5], "radius": 0.8},
        "tips": [{"id": "from-above", "text": "Add frames from above."}],
        "gsdMm": None,
        "medianViews": 31,
        "coverageUrl": "https://cdn.example.com/c.ply",
    }


# --- what a finished run leaves behind -------------------------------------------------


def test_a_summary_that_does_not_parse_costs_the_forecast_not_the_run() -> None:
    job_id = uuid.uuid4()
    assert registration.capture_quality(None, job_id, None) is None
    broken = {**SUMMARY, "roi": {"center": [1, 2], "radius": -1}}
    assert registration.capture_quality(broken, job_id, None) is None
    odd = registration.capture_quality({"mode": "draft", "bar": "lenient"}, job_id, None)
    assert odd is not None
    assert (odd.mode, odd.bar, odd.roi, odd.tips) == ("refine", "everything", None, [])


def test_registration_publishes_the_coverage_and_records_the_verdict(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = Capture(slug="spool-table", name="Spool table", kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.flush()
    job = Job(capture_id=capture.id, recipe="photo-reconstruct", recipe_version="8", params={})
    db.add(job)
    db.commit()
    storage.put_object(f"runs/{job.id}/package/splat/tileset.json", b"{}", "application/json")
    storage.put_object(
        f"runs/{job.id}/place/coverage_enu.ply", b"ply\n", "application/octet-stream"
    )

    def register(document: dict[str, Any], job_id: uuid.UUID) -> None:
        path = tmp_path / f"{job_id}.json"
        path.write_text(json.dumps(document))
        registration.register(
            db,
            storage,
            capture=capture,
            job_id=job_id,
            registration=registration.Registration.read(path),
            tiles_stage_id="package",
            coverage_stage_id="place",
        )

    georef = {"lat": 51.5, "lon": -0.12, "height": 10.0, "georefMethod": "manual"}
    register({"georef": georef, "quality": SUMMARY, "coverage": "coverage_enu.ply"}, job.id)

    db.refresh(capture)
    assert capture.quality is not None
    assert capture.quality["mode"] == "preview"
    assert capture.quality["roi"] == {"center": [0.1, -0.2, 3.5], "radius": 0.8}
    url = f"https://cdn.example.com/twin-worker-test/runs/{job.id}/place/coverage_enu.ply"
    assert capture.quality["coverageUrl"] == url
    site = db.get(Site, capture.site_id)
    assert site is not None and site.metadata_["coverageUrl"] == url

    # A later run with no quality stage clears both: a verdict and an overlay describe
    # the run that measured them, and that is no longer the one on the globe.
    register({"georef": georef}, job.id)
    db.refresh(capture)
    db.refresh(site)
    assert capture.quality is None
    assert "coverageUrl" not in site.metadata_


# --- the worker's guard ----------------------------------------------------------------


def test_a_region_is_only_passed_on_over_the_poses_it_was_measured_in() -> None:
    plan = plan_recipe(load_recipe("photo-reconstruct"))
    resolved = {
        "train": {"cap_max": 500_000, "roi": {"center": [0, 0, 0], "radius": 1}},
        "quality": {"mode": "refine"},
    }

    kept, dropped = without_stale_roi(resolved, plan, {"normalize", "pose", "mask"})
    assert kept == resolved and dropped == []

    fresh, dropped = without_stale_roi(resolved, plan, set())
    assert dropped == ["train"]
    assert fresh == {"train": {"cap_max": 500_000}, "quality": {"mode": "refine"}}
    assert resolved["train"]["roi"]  # the caller's copy is untouched
