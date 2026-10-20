"""Epoch 2 is seeded with epoch 1 as its reference, registers, and the API says how.

The real supervisor, child, executor, artifact upload, seed and `register`, against
PostGIS and moto (`tests/recipes/t-align.yaml`). Only the reconstruction is a stand-in:
the alignment each run reports is **recorded** -- what the real `align_epoch` wrote for
the synthetic tree on pycolmap 4.2 (`tests/fixtures/alignment/`, from
`tools/pipeline/tests/registration_scene.py`'s e1, e2_drone and e2_phone, each run seeded
by this worker's own `site_context` seeder) -- and replayed only after the stand-in has
checked that the seeded context holds the reference's files and the site's targets.
CI has no pycolmap in this venv; the live version of this test is the PR's evidence.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, Job
from app.models.enums import CaptureKind, GeorefMethod, RunStatus, ScaleSource
from app.services import epochs
from app.storage import S3Storage
from app.worker.runner import JobSupervisor
from tests.test_worker import config, queue_job, steps_by_stage
from tests.test_worker_stops import claimed

RECORDED = Path(__file__).resolve().parent / "fixtures" / "alignment"
LAT, LON = 28.0389, -82.6966
D = 40.0 / 111_320.0
BOUNDARY = {
    "type": "Polygon",
    "coordinates": [
        [[LON - D, LAT - D], [LON + D, LAT - D], [LON + D, LAT + D], [LON - D, LAT + D],
         [LON - D, LAT - D]]
    ],
}  # fmt: skip
TARGET = {
    "role": "check",
    "lat": LAT,
    "lon": LON,
    "height": 10.8,
    "heightDatum": "msl-unverified",
    "sigmaHM": 0.005,
    "sigmaVM": 0.01,
    "source": "synthetic",
}


def recorded(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((RECORDED / f"{name}.json").read_text())
    return document


def _epoch_run(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    site_id: str,
    name: str,
    kind: CaptureKind,
) -> tuple[Capture, Job]:
    capture = Capture(
        slug=name.replace("_", "-"), name=name, kind=kind, metadata_={"siteId": site_id}
    )
    db.add(capture)
    db.commit()
    job = queue_job(db, capture, "t-align")
    job.params = {"align": {"recorded": str(RECORDED / f"{name}.json")}}
    db.commit()
    claimed(sessions, job)
    outcome = JobSupervisor(sessions, storage, config(tmp_path / name)).run(job.id)
    db.expire_all()
    done = db.get(Job, job.id)
    assert done is not None
    assert outcome == "complete", done.error
    db.refresh(capture)
    return capture, done


def test_epoch_2_is_aligned_to_epoch_1_and_the_api_shows_its_alignment(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    client: TestClient,
) -> None:
    from tests.conftest import site_payload

    site = client.post(
        "/api/v1/sites", json=site_payload(slug="synthetic-tree", boundary=BOUNDARY, assets=[])
    )
    assert site.status_code == 201, site.text
    site_id = site.json()["id"]
    targets = [{**TARGET, "label": f"T{i}"} for i in range(1, 4)]
    put = client.put(f"/api/v1/sites/{site_id}/check-targets", json={"targets": targets})
    assert put.status_code == 200, put.text

    first, one = _epoch_run(db, sessions, storage, tmp_path, site_id, "e1", CaptureKind.IMAGES)
    seen = steps_by_stage(db, one.id)["align"].metrics
    assert (seen["referenceEpochId"], seen["targets"]) == ("", 3)
    epoch_1 = epochs.epoch_of_capture(db, first.id)
    assert epoch_1 is not None and str(epoch_1.site_id) == site_id

    drone, two = _epoch_run(
        db, sessions, storage, tmp_path, site_id, "e2_drone", CaptureKind.IMAGES
    )
    phone, three = _epoch_run(
        db, sessions, storage, tmp_path, site_id, "e2_phone", CaptureKind.VIDEO
    )

    for job in (two, three):
        seen = steps_by_stage(db, job.id)["align"].metrics
        assert seen["referenceEpochId"] == str(epoch_1.id), "seeded with epoch 1"
        assert seen["referenceVerdict"] == "survey"
        # 3 frames, the 4 files of the model, placement.json and georef.json.
        assert (seen["referenceFiles"], seen["targets"]) == (9, 3)

    shown = client.get(f"/api/v1/sites/{site_id}/epochs").json()
    assert [e["ordinal"] for e in shown] == [1, 2, 3]
    assert [e["status"] for e in shown] == ["published"] * 3
    assert [e["platform"] for e in shown] == ["rtk-drone", "rtk-drone", "phone-video"]
    for epoch, name in zip(shown, ("e1", "e2_drone", "e2_phone"), strict=True):
        want = recorded(name)
        got = epoch["alignment"]
        assert {k: got[k] for k in ("verdict", "label", "uncertaintyM")} == {
            k: want[k] for k in ("verdict", "label", "uncertaintyM")
        }, name
    assert shown[0]["alignment"]["referenceEpochId"] is None
    assert shown[1]["alignment"]["referenceEpochId"] == str(epoch_1.id)
    assert shown[2]["alignment"]["referenceEpochId"] == str(epoch_1.id)
    # What the web shows, word for word (the D4 labels, written by align_epoch).
    assert shown[1]["alignment"]["label"].startswith("RTK drone, surveyed, check targets ±")
    assert shown[2]["alignment"]["label"].startswith("Phone video, aligned visually, ±")
    assert shown[2]["alignment"]["verdict"] == "coarse", "a phone is never surveyed"

    # Placed by epoch 1, whose own scale this stand-in run never measured (it was placed
    # by hand): the visual alignment carries that over as unresolved, not as metric.
    for capture, name in ((drone, "e2_drone"), (phone, "e2_phone")):
        assert capture.georef_method is GeorefMethod.VISUAL_EPOCH
        assert capture.scale_source is ScaleSource.UNRESOLVED
        assert capture.uncertainty_m == recorded(name)["uncertaintyM"]
    assert first.georef_method is GeorefMethod.MANUAL
    assert uuid.UUID(site_id) == first.site_id == drone.site_id == phone.site_id
    assert two.status is RunStatus.COMPLETE and three.status is RunStatus.COMPLETE
