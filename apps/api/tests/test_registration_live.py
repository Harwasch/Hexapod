"""The live version of `test_worker_registration`: the real pipeline aligns epoch 2.

Opt-in, because it needs pycolmap 4.2, which this venv does not have: set
`COLMAP_GLOBAL_PYTHON` to an interpreter that has it (`tools/pipeline/.venv/bin/python`
after `uv pip install pycolmap==4.2.0`); CI's api job does not, so it skips there. About
25 s on 16 cores.

Three epochs of the synthetic tree (`tools/pipeline/tests/registration_scene.py`): an RTK
drone survey (the site's first, scored against eight check targets), a second RTK flight,
and an iPhone-style video. Each run's `site_context` is seeded by the worker's seeder from
this database and bucket -- epochs 2 and 3 get epoch 1's frames, model, placement and
georef as their reference -- then the shipped stages normalize, pose, georeference,
align, place, package, manifest and catalog run on this machine (training, which needs a
GPU, replaced by the pose model's sparse points), later epochs given the site frame as
the worker gives it, and the worker's `register` writes each epoch. Asserted: the D4
words, the tiers, and that each later epoch names epoch 1 -- no figure measured here.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Artifact, Capture, CaptureFile, Job, JobStep
from app.models.enums import CaptureKind, GeorefMethod, RunStatus, ScaleSource, UploadStatus
from app.schemas.check_target import CheckTargetSet
from app.schemas.site import SiteCreate
from app.services import check_targets, epochs
from app.services import sites as site_service
from app.storage import S3Storage
from app.worker import registration, site_context
from app.worker.carry import CarryPlan
from app.worker.outputs import artifact_key, kind_for
from app.worker.pipeline_bridge import PIPELINE_DIR, Workdir


def _pycolmap() -> bool:
    interpreter = os.environ.get("COLMAP_GLOBAL_PYTHON")
    if not interpreter:
        return False
    probe = [interpreter, "-c", "import pycolmap; assert pycolmap.__version__ == '4.2.0'"]
    return subprocess.run(probe, capture_output=True, check=False).returncode == 0


pytestmark = pytest.mark.skipif(
    not _pycolmap(), reason="needs COLMAP_GLOBAL_PYTHON with pycolmap 4.2.0 (not in CI's api job)"
)


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PIPELINE_DIR / "tests"))
    # By name: the pipeline's test helpers are not on this project's type-checking path.
    return importlib.import_module("registration_scene")


def _stages(scene: Any, video: bool, frames: int, frame: dict[str, Any] | None) -> list[Any]:
    stages = [scene._shipped(s) for s in ("normalize", "pose", "georeference", "align")]
    stages[0]["params"].update({"select": "all", "keep": frames, "keep_video": frames})
    if video:
        stages[0]["params"]["fps"] = scene.VIDEO_FPS
    stages.append({"id": "train", "impl": "t_sparse_as_trained"})
    stages += [scene._shipped(s) for s in ("place", "package", "manifest", "register")]
    for stage in stages:
        if frame is not None and stage["impl"] in ("colmap", "exif_gps", "align_epoch"):
            stage["params"]["site_frame"] = frame  # as `worker/params.stage_params` does
    return stages


def _keep(db: Session, storage: S3Storage, job: Job, workdir: Workdir) -> None:
    """The reference artifacts, rowed and uploaded as `worker/outputs` leaves them."""
    named = (("normalize", "frames"), ("pose", "poses"), ("georeference", "georef.json"))
    for ordinal, (stage, name) in enumerate((*named, ("place", "placement.json"))):
        step = JobStep(job_id=job.id, stage_id=stage, ordinal=ordinal, impl=stage,
                       status=RunStatus.COMPLETE, metrics={})  # fmt: skip
        db.add(step)
        db.flush()
        key, source = artifact_key(job.id, stage, name), workdir.artifact_path(stage, name)
        members = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else []
        for member in members:
            relative = member.relative_to(source).as_posix()
            storage.upload_file(f"{key}/{relative}", member, "application/octet-stream")
        if not source.is_dir():
            storage.upload_file(key, source, "application/json")
        # Rowed before the register, as the worker uploads a stage's outputs as it ends.
        db.add(Artifact(job_step_id=step.id, kind=kind_for(name), storage_key=key,
                        created_at=job.created_at))  # fmt: skip
    db.commit()


def test_the_real_pipeline_aligns_epochs_2_to_epoch_1_and_the_api_shows_it(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient, scene: Any
) -> None:
    from executor import execute
    from runners import LocalRunner, RunnerSet

    lat, lon, _height = scene.ORIGIN
    d = 40.0 / 111_320.0
    ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d]]
    site = site_service.create_site(db, SiteCreate.model_validate({
        "name": "Synthetic tree", "slug": "synthetic-tree", "attribution": [], "assets": [],
        "boundary": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]},
    }))  # fmt: skip
    targets = [{**t, "source": "synthetic"} for t in scene.surveyed_targets()]
    check_targets.replace_active(db, site.id, CheckTargetSet.model_validate({"targets": targets}))

    def epoch(name: str, kind: CaptureKind, upload: Path, frames: int, clicks: Any) -> Capture:
        capture = Capture(slug=name.replace("_", "-"), name=name, kind=kind,
                          metadata_={"siteId": str(site.id)})  # fmt: skip
        db.add(capture)
        db.commit()
        if clicks:
            key = f"captures/{capture.id}/check_observations.json"
            body = json.dumps({"version": 1, "observations": clicks}).encode()
            storage.put_object(key, body, "application/json")
            db.add(CaptureFile(capture_id=capture.id, filename="check_observations.json",
                               storage_key=key, status=UploadStatus.COMPLETE))  # fmt: skip
            db.commit()
        frame = epochs.frame_for_run(db, capture)
        workdir = Workdir.create(tmp_path / name / "run")
        site_context.seed_site_context(db, storage, capture, workdir.input_path("site_context"))
        shutil.copytree(upload, workdir.input_path("upload"), dirs_exist_ok=True)
        recipe = scene.make_recipe(
            _stages(scene, kind is CaptureKind.VIDEO, frames, frame.param() if frame else None),
            name="photo-reconstruct", inputs=["upload", "site_context"],
        )  # fmt: skip
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
        job = Job(capture_id=capture.id, recipe="photo-reconstruct", recipe_version="13",
                  params={}, status=RunStatus.COMPLETE)  # fmt: skip
        db.add(job)
        db.commit()
        path = workdir.artifact_path("register", "registration.json")
        registration.register(
            db, storage, capture=capture, job_id=job.id,
            registration=registration.Registration.read(path), tiles_stage_id="package",
            published=registration.Published(
                tileset=f"https://cdn.example.com/t/runs/{job.id}/package/splat/tileset.json",
                thumbnail=None, coverage=None,
                carry=CarryPlan(based_on=registration.live_tileset_url(db, capture)),
            ),
        )  # fmt: skip
        if name == "e1":
            _keep(db, storage, job, workdir)
        db.refresh(capture)
        return capture

    upload, truth, originals = scene.render_drone(tmp_path, "e1", seed=1)
    clicks = scene.observations(truth, originals, seed=21)
    first = epoch("e1", CaptureKind.IMAGES, upload, len(truth.poses), clicks)
    upload, truth, originals = scene.render_drone(tmp_path, "e2_drone", seed=2)
    clicks = scene.observations(truth, originals, seed=22)
    drone = epoch("e2_drone", CaptureKind.IMAGES, upload, len(truth.poses), clicks)
    upload, truth, _rendered = scene.render_phone(tmp_path)
    phone = epoch("e2_phone", CaptureKind.VIDEO, upload, len(truth.poses), None)

    shown = client.get(f"/api/v1/sites/{site.id}/epochs").json()
    one = epochs.epoch_of_capture(db, first.id)
    assert one is not None
    assert [e["platform"] for e in shown] == ["rtk-drone", "rtk-drone", "phone-video"]
    words = [e["alignment"]["label"] for e in shown]
    assert words[0].startswith("RTK drone, surveyed, check targets ±"), words
    assert words[1].startswith("RTK drone, surveyed, check targets ±"), words
    assert words[2].startswith("Phone video, aligned visually, ±"), words
    assert [e["alignment"]["verdict"] for e in shown] == ["survey", "survey", "coarse"]
    assert [e["alignment"]["referenceEpochId"] for e in shown] == [None, str(one.id), str(one.id)]
    for e in shown:
        assert isinstance(e["alignment"]["uncertaintyM"], float)
        assert e["alignment"]["uncertaintyM"] > 0
    for capture in (drone, phone):
        # Placed by epoch 1, whose scale its GNSS pose priors measured.
        assert (capture.georef_method, capture.scale_source) == (
            GeorefMethod.VISUAL_EPOCH,
            ScaleSource.VISUAL_EPOCH,
        )
