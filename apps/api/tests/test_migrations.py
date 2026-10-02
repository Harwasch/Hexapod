from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alembic.config import Config
from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from alembic import command
from app.models import Capture, Job, JobStep
from app.models.enums import CaptureKind, RunStatus


def test_migrations_round_trip(alembic_config: Config, engine: Engine) -> None:
    command.downgrade(alembic_config, "base")
    assert not inspect(engine).has_table("sites")
    command.upgrade(alembic_config, "head")
    inspector = inspect(engine)
    for table in (
        "sites",
        "assets",
        "layers",
        "camera_bookmarks",
        "plans",
        "plan_revisions",
        "captures",
        "capture_files",
        "jobs",
        "job_steps",
        "artifacts",
    ):
        assert inspector.has_table(table)
    index_names = {index["name"] for index in inspector.get_indexes("sites")}
    assert "idx_sites_boundary" in index_names


def test_models_match_migrations(alembic_config: Config, engine: Engine) -> None:
    command.check(alembic_config)


def test_0008_resolves_duplicate_active_runs_before_it_forbids_them(
    alembic_config: Config, engine: Engine, db: Session
) -> None:
    """A database that already lost the double-click race holds two active runs of one
    capture, and a unique index cannot be built over that. The upgrade keeps the run a
    worker is on -- else the oldest queued -- and cancels the rest, saying why."""
    command.downgrade(alembic_config, "0007")
    try:
        now = datetime.now(tz=UTC)
        with Session(engine) as session:
            capture = Capture(slug="raced", name="Raced", kind=CaptureKind.VIDEO)
            calm = Capture(slug="calm", name="Calm", kind=CaptureKind.VIDEO)
            session.add_all([capture, calm])
            session.flush()
            queued = Job(
                capture_id=capture.id,
                recipe="splat-ingest",
                recipe_version="1",
                created_at=now - timedelta(minutes=5),
            )
            running = Job(
                capture_id=capture.id,
                recipe="splat-ingest",
                recipe_version="1",
                status=RunStatus.IN_PROGRESS,
                created_at=now,
            )
            finished = Job(
                capture_id=capture.id,
                recipe="splat-ingest",
                recipe_version="1",
                status=RunStatus.COMPLETE,
            )
            alone = Job(capture_id=calm.id, recipe="splat-ingest", recipe_version="1")
            session.add_all([queued, running, finished, alone])
            session.flush()
            session.add(JobStep(job_id=queued.id, stage_id="normalize", ordinal=0, impl="x"))
            session.commit()
            ids = {
                name: job.id
                for name, job in {
                    "queued": queued,
                    "running": running,
                    "finished": finished,
                    "alone": alone,
                }.items()
            }
    finally:
        command.upgrade(alembic_config, "head")

    status = {job.id: job for job in db.scalars(select(Job).where(Job.id.in_(ids.values())))}
    assert status[ids["running"]].status is RunStatus.IN_PROGRESS
    assert status[ids["queued"]].status is RunStatus.CANCELLED
    assert "migration 0008" in (status[ids["queued"]].error or "")
    assert status[ids["finished"]].status is RunStatus.COMPLETE
    assert status[ids["alone"]].status is RunStatus.NOT_STARTED
    step = db.scalar(select(JobStep).where(JobStep.job_id == ids["queued"]))
    assert step is not None and step.status is RunStatus.CANCELLED
    indexes = {index["name"] for index in inspect(engine).get_indexes("jobs")}
    assert "uq_jobs_one_active_per_capture" in indexes
