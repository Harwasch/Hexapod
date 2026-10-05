from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alembic.config import Config
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from alembic import command
from app.models import Asset, Capture, Job, JobStep
from app.models.enums import AssetProvider, CaptureKind, Representation, RunStatus, ScaleSource


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


def test_0008_keeps_the_run_a_worker_holds_not_the_oldest_in_progress_row(
    alembic_config: Config, engine: Engine, db: Session
) -> None:
    """In progress is not the same as running. An in-progress row whose worker died long
    ago -- created first, lease lapsed -- used to outrank the duplicate a worker was
    actually running, which the upgrade then cancelled. The live lease wins; of rows
    nobody holds, the newest claimed; of queued ones, the oldest."""
    command.downgrade(alembic_config, "0007")
    try:
        now = datetime.now(tz=UTC)
        with Session(engine) as session:
            held, lapsed, queued = (
                Capture(slug=slug, name=slug, kind=CaptureKind.VIDEO)
                for slug in ("held", "lapsed", "queued")
            )
            session.add_all([held, lapsed, queued])
            session.flush()

            def job(capture: Capture, created_min: int, **fields: object) -> Job:
                row = Job(
                    capture_id=capture.id,
                    recipe="splat-ingest",
                    recipe_version="1",
                    created_at=now - timedelta(minutes=created_min),
                    **fields,
                )
                session.add(row)
                return row

            running = RunStatus.IN_PROGRESS
            jobs = {
                # A worker died on this two hours ago; another is running the one below.
                "dead": job(
                    held,
                    120,
                    status=running,
                    claimed_at=now - timedelta(minutes=120),
                    lease_expires_at=now - timedelta(minutes=110),
                ),
                "live": job(
                    held,
                    1,
                    status=running,
                    claimed_at=now - timedelta(minutes=1),
                    lease_expires_at=now + timedelta(minutes=5),
                ),
                "waiting": job(held, 180),
                # Nobody holds either: the one claimed most recently is the one kept,
                # whichever was created first.
                "claimed-early": job(
                    lapsed,
                    30,
                    status=running,
                    claimed_at=now - timedelta(minutes=29),
                    lease_expires_at=now - timedelta(minutes=28),
                ),
                "claimed-late": job(
                    lapsed,
                    60,
                    status=running,
                    claimed_at=now - timedelta(minutes=5),
                    lease_expires_at=now - timedelta(minutes=4),
                ),
                "first-in-line": job(queued, 10),
                "second-in-line": job(queued, 5),
            }
            session.commit()
            ids = {name: row.id for name, row in jobs.items()}
    finally:
        command.upgrade(alembic_config, "head")

    status = {
        name: db.scalar(select(Job.status).where(Job.id == job_id)) for name, job_id in ids.items()
    }
    assert status == {
        "dead": RunStatus.CANCELLED,
        "live": RunStatus.IN_PROGRESS,
        "waiting": RunStatus.CANCELLED,
        "claimed-early": RunStatus.CANCELLED,
        "claimed-late": RunStatus.IN_PROGRESS,
        "first-in-line": RunStatus.NOT_STARTED,
        "second-in-line": RunStatus.CANCELLED,
    }


def test_0010_folds_an_estimated_scale_into_unresolved_on_the_way_down(
    alembic_config: Config, engine: Engine, db: Session
) -> None:
    """`camera-height-estimate` is a value 0009's enum cannot hold, and a value 0009's
    `Provenance` refuses to read. Down, a capture and its asset's provenance go back to
    what they were registered as before the estimate existed; up, the value is back."""
    with Session(engine) as session:
        capture = Capture(
            slug="estimated",
            name="Estimated",
            kind=CaptureKind.VIDEO,
            scale_source=ScaleSource.CAMERA_HEIGHT_ESTIMATE,
        )
        asset = Asset(
            name="Estimated splat",
            representation=Representation.GAUSSIAN_SPLAT,
            provider=AssetProvider.TILES_3D_URL,
            source={"type": "3d-tiles-url", "url": "https://cdn.example.com/t/tileset.json"},
            render_config={
                "provenance": {
                    "georefMethod": "exif-gps",
                    "scaleSource": "camera-height-estimate",
                    "uncertaintyM": 10.0,
                    "scaleUncertaintyPct": 22.4,
                }
            },
            attribution=[],
            default_visible=True,
        )
        session.add_all([capture, asset])
        session.commit()
        capture_id, asset_id = capture.id, asset.id

    command.downgrade(alembic_config, "0009")
    try:
        with engine.connect() as connection:
            source = connection.execute(
                text("SELECT scale_source::text FROM captures WHERE id = :id"), {"id": capture_id}
            ).scalar_one()
            provenance = connection.execute(
                text("SELECT render_config -> 'provenance' FROM assets WHERE id = :id"),
                {"id": asset_id},
            ).scalar_one()
            labels = connection.execute(
                text("SELECT unnest(enum_range(NULL::scale_source))::text")
            ).scalars()
            assert source == "unresolved"
            assert provenance == {
                "georefMethod": "exif-gps",
                "scaleSource": "unresolved",
                "uncertaintyM": 10.0,
            }
            assert list(labels) == ["arkit", "exif-gps", "manual", "unresolved"]
    finally:
        command.upgrade(alembic_config, "head")

    with engine.connect() as connection:
        labels = connection.execute(
            text("SELECT unnest(enum_range(NULL::scale_source))::text")
        ).scalars()
        assert list(labels) == [source.value for source in ScaleSource]
