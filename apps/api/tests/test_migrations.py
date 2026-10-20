from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, inspect, select, text
from sqlalchemy.orm import Session

from alembic import command
from app.models import Asset, Capture, Job, JobStep
from app.models.enums import AssetProvider, CaptureKind, Representation, RunStatus, ScaleSource
from app.storage import S3Storage


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
        "site_frames",
        "epochs",
    ):
        assert inspector.has_table(table)
    index_names = {index["name"] for index in inspector.get_indexes("sites")}
    assert {"idx_sites_boundary", "ix_sites_boundary_geography"} <= index_names
    assert "ix_assets_epoch_id" in {index["name"] for index in inspector.get_indexes("assets")}
    epoch_indexes = {index["name"] for index in inspector.get_indexes("epochs")}
    assert {"idx_epochs_footprint", "ix_epochs_site_id"} <= epoch_indexes


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


def test_0031_links_missions_that_land_actions_already_scheduled(
    alembic_config: Config, engine: Engine, db: Session, client: TestClient
) -> None:
    """Missions scheduled from land actions before 0031 hang off their land afterwards: the
    upgrade reads the land and boundary revision from the action revision that scheduled them."""
    from tests.test_workspace_missions import schedule_action

    land, _action, mission = schedule_action(client)
    # End the client's read transaction before DDL runs on another connection.
    db.rollback()
    command.downgrade(alembic_config, "0031-1")
    try:
        columns = {column["name"] for column in inspect(engine).get_columns("plans")}
        assert "land_id" not in columns and "boundary_revision" not in columns
    finally:
        command.upgrade(alembic_config, "head")
    row = db.execute(
        text("SELECT land_id, boundary_revision FROM plans WHERE id = :id"), {"id": mission["id"]}
    ).one()
    assert str(row.land_id) == land["id"] and row.boundary_revision == 1
    checks = {check["name"] for check in inspect(engine).get_check_constraints("plans")}
    assert "ck_plans_land_private" in checks


_SQUARE_WKT = (
    "MULTIPOLYGON(((-122.31 47.59, -122.29 47.59, -122.29 47.61, -122.31 47.61, -122.31 47.59)))"
)


def _raw_site(connection: Connection, slug: str, metadata: dict[str, object]) -> str:
    return str(
        connection.execute(
            text(
                "INSERT INTO sites (id, slug, name, boundary, centroid, metadata, attribution) "
                "VALUES (gen_random_uuid(), :slug, :slug, ST_GeomFromText(:wkt, 4326), "
                "ST_Centroid(ST_GeomFromText(:wkt, 4326)), CAST(:metadata AS jsonb), '[]') "
                "RETURNING id"
            ),
            {"slug": slug, "wkt": _SQUARE_WKT, "metadata": json.dumps(metadata)},
        ).scalar_one()
    )


def _raw_capture(
    connection: Connection,
    slug: str,
    site_id: str | None,
    *,
    captured_at: datetime | None,
    metadata: dict[str, object] | None = None,
    quality: dict[str, object] | None = None,
    capture_id: str | None = None,
) -> str:
    return str(
        connection.execute(
            text(
                "INSERT INTO captures (id, slug, name, status, kind, captured_at, metadata, "
                "attribution, site_id, quality) VALUES (COALESCE(CAST(:id AS uuid), "
                "gen_random_uuid()), :slug, :slug, 'complete', 'gaussian-splat', :captured_at, "
                "CAST(:metadata AS jsonb), '[]', CAST(:site AS uuid), CAST(:quality AS jsonb)) "
                "RETURNING id"
            ),
            {
                "id": capture_id,
                "slug": slug,
                "captured_at": captured_at,
                "metadata": json.dumps(metadata or {}),
                "site": site_id,
                "quality": None if quality is None else json.dumps(quality),
            },
        ).scalar_one()
    )


def _raw_job(connection: Connection, capture_id: str, *, finished: datetime) -> str:
    return str(
        connection.execute(
            text(
                "INSERT INTO jobs (id, capture_id, recipe, recipe_version, status, params, "
                "finished_at) VALUES (gen_random_uuid(), CAST(:capture AS uuid), 'splat-ingest', "
                "'3', 'complete', '{}', :finished) RETURNING id"
            ),
            {"capture": capture_id, "finished": finished},
        ).scalar_one()
    )


def _raw_splat(connection: Connection, site_id: str, url: str, created: datetime) -> str:
    return str(
        connection.execute(
            text(
                "INSERT INTO assets (id, site_id, name, representation, provider, source, "
                "attribution, render_config, default_visible, created_at, updated_at) VALUES "
                "(gen_random_uuid(), CAST(:site AS uuid), 'splat', 'gaussian-splat', "
                "'3d-tiles-url', CAST(:source AS jsonb), '[]', '{}', true, :created, :created) "
                "RETURNING id"
            ),
            {
                "site": site_id,
                "source": json.dumps({"type": "3d-tiles-url", "url": url}),
                "created": created,
            },
        ).scalar_one()
    )


def test_0032_backfills_a_frame_and_an_epoch_for_every_registered_site(
    alembic_config: Config, engine: Engine, db: Session
) -> None:
    """Every site the pipeline registered becomes epoch 1 of itself, in a frame made from
    the georef it was registered with; nothing else gains an epoch (decision D3)."""
    db.rollback()
    command.downgrade(alembic_config, "0031")
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    quality: dict[str, object] = {"verdict": "ok", "coverageUrl": "https://cdn.example.com/c.ply"}
    try:
        with engine.begin() as connection:
            # (A) a phone-key capture, no capture date, one complete run, two splats.
            a_capture = str(uuid.uuid4())
            a_job_placeholder = str(uuid.uuid4())
            a_site = _raw_site(
                connection,
                "site-a",
                {
                    "captureId": a_capture,
                    "jobId": a_job_placeholder,
                    "registration": {"georef": {"lat": 47.6, "lon": -122.3, "height": 30}},
                },
            )
            _raw_capture(
                connection,
                "cap-a",
                a_site,
                captured_at=None,
                metadata={"origin": "phone-key"},
                quality=quality,
                capture_id=a_capture,
            )
            a_job = _raw_job(connection, a_capture, finished=now)
            connection.execute(
                text(
                    "UPDATE sites SET metadata = jsonb_set(metadata, '{jobId}', to_jsonb(CAST(:j AS text)))"
                    " WHERE id = CAST(:s AS uuid)"
                ),
                {"j": a_job, "s": a_site},
            )
            a_first = _raw_splat(connection, a_site, "https://cdn.example.com/a1/tileset.json", now)
            a_second = _raw_splat(
                connection,
                a_site,
                "https://cdn.example.com/a2/tileset.json",
                now + timedelta(seconds=1),
            )
            # (B) a jobId naming no job, a capture date, two complete runs.
            b_capture = str(uuid.uuid4())
            b_site = _raw_site(
                connection,
                "site-b",
                {
                    "captureId": b_capture,
                    "jobId": str(uuid.uuid4()),
                    "registration": {"georef": {"lat": -33.9, "lon": 151.2}},
                },
            )
            _raw_capture(
                connection,
                "cap-b",
                b_site,
                captured_at=now - timedelta(days=3),
                capture_id=b_capture,
            )
            _raw_job(connection, b_capture, finished=now - timedelta(hours=2))
            b_newest = _raw_job(connection, b_capture, finished=now - timedelta(hours=1))
            # B's splat had a validity window somebody set (PATCH allowed it on any asset
            # before 0032). On an epoch asset `valid_to` means superseded, so it stays legacy.
            b_windowed = _raw_splat(
                connection, b_site, "https://cdn.example.com/b/tileset.json", now
            )
            window_end = now + timedelta(days=30)
            connection.execute(
                text("UPDATE assets SET valid_to = :t WHERE id = CAST(:a AS uuid)"),
                {"t": window_end, "a": b_windowed},
            )
            # (C) a seeded demo-ranch site, (D) captureId but no registration, (E) a
            # captureId that is not a uuid.
            c_site = _raw_site(connection, "site-c", {"origin": "demo-ranch"})
            _raw_splat(connection, c_site, "https://cdn.example.com/c/tileset.json", now)
            d_capture = str(uuid.uuid4())
            d_site = _raw_site(connection, "site-d", {"captureId": d_capture})
            _raw_capture(connection, "cap-d", d_site, captured_at=now, capture_id=d_capture)
            e_site = _raw_site(
                connection,
                "site-e",
                {
                    "captureId": "not-a-uuid",
                    "registration": {"georef": {"lat": 1.0, "lon": 2.0}},
                },
            )
            a_created = connection.execute(
                text("SELECT created_at FROM captures WHERE id = CAST(:c AS uuid)"),
                {"c": a_capture},
            ).scalar_one()
    finally:
        command.upgrade(alembic_config, "head")

    with engine.connect() as connection:
        frames = {
            str(row.site_id): row
            for row in connection.execute(text("SELECT * FROM site_frames")).all()
        }
        epochs = {
            str(row.site_id): row for row in connection.execute(text("SELECT * FROM epochs")).all()
        }
        assets = {
            str(row.id): row
            for row in connection.execute(
                text("SELECT id, epoch_id, observed_at, valid_to, source_kind FROM assets")
            ).all()
        }
        same_footprint = connection.execute(
            text(
                "SELECT ST_Equals(e.footprint, s.boundary) FROM epochs e JOIN sites s "
                "ON s.id = e.site_id WHERE s.id = CAST(:s AS uuid)"
            ),
            {"s": a_site},
        ).scalar_one()

    assert set(frames) == set(epochs) == {a_site, b_site}
    frame = frames[a_site]
    assert (frame.lat, frame.lon, frame.height) == (47.6, -122.3, 30.0)
    assert (frame.height_datum, frame.epsg, frame.source) == ("msl-unverified", 32610, "backfill")
    epoch = epochs[a_site]
    assert epoch.ordinal == 1
    assert epoch.observed_at == a_created
    assert epoch.observed_at_source == "created"
    assert (epoch.platform, epoch.join_method, epoch.status) == (
        "phone-video",
        "new-site",
        "published",
    )
    assert str(epoch.job_id) == a_job
    assert str(epoch.capture_id) == a_capture
    assert epoch.quality == quality
    assert epoch.alignment is None
    assert same_footprint is True
    # Only the first splat by creation is the epoch's, and it takes the epoch's date.
    assert assets[a_first].epoch_id == epoch.id
    assert assets[a_first].observed_at == a_created
    assert assets[a_second].epoch_id is None and assets[a_second].observed_at is None
    assert {row.source_kind for row in assets.values()} == {"observed"}

    b_epoch = epochs[b_site]
    assert b_epoch.observed_at_source == "captured-at"
    assert b_epoch.observed_at == now - timedelta(days=3)
    assert str(b_epoch.job_id) == b_newest
    assert b_epoch.platform == "unknown"
    # Its window is kept as it was, and the asset is not made the epoch's: linked, its
    # `valid_to` would read as superseded and the splat would vanish from `SiteRead`.
    assert assets[b_windowed].epoch_id is None
    assert assets[b_windowed].valid_to == window_end
    assert (frames[b_site].height, frames[b_site].epsg) == (0.0, 32756)
    for legacy in (c_site, d_site, e_site):
        assert legacy not in frames and legacy not in epochs


def test_the_epoch_vocabularies_match_their_check_constraints(engine: Engine) -> None:
    """The enums the API reads and writes are exactly what the database accepts."""
    from app.models.enums import (
        AssetSourceKind,
        EpochJoinMethod,
        EpochPlatform,
        EpochStatus,
        HeightDatum,
        ObservedAtSource,
        SiteFrameSource,
    )

    expected = {
        "ck_site_frames_height_datum": HeightDatum,
        "ck_site_frames_source": SiteFrameSource,
        "ck_epochs_status": EpochStatus,
        "ck_epochs_join_method": EpochJoinMethod,
        "ck_epochs_platform": EpochPlatform,
        "ck_epochs_observed_at_source": ObservedAtSource,
        "ck_assets_source_kind": AssetSourceKind,
    }
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = ANY(:names)"
            ),
            {"names": list(expected)},
        ).all()
    definitions: dict[str, str] = {str(row[0]): str(row[1]) for row in rows}
    assert set(definitions) == set(expected)
    for name, vocabulary in expected.items():
        quoted = set(re.findall(r"'([^']*)'", definitions[name]))
        assert quoted == {member.value for member in vocabulary}, name


def test_0032_downgrade_keeps_every_asset_and_its_source(
    alembic_config: Config, engine: Engine, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Two epochs, one with two revisions, then down to 0031: every asset row and every
    `source` survives; only the new tables and columns go."""
    from tests.test_registration_epochs import _capture, _echoed, run

    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    second = _capture(db, "second")
    run(db, storage, tmp_path, second, _echoed(first.site_id))
    db.commit()
    with engine.connect() as connection:
        before = {
            str(row.id): row.source
            for row in connection.execute(text("SELECT id, source FROM assets")).all()
        }
    assert len(before) == 3
    db.rollback()
    db.close()

    command.downgrade(alembic_config, "0031")
    try:
        inspector = inspect(engine)
        assert not inspector.has_table("epochs") and not inspector.has_table("site_frames")
        columns = {column["name"] for column in inspector.get_columns("assets")}
        assert "epoch_id" not in columns and "source_kind" not in columns
        with engine.connect() as connection:
            after = {
                str(row.id): row.source
                for row in connection.execute(text("SELECT id, source FROM assets")).all()
            }
        assert after == before
    finally:
        command.upgrade(alembic_config, "head")


def test_0032_phone_origin_is_the_routes() -> None:
    """The backfill spells the phone origin as a literal (migrations import no app code);
    it must stay the constant the phone route marks its captures with."""
    from app.api.v1.phone import ORIGIN

    path = Path(__file__).parents[1] / "alembic" / "versions" / "0032_site_frames_and_epochs.py"
    assert f"c.metadata ->> 'origin' = '{ORIGIN}'" in path.read_text(encoding="utf-8")


def test_0032_re_runs_after_an_index_build_fails(
    alembic_config: Config, engine: Engine, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0032's tables, columns and backfill commit in one transaction before its two
    indexes are built CONCURRENTLY, outside it. If a build then fails, 0032 is never
    recorded: the re-run must skip what already committed (not die at CREATE TABLE, not
    backfill twice), and replace the INVALID index a failed concurrent build leaves."""
    from alembic import op

    db.rollback()
    command.downgrade(alembic_config, "0031")
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    with engine.begin() as connection:
        capture = str(uuid.uuid4())
        site = _raw_site(
            connection,
            "site-r",
            {"captureId": capture, "registration": {"georef": {"lat": 47.6, "lon": -122.3}}},
        )
        _raw_capture(connection, "cap-r", site, captured_at=now, capture_id=capture)
        _raw_splat(connection, site, "https://cdn.example.com/r1/tileset.json", now)
        _raw_splat(connection, site, "https://cdn.example.com/r2/tileset.json", now)
    real = op.execute

    def failing(sql: Any, *args: Any, **kwargs: Any) -> Any:
        if "CREATE INDEX CONCURRENTLY" in str(sql) and "ix_sites_boundary_geography" in str(sql):
            raise RuntimeError("simulated: the index build failed")
        return real(sql, *args, **kwargs)

    monkeypatch.setattr(op, "execute", failing)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            command.upgrade(alembic_config, "0032")
    finally:
        monkeypatch.undo()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            "0031"
        )
        assert inspect(engine).has_table("epochs")
    # A concurrent build that fails leaves its index behind, INVALID: make one the way a
    # build of `ix_assets_epoch_id` could (a unique build over duplicate rows fails).
    with engine.connect() as connection:
        connection.execute(text("DROP INDEX IF EXISTS ix_assets_epoch_id"))
        connection.commit()
        autocommit = connection.execution_options(isolation_level="AUTOCOMMIT")
        with pytest.raises(Exception, match="could not create unique index"):
            autocommit.execute(
                text("CREATE UNIQUE INDEX CONCURRENTLY ix_assets_epoch_id ON assets (site_id)")
            )

    command.upgrade(alembic_config, "head")

    with engine.connect() as connection:
        epochs = connection.execute(
            text("SELECT count(*) FROM epochs WHERE site_id = CAST(:s AS uuid)"), {"s": site}
        ).scalar()
        assert epochs == 1
        indexes = {
            row.name: row
            for row in connection.execute(
                text(
                    "SELECT c.relname AS name, i.indisvalid AS valid, i.indisunique AS uniq, "
                    "pg_get_indexdef(i.indexrelid) AS definition "
                    "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname IN ('ix_assets_epoch_id', 'ix_sites_boundary_geography')"
                )
            ).all()
        }
    assert set(indexes) == {"ix_assets_epoch_id", "ix_sites_boundary_geography"}
    assert all(row.valid for row in indexes.values())
    assert not indexes["ix_assets_epoch_id"].uniq
    assert "(epoch_id)" in indexes["ix_assets_epoch_id"].definition


def test_0033_downgrade_folds_visual_epoch(
    alembic_config: Config, engine: Engine, db: Session
) -> None:
    """`visual-epoch` is a value 0032's enums cannot hold and 0032's `Provenance` refuses.
    Down, a visually aligned capture and its asset's provenance read as placed by no
    method of their own with an unresolved scale (their uncertainty kept); up, the value
    is back in both types, before each type's "nothing" member, as the enums declare."""
    from app.models.enums import GeorefMethod

    with Session(engine) as session:
        capture = Capture(
            slug="aligned",
            name="Aligned",
            kind=CaptureKind.VIDEO,
            georef_method=GeorefMethod.VISUAL_EPOCH,
            scale_source=ScaleSource.VISUAL_EPOCH,
        )
        asset = Asset(
            name="Aligned splat",
            representation=Representation.GAUSSIAN_SPLAT,
            provider=AssetProvider.TILES_3D_URL,
            source={"type": "3d-tiles-url", "url": "https://cdn.example.com/a/tileset.json"},
            render_config={
                "provenance": {
                    "georefMethod": "visual-epoch",
                    "scaleSource": "visual-epoch",
                    "uncertaintyM": 0.065,
                }
            },
            attribution=[],
            default_visible=True,
        )
        session.add_all([capture, asset])
        session.commit()
        capture_id, asset_id = capture.id, asset.id

    def labels(connection: Connection, type_name: str) -> list[str]:
        rows = connection.execute(text(f"SELECT unnest(enum_range(NULL::{type_name}))::text"))
        return [str(value) for value in rows.scalars()]

    command.downgrade(alembic_config, "0032")
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT georef_method::text, scale_source::text FROM captures WHERE id = :id"),
                {"id": capture_id},
            ).one()
            provenance = connection.execute(
                text("SELECT render_config -> 'provenance' FROM assets WHERE id = :id"),
                {"id": asset_id},
            ).scalar_one()
            assert tuple(row) == ("none", "unresolved")
            assert provenance == {
                "georefMethod": "none",
                "scaleSource": "unresolved",
                "uncertaintyM": 0.065,
            }
            assert labels(connection, "georef_method") == ["exif-gps", "arkit", "manual", "none"]
            assert labels(connection, "scale_source") == [
                "arkit",
                "exif-gps",
                "manual",
                "camera-height-estimate",
                "unresolved",
            ]
    finally:
        command.upgrade(alembic_config, "head")

    with engine.connect() as connection:
        assert labels(connection, "georef_method") == [m.value for m in GeorefMethod]
        assert labels(connection, "scale_source") == [s.value for s in ScaleSource]
