"""captures, uploaded files, and the job spine

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-21
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

CAPTURE_KIND = ("video", "images", "gaussian-splat", "point-cloud")
CAPTURE_STATUS = ("awaiting-files", "not-started", "in-progress", "complete", "error")
UPLOAD_STATUS = ("not-started", "in-progress", "complete", "error", "aborted")
RUN_STATUS = ("not-started", "in-progress", "complete", "error", "cancelled")
GEOREF_METHOD = ("exif-gps", "arkit", "manual", "none")
SCALE_SOURCE = ("arkit", "exif-gps", "manual", "unresolved")
ARTIFACT_KIND = (
    "frames",
    "poses",
    "masks",
    "splat",
    "deformation-field",
    "mesh",
    "point-cloud",
    "3d-tiles",
    "thumbnail",
    "ground-samples",
    "manifest",
    "clip",
)

ENUM_NAMES = (
    "artifact_kind",
    "scale_source",
    "georef_method",
    "run_status",
    "upload_status",
    "capture_status",
    "capture_kind",
)


def upgrade() -> None:
    capture_kind = postgresql.ENUM(*CAPTURE_KIND, name="capture_kind", create_type=False)
    capture_status = postgresql.ENUM(*CAPTURE_STATUS, name="capture_status", create_type=False)
    upload_status = postgresql.ENUM(*UPLOAD_STATUS, name="upload_status", create_type=False)
    run_status = postgresql.ENUM(*RUN_STATUS, name="run_status", create_type=False)
    georef_method = postgresql.ENUM(*GEOREF_METHOD, name="georef_method", create_type=False)
    scale_source = postgresql.ENUM(*SCALE_SOURCE, name="scale_source", create_type=False)
    artifact_kind = postgresql.ENUM(*ARTIFACT_KIND, name="artifact_kind", create_type=False)
    bind = op.get_bind()
    for enum in (
        capture_kind,
        capture_status,
        upload_status,
        run_status,
        georef_method,
        scale_source,
        artifact_kind,
    ):
        enum.create(bind, checkfirst=True)

    op.create_table(
        "captures",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", capture_status, nullable=False),
        sa.Column("kind", capture_kind, nullable=False),
        sa.Column("device", sa.String(200), nullable=True),
        sa.Column("sensor", sa.String(120), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("temporal_extent", postgresql.JSONB(), nullable=True),
        sa.Column("georef_method", georef_method, nullable=True),
        sa.Column("scale_source", scale_source, nullable=True),
        sa.Column("uncertainty_m", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("attribution", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("license", postgresql.JSONB(), nullable=True),
        sa.Column("provenance", postgresql.JSONB(), nullable=True),
        sa.Column(
            "site_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sites.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_captures_slug", "captures", ["slug"], unique=True)
    op.create_index("ix_captures_site_id", "captures", ["site_id"])

    op.create_table(
        "capture_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "capture_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("captures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(200), nullable=True),
        sa.Column("bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum", sa.String(128), nullable=True),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("status", upload_status, nullable=False),
        sa.Column("upload_id", sa.String(255), nullable=True),
        sa.Column("parts_total", sa.Integer(), nullable=True),
        sa.Column("parts_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_capture_files_capture_id", "capture_files", ["capture_id"])
    op.create_index("ix_capture_files_storage_key", "capture_files", ["storage_key"], unique=True)

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "capture_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("captures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("recipe", sa.String(120), nullable=False),
        sa.Column("recipe_version", sa.String(40), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", run_status, nullable=False),
        sa.Column("provider", sa.String(40), nullable=True),
        sa.Column("tier", sa.String(40), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(12, 4), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("claimed_by", sa.String(200), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_jobs_capture_id", "jobs", ["capture_id"])
    op.create_index("ix_jobs_status_lease_expires_at", "jobs", ["status", "lease_expires_at"])

    op.create_table(
        "job_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_id", sa.String(120), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("impl", sa.String(120), nullable=False),
        sa.Column("status", run_status, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("log_key", sa.String(1024), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("checkpoint_key", sa.String(1024), nullable=True),
        sa.Column("preempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("job_id", "ordinal", name="uq_job_step_ordinal"),
    )

    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_step_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job_steps.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", artifact_kind, nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum", sa.String(128), nullable=True),
        sa.Column("content_type", sa.String(200), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_artifacts_job_step_id", "artifacts", ["job_step_id"])
    op.create_index("ix_artifacts_storage_key", "artifacts", ["storage_key"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_storage_key", table_name="artifacts")
    op.drop_index("ix_artifacts_job_step_id", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_table("job_steps")
    op.drop_index("ix_jobs_status_lease_expires_at", table_name="jobs")
    op.drop_index("ix_jobs_capture_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_capture_files_storage_key", table_name="capture_files")
    op.drop_index("ix_capture_files_capture_id", table_name="capture_files")
    op.drop_table("capture_files")
    op.drop_index("ix_captures_site_id", table_name="captures")
    op.drop_index("ix_captures_slug", table_name="captures")
    op.drop_table("captures")
    for name in ENUM_NAMES:
        op.execute(f"DROP TYPE IF EXISTS {name}")
