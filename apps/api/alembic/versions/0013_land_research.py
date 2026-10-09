"""Durable research and evidence pinned to immutable land revisions.

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "land_investigations",
        sa.Column("land_id", sa.UUID(), nullable=False),
        sa.Column("boundary_revision", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["land_id", "boundary_revision"],
            ["land_boundary_revisions.land_id", "land_boundary_revisions.revision"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_land_investigations_land_id"), "land_investigations", ["land_id"], unique=False
    )
    op.create_table(
        "land_research_messages",
        sa.Column("investigation_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["investigation_id"], ["land_investigations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_land_research_messages_investigation_id"),
        "land_research_messages",
        ["investigation_id"],
        unique=False,
    )
    op.create_table(
        "land_research_runs",
        sa.Column("investigation_id", sa.UUID(), nullable=False),
        sa.Column("request_key", sa.UUID(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("budget", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("checkpoint", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("next_sequence", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.UUID(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'partial', 'failed', 'cancelled')"
        ),
        sa.ForeignKeyConstraint(
            ["investigation_id"], ["land_investigations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("investigation_id", "request_key"),
    )
    op.create_index(
        op.f("ix_land_research_runs_investigation_id"),
        "land_research_runs",
        ["investigation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_land_research_runs_status"), "land_research_runs", ["status"], unique=False
    )
    op.create_table(
        "land_evidence",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("source_key", sa.String(length=300), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["land_research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "source_key"),
    )
    op.create_index(op.f("ix_land_evidence_run_id"), "land_evidence", ["run_id"], unique=False)
    op.create_table(
        "land_findings",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("output_key", sa.String(length=200), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("disposition", sa.String(length=20), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["land_research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "output_key"),
    )
    op.create_index(op.f("ix_land_findings_run_id"), "land_findings", ["run_id"], unique=False)
    op.create_table(
        "land_research_artifacts",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("output_key", sa.String(length=200), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["land_research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "output_key"),
    )
    op.create_index(
        op.f("ix_land_research_artifacts_run_id"),
        "land_research_artifacts",
        ["run_id"],
        unique=False,
    )
    op.create_table(
        "land_research_events",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["land_research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "sequence"),
    )


def downgrade() -> None:
    op.drop_table("land_research_events")
    op.drop_index(op.f("ix_land_research_artifacts_run_id"), table_name="land_research_artifacts")
    op.drop_table("land_research_artifacts")
    op.drop_index(op.f("ix_land_findings_run_id"), table_name="land_findings")
    op.drop_table("land_findings")
    op.drop_index(op.f("ix_land_evidence_run_id"), table_name="land_evidence")
    op.drop_table("land_evidence")
    op.drop_index(op.f("ix_land_research_runs_status"), table_name="land_research_runs")
    op.drop_index(op.f("ix_land_research_runs_investigation_id"), table_name="land_research_runs")
    op.drop_table("land_research_runs")
    op.drop_index(
        op.f("ix_land_research_messages_investigation_id"), table_name="land_research_messages"
    )
    op.drop_table("land_research_messages")
    op.drop_index(op.f("ix_land_investigations_land_id"), table_name="land_investigations")
    op.drop_table("land_investigations")
