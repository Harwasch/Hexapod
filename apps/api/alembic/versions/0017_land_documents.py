"""Private immutable land records, extracted pages and document relationships."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def timestamps():
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.create_table(
        "land_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("extracted_characters", sa.Integer(), nullable=False),
        sa.Column("warnings", postgresql.JSONB(), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_land_documents_land_id", "land_documents", ["land_id"])
    op.create_table(
        "land_document_blobs",
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )
    op.create_table(
        "land_document_pages",
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("page", sa.Integer(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
    )
    op.create_index(
        "ix_land_document_pages_text_search",
        "land_document_pages",
        ["text"],
        postgresql_using="gin",
        postgresql_ops={"text": "gin_trgm_ops"},
    )
    op.create_table(
        "land_document_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "land_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "from_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("land_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.UniqueConstraint("land_id", "request_key"),
        *timestamps(),
    )
    op.create_index("ix_land_document_links_land_id", "land_document_links", ["land_id"])


def downgrade():
    op.drop_table("land_document_links")
    op.drop_table("land_document_pages")
    op.drop_table("land_document_blobs")
    op.drop_table("land_documents")
