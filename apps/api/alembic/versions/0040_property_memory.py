"""Use the Ask ledger as property memory, retaining request context.

Revision ID: 0040
Revises: 0039
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ask_runs", sa.Column("request", pg.JSONB(), nullable=False, server_default="{}"))
    op.add_column(
        "ask_runs", sa.Column("site_id", pg.UUID(), sa.ForeignKey("sites.id", ondelete="SET NULL"))
    )
    op.create_index("ix_ask_runs_site_id", "ask_runs", ["site_id"])
    op.execute("""
        CREATE VIEW property_entity_memory AS
        SELECT e.id, c.site_id, c.id AS scan_id, e.entity->>'label' AS label,
               e.entity->'attrs' AS attrs, e.entity, e.geo_anchor AS geo,
               CASE WHEN jsonb_typeof(e.entity #> '{attrs,georeference,uncertaintyM}') = 'number'
                    THEN (e.entity #>> '{attrs,georeference,uncertaintyM}')::double precision
               END AS uncertainty_m,
               COALESCE(ep.observed_at, c.captured_at, e.created_at) AS observed_at
        FROM scan_entities e JOIN captures c ON c.id = e.capture_id
        LEFT JOIN epochs ep ON ep.capture_id = c.id
    """)
    op.execute("""
        CREATE VIEW property_answer_memory AS
        SELECT r.id, COALESCE(c.site_id, r.site_id) AS site_id, r.capture_id AS scan_id,
               r.request->>'question' AS question, r.events AS answer, r.created_at AS observed_at
        FROM ask_runs r LEFT JOIN captures c ON c.id = r.capture_id WHERE r.state = 'complete'
    """)
    op.execute("CREATE INDEX ix_scan_entities_attrs ON scan_entities USING gin ((entity->'attrs'))")
    op.execute(
        "CREATE INDEX ix_scan_entities_geography ON scan_entities USING gist ((geo_anchor::geography))"
    )
    op.create_index("ix_scan_entities_site_time", "scan_entities", ["site_id", "created_at", "id"])


def downgrade() -> None:
    op.drop_index("ix_scan_entities_site_time", "scan_entities")
    op.drop_index("ix_scan_entities_geography", "scan_entities")
    op.drop_index("ix_scan_entities_attrs", "scan_entities")
    op.execute("DROP VIEW property_answer_memory")
    op.execute("DROP VIEW property_entity_memory")
    op.drop_index("ix_ask_runs_site_id", "ask_runs")
    op.drop_column("ask_runs", "site_id")
    op.drop_column("ask_runs", "request")
