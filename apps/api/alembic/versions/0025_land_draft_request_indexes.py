"""Index private action and scenario recovery by land and saved request key."""

import sqlalchemy as sa

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("land_action_revisions", "land_scenario_revisions"):
        op.create_index(
            f"ix_{table}_request", table, ["land_id", sa.text("(payload ->> 'request_key')")]
        )


def downgrade():
    for table in ("land_action_revisions", "land_scenario_revisions"):
        op.drop_index(f"ix_{table}_request", table_name=table)
