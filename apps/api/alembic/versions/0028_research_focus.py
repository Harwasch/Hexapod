"""Pin selected research map features and their evidence to follow-up runs."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("land_research_runs", sa.Column("focus", postgresql.JSONB(), nullable=True))
    op.add_column(
        "land_research_runs", sa.Column("focus_snapshot", postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("land_research_runs", "focus_snapshot")
    op.drop_column("land_research_runs", "focus")
