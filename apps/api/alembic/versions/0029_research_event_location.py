"""Let research events say where on the map they are looking or what they found."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("land_research_events", sa.Column("location", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("land_research_events", "location")
