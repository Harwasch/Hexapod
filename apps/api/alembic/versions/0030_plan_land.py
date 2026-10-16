"""Plans that hang off a saved Land: the land and the boundary revision they were planned on.

Land is workspace-private, so a check keeps it off public plans. Missions already scheduled
from land actions are linked through the action revision that scheduled them.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0030"
# merge: re-point to 0029 when research-status lands
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("plans", sa.Column("land_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("plans", sa.Column("boundary_revision", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_plans_land", "plans", "land_areas", ["land_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index("ix_plans_land_id", "plans", ["land_id"])
    op.execute(
        """
        UPDATE plans p
        SET land_id = r.land_id, boundary_revision = r.boundary_revision
        FROM land_action_revisions r
        WHERE r.mission_id = p.id
        """
    )
    op.create_check_constraint(
        "ck_plans_land_private", "plans", "land_id IS NULL OR workspace_id IS NOT NULL"
    )


def downgrade() -> None:
    # Only the link is lost: the plans stay private, and 0016's guard still refuses to make
    # them public on the way further down.
    op.drop_constraint("ck_plans_land_private", "plans", type_="check")
    op.drop_index("ix_plans_land_id", table_name="plans")
    op.drop_constraint("fk_plans_land", "plans", type_="foreignkey")
    op.drop_column("plans", "boundary_revision")
    op.drop_column("plans", "land_id")
