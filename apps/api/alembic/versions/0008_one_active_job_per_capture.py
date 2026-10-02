"""jobs: at most one queued-or-running job per capture, enforced by the database.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-02

`create_job` checked for an active run with a SELECT and then INSERTed, so two requests in
flight -- a double-click on Process, a phone retrying a POST it thought had failed -- both
passed the check and queued two runs of one capture, which the worker would then run side
by side over the same inputs. A partial unique index on `capture_id`, over the two active
statuses only, is the check that cannot race; finished runs are untouched by it, so a
capture keeps its whole history.

The index cannot be built over a table that already breaks it, so the upgrade first
resolves any capture with more than one active job: it keeps one and cancels the rest,
with their unfinished steps, saying why in `jobs.error`. A worker holding a cancelled job
sees the status on its next heartbeat and stops, exactly as for a cancel from the console.
The downgrade drops the index and does not un-cancel anything.

Which one is kept, in order:

1. **the one a worker is running now** -- in progress under a lease that has not run out.
   Cancelling that is cancelling real work (a GPU call, hours in), and it is the only
   duplicate whose worker would notice. "In progress" alone is not enough: it was the
   first rule once, and an in-progress row whose worker died long ago -- created earlier,
   lease lapsed -- outranked the job a worker was actually running, which was cancelled;
2. else **the newest claimed** in-progress one: of runs nobody holds, the one that got
   furthest most recently, and the one the next worker would resume;
3. else **the oldest** -- queued first, so first in line, as the worker would have taken it.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

INDEX = "uq_jobs_one_active_per_capture"
#: The same predicate the model declares (app/models/job.py), spelled the same way.
ACTIVE = "status IN ('not-started', 'in-progress')"
REASON = (
    "Cancelled by migration 0008: another run of this capture was already queued or "
    "running, and a capture is run one at a time."
)


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT id, row_number() OVER (
                    PARTITION BY capture_id
                    ORDER BY
                        (status = 'in-progress' AND lease_expires_at > now()) IS TRUE DESC,
                        (status = 'in-progress') DESC,
                        claimed_at DESC NULLS LAST,
                        created_at ASC,
                        id ASC
                ) AS n
                FROM jobs
                WHERE status IN ('not-started', 'in-progress')
            )
            UPDATE jobs
            SET status = 'cancelled', finished_at = now(), error = :reason
            FROM ranked
            WHERE jobs.id = ranked.id AND ranked.n > 1
            """
        ).bindparams(reason=REASON)
    )
    op.execute(
        sa.text(
            """
            UPDATE job_steps
            SET status = 'cancelled', finished_at = now()
            WHERE status IN ('not-started', 'in-progress')
              AND job_id IN (SELECT id FROM jobs WHERE status = 'cancelled' AND error = :reason)
            """
        ).bindparams(reason=REASON)
    )
    op.create_index(
        INDEX,
        "jobs",
        ["capture_id"],
        unique=True,
        postgresql_where=sa.text(ACTIVE),
    )


def downgrade() -> None:
    op.drop_index(INDEX, table_name="jobs")
