from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import ArtifactKind, RunStatus

if TYPE_CHECKING:
    from app.models.capture import Capture

# One PostgreSQL type shared by `jobs.status` and `job_steps.status`: the same five
# states, declared once so the two columns cannot drift apart.
RUN_STATUS = Enum(RunStatus, name="run_status", values_callable=lambda e: [m.value for m in e])


class Job(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One pipeline run over one capture.

    ``recipe``, ``recipe_version`` and the whole ``params`` set are recorded rather than
    summarised, because the console's reason to exist is comparing two runs of the same
    capture — and a parameter that was not written down is a comparison that cannot be
    made afterwards.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        # The claim loop, and only the claim loop: worker picks the oldest row that is
        # either unstarted or whose lease has lapsed. Leading `status` narrows to the
        # handful of claimable rows; `lease_expires_at` makes the reclaim half a range
        # scan instead of a sequential one over every job ever run.
        Index("ix_jobs_status_lease_expires_at", "status", "lease_expires_at"),
    )

    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("captures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    recipe: Mapped[str] = mapped_column(String(120), nullable=False)
    recipe_version: Mapped[str] = mapped_column(String(40), nullable=False)
    # The full resolved parameter set, every stage, after defaults and overrides.
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[RunStatus] = mapped_column(
        RUN_STATUS, nullable=False, default=RunStatus.NOT_STARTED
    )
    # Free strings, not enums: which GPU host and which card are market facts that change
    # faster than migrations, and nothing in the code branches on their values.
    provider: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tier: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- The claim is a committed lease, not a held row lock. ------------------------
    # A7 implements the claim loop against these three columns, and is not free to
    # reinvent it: A0 measured the alternative. `SELECT ... FOR UPDATE SKIP LOCKED` held
    # open for the job's duration releases in 0.02 s on SIGKILL, but a worker that is
    # merely frozen -- SIGSTOP, a wedged host, a network partition -- keeps the row
    # locked for about 2 h 51 min at default TCP keepalives, and the open transaction
    # pins the vacuum horizon while it waits (50 000 dead tuples measured
    # unreclaimable). So the claim UPDATE commits immediately and takes a short lease;
    # reclaim is then 3.06 s with a 3 s lease however the worker died. Throughput is not
    # the reason and should not be cited as one: 2883 claims/s locked vs 1565/s leased,
    # both about 1000x over any need here.
    #
    # A worker claims by committing `UPDATE jobs SET claimed_by=..., claimed_at=now(),
    # lease_expires_at=now() + lease`, and heartbeats by pushing `lease_expires_at`
    # forward. A row whose lease has passed is claimable by anyone.
    claimed_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    capture: Mapped[Capture] = relationship(back_populates="jobs")
    steps: Mapped[list[JobStep]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStep.ordinal"
    )


class JobStep(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One stage of one run.

    ``attempt``, ``checkpoint_key`` and ``preempted_at`` are not error handling. On an
    interruptible GPU tier the worker being killed mid-stage is ordinary operation, so a
    step records how many times it has been tried and where it left its checkpoint, and
    resumes rather than restarts.
    """

    __tablename__ = "job_steps"
    __table_args__ = (
        # A recipe is an ordered list, so a run has exactly one step per position. The
        # index this constraint creates leads with `job_id`, so it also serves "the steps
        # of this job, in order" -- no separate FK index is declared for that reason.
        UniqueConstraint("job_id", "ordinal", name="uq_job_step_ordinal"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    # `stage_id` is the recipe's name for the stage ("train"); `impl` is the registered
    # callable that ran it ("gsplat"). Swapping impl is the extensibility story, so the
    # pair is what makes a run reproducible -- both are registry names, hence strings.
    stage_id: Mapped[str] = mapped_column(String(120), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    impl: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        RUN_STATUS, nullable=False, default=RunStatus.NOT_STARTED
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    log_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    checkpoint_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    preempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped[Job] = relationship(back_populates="steps")
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="job_step", cascade="all, delete-orphan", order_by="Artifact.created_at"
    )


class Artifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One output object, attributed to the step that produced it.

    Making outputs explicit rather than implicit in the bucket is what gives the console
    an Outputs view, lets storage be reconciled against the database in both directions,
    and makes "retry from this stage" precise about what it is replacing.
    """

    __tablename__ = "artifacts"

    job_step_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_steps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[ArtifactKind] = mapped_column(
        Enum(ArtifactKind, name="artifact_kind", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    # Indexed but not unique: a retried stage writes the same key again, so one key can
    # legitimately have a row per attempt. Reconciliation asks "does the bucket have
    # this, does this have a bucket object", which needs the lookup, not the constraint.
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)

    job_step: Mapped[JobStep] = relationship(back_populates="artifacts")
