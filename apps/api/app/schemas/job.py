from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from app.models.enums import ArtifactKind, RunStatus
from app.schemas.base import CamelModel


class JobCreate(CamelModel):
    """Launch a run. `recipeVersion` is resolved from the recipe registry rather than
    supplied, so a caller cannot claim a run used a version that does not exist."""

    recipe: str = Field(min_length=1, max_length=120)
    params: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = Field(default=None, max_length=40)
    tier: str | None = Field(default=None, max_length=40)


class JobRetry(CamelModel):
    """Retry a finished run from one of its stages.

    `fromStage` is the stage to resume at; omitted, it is the stage that failed. Every
    stage before it keeps its `complete` step row and its artifacts, and the worker skips
    it — which is only possible because the run's workdir is still there (A6 keeps
    `checkpoint/` and clears `out/` at the start of each attempt).
    """

    from_stage: str | None = Field(default=None, max_length=120)


class JobStepLog(CamelModel):
    """One step's log, read back out of object storage.

    Logs are never in the database: `job_steps.log_key` is a key, and this endpoint is
    what turns it into text for the panel's log drawer.
    """

    step_id: uuid.UUID
    stage_id: str
    log_key: str
    text: str


class ArtifactRead(CamelModel):
    """Read model: every field is explicit (no defaults) so the OpenAPI contract marks it
    required."""

    id: uuid.UUID
    job_step_id: uuid.UUID
    kind: ArtifactKind
    storage_key: str
    bytes: int | None
    checksum: str | None
    content_type: str | None
    created_at: datetime
    updated_at: datetime


class ArtifactReference(CamelModel):
    """Something that points at an artifact, which is why it cannot be deleted.

    Today that is a site: `register` writes the packaged tileset's public URL onto an
    asset, and the thumbnail's onto the site itself. An artifact with no references is
    not automatically garbage — a `manifest.json` is read by people, not by rows — but
    the Outputs view's unreferenced list is where a cleanup starts.
    """

    #: `site-asset` or `site-thumbnail`.
    kind: str
    site_id: uuid.UUID
    site_slug: str
    label: str


class ArtifactRow(ArtifactRead):
    """One artifact with the run that produced it and whatever points at it.

    A join the console would otherwise do four requests to assemble: the Outputs view is
    a flat table of every artifact ever written, and `kind`, `bytes` and `storageKey`
    alone do not say which run to blame or whether anything still needs it.
    """

    stage_id: str
    impl: str
    job_id: uuid.UUID
    recipe: str
    capture_id: uuid.UUID
    capture_name: str
    references: list[ArtifactReference]


class JobStepRead(CamelModel):
    id: uuid.UUID
    job_id: uuid.UUID
    stage_id: str
    ordinal: int
    impl: str
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    metrics: dict[str, Any]
    log_key: str | None
    attempt: int
    checkpoint_key: str | None
    preempted_at: datetime | None
    artifacts: list[ArtifactRead]
    created_at: datetime
    updated_at: datetime


class JobRead(CamelModel):
    id: uuid.UUID
    capture_id: uuid.UUID
    recipe: str
    recipe_version: str
    params: dict[str, Any]
    status: RunStatus
    provider: str | None
    tier: str | None
    finished_at: datetime | None
    duration_s: float | None
    cost_usd: Decimal | None
    error: str | None
    # The lease, not a lock: see the comment on app.models.job.Job. `claimedBy` and
    # `leaseExpiresAt` are read-only facts about who holds the run right now.
    claimed_by: str | None
    claimed_at: datetime | None
    lease_expires_at: datetime | None
    steps: list[JobStepRead]
    created_at: datetime
    updated_at: datetime
