"""An epoch revision's per-cell maps (M4, ADR 0016): what was measured beside its tiles.

A run that measured how well each part of its capture was seen ships a `hexapod.cellgrid`
beside its tiles (`confidence.bin`, declared on the tileset's root `extras.confidence`).
`register` records it here, one row per **asset** (an epoch revision) and kind: a re-run
is a new asset with its own map, and the old revision keeps the map it was measured with,
as it keeps its tiles. The row holds the descriptor (the small JSON that says how to read
the file, its summary included) and the file's URL; the planes themselves stay in the
bucket, read on demand (`app/analysis/cell_grid.py`).

No map is not an error: an upload of a finished splat (`splat-ingest`) has no capture
evidence, so its epoch reads "not measured".
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin, utcnow
from app.models.enums import EpochMapKind


class EpochMap(UUIDPrimaryKeyMixin, Base):
    """One per-cell map of one epoch revision (its asset), as `register` recorded it."""

    __tablename__ = "epoch_maps"
    __table_args__ = (
        UniqueConstraint("asset_id", "kind", name="uq_epoch_maps_asset_kind"),
        CheckConstraint(
            f"kind IN ({', '.join(repr(k.value) for k in EpochMapKind)})",
            name="ck_epoch_maps_kind",
        ),
        CheckConstraint("cell_m > 0", name="ck_epoch_maps_cell_m"),
    )

    epoch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("epochs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    #: The run that measured it.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    #: The container (`hexapod.cellgrid`) and its version.
    format: Mapped[str] = mapped_column(String(40), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Where the file is: beside the tiles of the generation it was published in.
    url: Mapped[str] = mapped_column(Text, nullable=False)
    #: The finest level's cell, metres, and its lattice level.
    cell_m: Mapped[float] = mapped_column(Float, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The tileset's root extras value, verbatim: everything a reader needs.
    descriptor: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    #: `descriptor.summary`, kept apart so a list of epochs reads it without the levels.
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
