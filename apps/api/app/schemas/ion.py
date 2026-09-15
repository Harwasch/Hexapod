from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel

IonAssetStatus = Literal[
    "AWAITING_FILES", "NOT_STARTED", "IN_PROGRESS", "COMPLETE", "ERROR", "DATA_ERROR"
]


class IonReconstructionCapabilities(CamelModel):
    """What the backend can do with Cesium ion's reality reconstruction today."""

    monitor_jobs: bool
    register_assets: bool = True
    create_jobs: bool
    create_jobs_reason: str


class IonStatus(CamelModel):
    configured: bool
    api_base: str
    reconstruction: IonReconstructionCapabilities


class IonAssetMetadata(CamelModel):
    id: int
    name: str
    description: str | None = None
    type: str
    status: IonAssetStatus
    percent_complete: int = Field(ge=0, le=100)
    bytes: int | None = None
    attribution: str | None = None
    date_added: datetime | None = None
