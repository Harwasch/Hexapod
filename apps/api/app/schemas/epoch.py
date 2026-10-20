"""Read models of a site's frame and its epochs (decision D3)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.models.enums import (
    EpochJoinMethod,
    EpochPlatform,
    EpochStatus,
    HeightDatum,
    ObservedAtSource,
    SiteFrameSource,
)
from app.schemas.base import CamelModel
from app.schemas.capture import CaptureQuality
from app.schemas.confidence import EpochConfidenceRead
from app.schemas.geojson import Footprint


class SiteFrameRead(CamelModel):
    """The WGS84 origin every epoch of the site is placed about; fixed once created.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    site_id: uuid.UUID
    lat: float
    lon: float
    height: float
    height_datum: HeightDatum
    #: The UTM zone of the origin.
    epsg: int
    source: SiteFrameSource
    created_at: datetime


class EpochRead(CamelModel):
    """One dated capture of a site, expressed in the site's frame.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    id: uuid.UUID
    site_id: uuid.UUID
    #: 1 for the capture that defined the site's frame, then one more for each epoch.
    ordinal: int
    capture_id: uuid.UUID | None
    #: The run that produced the epoch's current revision.
    job_id: uuid.UUID | None
    observed_at: datetime
    #: Where `observedAt` came from; `created` means the upload time, not a capture date.
    observed_at_source: ObservedAtSource
    footprint: Footprint
    platform: EpochPlatform
    status: EpochStatus
    join_method: EpochJoinMethod
    #: How the epoch was aligned to the ones before it; null until alignment exists (M3).
    alignment: dict[str, Any] | None
    quality: CaptureQuality | None
    #: The epoch's newest observed asset that is not superseded; null when it has none.
    current_asset_id: uuid.UUID | None
    #: How well each part of it was seen: the confidence map its current revision's run
    #: measured (ADR 0016), summarised. Null is "not measured" -- that run measured none
    #: (a splat upload carries no capture evidence).
    confidence: EpochConfidenceRead | None
    created_at: datetime
    updated_at: datetime
