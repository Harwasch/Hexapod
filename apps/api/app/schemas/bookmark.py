from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.base import CamelModel


class CameraBookmarkBase(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    height: float = Field(description="Ellipsoidal height in metres")
    heading: float = Field(default=0.0, ge=-360, le=360, description="Degrees")
    pitch: float = Field(default=-30.0, ge=-90, le=90, description="Degrees")
    roll: float = Field(default=0.0, ge=-360, le=360, description="Degrees")
    is_default: bool = False


class CameraBookmarkCreate(CameraBookmarkBase):
    pass


class CameraBookmarkRead(CameraBookmarkBase):
    id: uuid.UUID
    site_id: uuid.UUID
    created_at: datetime
