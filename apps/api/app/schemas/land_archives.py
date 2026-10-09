from __future__ import annotations

from typing import Literal

from pydantic import Field, HttpUrl, field_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import MapGeometry


class ArchiveMedia(CamelModel):
    kind: Literal["photograph", "historical-map"]
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(max_length=3000)
    source_url: HttpUrl
    preview_url: HttpUrl
    download_url: HttpUrl | None = None
    creator: str = Field(max_length=2000)
    license: str = Field(min_length=1, max_length=500)
    license_url: HttpUrl
    source_date: str | None = Field(default=None, max_length=500)
    date_meaning: str = Field(max_length=1000)
    location: MapGeometry | None = None
    location_meaning: Literal["catalog-coordinate", "catalog-footprint"]
    relevance: str = Field(max_length=2000)
    source_version: str | None = Field(default=None, max_length=200)

    @field_validator("preview_url")
    @classmethod
    def registered_preview(cls, value: HttpUrl) -> HttpUrl:
        if (
            value.scheme != "https"
            or value.username
            or value.password
            or value.port not in (None, 443)
        ):
            raise ValueError("Archive previews require an approved public HTTPS image.")
        if value.host in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
            if not value.path or not value.path.startswith("/wikipedia/commons/"):
                raise ValueError("Only Commons media previews are supported.")
        elif value.host == "prd-tnm.s3.amazonaws.com":
            if not value.path or not value.path.startswith("/StagedProducts/Maps/HistoricalTopo/"):
                raise ValueError("Only historical USGS map previews are supported.")
        else:
            raise ValueError("This archive preview host is not registered.")
        return value
