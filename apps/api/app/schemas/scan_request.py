from __future__ import annotations

import re
import uuid
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint, Polygon, PolygonCoordinates

#: The same check the landing form makes (src/landing/form.ts): something@something.tld.
EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
#: A parcel the public lookup returned: `ca-parcel/<county FIPS>/<parcel number>`.
PARCEL_ID = r"^ca-parcel/06\d{3}/[^\s/]{1,150}$"


#: Vertices a submitted outline may have, as a saved land may (LandCreate.bounded_boundary).
MAX_VERTICES = 20_000


class ParcelsMultiPolygon(CamelModel):
    """Several parcels as sent: each part a valid polygon, though neighbours share edges, which
    a valid MultiPolygon may not. The service dissolves them into one outline on the way in."""

    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: Annotated[list[PolygonCoordinates], Field(min_length=1, max_length=50)]

    @model_validator(mode="after")
    def parts_are_polygons(self) -> ParcelsMultiPolygon:
        for part in self.coordinates:
            Polygon(coordinates=part)
        return self


SubmittedBoundary = Annotated[Polygon | ParcelsMultiPolygon, Field(discriminator="type")]


class ScanRequestCreate(CamelModel):
    """What the landing page's "Request a site scan" form sends (src/landing/form.ts)."""

    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=254)
    notes: str = Field(default="", max_length=5_000)
    boundary: SubmittedBoundary | None = None
    parcel_ids: list[str] = Field(default_factory=list, max_length=50)
    #: A field people never see and bots fill in: a request that carries one is answered as if
    #: received, and dropped.
    website: str = Field(default="", max_length=500)

    @field_validator("name", "email")
    @classmethod
    def stripped(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("email")
    @classmethod
    def looks_like_email(cls, value: str) -> str:
        if not EMAIL.fullmatch(value):
            raise ValueError("must be an email address like name@example.com")
        return value

    @field_validator("parcel_ids")
    @classmethod
    def parcel_ids_from_the_lookup(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(PARCEL_ID, value):
                raise ValueError("parcel ids are the public parcel lookup's (ca-parcel/...)")
        return values

    @field_validator("boundary")
    @classmethod
    def bounded(
        cls, boundary: Polygon | ParcelsMultiPolygon | None
    ) -> Polygon | ParcelsMultiPolygon | None:
        if boundary is None:
            return None
        polygons = [boundary.coordinates] if isinstance(boundary, Polygon) else boundary.coordinates
        rings = [ring for polygon in polygons for ring in polygon]
        if sum(len(ring) for ring in rings) > MAX_VERTICES:
            raise ValueError(f"an outline may have at most {MAX_VERTICES:,} vertices")
        if any(abs(a[0] - b[0]) > 180 for ring in rings for a, b in pairwise(ring)):
            raise ValueError("an outline may not cross the antimeridian")
        return boundary

    @model_validator(mode="after")
    def parcels_need_a_boundary(self) -> ScanRequestCreate:
        if self.parcel_ids and self.boundary is None:
            raise ValueError("parcel ids come with the boundary they make up")
        return self


class ScanRequestReceived(CamelModel):
    """The one answer to every accepted submission, stored or not: nothing to probe."""

    status: Literal["received"] = "received"


class ScanRequestRead(CamelModel):
    id: uuid.UUID
    created_at: datetime
    name: str
    email: str
    notes: str
    boundary: Footprint | None
    area_m2: float | None
    parcel_ids: list[str]
