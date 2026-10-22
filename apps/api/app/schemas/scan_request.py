from __future__ import annotations

import re
import uuid
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint, Polygon, PolygonCoordinates

#: An address as the landing form checks it (src/landing/form.ts): a plain local part, a domain
#: of dot-separated labels and a letter TLD. Stricter than RFC 5322 on purpose: no `?`, `&`,
#: `<`, spaces or quotes, so an address can go into a mailto link or a log line as it is.
EMAIL = re.compile(
    r"^[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$"
)
#: Control characters: none in a name or an email, and in notes only tab and line breaks.
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NOTES_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
#: A parcel the public lookup returned: `ca-parcel/<county FIPS>/<parcel number>`, the number
#: printable ASCII without `/` (no spaces, control characters or look-alikes).
PARCEL_ID = r"^ca-parcel/06\d{3}/[\x21-\x2e\x30-\x7e]{1,150}$"


#: What a submitted outline may hold, checked before PostGIS sees it: fifty parcels of a hundred
#: corners each, and their holes. Not the whole defence (two crossing combs of a few hundred
#: corners were slow to dissolve; services/scan_requests.py bounds that with a timeout), but
#: what keeps a body under the size cap from being a very large geometry.
MAX_VERTICES = 5_000
MAX_RINGS = 100


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
        if CONTROL.search(value):
            raise ValueError("must not contain control characters")
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("notes")
    @classmethod
    def plain_notes(cls, value: str) -> str:
        if NOTES_CONTROL.search(value):
            raise ValueError("must not contain control characters")
        return value

    @field_validator("email")
    @classmethod
    def looks_like_email(cls, value: str) -> str:
        if not EMAIL.fullmatch(value) or ".." in value:
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
        if len(rings) > MAX_RINGS:
            raise ValueError(f"an outline may have at most {MAX_RINGS} rings")
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
    #: True when a stored outline could not be read back (an old row): the rest is still shown.
    boundary_unreadable: bool = False
    #: True when the outline could not be cleaned in time and was kept as sent, for a person to
    #: look at (`raw_outline` in the table; not served here).
    outline_unvalidated: bool = False
    area_m2: float | None
    parcel_ids: list[str]
