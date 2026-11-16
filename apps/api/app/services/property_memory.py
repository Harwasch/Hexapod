"""Property-wide retrieval seam for the Ask reasoner; never owns Ask's tables.

Postgres adapters read two coordinator-provided views documented in docs/V2.md.
Tool callers bind site_id from the validated Ask request, not from model arguments.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.geojson import Polygon
from app.services.georef_entities import GeoPoint, geodetic_to_ecef


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    label: str | None = Field(default=None, min_length=1, max_length=200)
    attrs: dict[str, Any] = Field(default_factory=dict)
    since: datetime | None = None
    until: datetime | None = None
    lon: float | None = Field(default=None, ge=-180, le=180)
    lat: float | None = Field(default=None, ge=-90, le=90)
    radius_m: float | None = Field(default=None, gt=0, le=100000)
    nearest: bool = False
    polygon: Polygon | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0, le=100000)

    @model_validator(mode="after")
    def validate_query(self) -> MemoryQuery:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon are a pair")
        if (self.radius_m is not None or self.nearest) and self.lat is None:
            raise ValueError("radius/nearest needs lat and lon")
        if self.lat is not None and self.radius_m is None and not self.nearest:
            raise ValueError("point needs radius_m or nearest")
        for value in (self.since, self.until):
            if value is not None and value.utcoffset() is None:
                raise ValueError("time bounds must include a timezone")
        if self.since is not None and self.until is not None and self.since > self.until:
            raise ValueError("since must precede until")
        # Reject NaN in arbitrary JSON attrs before it reaches JSONB.
        json.dumps(self.attrs, allow_nan=False)
        return self


class MemoryRepository(Protocol):
    def entities(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]: ...
    def answers(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]: ...


class PostgresMemory:
    def __init__(self, db: Session) -> None:
        self.db = db

    def entities(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]:
        sql, params = entity_query(site_id, query)
        return [dict(row) for row in self.db.execute(text(sql), params).mappings()]

    def answers(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]:
        # Earlier answers are historical evidence, never silently recomputed current truth.
        params = _params(site_id, query)
        sql = """
            SELECT id::text AS id, scan_id::text AS "scanId", site_id::text AS "siteId",
                   question, answer, observed_at AS "observedAt"
            FROM property_answer_memory
            WHERE site_id = :site_id
              AND (CAST(:since AS timestamptz) IS NULL OR observed_at >= :since)
              AND (CAST(:until AS timestamptz) IS NULL OR observed_at <= :until)
              AND (CAST(:label AS text) IS NULL OR question ILIKE :label ESCAPE '!')
            ORDER BY observed_at DESC, id ASC LIMIT :limit OFFSET :offset
        """
        return [dict(row) for row in self.db.execute(text(sql), params).mappings()]


def _params(site_id: uuid.UUID, query: MemoryQuery) -> dict[str, Any]:
    label = query.label
    if label is not None:
        label = "%" + label.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
    return {
        "site_id": site_id,
        "since": query.since,
        "until": query.until,
        "label": label,
        "limit": query.limit,
        "offset": query.offset,
    }


def entity_query(site_id: uuid.UUID, query: MemoryQuery) -> tuple[str, dict[str, Any]]:
    """All values bound; metre distances use geography, polygon includes its boundary."""
    params = _params(site_id, query)
    params.update(
        attrs=json.dumps(query.attrs, allow_nan=False),
        lon=query.lon,
        lat=query.lat,
        radius=query.radius_m,
    )
    predicates = [
        "site_id = :site_id",
        "(CAST(:since AS timestamptz) IS NULL OR observed_at >= :since)",
        "(CAST(:until AS timestamptz) IS NULL OR observed_at <= :until)",
        "(CAST(:label AS text) IS NULL OR label ILIKE :label ESCAPE '!')",
        "attrs @> CAST(:attrs AS jsonb)",
    ]
    point = "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography"
    distance = f"ST_Distance(geo::geography, {point})"
    if query.radius_m is not None:
        predicates.append(f"ST_DWithin(geo::geography, {point}, :radius)")
    if query.polygon is not None:
        predicates.append("ST_Covers(ST_SetSRID(ST_GeomFromGeoJSON(:polygon),4326), geo)")
        params["polygon"] = query.polygon.model_dump_json(by_alias=True)
    if query.nearest:
        predicates.append("geo IS NOT NULL")
    order = f"{distance} ASC, id ASC" if query.nearest else "observed_at DESC, id ASC"
    # Only fixed SQL fragments above are interpolated. No table or column supplied by a caller.
    sql = f"""
        SELECT id::text AS id, scan_id::text AS "scanId", site_id::text AS "siteId",
               entity, observed_at AS "observedAt", uncertainty_m AS "uncertaintyM"
        FROM property_entity_memory WHERE {" AND ".join(predicates)}
        ORDER BY {order} LIMIT :limit OFFSET :offset
    """  # noqa: S608
    return sql, params


def search_property_entities(
    repository: MemoryRepository,
    site_id: uuid.UUID,
    arguments: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return repository.entities(site_id, MemoryQuery.model_validate(arguments))


def search_property_answers(
    repository: MemoryRepository,
    site_id: uuid.UUID,
    arguments: Mapping[str, Any],
) -> list[dict[str, Any]]:
    query = MemoryQuery.model_validate(arguments)
    if query.attrs or query.lat is not None or query.polygon is not None or query.nearest:
        raise ValueError("answer search accepts label, time and pagination only")
    return repository.answers(site_id, query)


def property_scope_site(request: Mapping[str, Any]) -> uuid.UUID | None:
    """Absent scope retains scan semantics. B calls this before registering property tools."""
    scope = request.get("scope", "scan")
    if scope == "scan":
        return None
    if scope != "property" or not request.get("siteId"):
        raise ValueError("property scope requires siteId")
    return uuid.UUID(str(request["siteId"]))


PROPERTY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_property_entities",
        "description": "Find observations across scans of the bound property.",
        "input_schema": MemoryQuery.model_json_schema(),
    },
    {
        "name": "search_property_answers",
        "description": "Retrieve earlier answers as dated evidence.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "maxLength": 200},
                "since": {"type": "string", "format": "date-time"},
                "until": {"type": "string", "format": "date-time"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "offset": {"type": "integer", "minimum": 0, "maximum": 100000},
            },
        },
    },
]


class Observation(BaseModel):
    """Dedup metadata envelope; `entity` remains the unchanged EntitySchema payload."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: uuid.UUID
    site_id: uuid.UUID
    scan_id: uuid.UUID
    label: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    uncertainty_m: float = Field(ge=0)
    observed_at: datetime
    attrs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def timezone_required(self) -> Observation:
        if not self.label.strip():
            raise ValueError("label must not be blank")
        if self.observed_at.utcoffset() is None:
            raise ValueError("observed_at requires timezone")
        return self


def dedupe_candidates(new: Observation, existing: Sequence[Observation]) -> dict[str, Any]:
    """Propose identity links, never destroy evidence or average correlated GPS errors.

    Conservative policy: cross-scan only, 30 days, fuzzy label >=.85, horizontal
    uncertainty <=5m each, distance <=1m + quadrature uncertainty. Ambiguity requires
    review; no transitive clustering. Dynamic attrs remain dated observations.
    """
    candidates: list[dict[str, Any]] = []
    a = geodetic_to_ecef(GeoPoint(new.lat, new.lon, 0))
    label = " ".join(new.label.casefold().split())
    for old in existing:
        if old.site_id != new.site_id or old.scan_id == new.scan_id or old.id == new.id:
            continue
        if max(new.uncertainty_m, old.uncertainty_m) > 5:
            continue
        if abs(new.observed_at - old.observed_at) > timedelta(days=30):
            continue
        similarity = SequenceMatcher(None, label, " ".join(old.label.casefold().split())).ratio()
        b = geodetic_to_ecef(GeoPoint(old.lat, old.lon, 0))
        distance = math.dist(a, b)
        if similarity < 0.85 or distance > 1 + math.hypot(new.uncertainty_m, old.uncertainty_m):
            continue
        candidates.append({"id": str(old.id), "distanceM": distance, "labelSimilarity": similarity})
    candidates.sort(key=lambda value: (value["distanceM"], value["id"]))
    decision: Literal["new", "link", "review"] = (
        "new" if not candidates else "link" if len(candidates) == 1 else "review"
    )
    return {
        "decision": decision,
        "canonicalId": candidates[0]["id"] if decision == "link" else None,
        "observationId": str(new.id),
        "candidates": candidates,
    }
