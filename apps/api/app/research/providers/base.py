from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import HttpUrl
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from app.schemas.geojson import Footprint
from app.schemas.research import EvidenceContent


@dataclass(frozen=True)
class SourceSpec:
    id: str
    name: str
    domain: str
    coverage: str
    resolution: str
    license: str
    attribution: str
    documentation_url: str
    endpoint: str


@dataclass
class SourceResult:
    provider: str
    status: Literal["available", "empty", "uncovered", "unavailable"]
    summary: str
    evidence: list[tuple[str, EvidenceContent]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceContext:
    boundary: Footprint

    @property
    def geometry(self) -> BaseGeometry:
        return shape(self.boundary.model_dump())

    @property
    def point(self) -> tuple[float, float]:
        point = self.geometry.representative_point()
        return point.x, point.y


class SourceError(Exception):
    pass


def fetch_json(
    client: httpx.Client,
    spec: SourceSpec,
    params: dict[str, str | int | float | bool],
    *,
    method: Literal["GET", "POST"] = "GET",
) -> tuple[dict[str, Any], str]:
    # Only registry endpoints are fetched. Redirects cannot turn a trusted endpoint
    # into an arbitrary URL, and response size is bounded even without Content-Length.
    with client.stream(
        method,
        spec.endpoint,
        params=params if method == "GET" else None,
        data=params if method == "POST" else None,
        follow_redirects=False,
        timeout=25,
    ) as response:
        response.raise_for_status()
        chunks = bytearray()
        for chunk in response.iter_bytes():
            chunks.extend(chunk)
            if len(chunks) > 2_000_000:
                raise SourceError("The provider response exceeded the analysis size limit.")
        result = json.loads(chunks)
        if not isinstance(result, dict):
            raise SourceError("The provider returned an unexpected response.")
        return result, str(response.url)


def evidence(
    spec: SourceSpec,
    url: str,
    title: str,
    data: dict[str, Any],
    relevance: str,
    *,
    regional: bool = False,
) -> EvidenceContent:
    snapshot = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return EvidenceContent(
        provider=spec.id,
        title=title,
        url=HttpUrl(url),
        license=spec.license,
        attribution=spec.attribution,
        retrieved_at=datetime.now(UTC),
        excerpt=snapshot[:30_000],
        spatial_relevance="regional" if regional else "within",
        relevance_note=relevance,
        snapshot_hash=hashlib.sha256(snapshot.encode()).hexdigest(),
    )
