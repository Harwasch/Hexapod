"""Bounded, revision-pinned inventory reads and evidence-backed mapped candidates."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.land_feature import FeatureInspection, LandFeature, LandFeatureRevision
from app.research.providers.base import SourceContext, SourceResult, evidence
from app.schemas.land_features import FeatureGeometryRequest, LandFeatureCreate
from app.schemas.land_selection import CandidateRequest
from app.services import land_features, land_selection
from app.services.errors import InvalidInputError, NotFoundError
from app.services.land import get_land

Section = Literal["overview", "geometry", "attributes", "inspections"]


def bounded_page(answer: dict[str, Any], records: list[Any], offset: int, count: int) -> None:
    answer["data"] = records[offset : offset + count]
    while len(json.dumps(answer, ensure_ascii=False)) > 28_500 and answer["data"]:
        answer["data"].pop()
    if records[offset:] and not answer["data"]:
        raise InvalidInputError("This inventory row exceeds the bounded read size.")
    following = offset + len(answer["data"])
    answer["offset"] = offset
    answer["nextOffset"] = following if following < len(records) else None


def read(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    feature_id: uuid.UUID | None,
    revision: int | None,
    section: Section,
    offset: int,
    count: int,
    as_of: datetime | None,
) -> dict[str, Any]:
    land = get_land(db, workspace_id, land_id)
    cutoff = as_of or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise InvalidInputError("Inventory read timestamps must include a timezone.")
    if feature_id is None:
        rows = list(
            db.scalars(
                select(LandFeature)
                .where(LandFeature.land_id == land_id, LandFeature.created_at <= cutoff)
                .order_by(LandFeature.id)
                .limit(min(count, 20) + 1)
                .offset(offset)
            )
        )
        selected = rows[: min(count, 20)]
        return {
            "kind": "catalog",
            "asOf": cutoff.isoformat(),
            "offset": offset,
            "nextOffset": offset + len(selected) if len(rows) > len(selected) else None,
            "data": [
                {
                    "id": str(row.id),
                    "revision": row.revision,
                    "name": row.content["name"],
                    "category": row.content["category"],
                    "status": row.content["status"],
                }
                for row in selected
            ],
            "interpretation": (
                "Current user-recorded inventory metadata. Read a specific revision before "
                "citing details; catalog pages can change as assets are added."
            ),
        }
    row = land_features.scoped(db, workspace_id, land_id, feature_id)
    snapshot = db.scalar(
        select(LandFeatureRevision).where(
            LandFeatureRevision.feature_id == row.id,
            LandFeatureRevision.revision == (revision if revision is not None else row.revision),
        )
    )
    if snapshot is None:
        raise NotFoundError("inventory revision", revision)
    feature = LandFeatureCreate.model_validate(snapshot.content)
    geometry = feature.geometry.model_dump(mode="json")
    answer: dict[str, Any] = {
        "kind": "feature",
        "featureId": str(row.id),
        "revision": snapshot.revision,
        "currentRevision": row.revision,
        "currentBoundaryRevision": land.revision,
        "section": section,
        "asOf": cutoff.isoformat(),
        "name": feature.name,
        "snapshotSha256": hashlib.sha256(
            json.dumps(snapshot.content, sort_keys=True).encode()
        ).hexdigest(),
        "interpretation": (
            "Private user-recorded asset data, not independently verified measurements or "
            "ownership. Read pages using this revision and asOf. Original evidence IDs "
            "retain their original investigation scope."
        ),
    }
    if section == "overview":
        answer["data"] = {
            key: value
            for key, value in feature.model_dump(mode="json").items()
            if key not in {"geometry", "attributes"}
        }
        answer["data"].update(
            {
                "geometryType": geometry["type"],
                "bounds": shape(geometry).bounds,
                "attributeCount": len(feature.attributes),
                "metrics": land_features.preview_geometry(
                    db, workspace_id, land_id, FeatureGeometryRequest(geometry=feature.geometry)
                ).model_dump(mode="json", exclude={"geometry"}),
            }
        )
    elif section == "geometry":
        parts = (
            [[[geometry["coordinates"]]]]
            if geometry["type"] == "Point"
            else [[geometry["coordinates"]]]
            if geometry["type"] == "LineString"
            else [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        records = [
            {"part": p, "ring": r, "index": i, "coordinates": coordinates}
            for p, part in enumerate(parts)
            for r, ring in enumerate(part)
            for i, coordinates in enumerate(ring)
        ]
        answer.update(
            {
                "geometryType": geometry["type"],
                "vertexCount": len(records),
                "partRingSizes": [[len(ring) for ring in part] for part in parts],
            }
        )
        bounded_page(answer, records, offset, count)
    elif section == "attributes":
        records = [
            {"name": key, "value": value} for key, value in sorted(feature.attributes.items())
        ]
        answer["attributeCount"] = len(records)
        bounded_page(answer, records, offset, count)
    else:
        inspections = list(
            db.scalars(
                select(FeatureInspection)
                .where(
                    FeatureInspection.feature_id == row.id,
                    FeatureInspection.feature_revision <= snapshot.revision,
                    FeatureInspection.created_at <= cutoff,
                )
                .order_by(FeatureInspection.created_at, FeatureInspection.id)
                .limit(min(count, 20) + 1)
                .offset(offset)
            )
        )
        records = []
        for inspection in inspections:
            item = land_features.inspection_read(inspection).model_dump(mode="json")
            if len(json.dumps(item)) > 25_000:
                item.pop("measurements", None)
                item.pop("measurement_units", None)
                item["measurementsOmitted"] = True
            records.append(item)
        bounded_page(answer, records, 0, min(count, 20))
        length = len(answer["data"])
        answer["offset"] = offset
        answer["nextOffset"] = offset + length if length < len(records) else None
    if len(json.dumps(answer, ensure_ascii=False)) > 29_000:
        raise InvalidInputError("This inventory section exceeds the bounded read size.")
    return answer


def mapped_search(
    db: Session, context: SourceContext, request: CandidateRequest, client: httpx.Client
) -> SourceResult:
    if request.kind == "parcel":
        raise InvalidInputError("Choose mapped physical features, not property parcels.")
    result = land_selection.candidates(db, request, client)
    output = SourceResult(
        "mapped-assets",
        result.status,
        result.message,
        data={
            "query": request.model_dump(mode="json"),
            "truncated": result.truncated,
            "candidates": [],
            "omitted": 0,
            "limitations": (
                "OSM nodes and ways near the query point only, not a complete asset inventory. "
                "Mapping is not surveyed accuracy, current condition, ownership or evidence of "
                "an easement. Building relations are not queried."
            ),
        },
    )
    if result.status == "unavailable":
        return output
    for candidate in result.candidates:
        snapshot = {
            "candidate": candidate.model_dump(mode="json"),
            "query": request.model_dump(mode="json"),
        }
        if len(json.dumps(snapshot, ensure_ascii=False)) > 29_000 or len(output.evidence) >= 20:
            output.data["omitted"] += 1
            continue
        intersects = context.geometry.intersects(shape(candidate.geometry.model_dump()))
        item = evidence(
            land_selection.OSM,
            str(candidate.source.url),
            candidate.label,
            snapshot,
            "Community-mapped physical feature. Geometry intersects the pinned study boundary."
            if intersects
            else "Community-mapped feature near the query point; it does not intersect the pinned study boundary.",
        )
        item.provider = "mapped-asset"
        item.spatial_relevance = "intersects" if intersects else "nearby"
        item.record_id = candidate.id
        output.evidence.append((candidate.id, item))
        output.data["candidates"].append(
            {
                "candidateId": candidate.id,
                "name": candidate.label,
                "geometryType": candidate.geometry.type,
                "distanceM": candidate.distance_m,
                "intersectsBoundary": intersects,
                "properties": candidate.properties,
            }
        )
    output.summary += (
        f" {len(output.evidence)} candidate snapshots retained; "
        f"{output.data['omitted']} omitted by snapshot limits."
    )
    return output


def mapped_feature(
    content: dict[str, Any],
    request_key: uuid.UUID,
    evidence_id: uuid.UUID,
    name: str | None,
    category: str | None,
    description: str,
) -> LandFeatureCreate:
    """The agent references a saved candidate; it cannot substitute arbitrary geometry."""
    from app.schemas.land_selection import LandCandidate
    from app.schemas.research import EvidenceContent

    item = EvidenceContent.model_validate(content)
    if item.provider != "mapped-asset":
        raise InvalidInputError(
            "Propose an asset using a retained mapped-asset candidate evidence ID."
        )
    payload = json.loads(item.excerpt)
    if not isinstance(payload, dict) or "candidate" not in payload:
        raise InvalidInputError("The saved mapped candidate snapshot is incomplete.")
    candidate = LandCandidate.model_validate(payload["candidate"])
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        digest != item.snapshot_hash
        or str(candidate.source.url) != str(item.url)
        or candidate.id != item.record_id
    ):
        raise InvalidInputError("The mapped candidate snapshot no longer matches its saved source.")
    tags = candidate.properties
    inferred = (
        "power"
        if tags.get("power")
        else "building"
        if tags.get("building")
        else "water"
        if tags.get("waterway")
        else "transport"
        if tags.get("highway") or tags.get("railway")
        else "equipment"
        if candidate.geometry.type == "Point"
        else "other"
    )
    return LandFeatureCreate.model_validate(
        {
            "requestKey": str(request_key),
            "name": name or candidate.label[:200],
            "category": category or inferred,
            "geometry": candidate.geometry.model_dump(mode="json"),
            "source": candidate.source.model_dump(mode="json"),
            "status": "candidate",
            "description": description,
            "attributes": candidate.properties,
            "evidenceIds": [str(evidence_id)],
            "externalRef": {
                "namespace": "openstreetmap",
                "recordId": candidate.id.removeprefix("osm/"),
            },
        }
    )
