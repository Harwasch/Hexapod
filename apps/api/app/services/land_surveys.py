from __future__ import annotations

import hashlib
import json
import uuid
from itertools import combinations

from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_survey import LandSurvey
from app.schemas.geojson import Footprint
from app.schemas.land_surveys import SurveyCreate, SurveyRead, SurveySpeciesSummary, SurveySummary
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import get_land

GEOD = Geod(ellps="WGS84")


def area(boundary: Footprint) -> float:
    geometry = shape(boundary.model_dump())
    assert isinstance(geometry, (Polygon, MultiPolygon))
    parts = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    return sum(abs(GEOD.geometry_area_perimeter(orient(part, sign=1))[0]) for part in parts)


def summarize(boundary: Footprint, payload: SurveyCreate) -> SurveySummary:
    land = shape(boundary.model_dump())
    west, south, east, north = land.bounds
    if east - west > 5 or north - south > 5 or max(abs(south), abs(north)) > 85:
        raise InvalidInputError(
            "Use a regional survey boundary within five degrees and below 85 degrees latitude."
        )
    plots = [shape(p.boundary.model_dump()) for p in payload.plots]
    if any(not land.covers(plot) for plot in plots):
        raise InvalidInputError(
            "Every survey plot must lie inside the pinned land boundary, including its exclusions."
        )
    if any(a.intersection(b).area > 1e-15 for a, b in combinations(plots, 2)):
        raise InvalidInputError(
            "Survey plots must not overlap; overlapping plots would count the same area twice."
        )
    areas = [area(p.boundary) for p in payload.plots]
    total, land_area = sum(areas), area(boundary)
    if min(areas) <= 0 or land_area <= 0:
        raise InvalidInputError("Survey plots need positive measurable surface areas.")
    if payload.design == "census" and abs(total / land_area - 1) > 0.00001:
        raise InvalidInputError(
            "A census must cover the whole pinned boundary. Use a sampling design for partial coverage."
        )
    # Species overlap is meaningful. Never normalize species/strata totals to 100%.
    keys = {
        (o.taxon.strip().casefold(), o.stratum): o.taxon.strip()
        for p in payload.plots
        for o in p.observations
    }
    summaries = []
    for (taxon, stratum), label in sorted(keys.items()):
        values: list[tuple[float, float]] = []
        identifications = set()
        for plot, plot_area in zip(payload.plots, areas, strict=True):
            observation = next(
                (
                    o
                    for o in plot.observations
                    if o.taxon.strip().casefold() == taxon and o.stratum == stratum
                ),
                None,
            )
            if observation is None:
                if plot.complete_inventory:
                    values.append((0.0, plot_area))
                continue
            cover = (
                observation.percent_cover
                if payload.method == "visual-cover"
                else 100 * (observation.hits or 0) / (plot.sample_points or 1)
            )
            assert cover is not None
            values.append((cover, plot_area))
            identifications.add(observation.identification)
        assessed = sum(weight for _, weight in values)
        summaries.append(
            SurveySpeciesSummary(
                taxon=label,
                stratum=stratum,
                identifications=sorted(identifications),
                mean_percent=min(
                    100, max(0, sum(value * weight for value, weight in values) / assessed)
                ),
                minimum_percent=min(value for value, _ in values),
                maximum_percent=max(value for value, _ in values),
                measured_plots=len(values),
                assessed_area_m2=assessed,
                assessed_sample_fraction=assessed / total,
            )
        )
    return SurveySummary(
        land_area_m2=land_area,
        sampled_area_m2=total,
        sampled_fraction=min(1.0, total / land_area),
        plot_areas_m2=areas,
        species=summaries,
        limitations=[
            "These are user-recorded field observations; identification and sampling "
            "quality are not independently verified.",
            "Species and vegetation strata can overlap. Their percentages must not be "
            "summed or normalized into exclusive land-cover classes.",
            "Means are weighted by mapped plot area. Missing taxa are unknown unless the "
            "observer explicitly marked the plot inventory complete for the assessed "
            "strata.",
            "A complete inventory records non-detection, not proof of biological absence. "
            "Survey season, detectability and observer effort affect results.",
            "Sample summaries describe measured plots, not a design-corrected whole-land "
            "estimate or statistical confidence interval. Point-intercept percentages count "
            "points with the taxon, not repeated contacts at a point.",
            "Polygon surface area is geodesic planimetric area, not terrain-adjusted area. "
            "Plot boundaries and positional error affect weighting.",
        ],
    )


def scoped(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, survey_id: uuid.UUID
) -> LandSurvey:
    get_land(db, workspace_id, land_id)
    row = db.scalar(
        select(LandSurvey).where(LandSurvey.land_id == land_id, LandSurvey.id == survey_id)
    )
    if row is None:
        raise NotFoundError("field survey", survey_id)
    return row


def read(db: Session, row: LandSurvey) -> SurveyRead:
    current = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return SurveyRead(
        **row.content,
        id=row.id,
        land_id=row.land_id,
        summary=SurveySummary.model_validate(row.summary),
        sha256=row.sha256,
        recorded_by=row.recorded_by,
        created_at=row.created_at,
        stale=current != row.boundary_revision,
    )


def preview(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: SurveyCreate
) -> SurveySummary:
    get_land(db, workspace_id, land_id)
    snapshot = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == land_id,
            LandBoundaryRevision.revision == payload.boundary_revision,
        )
    )
    if snapshot is None:
        raise NotFoundError("boundary revision", payload.boundary_revision)
    if payload.supersedes_id:
        scoped(db, workspace_id, land_id, payload.supersedes_id)
    from app.services.land import FOOTPRINT

    return summarize(FOOTPRINT.validate_python(snapshot.boundary), payload)


def create(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, principal: str, payload: SurveyCreate
) -> SurveyRead:
    get_land(db, workspace_id, land_id)
    db.execute(select(LandArea.id).where(LandArea.id == land_id).with_for_update())
    identifier = uuid.uuid5(land_id, f"survey/{payload.request_key}")
    content = payload.model_dump(mode="json")
    existing = db.get(LandSurvey, identifier)
    if existing:
        if existing.content != content:
            raise ConflictError(
                "This survey request was already saved with different observations."
            )
        return read(db, existing)
    result = preview(db, workspace_id, land_id, payload)
    summary = result.model_dump(mode="json")
    canonical = json.dumps(
        {"content": content, "summary": summary}, sort_keys=True, separators=(",", ":")
    )
    row = LandSurvey(
        id=identifier,
        land_id=land_id,
        boundary_revision=payload.boundary_revision,
        content=content,
        summary=summary,
        sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        recorded_by=principal,
    )
    db.add(row)
    db.commit()
    return read(db, row)
