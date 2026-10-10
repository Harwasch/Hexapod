from __future__ import annotations

import uuid

from app.research.ecology_outputs import ecology_outputs
from app.research.providers.archives import ARCHIVE_SOURCES
from app.research.providers.base import SourceResult
from app.research.providers.ecology import ECOLOGY_SOURCES
from app.research.providers.open_data import SOURCES
from app.schemas.research import (
    ArtifactContent,
    ChartOutput,
    ChartSeries,
    FindingContent,
    GalleryOutput,
    MapFeature,
    MapOutput,
    TableColumn,
    TableOutput,
)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def overview_outputs(
    result: SourceResult, ids: list[uuid.UUID]
) -> tuple[FindingContent | None, list[ArtifactContent]]:
    if result.provider in ECOLOGY_SOURCES:
        return ecology_outputs(result, ids)
    if not ids or result.status != "available":
        return None, []
    category = SOURCES[result.provider].domain
    finding = FindingContent.model_validate(
        {
            "title": SOURCES[result.provider].name,
            "summary": result.summary,
            "category": category,
            "evidence_ids": ids,
            "confidence": "supported",
            "uncertainty": result.evidence[0][1].relevance_note,
            "suggested_questions": {
                "usgs-elevation": ["How do slope and drainage vary across this land?"],
                "nasa-power": [
                    "What solar generation could this land support under my assumptions?"
                ],
                "gbif-occurrences": [
                    "Which observations are recent and which need field verification?"
                ],
                "usda-soils": ["What field tests would check these mapped soil properties?"],
                "fema-flood-zones": ["What effective maps and amendments apply to this land?"],
            }.get(result.provider, []),
        }
    )
    artifacts: list[ArtifactContent] = []
    if result.provider in ARCHIVE_SOURCES:
        finding.suggested_questions = [
            "What do these sources show about the land, and which dates and locations still need verification?"
        ]
        artifacts.append(
            ArtifactContent(
                title=SOURCES[result.provider].name,
                method=str(result.data.get("coverageNote", "Bounded public catalog search."))
                + " Catalog metadata and file-version references are saved; remote preview images may change. "
                "Coordinates are catalog locations and sheet footprints, not verified image georeferencing.",
                evidence_ids=ids,
                output=GalleryOutput(kind="gallery", evidence_ids=ids),
            )
        )
    if result.provider == "nasa-power":
        for key, values in result.data["parameters"].items():
            definition = result.data["definitions"].get(key, {})
            title = definition.get("longname", key)
            unit = definition.get("units", "source units")
            artifacts.append(
                ArtifactContent(
                    title=title,
                    method=(
                        "NASA POWER monthly climatology at a representative point. "
                        f"Period: {result.data['period']}. Regional estimates; not local measurements."
                    ),
                    evidence_ids=ids,
                    output=ChartOutput(
                        kind="chart",
                        chart_type="bar",
                        x_label="Month",
                        y_label=title,
                        unit=unit,
                        labels=MONTHS,
                        series=[
                            ChartSeries(label=title, values=[values.get(month) for month in MONTHS])
                        ],
                    ),
                )
            )
    if result.provider == "gbif-occurrences":
        records = result.data["records"]
        artifacts.append(
            ArtifactContent(
                title="Biodiversity observation sample",
                method=(
                    f"First {result.data['sampleSize']} matching bounding-box records, "
                    "filtered to polygon and compatible licenses. Generalized/withheld records excluded. "
                    "This sample is not a species inventory. Coordinates are not displayed."
                ),
                evidence_ids=ids,
                output=TableOutput(
                    kind="table",
                    columns=[
                        TableColumn(key="species", label="Recorded taxon"),
                        TableColumn(key="date", label="Observation date"),
                        TableColumn(
                            key="coordinateUncertaintyM", label="Position uncertainty", unit="m"
                        ),
                        TableColumn(key="basis", label="Record type"),
                        TableColumn(key="publisher", label="Publisher"),
                    ],
                    rows=[
                        {
                            k: r.get(k)
                            for k in (
                                "species",
                                "date",
                                "coordinateUncertaintyM",
                                "basis",
                                "publisher",
                            )
                        }
                        for r in records
                    ],
                ),
            )
        )

    if result.provider == "usda-soils":
        artifacts.append(
            ArtifactContent(
                title="Soil map-unit components at the sample point",
                method="Major components of the soil survey map unit at a representative point; map-unit "
                "proportions are not land-wide measurements.",
                evidence_ids=ids,
                output=TableOutput(
                    kind="table",
                    columns=[
                        TableColumn(key="muname", label="Mapped soil unit"),
                        TableColumn(key="compname", label="Component"),
                        TableColumn(key="comppct_r", label="Map-unit component", unit="%"),
                        TableColumn(key="drainagecl", label="Drainage class"),
                        TableColumn(key="hydgrp", label="Hydrologic group"),
                        TableColumn(
                            key="slope_r", label="Representative component slope", unit="%"
                        ),
                    ],
                    rows=result.data["records"],
                ),
            )
        )
    if result.provider == "fema-flood-zones":
        records = result.data["records"]
        artifacts.append(
            ArtifactContent(
                title="Flood zones intersecting the land",
                method="Returned FEMA NFHL polygons clipped to the pinned land boundary. Geodesic "
                "intersection areas, WGS 84. Polygons may overlap; do not sum them as total coverage.",
                evidence_ids=ids,
                output=MapOutput(
                    kind="map",
                    unit="m²",
                    legend="Mapped zone intersections; not a forecast or complete flood-risk assessment.",
                    features=[
                        MapFeature(
                            label=f"Zone {row['zone']} · {row['subtype']}",
                            geometry=row["geometry"],
                            value=row["intersectedAreaM2"],
                        )
                        for row in records
                    ],
                ),
            )
        )
        artifacts.append(
            ArtifactContent(
                title="Flood-map intersection records",
                method="Each row is one returned mapped polygon clipped to the land; overlaps and incomplete "
                "coverage are possible.",
                evidence_ids=ids,
                output=TableOutput(
                    kind="table",
                    columns=[
                        TableColumn(key="zone", label="Zone"),
                        TableColumn(key="subtype", label="Zone description"),
                        TableColumn(key="specialFloodHazard", label="Special flood hazard area"),
                        TableColumn(key="intersectedAreaM2", label="Intersection area", unit="m²"),
                        TableColumn(key="mapId", label="FIRM database"),
                        TableColumn(key="recordId", label="Flood-area record"),
                    ],
                    rows=[
                        {key: value for key, value in row.items() if key != "geometry"}
                        for row in records
                    ],
                ),
            )
        )
    return finding, artifacts
