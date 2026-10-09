from __future__ import annotations

import uuid

from app.research.providers.base import SourceResult
from app.research.providers.open_data import SOURCES
from app.schemas.research import (
    ArtifactContent,
    ChartOutput,
    ChartSeries,
    FindingContent,
    TableColumn,
    TableOutput,
)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def overview_outputs(
    result: SourceResult, ids: list[uuid.UUID]
) -> tuple[FindingContent | None, list[ArtifactContent]]:
    if not ids or result.status != "available":
        return None, []
    category = {
        "usgs-elevation": "physical",
        "nasa-power": "energy",
        "gbif-occurrences": "ecology",
    }[result.provider]
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
            }[result.provider],
        }
    )
    artifacts: list[ArtifactContent] = []
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
    return finding, artifacts
