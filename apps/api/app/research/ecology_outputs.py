from __future__ import annotations

import uuid

from app.research.providers.base import SourceResult
from app.research.providers.ecology import ECOLOGY_SOURCES
from app.research.providers.taxonomy import SPEC
from app.schemas.research import ArtifactContent, FindingContent, TableColumn, TableOutput


def ecology_outputs(
    result: SourceResult, ids: list[uuid.UUID]
) -> tuple[FindingContent | None, list[ArtifactContent]]:
    if not ids:
        return None, []
    spec = SPEC if result.provider == SPEC.id else ECOLOGY_SOURCES[result.provider]
    data = result.data
    finding = FindingContent(
        title=(f"Name review · {data['query']['scientific_name']}" if spec == SPEC else spec.name),
        summary=result.summary,
        category="ecology",
        evidence_ids=ids,
        confidence="supported" if result.status == "available" else "uncertain",
        uncertainty=str(data["limitation"]),
        suggested_questions=[
            "What field evidence and local reference sites would help define restoration targets?"
        ],
    )
    if spec == SPEC:
        if data["matchType"] != "EXACT" or data["alternatives"]:
            finding.confidence = "uncertain"
        used = data["usage"] or {}
        accepted = data["acceptedUsage"] or (used if used.get("status") == "ACCEPTED" else {})
        columns = [
            ("supplied", "Supplied name"),
            ("matched", "Matched usage"),
            ("accepted", "Accepted usage"),
            ("rank", "Rank"),
            ("matchType", "Match type"),
            ("synonym", "Synonym"),
            ("index", "Matching index"),
        ]
        rows = [
            {
                "supplied": data["query"]["scientific_name"],
                "matched": used.get("name"),
                "accepted": accepted.get("name"),
                "rank": accepted.get("rank") or used.get("rank"),
                "matchType": data["matchType"],
                "synonym": data["synonym"],
                "index": data["indexObserved"]["datasetAlias"],
            }
        ]
        method = (
            "Scientific name lookup against Catalogue of Life Extended Release via GBIF; "
            "the matching-service index metadata was observed immediately before lookup. "
            "The source preserves diagnostics and up to five alternatives. " + data["limitation"]
        )
    elif result.provider == "epa-ecoregions":
        columns = [
            ("US_L4CODE", "Level IV code"),
            ("US_L4NAME", "Level IV region"),
            ("US_L3NAME", "Level III region"),
            ("NA_L1NAME", "Continental context"),
        ]
        rows = data["records"]
        method = data["relation"] + " " + data["limitation"]
    else:
        columns = [
            ("muname", "Soil map unit"),
            ("compname", "Component"),
            ("comppct_r", "Map-unit component (%)"),
            ("ecoclassid", "Reference code"),
            ("ecoclassname", "Reference name"),
            ("ecoclasstypename", "Reference type"),
            ("descriptionUrl", "Candidate description link"),
        ]
        rows = data["records"]
        method = (
            data["limitation"]
            + " Description links are derived from standard identifiers; contents have not been retrieved."
        )
    if data.get("truncated"):
        method += " The bounded source sample is truncated."
    return finding, [
        ArtifactContent(
            title=finding.title,
            method=method,
            evidence_ids=ids,
            output=TableOutput(
                kind="table",
                columns=[TableColumn(key=key, label=label) for key, label in columns],
                rows=[{key: row.get(key) for key, _ in columns} for row in rows],
            ),
        )
    ]
