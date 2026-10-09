"""Bounded reads of immutable scenario revisions for agent follow-ups."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scenario import LandScenarioRevision
from app.services import scenarios
from app.services.errors import InvalidInputError, NotFoundError

Section = Literal["overview", "cover", "treatments", "species-targets", "species-results"]


def read(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    scenario_id: uuid.UUID,
    revision: int | None,
    section: Section,
    offset: int,
    count: int,
) -> dict[str, Any]:
    row = scenarios.scoped(db, workspace_id, land_id, scenario_id)
    snapshot = db.scalar(
        select(LandScenarioRevision).where(
            LandScenarioRevision.scenario_id == row.id,
            LandScenarioRevision.revision == (revision if revision is not None else row.revision),
        )
    )
    if snapshot is None:
        raise NotFoundError("scenario revision", revision)
    data = scenarios.read(db, row, snapshot).model_dump(mode="json")
    inputs, result = data["inputs"], data["result"]
    ecology = inputs.get("ecology") or {}
    digest = hashlib.sha256(
        json.dumps(
            {"payload": snapshot.payload, "result": snapshot.result}, sort_keys=True
        ).encode()
    ).hexdigest()
    answer: dict[str, Any] = {
        "scenarioId": str(row.id),
        "revision": snapshot.revision,
        "boundaryRevision": snapshot.boundary_revision,
        "stale": data["stale"],
        "name": data["name"],
        "snapshotSha256": digest,
        "section": section,
        "counts": {
            "cover": len(inputs.get("cover", [])),
            "treatments": len(inputs.get("treatments", [])),
            "species-targets": len(ecology.get("species_targets", [])),
            "species-results": len((result.get("ecology") or {}).get("species", [])),
        },
        "interpretation": "Saved scenario assumptions and calculation outputs, not new measurements. "
        "Source reference IDs remain attached to their original investigations; validate citations before publishing.",
    }
    if section == "overview":
        answer["data"] = {
            "inputs": {
                key: value
                for key, value in inputs.items()
                if key not in {"ecology", "cover", "treatments"}
            },
            "ecology": {key: value for key, value in ecology.items() if key != "species_targets"},
            "summary": result["summary"],
            "limitations": result["limitations"],
            "fieldSurveyIds": data["field_survey_ids"],
            "evidenceIds": data["evidence_ids"],
            "solarAssessmentId": data["solar_assessment_id"],
        }
    else:
        records = (
            ecology.get("species_targets", [])
            if section == "species-targets"
            else (result.get("ecology") or {}).get("species", [])
            if section == "species-results"
            else inputs.get(section, [])
        )
        answer["data"] = records[offset : offset + count]
        answer["offset"] = offset
        while len(json.dumps(answer, ensure_ascii=False)) > 28_500 and answer["data"]:
            answer["data"].pop()
        if records[offset:] and not answer["data"]:
            raise InvalidInputError("This scenario row exceeds the bounded agent read size.")
        following = offset + len(answer["data"])
        answer["nextOffset"] = following if following < len(records) else None
    if len(json.dumps(answer, ensure_ascii=False)) > 29_000:
        raise InvalidInputError("This scenario section exceeds the bounded agent read size.")
    return answer
