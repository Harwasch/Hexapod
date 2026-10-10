from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.research import ResearchArtifact, ResearchRun
from app.research import queue
from app.research.calculations import METHOD, calculate, page
from app.research.model import (
    CalculateAction,
    CalculationReadAction,
    CompleteAction,
    DecisionResult,
    ResearchDecision,
)
from app.research.worker import ResearchWorker
from app.schemas.calculations import CalculationRequest
from app.schemas.research import ArtifactContent, DocumentOutput
from app.services.errors import InvalidInputError
from tests.test_research import evidence, start


def recipe(expression: str = "area * rate") -> CalculationRequest:
    return CalculationRequest.model_validate(
        {
            "purpose": "Compare hypothetical planting costs.",
            "limitations": "Synthetic assumptions; excludes maintenance and inflation.",
            "rowLabels": ["No planting", "Small area", "Unmeasured"],
            "inputs": [
                {
                    "name": "area",
                    "label": "Area",
                    "unit": "ha",
                    "values": [0, 2.5, None],
                    "origin": "assumption",
                    "basis": "Hypothetical comparison only.",
                },
                {
                    "name": "rate",
                    "label": "Planting rate",
                    "unit": "USD/ha",
                    "values": [1200],
                    "origin": "question",
                    "basis": "User supplied a hypothetical rate.",
                },
            ],
            "formulas": [
                {"name": "cost", "label": "Planting cost", "unit": "USD", "expression": expression},
                {
                    "name": "reserve",
                    "label": "Including reserve",
                    "unit": "USD",
                    "expression": "cost * 1.1",
                },
            ],
        }
    )


def artifact(request: CalculationRequest) -> ArtifactContent:
    return ArtifactContent(
        title="Cost comparison",
        method="Agent text",
        evidence_ids=[identifier for item in request.inputs for identifier in item.evidence_ids],
        output=calculate(request),
    )


def test_arithmetic_broadcast_dependencies_precision_hash_and_page() -> None:
    output = calculate(recipe())
    assert output.rows[0] == {"cost": 0, "reserve": 0}
    assert output.rows[1] == {"cost": 3000, "reserve": 3300.0000000000005}
    assert output.rows[2] == {"cost": None, "reserve": None}
    assert [issue.kind for issue in output.issues] == ["missing", "missing"]
    assert calculate(CalculationRequest.model_validate_json(recipe().model_dump_json())) == output
    assert calculate(recipe("area * rate + 1")).request_sha256 != output.request_sha256
    second = page(output, 1, 1)
    assert second["labels"] == ["Small area"] and second["nextOffset"] == 2
    assert second["inputs"][0]["values"] == [2.5]
    assert second["inputs"][1]["values"] == [1200]
    assert page(output, 2, 50)["nextOffset"] is None
    for offset, count in [(-1, 1), (3, 1), (0, 0), (0, 51)]:
        with pytest.raises(InvalidInputError):
            page(output, offset, count)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "area.real",
        "area[0]",
        "[area]",
        "True",
        "'text'",
        "lambda: 1",
        "[x for x in range(10)]",
        "1 if area else 0",
        "reserve + 1",
        "missing + 1",
        "max()",
        "round(area, ndigits=2)",
        "min(*area)",
        "1e101",
        "+".join(["area"] * 60),
    ],
)
def test_expression_language_rejects_code_and_unknown_or_future_values(expression: str) -> None:
    with pytest.raises(InvalidInputError):
        calculate(recipe(expression))


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("round(2.5)", 2),
        ("round(2.555, 2)", 2.56),
        ("sum(1, 2, 3)", 6),
        ("sqrt(9) + abs(-2) + floor(2.9) + ceil(0.1)", 8),
        ("max(1, 4) + min(2, 3) + log(8, 2) + log10(100)", 11),
        ("-2 + +3 ** 2 % 5 + 10 // 3", 5),
        ("exp(0)", 1),
    ],
)
def test_registered_arithmetic(expression: str, expected: float) -> None:
    assert calculate(recipe(expression)).rows[0]["cost"] == expected


@pytest.mark.parametrize(
    "expression", ["1 / 0", "sqrt(-1)", "1e100 * 10", "2 ** 101", "round(1, 1.5)", "exp(1000)"]
)
def test_domain_and_range_errors_are_saved_gaps(expression: str) -> None:
    output = calculate(recipe(expression))
    assert output.rows[0]["cost"] is None
    assert output.issues[0].kind == "undefined"
    assert output.issues[1].kind == "missing"


def test_recipe_rejects_dimensions_citation_and_numeric_errors() -> None:
    for patch in [
        {"values": [1, 2]},
        {"values": [True]},
        {"values": [float("nan")]},
        {"values": [float("inf")]},
        {"origin": "evidence"},
        {"evidence_ids": [str(uuid.uuid4())]},
    ]:
        data = recipe().model_dump()
        data["inputs"][0].update(patch)
        with pytest.raises(ValidationError):
            CalculationRequest.model_validate(data)
    data = recipe().model_dump()
    data["inputs"][0]["name"] = "rate"
    with pytest.raises(ValidationError):
        CalculationRequest.model_validate(data)
    data["inputs"][0]["name"] = "sum"
    with pytest.raises(InvalidInputError):
        calculate(CalculationRequest.model_validate(data))
    with pytest.raises(ValidationError):
        ArtifactContent(
            title="Unsupported",
            method="No source",
            evidence_ids=[],
            output=DocumentOutput(kind="document", markdown="Claim"),
        )


def test_saved_calculations_are_recomputed_cited_and_idempotent(
    client: TestClient, db: Session
) -> None:
    start(client)
    claimed = queue.claim(db)
    assert claimed
    run_id, token = claimed
    content = artifact(recipe())
    identifier = queue.save_artifact(db, run_id, token, "calculation", content)
    assert queue.save_artifact(db, run_id, token, "calculation", content) == identifier
    saved = db.get(ResearchArtifact, identifier)
    assert saved and saved.content["method"] == METHOD and saved.content["evidence_ids"] == []
    for field, value in [
        ("rows", [{"cost": 999, "reserve": 0}]),
        ("request_sha256", "0" * 64),
        ("issues", []),
    ]:
        data = content.model_dump()
        data["output"][field] = value
        with pytest.raises(InvalidInputError, match="match the saved recipe"):
            queue.save_artifact(db, run_id, token, "forged", ArtifactContent.model_validate(data))
        db.rollback()
    source = queue.save_evidence(db, run_id, token, "survey", evidence())
    data = recipe().model_dump()
    data["inputs"][0].update(origin="evidence", evidence_ids=[source])
    cited = artifact(CalculationRequest.model_validate(data))
    with pytest.raises(ValidationError):
        ArtifactContent.model_validate({**cited.model_dump(), "evidence_ids": []})
    queue.save_artifact(db, run_id, token, "cited", cited)
    queue.finish(db, run_id, token, "succeeded", "Done")
    start(client)
    other = queue.claim(db)
    assert other
    with pytest.raises(InvalidInputError, match="this investigation"):
        queue.save_artifact(db, *other, "foreign", cited)
    db.rollback()


class CalculationModel:
    def __init__(self, foreign: uuid.UUID) -> None:
        self.foreign = foreign

    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        state = json.loads(context)
        actions = state["previousActions"]
        assert all(item["id"] != str(self.foreign) for item in state["savedCalculations"])
        action: CalculateAction | CalculationReadAction | CompleteAction
        if not actions:
            action = CalculationReadAction(kind="read_calculation", artifact_id=self.foreign)
        elif len(actions) == 1:
            assert "this investigation only" in actions[-1]["result"]
            action = CalculateAction(
                kind="calculate", title="Hypothetical planting comparison", request=recipe()
            )
        elif len(actions) == 2:
            result = json.loads(actions[-1]["result"])
            assert result["page"]["rows"][1]["cost"] == 3000
            action = CalculationReadAction(
                kind="read_calculation", artifact_id=result["artifactId"], offset=2, count=1
            )
        else:
            result = json.loads(actions[-1]["result"])
            assert result["rows"] == [{"cost": None, "reserve": None}]
            action = CompleteAction(
                kind="complete",
                summary="Hypothetical planting costs calculated; unmeasured area remains unknown.",
            )
        return DecisionResult(
            ResearchDecision(progress="Comparing supplied assumptions", action=action), 200
        )


def test_worker_saves_and_pages_calculations_with_scope_checks(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    start(client)
    claimed = queue.claim(db)
    assert claimed
    foreign = queue.save_artifact(db, *claimed, "foreign", artifact(recipe()))
    queue.finish(db, *claimed, "succeeded", "Done")
    _, inv, run, _ = start(client)
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row
    row.kind = "investigation"
    db.commit()
    assert ResearchWorker(
        sessions, Settings(_env_file=None), model=CalculationModel(foreign)
    ).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert detail["artifacts"][0]["output"]["rows"][1]["cost"] == 3000
    assert len(detail["artifacts"][0]["output"]["issues"]) == 2
    assert detail["artifacts"][0]["method"] == METHOD
