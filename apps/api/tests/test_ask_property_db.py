"""Integrated synthetic bundle -> Ask -> anchored property observations on PostGIS."""

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.api.v1.ask import reasoner
from app.models import AskRun
from app.schemas.ask import Entity
from app.schemas.capture import CaptureCreate
from app.services import ask_entities as ledger
from app.services.ask_reasoner import Reasoner, ScriptedAnswerModel
from app.services.captures import create_capture
from app.services.property_memory import MemoryQuery, PostgresMemory
from tests.conftest import site_payload
from tests.test_ask import REQUEST, events

ROOT = Path(__file__).resolve().parents[3]
# Synthetic control-point placement, explicitly verified for this fixture only.
PLACEMENT = {
    "georef": {
        "lat": 47.64,
        "lon": -122.14,
        "height": 100,
        "uncertaintyM": 0.5,
        "frame": {"headingSource": "capture"},
    },
    "placement": {
        "scale": 1,
        "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "translation": [0, 0, 0],
    },
    "arkitTrainingFrameVerified": True,
    "heightDatum": "ellipsoid-wgs84",
}


class FixtureAnalysis:
    def __init__(self, output: Path) -> None:
        bundle = ROOT / "apps/ios/Fixtures/bundles/yard-small"
        result = subprocess.run(
            [
                str(ROOT / "tools/analysis/.venv/bin/python"),
                "-m",
                "hex_analysis.cli",
                "count_measure",
                "--bundle",
                str(bundle),
                "--prompt",
                "tree",
                "--output",
                str(output),
            ],
            cwd=ROOT / "tools/analysis",
            capture_output=True,
            text=True,
            check=True,
        )
        self.entities = [Entity.model_validate(value) for value in json.loads(result.stdout)]

    async def context(self, request: Any, bundle_url: Any) -> Any:
        return []

    async def run(self, name: Any, arguments: Any, request: Any, bundle_url: Any) -> Any:
        assert name == "count_measure"
        return self.entities


def test_yard_bundle_scan_and_property_questions_save_located_entities(
    client: Any,
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = client.post("/api/v1/sites", json=site_payload()).json()["id"]
    scan = create_capture(
        db,
        CaptureCreate(
            name="Simulated yard-small",
            kind="video",
            site_id=uuid.UUID(site),
            metadata={"entityPlacement": PLACEMENT},
        ),
    )
    analysis = FixtureAnalysis(tmp_path)
    client.app.dependency_overrides[reasoner] = lambda: Reasoner(
        ScriptedAnswerModel(), analysis, PostgresMemory(db)
    )
    monkeypatch.setattr(client.app.state.settings, "ask_reasoner", "mock")
    monkeypatch.setattr(client.app.state.settings, "ask_mode", "live")
    scan_request = {
        **REQUEST,
        "id": "yard-scan",
        "scanId": str(scan.id),
        "question": "How tall is the tree?",
    }
    first = client.post("/v1/ask", json=scan_request)
    assert first.status_code == 200 and events(first)[-1] == {"type": "done"}
    rows = ledger.rows(db, scan)
    assert len(rows) == 1 and rows[0].entity["anchor"]["geo"]["lat"] > 47
    assert rows[0].entity["attrs"]["georeference"]["headingKnown"] is True
    assert events(client.post("/v1/ask", json=scan_request)) == events(first)
    property_request = {
        **REQUEST,
        "id": "yard-property",
        "scope": "property",
        "siteId": site,
        "question": "How many trees?",
    }
    answer = client.post("/v1/ask", json=property_request)
    assert events(answer)[-1] == {"type": "done"}
    assert "1 saved observation" in events(answer)[-2]["text"]
    assert not any(e["type"] == "entity" for e in events(answer))
    assert len(PostgresMemory(db).answers(uuid.UUID(site), MemoryQuery())) == 2
    page = client.get(f"/api/v1/properties/{site}/entities").json()
    assert page["items"][0]["scanId"] == str(scan.id)
    assert page["items"][0]["entity"]["anchor"]["geo"] == rows[0].entity["anchor"]["geo"]
    # Another scan links identity without deleting either observation.
    second = create_capture(
        db,
        CaptureCreate(
            name="Simulated repeat",
            kind="video",
            site_id=uuid.UUID(site),
            metadata={"entityPlacement": PLACEMENT},
        ),
    )
    repeated = ledger.add(db, second, analysis.entities[0])
    db.commit()
    assert repeated.entity["attrs"]["identity"]["canonicalId"] == str(rows[0].id)
    assert len(client.get(f"/api/v1/properties/{site}/entities").json()["items"]) == 2
    # Property recorded replay reads memory too, without requiring a scan frame.
    client.app.dependency_overrides.pop(reasoner)
    client.app.state.settings.ask_mode = "replay"
    replay = client.post("/v1/ask", json={**property_request, "id": "yard-property-replay"})
    assert "2 saved observation" in events(replay)[-2]["text"]
    saved = db.get(AskRun, "yard-property-replay")
    assert saved is not None and saved.state == "complete"
