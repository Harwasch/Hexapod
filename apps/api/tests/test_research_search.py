from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.config import Settings
from app.models.research import ResearchRun
from app.research import queue
from app.research.model import CompleteAction, DecisionResult, ResearchDecision, SearchAction
from app.research.providers.base import SourceContext
from app.research.search import SearchResponse, search_result
from app.research.worker import ResearchWorker
from app.schemas.geojson import Polygon
from app.schemas.research import ResearchBudget
from app.services.errors import InvalidInputError
from tests.test_land import BODY
from tests.test_research import start

URL = "https://www.loc.gov/item/test/"
PAYLOAD = {
    "content": [
        {
            "type": "web_search_tool_result",
            "content": [
                {"type": "web_search_result", "url": URL, "title": "Historic map"},
                {
                    "type": "web_search_result",
                    "url": "https://example.org/lead",
                    "title": "Unquoted lead",
                },
            ],
        },
        {
            "type": "text",
            "text": "An unsupported claim must not become an excerpt.",
            "citations": [
                {"type": "web_search_result_location", "url": URL, "cited_text": "Map dated 1876."},
                {
                    "type": "web_search_result_location",
                    "url": "https://invented.example/",
                    "cited_text": "Invented citation",
                },
            ],
        },
    ]
}


def test_only_verified_result_citations_become_evidence():
    result = search_result(PAYLOAD, "historical maps")
    assert len(result.data["leads"]) == 2
    assert len(result.evidence) == 1
    item = result.evidence[0][1]
    assert str(item.url) == URL and item.excerpt == "Map dated 1876."
    assert item.spatial_relevance == "unresolved"
    assert "unverified" in item.license
    assert "unsupported claim" not in item.excerpt
    assert (
        search_result(
            {
                "content": [
                    {"type": "web_search_tool_result", "content": {"error_code": "unavailable"}}
                ]
            },
            "maps",
        ).status
        == "unavailable"
    )


class Searcher:
    def __init__(self):
        self.calls = 0

    def search(self, query, bounds, max_tokens, max_searches, domains):
        self.calls += 1
        assert max_searches <= 2 and max_tokens <= 2000
        assert len(bounds) == 4
        return SearchResponse(search_result(PAYLOAD, query), 100, 1)


class Model:
    def decide(self, context, max_tokens):
        state = json.loads(context)
        if not state["retrieved"]:
            action = SearchAction(
                kind="search_public_sources", query="Historical maps", domains=["loc.gov"]
            )
        else:
            item = next(iter(state["retrieved"].values()))
            action = CompleteAction(
                kind="complete",
                summary="A historical map is a lead; location remains unresolved.",
                evidence_ids=item["evidenceIds"],
            )
        return DecisionResult(
            ResearchDecision(progress="Looking for archival records", action=action), 100
        )


def test_worker_discovers_evidence_and_preserves_budget_and_citations(client, db, sessions):
    _, inv, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    row.kind = "investigation"
    db.commit()
    searcher = Searcher()
    assert ResearchWorker(
        sessions, Settings(_env_file=None), model=Model(), searcher=searcher
    ).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert searcher.calls == 1 and row.checkpoint["web_searches"] == 1
    assert detail["evidence"][0]["id"] in detail["messages"][-1]["content"]


def test_search_replay_is_idempotent_and_exhausted_budget_blocks_network(client, db, sessions):
    _, _, _run, _ = start(client)
    run_id, token = queue.claim(db)
    context = SourceContext(Polygon.model_validate(BODY["boundary"]))
    searcher = Searcher()
    worker = ResearchWorker(sessions, Settings(_env_file=None), searcher=searcher)
    state = {"sources": {}, "output_tokens": 0, "steps": 1, "actions": []}
    budget = ResearchBudget(max_web_searches=1)
    action = SearchAction(kind="search_public_sources", query="Historic maps")
    worker._search(db, run_id, token, action, context, state, budget)
    worker._search(db, run_id, token, action, context, state, budget)
    assert searcher.calls == 1
    with pytest.raises(InvalidInputError, match="budget"):
        worker._search(
            db,
            run_id,
            token,
            SearchAction(kind="search_public_sources", query="Different query"),
            context,
            state,
            budget,
        )
    assert searcher.calls == 1


def test_checkpoint_updates_nested_state_and_fences_a_cached_run(client, db, sessions):
    start(client)
    run_id, token = queue.claim(db)
    cached = queue.locked(db, run_id, token)
    db.commit()
    state = {"nested": {"tokens": 1}}
    queue.checkpoint(db, run_id, token, state)
    state["nested"]["tokens"] = 200
    queue.checkpoint(db, run_id, token, state)
    with sessions() as other:
        persisted = other.get(ResearchRun, run_id)
        assert persisted.checkpoint == {"nested": {"tokens": 200}}
        persisted.status = "cancelled"
        other.commit()
    assert cached.status == "running"  # The old identity-map instance must not authorize a write.
    with pytest.raises(queue.LeaseLostError):
        queue.checkpoint(db, run_id, token, {"should": "never persist"})
