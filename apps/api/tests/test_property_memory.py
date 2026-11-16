import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.api.property_memory import DedupeRequest, dedupe, router
from app.services.property_memory import (
    MemoryQuery,
    Observation,
    dedupe_candidates,
    entity_query,
    property_scope_site,
    search_property_answers,
    search_property_entities,
)

SITE = uuid.UUID(int=1)


class FakeMemory:
    def entities(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]:
        return [{"id": "oak", "siteId": str(site_id), "label": query.label}]

    def answers(self, site_id: uuid.UUID, query: MemoryQuery) -> list[dict[str, Any]]:
        return [{"siteId": str(site_id), "question": query.label}]


def test_tools_bound_scope_and_fakes() -> None:
    assert property_scope_site({}) is None
    assert property_scope_site({"scope": "property", "siteId": str(SITE)}) == SITE
    with pytest.raises(ValueError):
        property_scope_site({"scope": "property"})
    assert search_property_entities(FakeMemory(), SITE, {"label": "oak"})[0]["label"] == "oak"
    assert (
        search_property_answers(FakeMemory(), SITE, {"label": "how many"})[0]["question"]
        == "how many"
    )
    with pytest.raises(ValidationError):
        search_property_entities(FakeMemory(), SITE, {"siteId": str(uuid.uuid4())})
    with pytest.raises(ValueError):
        search_property_answers(FakeMemory(), SITE, {"attrs": {"color": "red"}})


@pytest.mark.parametrize(
    "args",
    [
        {"lat": 1},
        {"nearest": True},
        {"radius_m": -1},
        {"limit": 501},
        {"since": "2026-01-01"},
        {"since": "2026-02-01T00:00:00Z", "until": "2026-01-01T00:00:00Z"},
        {"attrs": {"a": float("nan")}},
    ],
)
def test_reject_invalid_queries(args: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MemoryQuery.model_validate(args)


def test_bound_sql_spatial_metres_attrs_and_literal_label() -> None:
    query = MemoryQuery(
        label="oak%'_", attrs={"height": 3}, lon=179.999, lat=80, radius_m=20, nearest=True
    )
    sql, params = entity_query(SITE, query)
    assert "ST_DWithin(geo::geography" in sql and "ST_Distance" in sql
    assert "site_id = :site_id" in sql and "attrs @> CAST(:attrs AS jsonb)" in sql
    assert "oak" not in sql and params["label"] == "%oak!%'!_%"
    assert params["site_id"] == SITE


def observation(**changes: Any) -> Observation:
    return Observation.model_validate(
        {
            "id": uuid.UUID(int=3),
            "site_id": SITE,
            "scan_id": uuid.UUID(int=2),
            "label": "Oak Tree",
            "lat": 0,
            "lon": 0,
            "uncertainty_m": 0.5,
            "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
            **changes,
        }
    )


def test_dedupe_cross_scan_ambiguity_and_observation_preservation() -> None:
    new = observation()
    old = observation(
        id=uuid.UUID(int=4),
        scan_id=uuid.UUID(int=5),
        label=" oak  tree ",
        lon=0.000005,
        attrs={"health": "dead"},
    )
    result = dedupe_candidates(new, [old])
    assert result["decision"] == "link" and result["canonicalId"] == str(old.id)
    assert old.attrs == {"health": "dead"} and new.attrs == {}
    assert (
        dedupe_candidates(new, [old, old.model_copy(update={"id": uuid.UUID(int=6)})])["decision"]
        == "review"
    )
    for changes in (
        {"site_id": uuid.UUID(int=9)},
        {"scan_id": new.scan_id},
        {"lon": 0.01},
        {"uncertainty_m": 20},
        {"label": "fence"},
        {"observed_at": new.observed_at + timedelta(days=31)},
    ):
        assert dedupe_candidates(new, [old.model_copy(update=changes)])["decision"] == "new"


def test_independent_router_dedupe_site_check_and_openapi() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    assert "/api/v1/properties/{site_id}/entities" in app.openapi()["paths"]
    payload = DedupeRequest(observation=observation(), candidates=[])
    assert dedupe(SITE, payload)["decision"] == "new"
    with pytest.raises(HTTPException) as error:
        dedupe(uuid.UUID(int=99), payload)
    assert error.value.status_code == 422


def test_dedupe_requires_label_time_and_measured_uncertainty() -> None:
    for changes in (
        {"label": "   "},
        {"uncertainty_m": -1},
        {"uncertainty_m": float("nan")},
        {"observed_at": datetime(2026, 1, 1)},
    ):
        with pytest.raises(ValueError):
            observation(**changes)


def test_router_search_uses_bound_property_and_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.orm import Session

    from app.api import property_memory as routes

    monkeypatch.setattr(routes, "PostgresMemory", lambda db: FakeMemory())
    with Session() as db:
        result = routes.entities(SITE, db, MemoryQuery(limit=1, label="oak"))
        assert result["nextOffset"] == 1 and result["items"][0]["siteId"] == str(SITE)
        assert routes.search(SITE, MemoryQuery(limit=2), db)["nextOffset"] is None
