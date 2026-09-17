from __future__ import annotations

from datetime import date
from itertools import pairwise
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.schemas.agent import PlanDraftRequest, PlannerMachine, PlannerZone
from app.services.planner import ClaudePlanner, ModelPlanDraft, RulesPlanner, build_planner

ZONES = [
    PlannerZone(id="Z-14", name="West bench", acres=310, task="Mow thistle", progress_pct=40),
    PlannerZone(id="Z-21", name="Z-21 North fence", acres=220, task="Mow thistle", progress_pct=0),
    PlannerZone(
        id="Z-08", name="Corridor", acres=90, task="Inspect", treated=True, progress_pct=100
    ),
]
MACHINES = [
    PlannerMachine(id="TR-04", name="Kestrel", status="idle", battery_pct=88),
    PlannerMachine(id="TR-07", name="Harrier", status="attention", battery_pct=22),
    PlannerMachine(id="TR-12", name="Merlin", status="working", battery_pct=64),
]


def request(goal: str, **overrides: Any) -> PlanDraftRequest:
    fields: dict[str, Any] = {
        "goal": goal,
        "project_name": "Blackrock Mesa",
        "machines": MACHINES,
        "zones": ZONES,
        "today": date(2026, 9, 17),
    }
    fields.update(overrides)
    return PlanDraftRequest(**fields)


def test_rules_planner_uses_named_zones_and_machine_count() -> None:
    draft = RulesPlanner().draft(
        request("Clear the star thistle from Z-14 and Z-21 with two mowers by Friday")
    )
    assert draft.source == "rules"
    assert draft.zone_ids == ["Z-14", "Z-21"]
    assert draft.machine_ids == ["TR-04", "TR-12"], "idle first, then working, never attention"
    assert draft.estimates.acres == 530
    assert draft.cadence == "once" and draft.end_date is not None
    assert [s.zone_id for s in draft.steps[1:3]] == ["Z-14", "Z-21"]
    assert [s.title for s in draft.steps[1:3]] == [
        "Treat Z-14 West bench",
        "Treat Z-21 North fence",
    ]
    assert not draft.questions, "a deadline was given and zones matched"


def test_rules_planner_defaults_to_untreated_zones_and_flags_attention() -> None:
    draft = RulesPlanner().draft(
        request("Inspect the fence line monthly", preferred_machine_ids=["TR-07"])
    )
    assert draft.zone_ids == ["Z-14", "Z-21"], "treated zones are skipped"
    assert draft.machine_ids == ["TR-07"], "an operator pre-selection wins"
    assert draft.cadence == "monthly" and draft.end_date is None
    assert any("attention" in risk for risk in draft.risks)
    assert any("22%" in risk for risk in draft.risks)


def test_rules_planner_asks_when_nothing_matches() -> None:
    draft = RulesPlanner().draft(request("Do something", zones=[], machines=[]))
    assert draft.zone_ids == [] and draft.machine_ids == []
    assert len(draft.questions) >= 2


def test_build_planner_without_key_is_rules() -> None:
    planner = build_planner(Settings(_env_file=None))
    assert planner.status().provider == "rules"
    assert (
        build_planner(Settings(ANTHROPIC_API_KEY="k", _env_file=None)).status().provider == "claude"
    )


class _FakeMessages:
    def __init__(self, parsed: ModelPlanDraft) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class Response:
            stop_reason = "end_turn"
            parsed_output = self.parsed

        return Response()


class _FakeClient:
    def __init__(self, parsed: ModelPlanDraft) -> None:
        self.messages = _FakeMessages(parsed)


def test_claude_planner_filters_unknown_ids_and_labels_source() -> None:
    parsed = ModelPlanDraft(
        title="Clear thistle on the west bench",
        objective="Mow Z-14 with TR-04 at 1.2 acres/hour.",
        zone_ids=["Z-14", "Z-99"],
        machine_ids=["TR-04", "TR-00"],
        cadence="once",
        start_date="2026-09-18",
        end_date="2026-09-25",
        estimated_acres=310,
        estimated_machine_hours=258,
        estimated_calendar_days=7,
        steps=[
            {
                "title": "Mow",
                "detail": "Mow the bench",
                "machine_ids": ["TR-04", "TR-00"],
                "zone_id": "Z-99",
                "when": "Day 1",
                "start_day": 0,
                "days": 7,
            }
        ],
        assumptions=["1.2 acres/hour"],
        risks=[],
        questions=[],
    )
    settings = Settings(ANTHROPIC_API_KEY="k", ANTHROPIC_MODEL="claude-opus-5", _env_file=None)
    client = _FakeClient(parsed)
    draft = ClaudePlanner(settings, client=client).draft(request("Clear the thistle"))  # type: ignore[arg-type]
    assert draft.source == "claude" and draft.model == "claude-opus-5"
    assert draft.zone_ids == ["Z-14"] and draft.machine_ids == ["TR-04"]
    assert draft.steps[0].machine_ids == ["TR-04"] and draft.steps[0].zone_id is None
    assert draft.steps[0].days == 7 and draft.assumptions == ["1.2 acres/hour"]
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5" and "Z-14" in call["messages"][0]["content"]


def test_plan_draft_endpoint(client: TestClient) -> None:
    body = {
        "goal": "Mow Z-21 weekly with one mower",
        "projectName": "Blackrock Mesa",
        "zones": [z.model_dump(by_alias=True) for z in ZONES],
        "machines": [m.model_dump(by_alias=True) for m in MACHINES],
        "today": "2026-09-17",
    }
    response = client.post("/api/v1/agent/plan-draft", json=body)
    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["source"] in {"rules", "claude"}
    assert draft["zoneIds"] == ["Z-21"] if draft["source"] == "rules" else True
    assert client.get("/api/v1/agent/status").json()["provider"] in {"rules", "claude"}
    assert client.post("/api/v1/agent/plan-draft", json={"goal": "x"}).status_code == 422


def test_rules_planner_refinement_overrides_count_and_exclusions() -> None:
    zones = [*ZONES, PlannerZone(id="Z-08", name="Draw", acres=90, task="Survey", progress_pct=10)]
    draft = RulesPlanner().draft(
        request(
            "Clear Z-14 and Z-21 with two mowers, then survey Z-08",
            zones=zones,
            refinement="use three machines and skip Z-08",
        )
    )
    assert draft.zone_ids == ["Z-14", "Z-21"]
    assert len(draft.machine_ids) == 3
    # A crew with two zones does them one after the other, never in parallel.
    by_machine: dict[str, list[tuple[int, int]]] = {}
    for step in draft.steps[1:-1]:
        for m in step.machine_ids:
            by_machine.setdefault(m, []).append((step.start_day, step.start_day + step.days))
    for spans in by_machine.values():
        spans.sort()
        for (_, end), (start, _) in pairwise(spans):
            assert start >= end


def test_rules_planner_sequences_one_crew_over_two_zones() -> None:
    draft = RulesPlanner().draft(request("Mow Z-14 and Z-21 with TR-04"))
    assert draft.machine_ids == ["TR-04"]
    first, second = draft.steps[1], draft.steps[2]
    assert second.start_day == first.start_day + first.days
    assert draft.steps[-1].start_day == second.start_day + second.days


def test_rules_planner_uses_learned_rates_and_avoids_booked_machines() -> None:
    from app.schemas.agent import BusyWindow, PlannerExistingPlan, PlannerRate

    draft = RulesPlanner().draft(
        request(
            "Mow Z-21 with one mower",
            rates=[
                PlannerRate(
                    task="Mow pass 2", acres_per_machine_hour=11.0, samples=6, machine_ids=["TR-04"]
                )
            ],
            existing_plans=[
                PlannerExistingPlan(
                    id="p1",
                    title="Corridor sweep",
                    zone_ids=["Z-08"],
                    status="dispatched",
                    busy=[
                        BusyWindow(
                            machine_id="TR-04",
                            start_date=date(2026, 9, 17),
                            end_date=date(2026, 9, 30),
                        )
                    ],
                )
            ],
        )
    )
    # 220 acres at 11 acres/hour = 20 machine-hours, not 147 at the default rate.
    assert draft.estimates.machine_hours == 20
    assert any("learned from 6 logged runs" in a for a in draft.assumptions)
    # TR-04 is idle but booked; TR-12 (working, free) is chosen instead.
    assert draft.machine_ids == ["TR-12"]
    named = RulesPlanner().draft(
        request(
            "Mow Z-21 with TR-04",
            existing_plans=[
                PlannerExistingPlan(
                    id="p1",
                    title="Corridor sweep",
                    status="scheduled",
                    busy=[
                        BusyWindow(
                            machine_id="TR-04",
                            start_date=date(2026, 9, 17),
                            end_date=date(2026, 9, 30),
                        )
                    ],
                )
            ],
        )
    )
    assert named.machine_ids == ["TR-04"], "an operator's choice stands"
    assert any("already booked" in r for r in named.risks)
