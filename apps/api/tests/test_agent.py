from __future__ import annotations

import base64
import json
import re
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
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
        clarifications=[
            {
                "id": "deadline",
                "question": "When?",
                "kind": "range",
                "options": [],
                "min": 1,
                "max": 30,
                "step": 1,
                "unit": "days",
                "default": "7",
                "why": "Sets the end date.",
            }
        ],
    )
    settings = Settings(ANTHROPIC_API_KEY="k", ANTHROPIC_MODEL="claude-opus-5", _env_file=None)
    client = _FakeClient(parsed)
    draft = ClaudePlanner(settings, client=client).draft(request("Clear the thistle"))  # type: ignore[arg-type]
    assert draft.source == "claude" and draft.model == "claude-opus-5"
    assert draft.zone_ids == ["Z-14"] and draft.machine_ids == ["TR-04"]
    assert draft.steps[0].machine_ids == ["TR-04"] and draft.steps[0].zone_id is None
    assert draft.steps[0].days == 7 and draft.assumptions == ["1.2 acres/hour"]
    assert draft.clarifications[0].id == "deadline" and draft.clarifications[0].kind == "range"
    answered = ClaudePlanner(settings, client=client).draft(  # type: ignore[arg-type]
        request("Clear the thistle", answers={"deadline": 7})
    )
    assert answered.clarifications == [], "an answered clarification is not asked again"
    assert "deadline" in client.messages.calls[-1]["messages"][0]["content"]
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


def test_rules_planner_asks_structured_questions_and_applies_answers() -> None:
    first = RulesPlanner().draft(request("Survey the north fence"))
    ids = [c.id for c in first.clarifications]
    assert "deadline" in ids and "resolution" in ids and "cadence" in ids
    deadline = next(c for c in first.clarifications if c.id == "deadline")
    assert deadline.kind == "range" and deadline.unit == "days" and deadline.default == 14
    assert all(c.kind != "area" for c in first.clarifications), "no drawn area in scope"

    answered = RulesPlanner().draft(
        request(
            "Survey the north fence",
            answers={"deadline": 3, "resolution": "2", "cadence": "monthly", "crew": "3"},
        )
    )
    assert [c.id for c in answered.clarifications] == []
    assert answered.cadence == "monthly" and answered.end_date is None
    assert answered.estimates.machine_hours == pytest.approx(
        first.estimates.machine_hours * 2, abs=0.2
    )
    assert len(answered.machine_ids) == 3

    tight = RulesPlanner().draft(
        request("Mow Z-14 and Z-21", answers={"deadline": 2, "passes": "two"})
    )
    assert any("do not fit the 2-day deadline" in r for r in tight.risks)
    assert tight.estimates.machine_hours == pytest.approx(
        RulesPlanner()
        .draft(request("Mow Z-14 and Z-21", answers={"deadline": 2}))
        .estimates.machine_hours
        * 2,
        abs=0.2,
    )

    drawn = RulesPlanner().draft(
        request(
            "Survey the lake shore",
            zones=[PlannerZone(id="A-01", name="View area 01", acres=500, task="Treatment")],
            preferred_zone_ids=["A-01"],
        )
    )
    area = next(c for c in drawn.clarifications if c.id == "area")
    assert area.kind == "area" and [o.value for o in area.options][:2] == ["keep", "edit"]


def test_rules_planner_plans_a_3d_scan_with_capture_and_reconstruction() -> None:
    field = PlannerZone(id="A-01", name="View area 01", acres=120, task="Treatment")
    scan = RulesPlanner().draft(
        request("3D scan this field for a splat", zones=[field], preferred_zone_ids=["A-01"])
    )
    titles = [s.title for s in scan.steps]
    assert titles == [
        "Capture plan",
        "Capture A-01 View area 01",
        "Reconstruction",
        "Register and QA",
    ]
    assert not any("Treat" in t for t in titles)
    assert scan.steps[2].machine_ids == [], "reconstruction is compute, not machine time"
    assert any("capture" in a.lower() for a in scan.assumptions)
    ids = [c.id for c in scan.clarifications]
    assert ids[:4] == ["area", "output", "resolution", "views"] and "gcp" in ids
    # 120 acres at 20 ac/h with obliques (x2) = 12 machine-hours.
    assert scan.estimates.machine_hours == pytest.approx(12, abs=0.2)

    fine = RulesPlanner().draft(
        request(
            "3D scan this field for a splat",
            zones=[field],
            preferred_zone_ids=["A-01"],
            answers={"resolution": "1", "views": "nadir", "gcp": "yes", "output": "splat"},
        )
    )
    assert [s.title for s in fine.steps][:2] == ["Capture plan", "Ground control"]
    assert "gaussian splat" in fine.steps[-2].detail
    assert "markers" in fine.steps[-1].detail
    # 120 acres at 6 ac/h nadir only = 20 machine-hours.
    assert fine.estimates.machine_hours == pytest.approx(20, abs=0.2)
    assert not any(c.id in {"resolution", "views", "gcp", "output"} for c in fine.clarifications)

    survey = RulesPlanner().draft(request("Inspect the north fence for erosion"))
    assert survey.steps[0].title == "Survey plan"
    assert not any("Treat" in s.title for s in survey.steps)
    assert survey.steps[-1].title == "Review and report"


def test_outliner_clamps_points_and_labels_source() -> None:
    from app.schemas.agent import OutlinePoint, OutlineRequest
    from app.services.vision import ModelOutline, Outliner

    parsed = ModelOutline(
        points=[
            {"x": -0.1, "y": 0.2},
            {"x": 0.6, "y": 0.2},
            {"x": 0.6, "y": 1.4},
            {"x": 0.1, "y": 0.8},
        ],
        label="  orchard block ",
        confidence=1.7,
        note="Followed the fence lines.",
    )

    class FakeMessages:
        def parse(self, **kwargs: Any) -> Any:
            assert kwargs["output_format"] is ModelOutline
            content = kwargs["messages"][0]["content"]
            assert (
                content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
            )
            assert "x=0.500" in content[1]["text"]
            return type("R", (), {"stop_reason": "end_turn", "parsed_output": parsed})()

    class FakeClient:
        messages = FakeMessages()

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 80).decode()
    outline = Outliner(
        Settings(anthropic_api_key="k", anthropic_model="m"),
        client=FakeClient(),  # type: ignore[arg-type]
    ).outline(
        OutlineRequest(
            image=f"data:image/png;base64,{png}",
            width=800,
            height=600,
            point=OutlinePoint(x=0.5, y=0.5),
            hint="scan the orchard",
        )
    )
    assert outline.source == "claude" and outline.label == "orchard block"
    assert outline.confidence == 1.0
    assert [(p.x, p.y) for p in outline.points] == [(0, 0.2), (0.6, 0.2), (0.6, 1), (0.1, 0.8)]


def test_outline_endpoint_says_when_no_model_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    client = TestClient(create_app())
    png = base64.b64encode(b"\x89PNG" + b"0" * 80).decode()
    response = client.post(
        "/api/v1/agent/outline",
        json={"image": png, "width": 100, "height": 100, "point": {"x": 0.5, "y": 0.5}},
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


# A fence line near Redmond (~300 m east-west) and a ~200 x 100 m paddock, as GeoJSON.
FENCE = {"type": "LineString", "coordinates": [[-122.139, 47.644], [-122.135, 47.644]]}
PADDOCK = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.139, 47.644],
            [-122.136338, 47.644],
            [-122.136338, 47.6449],
            [-122.139, 47.6449],
            [-122.139, 47.644],
        ]
    ],
}


def land_request(goal: str, zones: list[PlannerZone], **overrides: Any) -> PlanDraftRequest:
    return request(goal, zones=zones, machines=[], **overrides)


def test_monitoring_and_checking_are_surveys_while_mowing_along_stays_treatment() -> None:
    from app.services.planner import _task_kind

    assert _task_kind("monitor the fence every week") == "survey"
    assert _task_kind("check the troughs") == "survey"
    assert _task_kind("scout the north pasture") == "survey"
    assert _task_kind("mow along the fence") == "treat"
    assert _task_kind("patrolling the boundary") == "survey"
    # The looking verbs are whole words: a county road, a checkpoint and a country lane do not
    # turn a repair into a survey.
    assert _task_kind("repair the fence along the county road") == "treat"
    assert _task_kind("fix the gate at the checkpoint") == "treat"
    assert _task_kind("grade the country lane") == "treat"


def test_a_line_zone_is_followed_by_the_assumed_drone_and_sized_by_its_length() -> None:
    from app.services.coverage import measure
    from app.services.machine_classes import MACHINE_CLASSES

    fence = PlannerZone(id="L-1a2b3c4d", name="North fence", acres=0.7, geometry=FENCE)
    draft = RulesPlanner().draft(
        land_request("monitor the fence every week", [fence], preferred_zone_ids=["L-1a2b3c4d"])
    )
    length = measure(FENCE, 1)
    assert length is not None
    assert draft.cadence == "weekly" and draft.zone_ids == ["L-1a2b3c4d"]
    assert not any(step.title.startswith("Treat") for step in draft.steps)
    assert all("L-1a2b3c4d" not in step.title for step in draft.steps)
    assert any(step.title == "Survey North fence" for step in draft.steps)
    assert draft.estimates.path_m == pytest.approx(length.length_m, rel=0.001)
    assert draft.estimates.acres == 0
    sized = draft.coverage[0]
    speed = MACHINE_CLASSES.drone.speed_mps
    assert sized.kind == "line" and sized.machine_class == "drone"
    assert sized.hours == pytest.approx(length.length_m / speed / 3600)
    assert "Mapping drone, Mavic 3 Enterprise class (assumed)" in sized.basis
    step = next(s for s in draft.steps if s.title == "Survey North fence")
    assert step.detail == "Follow 301 m of North fence (0.01 machine-hours)."
    path = next(a for a in draft.assumptions if a.startswith("Path from the outline"))
    assert "assumed" in path and "one run" in path and "it repeats weekly" in path


def test_an_outlined_area_takes_its_geodesic_acres_and_the_mower_path() -> None:
    from app.services.coverage import measure

    paddock = PlannerZone(id="A-01", name="Paddock", acres=999, task="Mow", geometry=PADDOCK)
    draft = RulesPlanner().draft(
        land_request("mow the paddock", [paddock], preferred_zone_ids=["A-01"])
    )
    measured = measure(PADDOCK, 2.0)
    assert measured is not None and measured.area_m2 is not None
    # Acres from the outline (~4.9), never the 999 the client stated.
    assert draft.estimates.acres == pytest.approx(measured.area_m2 / 4046.8564224, abs=0.05)
    hours = ((measured.length_m + measured.transit_m) / 1.4 + measured.spans * 8) / 3600
    assert draft.coverage[0].hours == pytest.approx(hours)
    assert draft.coverage[0].machine_class == "mower" and draft.coverage[0].spacing_m == 2.0
    assert draft.estimates.machine_hours == pytest.approx(hours, abs=0.05)
    assert not any("acres per machine-hour assumed" in a for a in draft.assumptions)
    two = RulesPlanner().draft(
        land_request(
            "mow the paddock", [paddock], preferred_zone_ids=["A-01"], answers={"passes": "two"}
        )
    )
    assert two.coverage[0].hours == pytest.approx(2 * hours)
    seeded = RulesPlanner().draft(
        land_request("seed the paddock", [paddock], preferred_zone_ids=["A-01"])
    )
    assert seeded.coverage[0].machine_class == "seeder" and seeded.coverage[0].spacing_m == 3.0


def test_a_learned_rate_still_sets_the_hours_of_an_outlined_area() -> None:
    from app.schemas.agent import PlannerRate

    paddock = PlannerZone(id="A-01", name="Paddock", acres=999, task="Mow", geometry=PADDOCK)
    draft = RulesPlanner().draft(
        land_request(
            "mow the paddock",
            [paddock],
            preferred_zone_ids=["A-01"],
            rates=[PlannerRate(task="Mow", acres_per_machine_hour=2.0, samples=4)],
        )
    )
    area = draft.coverage[0].area_m2
    assert area is not None
    assert draft.coverage[0].hours == pytest.approx(area / 4046.8564224 / 2.0)
    assert "measured rate" in draft.coverage[0].basis
    assert any("learned from 4 logged runs" in a for a in draft.assumptions)


def test_survey_detail_scales_the_drone_but_not_a_ground_rover() -> None:
    paddock = PlannerZone(id="A-01", name="Paddock", acres=999, geometry=PADDOCK)

    def sized(goal: str, detail: str) -> tuple[float, str, float | None]:
        draft = RulesPlanner().draft(
            land_request(
                goal, [paddock], preferred_zone_ids=["A-01"], answers={"resolution": detail}
            )
        )
        c = draft.coverage[0]
        return c.hours, c.basis, c.altitude_m

    flown, flown_basis, altitude = sized("survey the paddock", "2")
    assert flown == pytest.approx(2 * sized("survey the paddock", "5")[0])
    assert "2 times the flying for 2 cm detail" in flown_basis and altitude == 60
    rover, rover_basis, rover_altitude = sized("survey the paddock with the rover", "2")
    assert rover == pytest.approx(sized("survey the paddock with the rover", "5")[0])
    assert "flying" not in rover_basis and rover_altitude is None


def test_a_scan_reports_the_altitude_its_line_spacing_came_from() -> None:
    from app.services.machine_classes import MACHINE_CLASSES

    paddock = PlannerZone(id="A-01", name="Paddock", acres=999, geometry=PADDOCK)
    draft = RulesPlanner().draft(
        land_request(
            "3d scan the paddock",
            [paddock],
            preferred_zone_ids=["A-01"],
            answers={"resolution": "2"},
        )
    )
    c = draft.coverage[0]
    assert c.altitude_m == 70
    assert c.spacing_m == pytest.approx(MACHINE_CLASSES.drone.line_spacing_m(70))


def test_one_invalid_outline_out_of_scope_does_not_block_the_draft(client: TestClient) -> None:
    """A bow-tie left in the project (a corner dragged across an edge) is repaired or
    dropped, never a 422 for every later draft on the project."""
    bow_tie = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.139, 47.644],
                [-122.137, 47.645],
                [-122.137, 47.644],
                [-122.139, 47.645],
                [-122.139, 47.644],
            ]
        ],
    }
    paddock = PlannerZone(id="A-01", name="Paddock", acres=5, task="Mow", geometry=PADDOCK)
    body: dict[str, Any] = {
        "goal": "mow the paddock",
        "projectName": "Oak Bench",
        "zones": [
            paddock.model_dump(by_alias=True, mode="json", exclude_none=True),
            {"id": "A-02", "name": "Old sketch", "acres": 3, "geometry": bow_tie},
            {
                "id": "A-03",
                "name": "Broken",
                "acres": 2,
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]]},
            },
        ],
        "preferredZoneIds": ["A-01"],
        "today": "2026-09-17",
    }
    response = client.post("/api/v1/agent/plan-draft", json=body)
    assert response.status_code == 200, response.text
    if response.json()["source"] == "rules":
        assert [c["zoneId"] for c in response.json()["coverage"]] == ["A-01"]
    zones = PlanDraftRequest.model_validate(body).zones
    # The bow-tie keeps its two triangles; an outline with nothing left falls back to acres.
    assert zones[1].geometry is not None and zones[1].geometry.type == "MultiPolygon"
    assert zones[2].geometry is None


def test_zones_without_outlines_are_sized_exactly_as_before() -> None:
    draft = RulesPlanner().draft(request("Mow Z-21 with one mower"))
    assert draft.coverage == [] and draft.estimates.path_m is None
    assert draft.estimates.machine_hours == pytest.approx(220 / 1.5, abs=0.05)
    assert not any(a.startswith("Path from the outline") for a in draft.assumptions)


def test_claude_planner_sends_paths_not_outlines_and_returns_server_coverage() -> None:
    parsed = ModelPlanDraft(
        title="Weekly fence check",
        objective="Fly the fence weekly.",
        zone_ids=["L-1a2b3c4d"],
        machine_ids=[],
        cadence="weekly",
        start_date="2026-09-18",
        end_date=None,
        estimated_acres=12,
        estimated_machine_hours=0.5,
        estimated_calendar_days=1,
        steps=[],
        assumptions=["8 m/s"],
        risks=[],
        questions=[],
        clarifications=[],
    )
    settings = Settings(ANTHROPIC_API_KEY="k", ANTHROPIC_MODEL="claude-opus-5", _env_file=None)
    client = _FakeClient(parsed)
    fence = PlannerZone(id="L-1a2b3c4d", name="North fence", acres=0.7, geometry=FENCE)
    draft = ClaudePlanner(settings, client=client).draft(  # type: ignore[arg-type]
        land_request("monitor the fence every week", [fence], preferred_zone_ids=["L-1a2b3c4d"])
    )
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert '"geometry"' not in prompt and "-122.139" not in prompt
    assert '"coverage"' in prompt and '"lengthM"' in prompt
    assert draft.coverage[0].zone_id == "L-1a2b3c4d" and draft.coverage[0].kind == "line"
    assert draft.estimates.path_m == pytest.approx(draft.coverage[0].length_m, abs=0.1)
    # A line has no area, so the model's acres stand; its hours always stand.
    assert draft.estimates.acres == 12 and draft.estimates.machine_hours == 0.5


def test_plan_draft_endpoint_accepts_a_previous_draft_with_coverage(client: TestClient) -> None:
    fence = PlannerZone(id="L-1a2b3c4d", name="North fence", acres=0.7, geometry=FENCE)
    body: dict[str, Any] = {
        "goal": "monitor the fence every week",
        "projectName": "Oak Bench",
        "zones": [fence.model_dump(by_alias=True, mode="json", exclude_none=True)],
        "preferredZoneIds": ["L-1a2b3c4d"],
        "today": "2026-09-17",
    }
    first = client.post("/api/v1/agent/plan-draft", json=body)
    assert first.status_code == 200, first.text
    draft = first.json()
    if draft["source"] != "rules":
        return
    assert draft["coverage"][0]["kind"] == "line" and draft["estimates"]["pathM"] > 0
    previous = {k: v for k, v in draft.items() if k not in {"source", "model", "note"}}
    again = client.post(
        "/api/v1/agent/plan-draft",
        json={**body, "refinement": "every month", "previousDraft": previous},
    )
    assert again.status_code == 200, again.text


def test_the_fog_goal_drafts_a_survey_flown_at_the_scout_capture() -> None:
    """ "Lift the fog" proposes scout areas carrying their capture numbers: the goal reads as
    a survey (never a treatment, so it must not say "clear"), and the planner flies each area
    at its own altitude, overlap and speed rather than its detail table."""
    from app.schemas.scout import ScoutCapture
    from app.services.machine_classes import MACHINE_CLASSES
    from app.services.planner import _task_kind

    goal = "Survey the foggy ground from the air at 2 cm per pixel so the fog can lift"
    assert _task_kind(goal) == "survey"
    assert (
        _task_kind("Survey the foggy ground from the air at 1.7 cm per pixel so the fog can lift")
        == "survey"
    )
    assert _task_kind("scout the creek") == "survey"
    capture = ScoutCapture(altitude_m=60, forward_overlap=0.75, side_overlap=0.7, speed_ms=8)
    patch = PlannerZone(id="A-01", name="Fog patch 1", acres=999, geometry=PADDOCK, capture=capture)
    draft = RulesPlanner().draft(
        land_request(goal, [patch], preferred_zone_ids=["A-01"], answers={"resolution": "2"})
    )
    c = draft.coverage[0]
    drone = MACHINE_CLASSES.drone
    assert c.machine_class == "drone" and c.altitude_m == 60 and c.speed_mps == 8
    assert c.spacing_m == pytest.approx(drone.footprint_width_m(60) * 0.3)
    # The capture already says how it is flown: no detail multiplier on top.
    assert "times the flying" not in c.basis and "scout capture" in c.basis
    plain = PlannerZone(id="A-01", name="Fog patch 1", acres=999, geometry=PADDOCK)
    doubled = (
        RulesPlanner()
        .draft(
            land_request(goal, [plain], preferred_zone_ids=["A-01"], answers={"resolution": "2"})
        )
        .coverage[0]
    )
    assert doubled.hours == pytest.approx(2 * c.hours, rel=0.01)


def test_a_name_that_leads_with_its_id_answers_to_the_rest_of_it() -> None:
    """People say "the west grassland", not "Z-14 West grassland", and "Kestrel", not "TR-04
    Kestrel": the rest of a name that leads with its id is matched too, and excluded too."""
    zones = [
        PlannerZone(id="Z-14", name="Z-14 West grassland", acres=414, task="Mow thistle"),
        PlannerZone(id="Z-21", name="Z-21 Fence line", acres=34, task="Mow"),
    ]
    fleet = [
        PlannerMachine(id="TR-04", name="TR-04 Kestrel", model="mower", status="working"),
        PlannerMachine(id="TR-12", name="TR-12 Pipit", model="mower", status="idle"),
    ]
    draft = RulesPlanner().draft(
        request("Mow the west grassland with Kestrel", zones=zones, machines=fleet)
    )
    assert draft.zone_ids == ["Z-14"]
    assert draft.machine_ids == ["TR-04"]
    skipped = RulesPlanner().draft(
        request("Mow everything but skip the fence line", zones=zones, machines=fleet)
    )
    assert skipped.zone_ids == ["Z-14"]
    # A short remainder is not a name: "Z-1 N" does not answer to every "n".
    from app.services.planner import _mentioned

    assert _mentioned("north and south", [("Z-1", "Z-1 N")]) == []


def test_the_default_crew_is_the_machines_that_suit_the_work() -> None:
    """With no count asked for, mowing takes the mowers and drilling the one seed drill, not
    whichever machines happen to be idle; a survey in a fleet with no drone goes to its survey
    scouts. A count asked for still fills up from the rest."""
    fleet = [
        PlannerMachine(
            id="TR-04", name="TR-04 Kestrel", model="Ranger 2 · mower", status="working"
        ),
        PlannerMachine(id="TR-12", name="TR-12 Pipit", model="Ranger 2 · mower", status="working"),
        PlannerMachine(id="TR-11", name="TR-11 Vireo", model="Scout 1 · survey", status="idle"),
        PlannerMachine(id="TR-15", name="TR-15 Towhee", model="Drill 1 · seeder", status="idle"),
    ]
    zone = [PlannerZone(id="Z-14", name="Z-14 West grassland", acres=414, task="Mow thistle")]

    def crew(goal: str, machines: list[PlannerMachine] = fleet) -> list[str]:
        return RulesPlanner().draft(request(goal, zones=zone, machines=machines)).machine_ids

    assert crew("Mow the west grassland") == ["TR-04", "TR-12"]
    assert crew("Drill native seed on the west grassland") == ["TR-15"]
    assert crew("Survey the west grassland every month") == ["TR-11"]
    assert crew("Drill seed on the west grassland with two machines") == ["TR-15", "TR-11"]
    # A drone in the fleet flies the survey, and only machines that may fly fill a flight.
    drone = PlannerMachine(id="UA-01", name="UA-01 Kite", model="Mavic 3E drone", status="idle")
    assert crew("Survey the west grassland every month", [*fleet, drone]) == ["UA-01"]
    assert crew("3D scan the west grassland with two machines", [*fleet, drone]) == ["UA-01"]


DEMO_NAMES = {"Z-14": "Z-14 West grassland", "Z-21": "Z-21 Fence line", "Z-08": "Z-08 Creek draw"}
DEMO_TASKS = {"Z-14": "Mow yellow star thistle", "Z-21": "Buffer + mow queued", "Z-08": "Survey"}
# The simulated fleet as apps/web/src/missions/demo.ts sends it.
DEMO_FLEET = [
    PlannerMachine(id="TR-04", name="TR-04 Kestrel", model="Ranger 2 · mower", status="working"),
    PlannerMachine(id="TR-12", name="TR-12 Pipit", model="Ranger 2 · mower", status="working"),
    PlannerMachine(id="TR-07", name="TR-07 Harrier", model="Ranger 2 · mower", status="attention"),
    PlannerMachine(id="TR-18", name="TR-18 Shrike", model="Scout 1 · survey", status="attention"),
    PlannerMachine(id="TR-11", name="TR-11 Vireo", model="Scout 1 · survey", status="idle"),
    PlannerMachine(id="TR-15", name="TR-15 Towhee", model="Drill 1 · seeder", status="idle"),
]
REPO = Path(__file__).resolve().parents[3]


def demo_ranch() -> tuple[dict[str, Any], list[PlannerZone]]:
    """The demo ranch's derived zones and the planner zones the console sends for them: each
    with its footprint, at its footprint's measured acres."""
    ranch = json.loads((REPO / "apps/api/app/seed/demo_ranch.json").read_text())
    derived = ranch["derived"]["zones"]
    zones = [
        PlannerZone(
            id=zid,
            name=name,
            acres=round(derived[zid]["areaM2"] / 4046.8564224),
            task=DEMO_TASKS[zid],
            treated=zid == "Z-14",
            geometry=derived[zid]["footprint"],
        )
        for zid, name in DEMO_NAMES.items()
    ]
    return derived, zones


def demo_draft(goal: str) -> Any:
    _, zones = demo_ranch()
    return RulesPlanner().draft(
        request(goal, zones=zones, machines=DEMO_FLEET, project_name="Oak Bench Ranch")
    )


def demo_examples() -> list[str]:
    """The example chips the plan composer offers on the demo ranch, read from the console."""
    source = (REPO / "apps/web/src/missions/demo.ts").read_text()
    block = re.search(r"export const DEMO_EXAMPLES = \[(.*?)\];", source, re.S)
    assert block, "DEMO_EXAMPLES moved: point this test at it"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_the_demo_ranch_drafts_from_its_outlines_with_the_assumptions_shown() -> None:
    """The simulated demo's zones now send their outlines (their acres are the footprints'
    measured acres), so its drafts are sized from path length under the assumed machine
    classes, and say so. The numbers stay in a field's range: a 2 m mower at 1.4 m/s does
    about 2.4 acres an hour, a 3 m drill at 1.2 m/s about 3."""
    derived, _ = demo_ranch()
    mow = demo_draft("Mow the star thistle on the west grassland by Friday")
    assert mow.zone_ids == ["Z-14"] and mow.machine_ids == ["TR-04", "TR-12"]
    [c] = mow.coverage
    assert c.machine_class == "mower" and c.area_m2 == pytest.approx(derived["Z-14"]["areaM2"])
    assert mow.estimates.acres == pytest.approx(derived["Z-14"]["areaM2"] / 4046.8564224, abs=0.1)
    assert 2.0 < mow.estimates.acres / mow.estimates.machine_hours < 2.6
    assert "Robotic brush mower (assumed)" in c.basis and "not a field-ready route" in c.basis
    assert any(a.startswith("Path from the outline: Z-14 West grassland") for a in mow.assumptions)
    assert not any("logged run" in a for a in mow.assumptions), "nothing measured is claimed"

    # The 30 m strip inside the fence turns far more often than the open grassland, so its
    # rate is a little lower.
    fence = demo_draft("Mow the fence line")
    assert fence.zone_ids == ["Z-21"] and fence.coverage[0].machine_class == "mower"
    assert 1.7 < fence.estimates.acres / fence.estimates.machine_hours < 2.6

    seed = demo_draft("Drill native seed on the west grassland")
    assert seed.machine_ids == ["TR-15"] and seed.coverage[0].machine_class == "seeder"
    assert 2.5 < seed.estimates.acres / seed.estimates.machine_hours < 3.2
    assert "Seed drill (assumed)" in seed.coverage[0].basis


def test_the_demo_examples_are_sized_for_the_crew_that_does_them() -> None:
    """Every example chip on the demo ranch drafts work its simulated fleet can do, sized for
    the machines it is given. The fleet has no drone: the creek survey goes to the survey
    scouts at the survey rover's assumptions (10 m swath at 1.5 m/s is about 13 acres an hour
    before turns), and the scan, which needs a mapping drone, gets no ground crew and says the
    drone is not in this fleet."""
    from app.services.machine_classes import MACHINE_CLASSES

    mow_goal, survey_goal, scan_goal = demo_examples()
    mow = demo_draft(mow_goal)
    assert mow.machine_ids == ["TR-04", "TR-12"]
    assert {c.machine_class for c in mow.coverage} == {"mower"}

    survey = demo_draft(survey_goal)
    assert survey.zone_ids == ["Z-08"], "the creek survey stays on the creek"
    assert survey.machine_ids == ["TR-11", "TR-18"] and survey.cadence == "monthly"
    [c] = survey.coverage
    rover = MACHINE_CLASSES.survey
    assert c.machine_class == "survey" and c.speed_mps == rover.speed_mps
    assert c.spacing_m == rover.swath_m and c.altitude_m is None
    assert "Survey rover (assumed)" in c.basis and "Mapping drone" not in c.basis
    acres_per_hour = survey.estimates.acres / survey.estimates.machine_hours
    assert 8 < acres_per_hour < 13.4, "a rover's pace, not a drone's"
    assert 3 < survey.estimates.machine_hours < 5
    assert "before driving them" in survey.steps[0].detail
    assert any("by a survey rover (assumed)" in a for a in survey.assumptions)
    assert all(c.id != "resolution" for c in survey.clarifications), "detail is a flight's"

    scan = demo_draft(scan_goal)
    assert scan.machine_ids == [] and all(not s.machine_ids for s in scan.steps)
    assert {c.machine_class for c in scan.coverage} == {"drone"}
    assert all("(assumed), not in this fleet" in c.basis for c in scan.coverage)
    assert any("not in this fleet" in a for a in scan.assumptions)
    assert any(q.startswith("Which drone flies this?") for q in scan.questions)
    assert not any("No machines are assigned" in q for q in scan.questions)
    assert all(c.id != "crew" for c in scan.clarifications), "no count of nothing to ask for"

    # A ground machine named for a flight keeps the operator's choice, with the mismatch said.
    named = demo_draft(f"{scan_goal} with TR-11")
    assert named.machine_ids == ["TR-11"]
    assert any("TR-11 Vireo works on the ground" in r for r in named.risks)
    # Asked for from the air, a survey stays a flight even without a drone to fly it.
    aerial = demo_draft("Survey the creek draw from the air")
    assert aerial.machine_ids == [] and aerial.coverage[0].machine_class == "drone"
