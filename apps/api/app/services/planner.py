"""Mission plan drafting.

The operator states a goal ("clear the star thistle from the west bench before seed set with
two mowers"); the planner turns it, plus the project's zones, machines and existing plans,
into a structured draft the console shows for review. Two providers share one interface:

* ``ClaudePlanner`` calls Claude through the official SDK with a structured output schema.
  It is used whenever ``ANTHROPIC_API_KEY`` is configured.
* ``RulesPlanner`` is a deterministic fallback so the flow works without a key. Every draft
  says which one produced it; a rule-based draft is never presented as a model's.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from typing import Literal, NamedTuple, Protocol

import anthropic
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.schemas.agent import (
    Cadence,
    PlanDraft,
    PlanDraftBody,
    PlanDraftRequest,
    PlanEstimates,
    PlannerStatus,
    PlanStep,
)

# Assumed treatment rate for the rule-based estimate (a mid-size mower on brush); the note on
# every rules draft states it so the number is never taken as measured.
RULES_ACRES_PER_MACHINE_HOUR = 1.5
RULES_HOURS_PER_DAY = 8
RULES_MAX_ZONES = 3

CADENCE_WORDS: dict[Cadence, tuple[str, ...]] = {
    "daily": ("daily", "every day", "each day"),
    "weekly": ("weekly", "every week", "each week"),
    "monthly": ("monthly", "every month", "each month", "once a month", "per month"),
    "seasonal": ("seasonal", "every season", "each season", "per season", "spring and"),
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


class PlannerError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class Planner(Protocol):
    def status(self) -> PlannerStatus: ...

    def draft(self, request: PlanDraftRequest) -> PlanDraft: ...


def _today(request: PlanDraftRequest) -> date:
    return request.today or date.today()


def _mentioned(text: str, candidates: list[tuple[str, str]]) -> list[str]:
    """Ids whose id or name appears in the text (word-bounded, case-insensitive)."""
    lower = text.lower()
    found: list[str] = []
    for identifier, name in candidates:
        for needle in (identifier, name):
            if needle and re.search(rf"(?<![\w-]){re.escape(needle.lower())}(?![\w-])", lower):
                if identifier not in found:
                    found.append(identifier)
                break
    return found


def _requested_count(text: str, nouns: str) -> int | None:
    match = re.search(rf"\b(\d+|{'|'.join(NUMBER_WORDS)})\s+(?:{nouns})\b", text.lower())
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


EXCLUDE_WORDS = r"skip|without|drop|remove|exclude|except|not|no|leave out|leave"


def _excluded(text: str, candidates: list[tuple[str, str]]) -> list[str]:
    """Ids named right after an exclusion word ("skip Z-21", "without TR-07, TR-12")."""
    lower = text.lower()
    out: list[str] = []
    for match in re.finditer(
        rf"\b(?:{EXCLUDE_WORDS})\b((?:\s+(?:and\s+|,\s*)?[\w-]+){{1,4}})", lower
    ):
        for identifier in _mentioned(match.group(1), candidates):
            if identifier not in out:
                out.append(identifier)
    return out


def _cadence(text: str) -> Cadence:
    lower = text.lower()
    for cadence, words in CADENCE_WORDS.items():
        if any(word in lower for word in words):
            return cadence
    return "once"


def _label(identifier: str, name: str) -> str:
    """ "Z-14 West bench" once, whether or not the name already carries the id."""
    return name if name.startswith(identifier) else f"{identifier} {name}"


def _task_family(task: str) -> str:
    """ "Mow pass 2" and "Mow thistle" share the family "mow"."""
    words = re.findall(r"[a-z]+", task.lower())
    return words[0] if words else ""


class Rate(NamedTuple):
    acres_per_hour: float
    note: str


def _rate_for(task: str, request: PlanDraftRequest) -> Rate:
    family = _task_family(task)
    for rate in request.rates:
        if _task_family(rate.task) == family and family:
            who = f" ({', '.join(rate.machine_ids[:3])})" if rate.machine_ids else ""
            return Rate(
                rate.acres_per_machine_hour,
                f"{family.capitalize()}: {rate.acres_per_machine_hour:.1f} acres per machine-hour, "
                f"learned from {rate.samples} logged run{'s' if rate.samples != 1 else ''}{who}.",
            )
    return Rate(
        RULES_ACRES_PER_MACHINE_HOUR,
        f"{family.capitalize() or 'Treatment'}: {RULES_ACRES_PER_MACHINE_HOUR:g} acres per "
        "machine-hour assumed (no logged runs for this task yet).",
    )


def _busy_machines(request: PlanDraftRequest, start: date, end: date) -> dict[str, str]:
    """Machines another plan books inside [start, end) → the title that books them."""
    busy: dict[str, str] = {}
    for plan in request.existing_plans:
        if plan.status in {"done", "cancelled"}:
            continue
        for window in plan.busy:
            if window.start_date < end and start < window.end_date:
                busy.setdefault(window.machine_id, plan.title)
    return busy


def _title(goal: str) -> str:
    text = re.sub(r"\s+", " ", goal.strip()).rstrip(".!")
    text = text[0].upper() + text[1:] if text else "New plan"
    return text if len(text) <= 72 else text[:69].rstrip() + "…"


class RulesPlanner:
    """Deterministic drafter: zones and machines named in the goal (or the operator's
    pre-selection), untreated zones otherwise, idle machines first, a survey / treat / verify
    sequence and an estimate from an assumed treatment rate."""

    def status(self) -> PlannerStatus:
        return PlannerStatus(configured=False, provider="rules", model=None)

    def draft(self, request: PlanDraftRequest) -> PlanDraft:
        text = request.goal if not request.refinement else f"{request.goal}\n{request.refinement}"
        zone_ids = self._zones(request, text)
        machine_ids = self._machines(request, text)
        cadence = (
            _cadence(request.refinement or "")
            if request.refinement and _cadence(request.refinement) != "once"
            else _cadence(text)
        )
        today = _today(request)
        zone_by_id = {z.id: z for z in request.zones}
        rate_by_zone = {z: _rate_for(zone_by_id[z].task, request) for z in zone_ids}
        acres = sum(z.acres for z in request.zones if z.id in zone_ids)
        machine_count = max(1, len(machine_ids))
        hours = sum(
            zone_by_id[z].acres / rate_by_zone[z].acres_per_hour
            for z in zone_ids
            if zone_by_id[z].acres
        )
        days = max(1, math.ceil(hours / (machine_count * RULES_HOURS_PER_DAY))) if hours else 1
        steps = [
            PlanStep(
                title="Survey pass",
                detail="Fly or drive the boundary of every zone in scope and confirm access, "
                "obstacles and current cover before treatment starts.",
                machine_ids=machine_ids[:1],
                zone_id=zone_ids[0] if zone_ids else None,
                when=f"Day 1 · {today.isoformat()}",
                start_day=0,
                days=1,
            )
        ]
        # Zones run in parallel, one crew each, from day 2; a crew's zone takes as many days
        # as its acreage needs at the assumed rate, and a crew with two zones does them in turn.
        last_day = 1
        # Each crew starts its next zone the day after it finishes the previous one.
        crew_free_day = dict.fromkeys(machine_ids, 1)
        for index, zone_id in enumerate(zone_ids):
            zone = zone_by_id[zone_id]
            crew = machine_ids if len(zone_ids) == 1 else [machine_ids[index % machine_count]]
            crew = [m for m in crew if m]
            zone_hours = zone.acres / rate_by_zone[zone_id].acres_per_hour if zone.acres else 0.0
            zone_days = (
                max(1, math.ceil(zone_hours / (max(1, len(crew)) * RULES_HOURS_PER_DAY)))
                if zone_hours
                else 1
            )
            start_day = max([crew_free_day.get(m, 1) for m in crew] or [1])
            for m in crew:
                crew_free_day[m] = start_day + zone_days
            last_day = max(last_day, start_day + zone_days)
            span = f"Day {start_day + 1}" + (f"-{start_day + zone_days}" if zone_days > 1 else "")
            steps.append(
                PlanStep(
                    title=f"Treat {_label(zone.id, zone.name)}",
                    detail=(zone.task or "Treatment")
                    + f" across {zone.acres:g} acres"
                    + (f" ({zone_hours:.0f} machine-hours)." if zone_hours else "."),
                    machine_ids=crew,
                    zone_id=zone.id,
                    when=span,
                    start_day=start_day,
                    days=zone_days,
                )
            )
        steps.append(
            PlanStep(
                title="Verification pass",
                detail="Capture imagery over the treated zones and log residual cover against "
                "the target.",
                machine_ids=machine_ids[:1],
                zone_id=None,
                when=f"Day {last_day + 1}",
                start_day=last_day,
                days=1,
            )
        )
        days = max(days, last_day + 1)
        risks: list[str] = []
        end = today + timedelta(days=days)
        busy = _busy_machines(request, today, end)
        for machine_id in machine_ids:
            if machine_id in busy:
                risks.append(
                    f"{machine_id} is already booked by “{busy[machine_id]}” in this window."
                )
        for machine in request.machines:
            if machine.id in machine_ids and machine.battery_pct < 30:
                risks.append(
                    f"{_label(machine.id, machine.name)} is at {machine.battery_pct}% battery."
                )
            if machine.id in machine_ids and machine.status == "attention":
                risks.append(
                    f"{_label(machine.id, machine.name)} needs attention before it can run."
                )
        for plan in request.existing_plans:
            overlap = [z for z in plan.zone_ids if z in zone_ids]
            if overlap:
                risks.append(f"“{plan.title}” already covers {', '.join(overlap)}.")
        questions: list[str] = []
        if not zone_ids:
            questions.append("Which zones should this plan cover? None matched the goal.")
        if not machine_ids:
            questions.append("No machines are assigned to this project yet.")
        if cadence == "once" and not re.search(r"\b(by|before|until|within)\b", text.lower()):
            questions.append("Is there a deadline? None was given, so the plan starts now.")
        ongoing = cadence != "once"
        return PlanDraft(
            title=_title(request.goal),
            objective=request.goal.strip(),
            zone_ids=zone_ids,
            machine_ids=machine_ids,
            cadence=cadence,
            start_date=today,
            end_date=None if ongoing else today + timedelta(days=days),
            estimates=PlanEstimates(
                acres=round(acres, 1), machine_hours=round(hours, 1), calendar_days=days
            ),
            steps=steps,
            assumptions=[
                *dict.fromkeys(rate.note for rate in rate_by_zone.values()),
                f"{RULES_HOURS_PER_DAY} working hours per machine per day, no weather days.",
                "Zones run in parallel, one crew each; a crew with two zones does them in turn.",
            ],
            risks=risks,
            questions=questions,
            source="rules",
            model=None,
            note="Rule-based draft: no model is configured (set ANTHROPIC_API_KEY).",
        )

    @staticmethod
    def _zones(request: PlanDraftRequest, text: str) -> list[str]:
        candidates = [(z.id, z.name) for z in request.zones]
        excluded = _excluded(text, candidates)
        mentioned = [z for z in _mentioned(text, candidates) if z not in excluded]
        chosen = list(dict.fromkeys([*request.preferred_zone_ids, *mentioned]))
        chosen = [
            z for z in chosen if z not in excluded and any(zone.id == z for zone in request.zones)
        ]
        if not chosen:
            pending = [
                z.id
                for z in request.zones
                if not z.treated and z.progress_pct < 100 and z.id not in excluded
            ]
            chosen = pending[:RULES_MAX_ZONES]
        return chosen

    @staticmethod
    def _machines(request: PlanDraftRequest, text: str) -> list[str]:
        candidates = [(m.id, m.name) for m in request.machines]
        excluded = _excluded(text, candidates)
        mentioned = [m for m in _mentioned(text, candidates) if m not in excluded]
        nouns = "machines?|mowers?|robots?|tractors?|units?|drones?"
        # A count in the refinement ("use three machines") overrides one in the goal.
        wanted = _requested_count(request.refinement or "", nouns) or _requested_count(text, nouns)
        chosen = list(dict.fromkeys([*request.preferred_machine_ids, *mentioned]))
        chosen = [
            m
            for m in chosen
            if m not in excluded and any(machine.id == m for machine in request.machines)
        ]
        if chosen and (wanted is None or len(chosen) >= wanted):
            return chosen
        wanted = wanted or min(2, len(request.machines))
        order = {"idle": 0, "working": 1, "attention": 2}
        today = _today(request)
        # A rough window: nothing is scheduled yet, so "busy in the next two weeks" is the test.
        booked = _busy_machines(request, today, today + timedelta(days=14))
        ranked = sorted(
            request.machines,
            key=lambda m: (m.id in booked, order.get(m.status, 1), -m.battery_pct),
        )
        for machine in ranked:
            if len(chosen) >= wanted:
                break
            if machine.id not in chosen and machine.id not in excluded:
                chosen.append(machine.id)
        return chosen


class ModelPlanDraft(BaseModel):
    """What Claude returns: plain fields only, so the structured-output schema stays simple."""

    title: str
    objective: str
    zone_ids: list[str]
    machine_ids: list[str]
    cadence: Literal["once", "daily", "weekly", "monthly", "seasonal"]
    start_date: str = Field(description="ISO date, YYYY-MM-DD")
    end_date: str | None = Field(description="ISO date or null when the plan is ongoing")
    estimated_acres: float
    estimated_machine_hours: float
    estimated_calendar_days: int
    steps: list[ModelPlanStep]
    assumptions: list[str] = Field(
        description="What the estimate rests on: rates, hours per day, weather, access"
    )
    risks: list[str]
    questions: list[str]


class ModelPlanStep(BaseModel):
    title: str
    detail: str
    machine_ids: list[str]
    zone_id: str | None
    when: str = Field(description="Short human label, e.g. 'Day 2-4' or 'Thu 06:00'")
    start_day: int = Field(description="Offset from the plan start in whole days, 0 = first day")
    days: int = Field(description="Duration in whole days, at least 1")


ModelPlanDraft.model_rebuild()

SYSTEM_PROMPT = """You are the mission planner for a fleet of autonomous land-management \
robots. The operator states a goal; you turn it into one concrete plan for the machines and \
zones of the project you are given.

Use only zone ids and machine ids that exist in the project context. Prefer the zones and \
machines the operator named or pre-selected. Prefer idle machines over working ones, never \
schedule a machine another plan already books in the same days (existingPlans.busy) unless \
the operator named it, and never schedule a machine that needs attention without listing that \
as a risk. Use the learned rates (rates, per task family) for estimates when one matches the \
zone's task; otherwise state the rate you assumed. Keep steps concrete \
and ordered: a survey pass, treatment per zone with the machines assigned, and a verification \
pass. Give estimates from the zone acreage and a realistic treatment rate for the task, and \
list every rate and window you assumed under assumptions. Schedule each step with start_day \
(offset from the plan start) and days (duration) so the steps form a timeline per machine; two \
steps that share a machine must not overlap. Ask a question only when the goal cannot be \
planned without the answer; otherwise make the reasonable choice and note it under risks. \
Dates are ISO (YYYY-MM-DD); end_date is null when the plan repeats on a cadence."""


class ClaudePlanner:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._settings = settings
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=120.0
        )

    def status(self) -> PlannerStatus:
        return PlannerStatus(
            configured=True, provider="claude", model=self._settings.anthropic_model
        )

    def draft(self, request: PlanDraftRequest) -> PlanDraft:
        context = request.model_dump(mode="json", by_alias=True, exclude={"goal", "refinement"})
        context["today"] = _today(request).isoformat()
        parts = [
            f"Project context (JSON):\n{json.dumps(context, indent=1)}",
            f"Operator goal: {request.goal.strip()}",
        ]
        if request.refinement:
            parts.append(
                f"The operator reviewed the previous draft and asks: {request.refinement.strip()}"
            )
        try:
            response = self._client.messages.parse(
                model=self._settings.anthropic_model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "\n\n".join(parts)}],
                output_format=ModelPlanDraft,
            )
        except anthropic.AuthenticationError as exc:
            raise PlannerError(503, "The planning model rejected the configured API key.") from exc
        except anthropic.RateLimitError as exc:
            raise PlannerError(
                429, "The planning model is rate-limited; try again shortly."
            ) from exc
        except anthropic.APIStatusError as exc:
            raise PlannerError(
                502, f"The planning model returned an error ({exc.status_code})."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise PlannerError(502, "The planning model could not be reached.") from exc
        if response.stop_reason == "refusal":
            raise PlannerError(422, "The planning model declined this goal.")
        parsed = response.parsed_output
        if parsed is None:
            raise PlannerError(502, "The planning model returned no plan.")
        return self._to_draft(parsed, request)

    def _to_draft(self, parsed: ModelPlanDraft, request: PlanDraftRequest) -> PlanDraft:
        zone_ids = {z.id for z in request.zones}
        machine_ids = {m.id for m in request.machines}
        today = _today(request)
        body = PlanDraftBody(
            title=parsed.title,
            objective=parsed.objective,
            zone_ids=[z for z in parsed.zone_ids if z in zone_ids],
            machine_ids=[m for m in parsed.machine_ids if m in machine_ids],
            cadence=parsed.cadence,
            start_date=_parse_date(parsed.start_date, today),
            end_date=_parse_date(parsed.end_date, None) if parsed.end_date else None,
            estimates=PlanEstimates(
                acres=max(0.0, parsed.estimated_acres),
                machine_hours=max(0.0, parsed.estimated_machine_hours),
                calendar_days=max(0, parsed.estimated_calendar_days),
            ),
            steps=[
                PlanStep(
                    title=step.title,
                    detail=step.detail,
                    machine_ids=[m for m in step.machine_ids if m in machine_ids],
                    zone_id=step.zone_id if step.zone_id in zone_ids else None,
                    when=step.when,
                    start_day=max(0, step.start_day),
                    days=max(0, step.days),
                )
                for step in parsed.steps
            ],
            assumptions=parsed.assumptions,
            risks=parsed.risks,
            questions=parsed.questions,
        )
        return PlanDraft(
            **body.model_dump(),
            source="claude",
            model=self._settings.anthropic_model,
            note=f"Drafted by Claude ({self._settings.anthropic_model}). Review before approving.",
        )


def _parse_date(value: str, fallback: date | None) -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        if fallback is None:
            raise
        return fallback


def build_planner(settings: Settings | None = None) -> Planner:
    settings = settings or get_settings()
    if settings.anthropic_api_key:
        return ClaudePlanner(settings)
    return RulesPlanner()
