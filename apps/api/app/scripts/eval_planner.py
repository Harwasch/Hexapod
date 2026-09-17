"""Planner evaluation: goals over a fixed project context with structural expectations.

Runs every case in ``evals/planner_cases.json`` against the rule-based planner, and against
Claude when ``ANTHROPIC_API_KEY`` is set, and prints one row per case and planner. The
checks are structural (which zones, how many machines, cadence, no double-booked machine,
questions when the goal is under-specified) so they hold for any planner.

    uv run python -m app.scripts.eval_planner            # rules, plus Claude when configured
    uv run python -m app.scripts.eval_planner --rules    # rules only
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas.agent import PlanDraft, PlanDraftRequest
from app.services.planner import ClaudePlanner, Planner, RulesPlanner

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "planner_cases.json"


def load_cases() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CASES_PATH.read_text())
    return data


def request_for(context: dict[str, Any], case: dict[str, Any]) -> PlanDraftRequest:
    payload = {**context, "goal": case["goal"]}
    if case.get("refinement"):
        payload["refinement"] = case["refinement"]
    return PlanDraftRequest.model_validate(payload)


def overlaps(draft: PlanDraft) -> list[str]:
    """Machines booked by two steps at once."""
    spans: dict[str, list[tuple[int, int]]] = {}
    for step in draft.steps:
        for machine in step.machine_ids:
            spans.setdefault(machine, []).append((step.start_day, step.start_day + step.days))
    bad: list[str] = []
    for machine, windows in spans.items():
        windows.sort()
        for (_, end), (start, _) in pairwise(windows):
            if start < end:
                bad.append(machine)
                break
    return bad


def check(draft: PlanDraft, expect: dict[str, Any]) -> list[str]:
    """Names of the expectations the draft fails."""
    failed: list[str] = []
    if "zone_ids" in expect and sorted(draft.zone_ids) != sorted(expect["zone_ids"]):
        failed.append(f"zone_ids={draft.zone_ids}")
    if "zone_ids_subset" in expect and not set(draft.zone_ids) <= set(expect["zone_ids_subset"]):
        failed.append(f"zone_ids_subset={draft.zone_ids}")
    if "min_zones" in expect and len(draft.zone_ids) < expect["min_zones"]:
        failed.append("min_zones")
    if "machine_count" in expect and len(draft.machine_ids) != expect["machine_count"]:
        failed.append(f"machine_count={len(draft.machine_ids)}")
    if "machine_ids" in expect and sorted(draft.machine_ids) != sorted(expect["machine_ids"]):
        failed.append(f"machine_ids={draft.machine_ids}")
    if "machine_ids_exclude" in expect and set(draft.machine_ids) & set(
        expect["machine_ids_exclude"]
    ):
        failed.append(f"machine_ids_exclude={draft.machine_ids}")
    if "cadence" in expect and draft.cadence != expect["cadence"]:
        failed.append(f"cadence={draft.cadence}")
    if "end_date_null" in expect and (draft.end_date is None) != expect["end_date_null"]:
        failed.append(f"end_date={draft.end_date}")
    if "max_questions" in expect and len(draft.questions) > expect["max_questions"]:
        failed.append(f"questions={len(draft.questions)}")
    if "min_questions" in expect and len(draft.questions) < expect["min_questions"]:
        failed.append(f"questions={len(draft.questions)}")
    if "risks_mention" in expect and not any(expect["risks_mention"] in r for r in draft.risks):
        failed.append("risks_mention")
    if "assumptions_mention" in expect and not any(
        expect["assumptions_mention"] in a for a in draft.assumptions
    ):
        failed.append("assumptions_mention")
    if (
        "max_machine_hours" in expect
        and draft.estimates.machine_hours > expect["max_machine_hours"]
    ):
        failed.append(f"machine_hours={draft.estimates.machine_hours}")
    if draft.zone_ids and not draft.steps:
        failed.append("no_steps")
    if bad := overlaps(draft):
        failed.append(f"overlap={bad}")
    return failed


def run(planner: Planner, cases: dict[str, Any]) -> list[tuple[str, list[str]]]:
    results: list[tuple[str, list[str]]] = []
    for case in cases["cases"]:
        draft = planner.draft(request_for(cases["context"], case))
        results.append((case["id"], check(draft, case["expect"])))
    return results


def main(argv: list[str]) -> int:
    cases = load_cases()
    planners: list[tuple[str, Planner]] = [("rules", RulesPlanner())]
    settings = get_settings()
    if "--rules" not in argv and settings.anthropic_api_key:
        planners.append((settings.anthropic_model, ClaudePlanner(settings)))
    exit_code = 0
    for name, planner in planners:
        results = run(planner, cases)
        passed = sum(1 for _, failed in results if not failed)
        sys.stdout.write(f"\n{name}: {passed}/{len(results)} cases pass\n")
        for case_id, failed in results:
            detail = f"  ({', '.join(failed)})" if failed else ""
            sys.stdout.write(f"  {'ok  ' if not failed else 'FAIL'} {case_id}{detail}\n")
        if passed != len(results) and name == "rules":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
