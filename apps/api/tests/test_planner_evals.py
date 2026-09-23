from __future__ import annotations

from app.scripts.eval_planner import load_cases, run
from app.services.planner import RulesPlanner


def test_rules_planner_passes_every_eval_case() -> None:
    results = run(RulesPlanner(), load_cases())
    failures = {case_id: failed for case_id, failed in results if failed}
    assert not failures, failures
