"""The licence scan (scripts/licence_scan.py), run with the pipeline's tests as well as with
scripts/tests: CI path-gates its jobs, and the pipeline job runs for tools/ and infra/modal/
changes, where the markers this scan looks for would come from."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _scan() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "licence_scan", ROOT / "scripts" / "licence_scan.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_repository_is_clean() -> None:
    assert _scan().main(["--root", str(ROOT)]) == 0


def test_the_rules_are_the_scripts() -> None:
    scan = _scan()
    ids = {rule.id for rule in scan.RULES} | {scan.NOTICE_RULE}
    assert ids == {
        "nc-code-marker",
        "nc-derivation",
        "nc-project-in-source",
        "nc-model-id",
        "nc-licence-in-source",
        "withdrawn-asset",
        "notice-names-missing-file",
    }
