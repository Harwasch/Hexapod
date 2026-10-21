"""This environment's installed packages pass the dependency licence check
(scripts/dependency_licences.py): the pipeline job syncs this venv, so CI reads the licence of
every package it installs, pycolmap included, with no step of its own."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _deps() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "dependency_licences", ROOT / "scripts" / "dependency_licences.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_installed_package_forbids_commercial_use() -> None:
    deps = _deps()
    own = deps.project_name(ROOT, "tools/pipeline")
    assert own == "twin-pipeline"
    argv = ["--root", str(ROOT), "--python", sys.executable, "--skip", own]
    assert deps.main(argv) == 0
