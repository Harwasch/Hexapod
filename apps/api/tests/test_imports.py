"""What `import app.main` costs, and what it must not pull in.

The Anthropic SDK was imported at the top of app/services/planner.py and vision.py, which
app.api.deps imports -- so every boot paid ~0.5 s of a ~1.9 s import for a client that only
the agent routes use, and only when ANTHROPIC_API_KEY is set. It is imported where it is
called now. These tests run in a fresh interpreter because this one has usually imported
the SDK already by the time they run (test_agent builds a real planner).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=API_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_booting_the_app_does_not_import_the_model_sdk() -> None:
    """Nor sentry-sdk, which only a deployment with SENTRY_DSN set loads."""
    result = _run(
        """
        import os, sys
        os.environ.pop("SENTRY_DSN", None)
        import app.main
        heavy = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"anthropic", "sentry_sdk"}
        )
        assert not heavy, heavy
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_model_path_still_imports_and_builds_its_client() -> None:
    """Deferred, not dropped: a configured key still gets a real client and the SDK's
    exception types are still the ones the planner and the outliner catch."""
    result = _run(
        """
        import sys
        from app.config import Settings
        from app.services.planner import ClaudePlanner, build_planner
        from app.services.vision import Outliner, build_outliner

        settings = Settings(anthropic_api_key="k", anthropic_model="m")
        assert "anthropic" not in sys.modules
        assert isinstance(build_planner(settings), ClaudePlanner)
        assert isinstance(build_outliner(settings), Outliner)
        import anthropic
        assert isinstance(ClaudePlanner(settings)._client, anthropic.Anthropic)
        """
    )
    assert result.returncode == 0, result.stderr
