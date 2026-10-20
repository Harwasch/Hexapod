"""Tests for check_fly_toml.py. Run: python3 -m unittest discover -s .github/scripts"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import Any

from check_fly_toml import commands, problems

ROOT = Path(__file__).resolve().parents[2]


def config(
    *, provision: str = "false", auto_stop: str = "suspend", minimum: int = 0
) -> dict[str, Any]:
    return {
        "kill_timeout": "30s",
        "env": {
            "WORLD_RUNPOD_ALLOW_PROVISION": "false",
            "WORLD_MODAL_ALLOW_PROVISION": provision,
        },
        "http_service": {"auto_stop_machines": auto_stop, "min_machines_running": minimum},
        "processes": {"app": "uvicorn app.main:app", "worker": "python -m app.worker"},
        "deploy": {"release_command": "alembic upgrade head"},
    }


class SleepingApiTests(unittest.TestCase):
    def test_the_committed_fly_toml_passes(self) -> None:
        with open(ROOT / "fly.toml", "rb") as handle:
            committed = tomllib.load(handle)
        self.assertEqual(problems(committed), [])

    def test_a_sleeping_api_without_paid_provisioning_passes(self) -> None:
        self.assertEqual(problems(config()), [])

    def test_paid_provisioning_while_the_api_may_suspend_fails(self) -> None:
        found = problems(config(provision="true"))
        self.assertEqual(len(found), 1)
        self.assertIn("WORLD_MODAL_ALLOW_PROVISION", found[0])

    def test_paid_provisioning_while_the_api_may_stop_fails(self) -> None:
        self.assertTrue(problems(config(provision="true", auto_stop="stop", minimum=1)))

    def test_paid_provisioning_with_no_machine_kept_running_fails(self) -> None:
        self.assertTrue(problems(config(provision="true", auto_stop="off", minimum=0)))

    def test_paid_provisioning_on_an_always_on_api_passes(self) -> None:
        self.assertEqual(problems(config(provision="true", auto_stop="off", minimum=1)), [])


class LayoutTests(unittest.TestCase):
    def test_kill_timeout_must_be_top_level(self) -> None:
        broken = config()
        del broken["kill_timeout"]
        self.assertEqual(problems(broken), ["kill_timeout is not a top-level key in fly.toml"])

    def test_commands_are_each_process_and_the_release_command(self) -> None:
        self.assertEqual(commands(config()), ["uvicorn", "python", "alembic"])


if __name__ == "__main__":
    unittest.main()
