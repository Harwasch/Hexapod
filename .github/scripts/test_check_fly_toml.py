"""Tests for check_fly_toml.py. Run: python3 -m unittest discover -s .github/scripts"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import Any

from check_fly_toml import commands, problems

ROOT = Path(__file__).resolve().parents[2]


def config(
    *, provision: Any = "false", auto_stop: Any = "suspend", minimum: int = 0
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

    def test_every_value_the_api_reads_as_true_counts(self) -> None:
        # WorldsSettings parses these with pydantic, which reads all of them as True.
        for value in ("1", "yes", "on", "t", "y", "TRUE", "Yes", "ON", True, 1):
            with self.subTest(value=value):
                found = problems(config(provision=value))
                self.assertEqual(len(found), 1)
                self.assertIn("while the API may sleep", found[0])

    def test_every_value_the_api_reads_as_false_passes(self) -> None:
        for value in ("0", "no", "off", "f", "n", "FALSE", False, 0):
            with self.subTest(value=value):
                self.assertEqual(problems(config(provision=value)), [])

    def test_a_value_the_api_cannot_read_fails(self) -> None:
        found = problems(config(provision=" true"))
        self.assertEqual(len(found), 1)
        self.assertIn("is not a boolean", found[0])

    def test_lowercase_key_names_are_read(self) -> None:
        # pydantic-settings matches environment names case-insensitively.
        lowered = config()
        lowered["env"]["world_lambda_allow_provision"] = "yes"
        found = problems(lowered)
        self.assertEqual(len(found), 1)
        self.assertIn("WORLD_LAMBDA_ALLOW_PROVISION", found[0])

    def test_auto_stop_false_is_off(self) -> None:
        self.assertEqual(problems(config(provision="1", auto_stop=False, minimum=1)), [])
        self.assertTrue(problems(config(provision="1", auto_stop=True, minimum=1)))


class LayoutTests(unittest.TestCase):
    def test_kill_timeout_must_be_top_level(self) -> None:
        broken = config()
        del broken["kill_timeout"]
        self.assertEqual(problems(broken), ["kill_timeout is not a top-level key in fly.toml"])

    def test_commands_are_each_process_and_the_release_command(self) -> None:
        self.assertEqual(commands(config()), ["uvicorn", "python", "alembic"])


if __name__ == "__main__":
    unittest.main()
