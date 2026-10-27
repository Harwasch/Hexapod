"""Production corrections' dispatch boundary, with its own step scripts run against a fake
flyctl: what reaches the shell line, and what counts as success."""

import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/production-corrections.yml"
EPOCH = "a3474be6-50b8-4e96-974d-aa1c209b8d8e"
SITE = "123e90c7-0f93-4a66-9d6e-0f84ec3a797c"
MARKER = "CORRECTION-EXIT-0"


class ProductionCorrectionsTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.steps = {step.get("id"): step for step in self.workflow["jobs"]["correct"]["steps"]}
        self.tmp = Path(tempfile.mkdtemp())

    def run_step(self, step_id, env, flyctl=None):
        """The step's own `run:` under bash, as Actions runs it, in a scratch directory."""
        environment = {"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(self.tmp / "output")}
        if flyctl is not None:
            bin_dir = self.tmp / "bin"
            bin_dir.mkdir(exist_ok=True)
            fake = bin_dir / "flyctl"
            fake.write_text(flyctl)
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
        environment.update(env)
        (self.tmp / "output").write_text("")
        result = subprocess.run(
            ["bash", "-e", "-c", self.steps[step_id]["run"]],
            cwd=self.tmp,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        outputs = dict(
            line.split("=", 1) for line in (self.tmp / "output").read_text().splitlines() if line
        )
        return result, outputs

    def plan(self, correction="redate-epochs", ids=EPOCH, apply="false", confirm=""):
        return self.run_step(
            "plan", {"CORRECTION": correction, "IDS": ids, "APPLY": apply, "CONFIRM": confirm}
        )

    def correct(self, printed, exit_code=0, messages=""):
        """The correction step against a flyctl that records its argv, writes `messages` to
        stderr after it prints `printed` (as its closing message comes), and exits
        `exit_code`."""
        flyctl = (
            "#!/bin/sh\n"
            f'printf "%s\\n" "$@" > "{self.tmp}/argv"\n'
            f"printf '%s' '{printed}'\n"
            f"printf '%s' '{messages}' >&2\n"
            f"exit {exit_code}\n"
        )
        env = {
            "FLY_API_TOKEN": "test-only",
            "APP": "twin-api",
            "MACHINE": "m1",
            "MODULE": "app.scripts.redate_epochs",
            "FLAGS": f" --epoch {EPOCH}",
        }
        result, _ = self.run_step("correct", env, flyctl)
        return result

    def test_dispatch_only_with_read_permissions(self):
        triggers = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        # `actions: read` only for `gh run list` (refusing during a deploy).
        self.assertEqual(self.workflow["permissions"], {"contents": "read", "actions": "read"})
        for job in self.workflow["jobs"].values():
            self.assertNotIn("permissions", job)
        self.assertFalse(self.workflow["concurrency"]["cancel-in-progress"])

    def test_no_inputs_or_secrets_interpolated_into_any_run(self):
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "run":
                        self.assertEqual(re.findall(r"\$\{\{", child), [], child[:80])
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.workflow)

    def test_inputs_name_commands_that_exist_and_require_ids(self):
        inputs = self.workflow[True]["workflow_dispatch"]["inputs"]
        self.assertTrue(inputs["ids"]["required"])
        self.assertFalse(inputs["apply"]["default"])
        for correction in inputs["correction"]["options"]:
            module = correction.replace("-", "_")
            self.assertTrue((ROOT / f"apps/api/app/scripts/{module}.py").is_file(), module)

    def test_named_ids_become_flags(self):
        result, outputs = self.plan(ids=f"{EPOCH}  {EPOCH.replace('a', 'b')}")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(outputs["module"], "app.scripts.redate_epochs")
        self.assertEqual(outputs["flags"], f" --epoch {EPOCH} --epoch {EPOCH.replace('a', 'b')}")
        result, outputs = self.plan(correction="fit-site-boundaries", ids=SITE)
        self.assertEqual((outputs["module"], outputs["flags"]), ("app.scripts.fit_site_boundaries", f" --site {SITE}"))

    def test_hostile_or_missing_ids_are_refused_before_anything_runs(self):
        # A file whose name is an id: a glob that matched it would pass for one.
        (self.tmp / EPOCH).write_text("")
        for ids in (
            "",
            "   ",
            "all",
            "*",
            f"{EPOCH[:-1]}?",
            f"{EPOCH};rm -rf /",
            f"{EPOCH} $(id)",
            f"{EPOCH.upper()}",
            f"{EPOCH}'",
            f"{EPOCH[:-1]}",
            f"--apply {EPOCH}",
        ):
            with self.subTest(ids=ids):
                result, outputs = self.plan(ids=ids)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("flags", outputs)

    def test_apply_needs_the_confirm_text(self):
        for confirm in ("", "yes", "correct-production "):
            with self.subTest(confirm=confirm):
                result, outputs = self.plan(apply="true", confirm=confirm)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("flags", outputs)
        result, outputs = self.plan(apply="true", confirm="correct-production")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(outputs["flags"].endswith(" --apply"))
        result, outputs = self.plan(correction="everything")
        self.assertNotEqual(result.returncode, 0)

    def test_the_marker_is_printed_only_after_the_command_succeeded(self):
        self.correct(f"done\n{MARKER}\n")
        argv = (self.tmp / "argv").read_text()
        self.assertIn(
            f"/app/.venv/bin/python -m app.scripts.redate_epochs --epoch {EPOCH} && echo {MARKER}",
            argv,
        )

    def test_success_is_the_marker_as_the_machines_last_line(self):
        ok = self.correct(f"epoch {EPOCH}: ...\nREDATE-EPOCHS-OK\n{MARKER}\r\n\n")
        self.assertEqual(ok.returncode, 0, ok.stdout)
        # flyctl's own messages after it do not count: they are not the machine's output.
        ok = self.correct(f"REDATE-EPOCHS-OK\n{MARKER}\n", messages="Connection closed.\n")
        self.assertEqual(ok.returncode, 0, ok.stdout)

    def test_anything_else_is_a_failure(self):
        for printed, code in (
            ("REDATE-EPOCHS-OK\n", 0),  # the command's own line, but no exit-0 marker
            (f"site ({MARKER})\n{MARKER}\nTraceback: ...\n", 0),  # marker, then more output
            (f"{MARKER}\nREDATE-EPOCHS-OK\n", 0),
            ("", 0),
            (f"{MARKER}\n", 1),  # flyctl failed
        ):
            with self.subTest(printed=printed, code=code):
                self.assertNotEqual(self.correct(printed, code).returncode, 0)


if __name__ == "__main__":
    unittest.main()
