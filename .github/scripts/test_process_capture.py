"""Process capture's dispatch boundary and scripts, with every HTTP call mocked."""

import io
import json
import os
import re
import subprocess
import tempfile
import tomllib
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/process-capture.yml"
CAPTURE_ID = "510d624e-dad2-4b7b-966b-357c4447af02"
JOB_ID = "87d772f0-92c9-49be-8570-3c830425d3a3"


class ProcessCaptureTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.job = self.workflow["jobs"]["process"]
        self.steps = self.job["steps"]
        self.env = {
            "CAPTURE_ID": CAPTURE_ID,
            "RECIPE": "photo-reconstruct",
            "DRY_RUN": "false",
            "API_BASE_URL": "",
            "API_WRITE_TOKEN": "test-only-write-token",
        }
        self.capture = {
            "slug": "pumpkin",
            "kind": "photos",
            "status": "complete",
            "files": [{"status": "complete"}],
            "jobs": [],
        }

    def test_dispatch_only_and_read_only_permissions(self):
        # PyYAML's YAML 1.1 loader reads the unquoted `on` key as True.
        triggers = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        for job in self.workflow["jobs"].values():
            self.assertEqual(
                job.get("permissions", self.workflow["permissions"]), {"contents": "read"}
            )
        self.assertEqual(
            self.workflow["concurrency"],
            {"group": "process-capture-${{ inputs.capture_id }}", "cancel-in-progress": False},
        )

    def test_no_inputs_or_secrets_interpolated_into_any_run(self):
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "run":
                        for expression in re.findall(r"\$\{\{(.*?)\}\}", child, re.S):
                            self.assertNotRegex(expression, r"\b(?:inputs|secrets)\b")
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.workflow)

    def test_inputs_and_recipes_agree_with_registry(self):
        inputs = self.workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["capture_id"]["type"], "string")
        self.assertTrue(inputs["capture_id"]["required"])
        self.assertEqual(inputs["recipe"]["type"], "choice")
        self.assertEqual(inputs["recipe"]["default"], "photo-reconstruct")
        recipes = {
            yaml.safe_load(path.read_text())["name"]
            for path in (ROOT / "tools/pipeline/recipes").glob("*.yaml")
        }
        self.assertEqual(set(inputs["recipe"]["options"]), recipes)
        self.assertEqual(inputs["dry_run"]["type"], "boolean")
        self.assertIs(inputs["dry_run"]["default"], False)
        for recipe in recipes:
            with self.subTest(recipe=recipe):
                self.env["RECIPE"] = recipe
                code, calls, _, _ = self.execute(self.steps[0])
                self.assertEqual(code, 0)
                self.assertEqual(calls, [])

    def test_handover_names_every_secret_and_variable(self):
        handover = (ROOT / "docs/DEPLOYMENT.md").read_text()
        for name in re.findall(r"\b(?:secrets|vars)\.([A-Za-z_]\w*)", WORKFLOW.read_text()):
            self.assertIn(f"`{name}`", handover)

    def execute(self, step, responses=()):
        """Run the actual Python step; subprocess cannot reach curl or the network."""
        self.assertEqual(step["shell"], "python")
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, self.env | {"GITHUB_STEP_SUMMARY": str(summary)}, clear=True),
                mock.patch("subprocess.run", side_effect=responses) as request,
                chdir(ROOT),
                redirect_stdout(output),
            ):
                code = 0
                try:
                    exec(compile(step["run"], str(WORKFLOW), "exec"), {})
                except SystemExit as error:
                    code = error.code
            summary_text = summary.read_text() if summary.exists() else ""
            if self.env["API_WRITE_TOKEN"]:
                self.assertNotIn(self.env["API_WRITE_TOKEN"], output.getvalue() + summary_text)
            return code, request.call_args_list, output.getvalue(), summary_text

    def response(self, body, returncode=0):
        return subprocess.CompletedProcess([], returncode, json.dumps(body), "")

    def test_invalid_inputs_fail_in_first_step_without_requests(self):
        for key, value in (
            ("CAPTURE_ID", "$(touch injected); not-a-uuid"),
            ("CAPTURE_ID", CAPTURE_ID + "\nmalicious"),
            ("RECIPE", "unknown-recipe"),
            ("DRY_RUN", "yes"),
        ):
            with self.subTest(key=key, value=value), mock.patch.dict(self.env, {key: value}):
                code, calls, _, _ = self.execute(self.steps[0])
                self.assertNotEqual(code, 0)
                self.assertEqual(calls, [])

    def test_dry_run_reads_only_even_when_not_eligible_and_without_token(self):
        self.env.update(DRY_RUN="true", API_WRITE_TOKEN="")
        self.capture.update(status="awaiting-files", files=[])
        code, calls, output, summary = self.execute(self.steps[-1], [self.response(self.capture)])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--header", calls[0].args[0])
        self.assertIsNone(calls[0].kwargs.get("input"))
        self.assertIn("pumpkin | kind: photos | status: awaiting-files", output)
        self.assertIn("none (dry run; no POST)", summary)

    def test_queue_matches_api_body_and_public_follow_urls(self):
        self.env["API_BASE_URL"] = "https://example.invalid/"
        # A capture status alone is not a blocker in create_job, including a stale
        # in-progress status after cancellation. Only active job statuses block.
        self.capture.update(status="in-progress", jobs=[{"status": "cancelled"}])
        code, calls, output, summary = self.execute(
            self.steps[-1],
            [self.response(self.capture), self.response({"id": JOB_ID, "status": "not-started"})],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call.args[0][0], "curl")
            self.assertIn("--fail-with-body", call.args[0])
            for argument in call.args[0]:
                self.assertNotIn(self.env["API_WRITE_TOKEN"], argument)
        self.assertNotIn("--header", calls[0].args[0])
        self.assertIsNone(calls[0].kwargs.get("input"))
        argv = calls[1].args[0]
        self.assertEqual(argv[-1], f"https://example.invalid/api/v1/captures/{CAPTURE_ID}/process")
        self.assertEqual(argv[argv.index("--request") + 1], "POST")
        self.assertEqual(json.loads(argv[argv.index("--data") + 1]), {"recipe": "photo-reconstruct"})
        self.assertEqual(argv[argv.index("--header") + 1], "@-")
        self.assertEqual(calls[1].kwargs["input"], "Authorization: Bearer test-only-write-token\n")
        self.assertIn(f"Job id: {JOB_ID} | status: not-started", output)
        for expected in (
            CAPTURE_ID, "photo-reconstruct", f"/api/v1/jobs/{JOB_ID}",
            f"/api/v1/jobs?captureId={CAPTURE_ID}", "/steps/<step_id>/log",
        ):
            self.assertIn(expected, summary)

    def test_active_jobs_or_missing_upload_refuse_post(self):
        for status in ("not-started", "in-progress", None):
            with self.subTest(status=status):
                self.capture["jobs"] = [{"status": status}] if status else []
                self.capture["files"] = [{"status": "complete"}] if status else []
                code, calls, _, _ = self.execute(self.steps[-1], [self.response(self.capture)])
                self.assertNotEqual(code, 0)
                self.assertEqual(len(calls), 1)

    def test_missing_token_refuses_post(self):
        self.env["API_WRITE_TOKEN"] = ""
        code, calls, _, _ = self.execute(self.steps[-1], [self.response(self.capture)])
        self.assertIn("API_WRITE_TOKEN is not set", code)
        self.assertEqual(len(calls), 1)

    def test_http_failure_stops_without_retry_or_success_summary(self):
        for responses in (
            [self.response({"detail": "not found"}, 22)],
            [self.response(self.capture), self.response({"detail": "conflict"}, 22)],
        ):
            with self.subTest(request_count=len(responses)):
                code, calls, _, summary = self.execute(self.steps[-1], responses)
                self.assertIn("API request failed", code)
                self.assertEqual(len(calls), len(responses))
                self.assertNotIn("- Job id:", summary)
                app = tomllib.loads((ROOT / "fly.toml").read_text())["app"]
                self.assertEqual(
                    calls[0].args[0][-1], f"https://{app}.fly.dev/api/v1/captures/{CAPTURE_ID}"
                )


if __name__ == "__main__":
    unittest.main()
