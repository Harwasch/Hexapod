"""Import NWM's dispatch boundary and its Python steps, with every download mocked.

The archive is untrusted: these hold that nothing from it reaches $GITHUB_ENV (where a
newline in a folder name once set BASH_ENV for the step that holds the write token), that
the timestep is one of the record's eleven, and that the token never reaches an argv.
"""

import hashlib
import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/import-nwm.yml"
CAPTURE_ID = "c0ffee00-0000-0000-0000-000000000001"
#: The record's timesteps (doi:10.48436/bphp1-hbb20), as published on 2026-08-11.
TIMESTEPS = {
    "2025-08-25", "2025-11-11", "2025-11-22", "2025-11-23", "2025-12-04_1", "2025-12-04_2",
    "2025-12-08", "2026-01-10", "2026-01-10_detail_1", "2026-01-10_detail_2", "2026-03-28",
}


class ImportNwmTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.steps = {step.get("name"): step for step in self.workflow["jobs"]["import"]["steps"]}
        self.env = {"TIMESTEP": "2025-08-25", "DRY_RUN": "false", "PROCESS": "true", "CAPTURE_ID": ""}

    def runs(self):
        return [step["run"] for step in self.workflow["jobs"]["import"]["steps"] if "run" in step]

    def test_dispatch_only_and_read_only_permissions(self):
        triggers = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        for job in self.workflow["jobs"].values():
            self.assertNotIn("permissions", job)
        self.assertIs(triggers["workflow_dispatch"]["inputs"]["dry_run"]["default"], True)

    def test_no_expression_in_any_run_and_nothing_written_to_github_env(self):
        for run in self.runs():
            self.assertNotIn("${{", run)
            self.assertNotIn("GITHUB_ENV", run)
            self.assertNotIn("GITHUB_OUTPUT", run)

    def test_the_token_is_only_ever_an_environment_variable(self):
        upload = self.steps["Create the capture and upload the frames"]
        self.assertEqual(upload["env"]["API_WRITE_TOKEN"], "${{ secrets.API_WRITE_TOKEN }}")
        uses = re.findall(r".*API_WRITE_TOKEN.*", upload["run"])
        self.assertEqual(uses, ['if [ -z "$API_WRITE_TOKEN" ]; then',
                                '  echo "::error::API_WRITE_TOKEN is not set; add the repository '
                                'secret (docs/DEPLOYMENT.md)"'])
        for name, step in self.steps.items():
            if step is not upload:
                self.assertNotIn("secrets.", json.dumps(step), name)

    def test_handover_names_every_secret_and_variable(self):
        handover = (ROOT / "docs/DEPLOYMENT.md").read_text()
        for name in re.findall(r"\b(?:secrets|vars)\.([A-Za-z_]\w*)", WORKFLOW.read_text()):
            self.assertIn(f"`{name}`", handover)

    def test_timesteps_are_the_records_eleven_and_nothing_else_passes(self):
        inputs = self.workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(inputs["timestep"]["options"]), TIMESTEPS)
        for timestep in sorted(TIMESTEPS):
            with self.subTest(timestep=timestep):
                self.assertEqual(self.validate(TIMESTEP=timestep), 0)
        for timestep in ("2025-08-26", "2025-08-25\nBASH_ENV=/tmp/evil", "../2025-08-25", ""):
            with self.subTest(timestep=timestep):
                self.assertNotEqual(self.validate(TIMESTEP=timestep), 0)
        for key, value in (("CAPTURE_ID", "not-a-uuid"), ("DRY_RUN", "yes"), ("PROCESS", "1")):
            with self.subTest(key=key):
                self.assertNotEqual(self.validate(**{key: value}), 0)
        self.assertEqual(self.validate(CAPTURE_ID=CAPTURE_ID), 0)

    def validate(self, **env):
        step = self.steps["Validate inputs"]
        with mock.patch.dict(os.environ, self.env | env, clear=True):
            try:
                exec(compile(step["run"], str(WORKFLOW), "exec"), {})
            except SystemExit as error:
                return error.code or 0
        return 0

    def download(self, members, *, md5=None):
        """Run the download step against a fake record holding a zip of `members`."""
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in members.items():
                bundle.writestr(name, data)
        payload = archive.getvalue()
        record = {"entries": [{
            "key": "nwm_2025-08-25.zip",
            "size": len(payload),
            "checksum": f"md5:{md5 or hashlib.md5(payload).hexdigest()}",
        }]}

        def fake_curl(argv, check):
            target = Path(argv[argv.index("--output") + 1])
            url = argv[-1]
            target.write_bytes(payload if url.endswith("/content") else json.dumps(record).encode())

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        temp = Path(directory.name)
        github_env = temp / "github_env"
        github_env.write_text("")
        env = self.env | {"RUNNER_TEMP": str(temp), "GITHUB_ENV": str(github_env)}
        step = self.steps["Download the timestep and check its MD5"]
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("subprocess.run", side_effect=fake_curl),
            redirect_stdout(io.StringIO()),
        ):
            code = 0
            try:
                exec(compile(step["run"], str(WORKFLOW), "exec"), {})
            except SystemExit as error:
                code = error.code
        return code, temp, github_env.read_text()

    def test_a_folder_name_carrying_bash_env_goes_nowhere(self):
        evil = "nwm_2025-08-25\nBASH_ENV=/tmp/evil"
        code, temp, github_env = self.download({
            f"{evil}/camera_info.json": "[]",
            f"{evil}/input/a.png": b"png",
        })
        self.assertEqual(code, 0)
        self.assertEqual(github_env, "")
        self.assertTrue((temp / "nwm/dataset/camera_info.json").is_file())
        self.assertTrue((temp / "nwm/dataset/input/a.png").is_file())

    def test_a_wrong_md5_stops_before_anything_is_unpacked(self):
        code, temp, _ = self.download({"nwm/camera_info.json": "[]"}, md5="0" * 32)
        self.assertIn("does not match", code)
        self.assertFalse((temp / "nwm/raw").exists())
        self.assertFalse((temp / "nwm/dataset").exists())

    def test_two_camera_info_files_are_refused(self):
        code, _, _ = self.download({"a/camera_info.json": "[]", "b/camera_info.json": "[]"})
        self.assertIn("expected one camera_info.json", code)

    def summary(self, upload=None):
        step = self.steps["Summary"]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "prepare.json").write_text(json.dumps({
                "frames": 226, "recordedOn": ["2025-08-29"], "firstLocal": "2025-08-29T15:20:14+02:00",
                "firstUtc": "2025-08-29T13:20:14Z", "medianLat": 48.2, "medianLon": 16.2,
                "altitudeM": {"min": 317.3, "max": 336.2}, "fps": 29.97, "timezone": "Europe/Vienna",
            }))
            if upload is not None:
                (temp / "upload.json").write_text(json.dumps(upload))
            summary = temp / "summary.md"
            env = self.env | {"RUNNER_TEMP": str(temp), "GITHUB_STEP_SUMMARY": str(summary)}
            with mock.patch.dict(os.environ, env, clear=True), chdir(ROOT):
                exec(compile(step["run"], str(WORKFLOW), "exec"), {})
            return summary.read_text()

    def test_summary_names_the_capture_even_when_the_upload_stopped(self):
        text = self.summary({"captureId": CAPTURE_ID, "resumed": False})
        self.assertIn(CAPTURE_ID, text)
        self.assertIn("stopped part-way", text)
        text = self.summary({
            "captureId": CAPTURE_ID, "uploaded": 226, "resumedRows": 0, "alreadyThere": 0,
            "jobId": "job-1",
        })
        self.assertIn("226 uploaded", text)
        self.assertIn("/api/v1/jobs/job-1", text)
        self.assertIn("No capture was created", self.summary())


if __name__ == "__main__":
    unittest.main()
