"""Linux contract checks; no Apple tooling, network, or real credentials."""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import jsonschema
import yaml

IOS = Path(__file__).resolve().parents[1]
ROOT = IOS.parents[1]


class EntitySchemaTests(unittest.TestCase):
    def setUp(self):
        package = IOS / "Modules/EntitySchema"
        self.schema = json.loads((package / "schema/entity.schema.json").read_text())
        self.fixture = json.loads(
            (package / "Tests/EntitySchemaTests/Fixtures/entity.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(self.schema)
        self.validator = jsonschema.Draft202012Validator(self.schema)

    def test_fixture_matches_schema(self):
        self.validator.validate(self.fixture)

    def test_local_anchor_and_empty_collections(self):
        del self.fixture["anchor"]["geo"]
        self.fixture["masks"] = []
        self.fixture["attrs"] = {}
        self.validator.validate(self.fixture)

    def test_invalid_measurements_and_missing_provenance(self):
        for field, value in (("confidence", -0.1), ("confidence", 1.1), ("source", None)):
            with self.subTest(field=field, value=value):
                sample = copy.deepcopy(self.fixture)
                sample[field] = value
                self.assertFalse(self.validator.is_valid(sample))
        self.fixture["anchor"]["geo"]["lat"] = 91
        self.assertFalse(self.validator.is_valid(self.fixture))
        self.fixture["anchor"]["geo"]["lat"] = 34.6
        self.fixture["extent"]["x"] = -1
        self.assertFalse(self.validator.is_valid(self.fixture))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((ROOT / ".github/workflows/ios.yml").read_text())
        self.steps = self.workflow["jobs"]["testflight"]["steps"]
        self.check = self.steps[0]
        self.names = ("ASC_KEY_ID", "ASC_ISSUER_ID", "ASC_KEY_P8", "APPLE_TEAM_ID")

    def test_dispatch_only_and_upload_gates(self):
        triggers = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        self.assertIs(triggers["workflow_dispatch"]["inputs"]["testflight"]["default"], False)
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        job = self.workflow["jobs"]["testflight"]
        self.assertEqual(job["if"], "${{ inputs.testflight }}")
        self.assertEqual(job["needs"], "build")
        for step in self.steps[1:]:
            self.assertIn("steps.signing.outputs.ready == 'true'", step["if"])
        self.assertIn("always()", self.steps[-1]["if"])
        for job in self.workflow["jobs"].values():
            self.assertEqual(job["runs-on"], "macos-26")
            select = next(s for s in job["steps"] if s.get("name") == "Select Xcode 26 explicitly")
            self.assertIn("/Applications/Xcode_26.2.app/Contents/Developer", select["run"])

    def test_every_missing_secret_skips_without_disclosing_values(self):
        for missing in (*self.names, "all", None):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ)
                env.update({name: "private-fixture-value" for name in self.names})
                for name in self.names:
                    if missing in (name, "all"):
                        env[name] = ""
                output = Path(directory) / "output"
                summary = Path(directory) / "summary"
                env.update(GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary))
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", self.check["run"]],
                    env=env, capture_output=True, text=True, check=True,
                )
                self.assertEqual(output.read_text().strip(), f"ready={str(missing is None).lower()}")
                self.assertNotIn("private-fixture-value", result.stdout + result.stderr)
                if missing is not None:
                    self.assertIn("TestFlight skipped", summary.read_text())
                    if missing != "all":
                        self.assertIn(f"missing {missing}", result.stdout)

    def test_prepare_writes_private_key_and_upload_options(self):
        step = next(s for s in self.steps if s.get("name") == "Prepare API key and export options")
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, RUNNER_TEMP=directory, GITHUB_RUN_ID="123",
                       GITHUB_RUN_ATTEMPT="1", ASC_KEY_P8="fixture-key\n", APPLE_TEAM_ID="TEAM")
            subprocess.run(["bash", "-euo", "pipefail", "-c", step["run"]], env=env, check=True)
            signing = Path(directory) / "hex-signing-123-1"
            key = signing / "AuthKey.p8"
            self.assertEqual(key.read_text(), "fixture-key\n")
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            import plistlib
            options = plistlib.loads((signing / "ExportOptions.plist").read_bytes())
            self.assertEqual(options["destination"], "upload")
            self.assertEqual(options["method"], "app-store-connect")
            self.assertEqual(options["teamID"], "TEAM")
            self.assertEqual(options["signingStyle"], "automatic")
            subprocess.run(["bash", "-euo", "pipefail", "-c", self.steps[-1]["run"]], env=env, check=True)
            self.assertFalse(signing.exists())


if __name__ == "__main__":
    unittest.main()
