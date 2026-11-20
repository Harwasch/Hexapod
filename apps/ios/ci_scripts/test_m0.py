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
        self.assertEqual(job["needs"], ["packages", "app", "tests"])
        for step in self.steps[1:]:
            self.assertIn("steps.signing.outputs.ready == 'true'", step["if"])
        self.assertIn("always()", self.steps[-1]["if"])
        jobs = self.workflow["jobs"]
        self.assertEqual(set(jobs), {"packages", "app", "tests", "testflight"})
        self.assertNotIn("needs", jobs["packages"])
        self.assertNotIn("needs", jobs["app"])
        self.assertEqual(jobs["tests"]["needs"], "app")
        self.assertIs(self.workflow["concurrency"]["cancel-in-progress"], True)
        self.assertIn("github.ref", self.workflow["concurrency"]["group"])
        self.assertEqual(triggers["workflow_dispatch"]["inputs"]["testflight"]["type"], "boolean")
        self.assertEqual(set(triggers["workflow_dispatch"]["inputs"]), {"testflight", "bundle_id", "skip_ui_tests"})
        for name, job in jobs.items():
            self.assertEqual(job["timeout-minutes"], {"packages": 20, "app": 30, "tests": 45, "testflight": 45}[name])
            self.assertEqual(job["runs-on"], "macos-26")
            select = next(s for s in job["steps"] if s.get("name") == "Select Xcode 26 explicitly")
            self.assertIn("/Applications/Xcode_26*.app", select["run"])

    def test_build_logs_caches_and_ios_destination(self):
        jobs = self.workflow["jobs"]
        app = next(s for s in jobs["app"]["steps"] if s.get("name") == "Build without signing")
        for expected in ("generic/platform=iOS Simulator", "CODE_SIGNING_ALLOWED=NO", "-skipMacroValidation", "tee build/app.log | xcbeautify", "pipefail"):
            self.assertIn(expected, app["run"])
        errors = next(s for s in jobs["app"]["steps"] if s.get("name") == "Show every build error")
        self.assertEqual(errors["if"], "${{ always() }}")
        self.assertIn("grep 'error:'", errors["run"])
        self.assertNotIn("head", errors["run"])
        for name, job in jobs.items():
            cache = next(s for s in job["steps"] if s.get("name") == "Cache SwiftPM")
            for path in ("~/Library/Caches/org.swift.swiftpm", "apps/ios/Modules/*/.build", "apps/ios/App/.build"):
                self.assertIn(path, cache["with"]["path"])
            for manifest in ("Package.swift", "Package.resolved"):
                self.assertIn(manifest, cache["with"]["key"])
            if name in ("app", "testflight"):
                derived = next(s for s in job["steps"] if s.get("name") == "Cache DerivedData")
                self.assertIn("project.yml", derived["with"]["key"])
                self.assertIn("Package.swift", derived["with"]["key"])
            else:
                self.assertTrue(any("packages.py " in s.get("run", "") for s in job["steps"]))
            if name != "testflight":
                artifact = next(s for s in job["steps"] if s.get("name", "").startswith("Upload "))
                self.assertEqual(artifact["if"], "${{ always() }}")

    def test_phone_key_is_only_supplied_to_archive(self):
        jobs = self.workflow["jobs"]
        found = []
        for name, job in jobs.items():
            for step in job["steps"]:
                if "PHONE_API_TOKEN" in json.dumps(step):
                    found.append((name, step))
        self.assertEqual(len(found), 1)
        name, archive = found[0]
        self.assertEqual(name, "testflight")
        self.assertEqual(archive["env"]["HEX_API_PHONE_TOKEN"], "${{ secrets.PHONE_API_TOKEN }}")
        self.assertIn("xcodebuild archive", archive["run"])
        self.assertNotIn("secrets.API_WRITE_TOKEN", json.dumps(self.workflow))
        project = yaml.safe_load((IOS / "project.yml").read_text())
        self.assertEqual(project["settings"]["base"]["HEX_API_PHONE_TOKEN"], "")
        self.assertEqual(project["targets"]["Hex"]["info"]["properties"]["HEX_API_PHONE_TOKEN"], "$(HEX_API_PHONE_TOKEN)")

    def test_archive_redacts_phone_key_and_preserves_failures(self):
        run = next(s for s in self.steps if "xcodebuild archive" in s.get("run", ""))["run"]
        archive = run[:run.index("xcodebuild -exportArchive")]
        with tempfile.TemporaryDirectory() as directory:
            xcode = Path(directory) / "xcodebuild"
            xcode.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                "print(os.environ['HEX_API_PHONE_TOKEN'])\n"
                "print(os.environ['HEX_API_PHONE_TOKEN'], file=sys.stderr)\n"
                "sys.exit(int(os.environ['ARCHIVE_EXIT']))\n"
            )
            xcode.chmod(0o700)
            env = dict(os.environ)
            env.update(
                PATH=directory + os.pathsep + env["PATH"], RUNNER_TEMP=directory,
                GITHUB_RUN_ID="1", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_NUMBER="1",
                HEX_API_PHONE_TOKEN="private-phone-fixture", ASC_KEY_ID="id",
                ASC_ISSUER_ID="issuer", APPLE_TEAM_ID="team", HEX_BUNDLE_ID="app",
                HEX_API_BASE_URL="https://example.com", HEX_WEB_BASE_URL="https://example.com",
            )
            for code in (0, 17):
                env["ARCHIVE_EXIT"] = str(code)
                result = subprocess.run(["bash", "-eu", "-c", archive], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertNotIn(env["HEX_API_PHONE_TOKEN"], result.stdout + result.stderr)
                self.assertIn("<phone credential>", result.stdout)

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

    def test_archive_needs_no_device_profile(self):
        # Run 44592652993: an automatically signed archive asked for a development profile,
        # which needs a registered device. Archive ad hoc; only the export signs for the store.
        run = next(s for s in self.steps if "xcodebuild archive" in s.get("run", ""))["run"]
        archive = run[run.index("xcodebuild archive"):run.index("xcodebuild -exportArchive")]
        self.assertIn("CODE_SIGN_IDENTITY=-", archive)
        self.assertIn("AD_HOC_CODE_SIGNING_ALLOWED=YES", archive)
        self.assertNotIn("-allowProvisioningUpdates", archive)
        self.assertNotIn("auth[@]", archive)
        self.assertIn("auth[@]", run[run.index("xcodebuild -exportArchive"):])

    def test_app_target_is_iphone_only_with_an_icon(self):
        # Run 44597285483: App Store validation wanted iPad icons and orientations because the
        # target-level device family came from XcodeGen's preset, and no app icon existed.
        import json
        project = yaml.safe_load((ROOT / "apps/ios/project.yml").read_text())
        target = project["targets"]["Hex"]["settings"]["base"]
        self.assertEqual(str(target["TARGETED_DEVICE_FAMILY"]), "1")
        self.assertEqual(target["ASSETCATALOG_COMPILER_APPICON_NAME"], "AppIcon")
        icon_set = ROOT / "apps/ios/App/Assets.xcassets/AppIcon.appiconset"
        contents = json.loads((icon_set / "Contents.json").read_text())
        image = contents["images"][0]
        self.assertEqual((image["idiom"], image["size"]), ("universal", "1024x1024"))
        png = (icon_set / image["filename"]).read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(png[25], 2, "App Store icons must be RGB with no alpha channel")

    def test_app_sources_exclude_swiftpm_build_output(self):
        # Run 44655055029: swift test on App/Package.swift left App/.build behind, and the
        # regenerated project compiled it into the app ("Unexpected duplicate tasks").
        import yaml
        project = yaml.safe_load((ROOT / "apps/ios/project.yml").read_text())
        source = next(s for s in project["targets"]["Hex"]["sources"] if s["path"] == "App")
        for pattern in (".build/**", ".swiftpm/**", "Package.swift", "Tests/**"):
            self.assertIn(pattern, source["excludes"])


if __name__ == "__main__":
    unittest.main()
