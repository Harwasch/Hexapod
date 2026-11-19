"""Offline checks for generated bridge and bundle integration."""
import json
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]


class GlobeTests(unittest.TestCase):
    def test_generated_contract(self):
        subprocess.run(["node", "apps/web/scripts/generate-globe-bridge.mjs", "--check"], cwd=ROOT, check=True)
        module = ROOT / "apps/ios/Modules/Globe"
        schema = json.loads((module / "Bridge/schema.json").read_text())
        fixtures = json.loads((module / "Tests/GlobeTests/Fixtures/messages.json").read_text())
        self.assertEqual(set(schema["messages"]), {f["type"] for f in fixtures})

    def test_both_build_paths_generate_resources_before_xcode(self):
        workflow = (ROOT / ".github/workflows/ios.yml").read_text()
        jobs = yaml.safe_load(workflow)["jobs"]
        self.assertEqual(workflow.count("bash apps/ios/ci_scripts/build_globe.sh"), 3)
        for name in ("packages", "app", "testflight"):
            steps = jobs[name]["steps"]
            bundle = next(i for i, step in enumerate(steps) if step.get("name") == "Bundle local globe")
            build = next(i for i, step in enumerate(steps) if step.get("name") in (
                "Build every package for iOS Simulator", "Generate project"))
            self.assertLess(bundle, build)
        self.assertTrue(any(s.get("uses") == "actions/download-artifact@v4" for s in jobs["tests"]["steps"]))
        cloud = (ROOT / "apps/ios/ci_scripts/ci_post_clone.sh").read_text()
        self.assertLess(cloud.index("build_globe.sh"), cloud.index("xcodegen generate"))
        package = (ROOT / "apps/ios/Modules/Globe/Package.swift").read_text()
        self.assertIn('.copy("Resources/GlobeWeb")', package)
