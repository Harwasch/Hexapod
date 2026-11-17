"""Offline checks for generated bridge and bundle integration."""
import json
from pathlib import Path
import subprocess
import unittest

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
        self.assertEqual(workflow.count("bash apps/ios/ci_scripts/build_globe.sh"), 2)
        cloud = (ROOT / "apps/ios/ci_scripts/ci_post_clone.sh").read_text()
        self.assertLess(cloud.index("build_globe.sh"), cloud.index("xcodegen generate"))
        package = (ROOT / "apps/ios/Modules/Globe/Package.swift").read_text()
        self.assertIn('.copy("Resources/GlobeWeb")', package)
