"""Dispatch workflow and UI target contracts, without Apple tools."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml
from ui_tests import ui_command

ROOT = Path(__file__).resolve().parents[3]


class UITestWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((ROOT / '.github/workflows/ios.yml').read_text())
        self.project = yaml.safe_load((ROOT / 'apps/ios/project.yml').read_text())

    def test_simulator_ui_target_and_scheme(self):
        target = self.project['targets']['HexUITests']
        self.assertEqual(target['type'], 'bundle.ui-testing')
        self.assertEqual(target['dependencies'], [{'target': 'Hex'}])
        self.assertEqual(self.project['schemes']['HexUITests']['test']['targets'], ['HexUITests'])
        with patch.dict(os.environ, {'HEX_API_BASE_URL': 'https://custom.invalid'}):
            command = ui_command('platform=iOS Simulator,id=fixture')
        self.assertIn('platform=iOS Simulator,id=fixture', command)
        self.assertIn('HEX_API_BASE_URL=https://custom.invalid', command)
        self.assertIn('-resultBundlePath', command)

    def test_real_public_service_defaults_and_token_propagation(self):
        settings = self.project['settings']['base']
        properties = self.project['targets']['Hex']['info']['properties']
        self.assertEqual(settings['HEX_API_BASE_URL'], 'https://twin-api.fly.dev')
        self.assertEqual(settings['HEX_WEB_BASE_URL'], 'https://twin-web-f57.pages.dev')
        for key in ('HEX_API_BASE_URL', 'HEX_WEB_BASE_URL'):
            self.assertEqual(properties[key], '$(' + key + ')')
        self.assertIn('vars.VITE_API_BASE_URL', self.workflow['env']['HEX_API_BASE_URL'])
        self.assertNotIn('HEX_API_WRITE_TOKEN', properties)
        for name in ('packages', 'app', 'testflight'):
            bundle = next(s for s in self.workflow['jobs'][name]['steps'] if s.get('name') == 'Bundle local globe')
            self.assertEqual(bundle['env']['VITE_CESIUM_ION_ACCESS_TOKEN'], '${{ secrets.VITE_CESIUM_ION_ACCESS_TOKEN }}')

    def test_screenshots_export_even_after_failure_and_block_testflight(self):
        steps = self.workflow['jobs']['tests']['steps']
        self.assertTrue(any('ui_tests.py' in s.get('run', '') for s in steps))
        export = next(s for s in steps if s.get('name') == 'Export UI screenshots')
        self.assertEqual(export['if'], '${{ always() }}')
        self.assertIn('xcresulttool export attachments', export['run'])
        self.assertIn('--path apps/ios/build/HexUITests.xcresult', export['run'])
        artifact = next(s for s in steps if s.get('with', {}).get('name') == 'ios-ui-screenshots')
        self.assertEqual(artifact['if'], '${{ always() }}')
        self.assertIn('tests', self.workflow['jobs']['testflight']['needs'])

    def test_fixture_resources_are_debug_only(self):
        script = self.project['targets']['Hex']['postBuildScripts'][0]['script']
        self.assertIn('"$CONFIGURATION" = Debug', script)
        self.assertIn('rm -rf "$fixture_dir"', script)
        self.assertIn('UITestFixtures', script)
