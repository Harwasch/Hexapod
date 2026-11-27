"""Dispatch workflow and UI target contracts, without Apple tools."""
import json
import os
import re
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import yaml
from ui_tests import ui_command, boot_simulator

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
        self.assertEqual(command[command.index('-test-timeouts-enabled') + 1], 'YES')
        self.assertEqual(command[command.index('-default-test-execution-time-allowance') + 1], '300')
        self.assertEqual(command[command.index('-maximum-test-execution-time-allowance') + 1], '300')

    def test_boot_waits_before_testing_and_propagates_boot_failure(self):
        for already_booted in (False, True):
            devices = {'devices': {'runtime': [{'udid': 'fixture'}] if already_booted else []}}
            with patch('ui_tests.subprocess.run') as run:
                run.return_value = SimpleNamespace(stdout=json.dumps(devices))
                boot_simulator('platform=iOS Simulator,id=fixture')
                calls = [call.args[0] for call in run.call_args_list]
                self.assertEqual(['xcrun', 'simctl', 'boot', 'fixture'] in calls, not already_booted)
                self.assertEqual(calls[-1], ['xcrun', 'simctl', 'bootstatus', 'fixture', '-b'])
                self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

        failure = subprocess.CalledProcessError(7, ['xcrun', 'simctl', 'boot', 'fixture'])
        with patch('ui_tests.subprocess.run') as run:
            run.side_effect = [SimpleNamespace(stdout='{"devices": {}}'), failure]
            with self.assertRaises(subprocess.CalledProcessError):
                boot_simulator('platform=iOS Simulator,id=fixture')
            self.assertEqual(run.call_count, 2)

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
        steps = self.workflow['jobs']['ui-tests']['steps']
        self.assertTrue(any('ui_tests.py' in s.get('run', '') for s in steps))
        export = next(s for s in steps if s.get('name') == 'Export UI screenshots')
        self.assertEqual(export['if'], '${{ always() }}')
        self.assertIn('xcresulttool export attachments', export['run'])
        self.assertIn('--path apps/ios/build/HexUITests.xcresult', export['run'])
        artifact = next(s for s in steps if s.get('with', {}).get('name') == 'ios-ui-screenshots')
        self.assertEqual(artifact['if'], '${{ always() }}')
        self.assertEqual(self.workflow['jobs']['ui-tests']['needs'], 'app')
        self.assertEqual(self.workflow['jobs']['testflight']['needs'], ['packages', 'app', 'tests', 'ui-tests'])
        self.assertFalse(any('ui_tests.py' in s.get('run', '') for s in self.workflow['jobs']['tests']['steps']))

    def test_fixture_resources_are_debug_only(self):
        script = self.project['targets']['Hex']['postBuildScripts'][0]['script']
        self.assertIn('"$CONFIGURATION" = Debug', script)
        self.assertIn('rm -rf "$fixture_dir"', script)
        self.assertIn('UITestFixtures', script)

    def test_ui_identifiers_resolve_to_app_accessibility_identifiers(self):
        ios = ROOT / 'apps/ios'
        tests = (ios / 'UITests/HexUITests.swift').read_text()
        sources = list((ios / 'App').rglob('*.swift'))
        sources += list((ios / 'Modules').glob('*/Sources/**/*.swift'))
        declared = []
        contract = (ios / 'App/Support/TrainingExperience.swift').read_text()
        constants = dict(re.findall(r'static let (\w+) = "([^"]+)"', contract))
        for path in sources:
            source = path.read_text()
            # Include both arms of the two identifier ternaries in CameraScreen.
            # Build B keeps its public identifier constants in a portable enum.
            for name in re.findall(r'\bTrainingID\.(\w+)', source):
                if name in constants:
                    declared.append(re.escape(constants[name]))
            for expression in re.findall(r'\.accessibilityIdentifier\(([^\n]*(?:\n\s*"[^\n]*)?)', source):
                for identifier in re.findall(r'"([^"]+)"', expression):
                    literal = identifier.split('\\(', 1)[0]
                    pattern = re.escape(literal)
                    if literal != identifier:
                        pattern += '.*'
                    declared.append(pattern)
        used = set(re.findall(
            r'"((?:home|scan|fixture|splat|upload|capture|captures|globe|ask|answer|entity|settings|train|result|inplace|refine)\.[^"\\]+)"',
            tests,
        ))
        self.assertGreater(len(used), 25)
        for identifier in used:
            with self.subTest(identifier=identifier):
                # capture.row. is an NSPredicate prefix; all other values are exact IDs.
                if identifier.endswith('.'):
                    self.assertTrue(any(p.startswith(re.escape(identifier)) for p in declared))
                else:
                    self.assertTrue(any(re.fullmatch(p, identifier) for p in declared), identifier)
