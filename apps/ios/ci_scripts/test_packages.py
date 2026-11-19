"""Exercise collection and simulator selection without Apple tools or a network."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import packages


class PackageRunnerTests(unittest.TestCase):
    def test_build_collects_failures_and_still_builds_app_support(self):
        with tempfile.TemporaryDirectory() as directory:
            ios = Path(directory)
            for name in ('Modules/First', 'Modules/Second', 'App'):
                path = ios / name / 'Package.swift'
                path.parent.mkdir(parents=True)
                path.touch()
            calls = []

            def run(command, **kwargs):
                parent = kwargs['cwd'].name
                if command[0] == 'swift':
                    return subprocess.CompletedProcess(command, 0, json.dumps({
                        'name': 'AppSupport' if parent == 'App' else parent,
                        'targets': [{'type': 'test'}],
                    }))
                calls.append(command)
                kwargs['stdout'].write(f'raw log for {parent}\n')
                return subprocess.CompletedProcess(command, int(parent == 'First'))

            summary = ios / 'step-summary'
            with patch.object(packages.subprocess, 'run', side_effect=run), patch.dict(
                packages.os.environ, {'GITHUB_STEP_SUMMARY': str(summary)}
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(packages.run_packages('build', ios), 1)
            self.assertEqual([c[c.index('-scheme') + 1] for c in calls], ['First', 'Second', 'AppSupport'])
            for command in calls:
                self.assertIn('generic/platform=iOS Simulator', command)
                self.assertIn('-skipMacroValidation', command)
                self.assertIn('CODE_SIGNING_ALLOWED=NO', command)
            self.assertIn('| Modules/First | FAIL |', summary.read_text())
            self.assertIn('| App | PASS |', summary.read_text())
            self.assertEqual(len(list((ios / 'build/package-logs').glob('*.log'))), 3)

    def test_manifest_failure_does_not_stop_other_packages_and_tests_skip_empty_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            ios = Path(directory)
            for name in ('Modules/Broken', 'Modules/Empty', 'App'):
                (ios / name).mkdir(parents=True)
                (ios / name / 'Package.swift').touch()
            commands = []

            def run(command, **kwargs):
                parent = kwargs['cwd'].name
                if command[0] == 'swift':
                    if parent == 'Broken':
                        raise subprocess.CalledProcessError(1, command)
                    return subprocess.CompletedProcess(command, 0, json.dumps({
                        'name': 'AppSupport', 'targets': [] if parent == 'Empty' else [{'type': 'test'}],
                    }))
                commands.append(command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(packages.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(packages.run_packages('test', ios, 'platform=iOS Simulator,id=chosen'), 1)
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][1], 'test')
            self.assertIn('platform=iOS Simulator,id=chosen', commands[0])
            self.assertIn('No test targets', (ios / 'build/test-logs/summary.md').read_text())

    def test_simulator_preference_fallback_and_unavailable_devices(self):
        def device(name, udid, available=True):
            return {'name': name, 'udid': udid, 'isAvailable': available}

        devices = {
            'com.apple.CoreSimulator.SimRuntime.iOS-26-0': [device('iPhone 17 Pro', 'preferred')],
            'com.apple.CoreSimulator.SimRuntime.iOS-27-0': [
                device('iPhone 18', 'newest'), device('iPhone 18 Pro', 'missing', False),
            ],
        }
        self.assertTrue(packages.simulator_destination(devices).endswith('id=preferred'))
        del devices['com.apple.CoreSimulator.SimRuntime.iOS-26-0']
        self.assertTrue(packages.simulator_destination(devices).endswith('id=newest'))
        with self.assertRaisesRegex(RuntimeError, 'No available iPhone'):
            packages.simulator_destination({})
