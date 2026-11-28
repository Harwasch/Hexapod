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


def swift_group_end(source, start):
    """End of a Swift delimiter group, ignoring strings and comments."""
    pairs = {'(': ')', '{': '}', '[': ']'}
    stack = []
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|[(){}\[\]]'
    for match in re.finditer(tokens, source[start:]):
        token = match.group()
        if token in pairs:
            stack.append(pairs[token])
        elif token in pairs.values():
            if not stack or stack.pop() != token:
                raise AssertionError('Unbalanced Swift group')
            if not stack:
                return start + match.end()
    raise AssertionError('Unclosed Swift group')


def confirmation_cancel_identifiers(source, constants):
    """Find identifiers on cancel Buttons in confirmationDialog actions."""
    identifiers = set()
    for dialog in re.finditer(r'\.confirmationDialog\s*\(', source):
        start = dialog.end() - 1
        end = swift_group_end(source, start)
        trailing = re.match(r'\s*\{', source[end:])
        if trailing:
            end = swift_group_end(source, end + trailing.end() - 1)
        actions = source[start:end]
        for button in re.finditer(r'\bButton\s*\(', actions):
            head = button.end() - 1
            cursor = swift_group_end(actions, head)
            if not re.search(r'role\s*:\s*\.cancel\b', actions[head:cursor]):
                continue
            action = re.match(r'\s*\{', actions[cursor:])
            if action:
                cursor = swift_group_end(actions, cursor + action.end() - 1)
            label = re.match(r'\s*label\s*:\s*\{', actions[cursor:])
            if label:
                cursor = swift_group_end(actions, cursor + label.end() - 1)
            while modifier := re.match(r'\s*\.(\w+)\s*\(', actions[cursor:]):
                opening = cursor + modifier.end() - 1
                cursor = swift_group_end(actions, opening)
                if modifier[1] == 'accessibilityIdentifier':
                    value = actions[opening + 1:cursor - 1].strip()
                    if value.startswith('TrainingID.'):
                        identifiers.add(constants[value.split('.', 1)[1]])
                    else:
                        identifiers.add(json.loads(value))
    return identifiers


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

    def test_screen_identifiers_preserve_child_controls(self):
        # Run 22 rendered the controls, but SwiftUI propagated camera.screen to
        # every descendant. Declaration-only identifier audits cannot catch that.
        for filename, identifiers in {
            'CameraScreen.swift': ['camera.screen'],
            'HomeScreen.swift': ['capture.detail', 'globe.screen', 'settings.screen'],
        }.items():
            source = (ROOT / 'apps/ios/App' / filename).read_text()
            for identifier in identifiers:
                with self.subTest(identifier=identifier):
                    pattern = (r'\.accessibilityElement\(children: \.contain\)\s*'
                               r'\.accessibilityIdentifier\("' + re.escape(identifier) + r'"\)')
                    self.assertIsNotNone(re.search(pattern, source),
                                         f'{filename}: {identifier} needs its own container')

    def test_ui_fixture_keeps_uikit_presentations_available(self):
        source = (ROOT / 'apps/ios/App/HexApp.swift').read_text()
        # Keep native presentations working while SwiftUI quiets fixture views.
        self.assertNotIn('UIView.setAnimationsEnabled(false)', source)
        self.assertIn('transaction.disablesAnimations = true', source)

    def test_scan_sheet_identifiers_belong_to_buttons(self):
        source = (ROOT / 'apps/ios/Modules/Globe/Sources/Globe/GlobeView.swift').read_text()
        # Run 23 never presented the Menu. Use ordinary sheet Buttons, including
        # in production, so automation and VoiceOver follow the same controls.
        self.assertNotIn('Menu {', source)
        self.assertNotIn('Picker(', source)
        self.assertIn('Button { showScans = true } label:', source)
        self.assertIn('.sheet(isPresented: $showScans)', source)
        sheet = source.split('.sheet(isPresented: $showScans)', 1)[1]
        self.assertIn('.presentationDetents([.large])', sheet.split('private func scanButton', 1)[0])
        self.assertIn('Section("Scan")', source)
        self.assertIn('scanButton(scan)', source)
        button = source.split('private func scanButton', 1)[1]
        button = button.split('private func openScan', 1)[0]
        self.assertIn('let scanID = scan.captureId ?? scan.id', button)
        self.assertIn('selectedScan = scanID', button)
        self.assertIn('showScans = false', button)
        self.assertIn('if selectedScan == scanID', button)
        self.assertIn('Label(globeScanDate(scan.observedAt), systemImage: "checkmark")', button)
        self.assertIn('Text(globeScanDate(scan.observedAt))', button)
        self.assertIn('.accessibilityIdentifier("globe.scan.', button)
        self.assertIn('.accessibilityLabel(globeScanDate(scan.observedAt))', button)
        self.assertIn('.frame(minHeight: 44)', button)
        self.assertIn('.frame(maxWidth: .infinity, alignment: .leading)', button)
        self.assertIn('.contentShape(Rectangle())', button)

    def test_ui_queried_cancel_buttons_are_not_in_confirmation_popovers(self):
        ios = ROOT / 'apps/ios'
        tests = (ios / 'UITests/HexUITests.swift').read_text()
        # Include queries as well as taps; use actual TrainingID values, not names.
        used = set(re.findall(r'"([a-z][a-zA-Z0-9]*\.[^"\\]+)"', tests))
        contract = (ios / 'App/Support/TrainingExperience.swift').read_text()
        constants = dict(re.findall(r'static let (\w+) = "([^"]+)"', contract))
        sources = list((ios / 'App').rglob('*.swift'))
        sources += list((ios / 'Modules').glob('*/Sources/**/*.swift'))
        for path in sources:
            with self.subTest(path=path.relative_to(ios)):
                hidden = confirmation_cancel_identifiers(path.read_text(), constants)
                self.assertFalse(used & hidden, f'{path}: iOS popovers hide cancel buttons {used & hidden}')

    def test_cancel_popover_guard_handles_nested_actions_and_ignores_alerts(self):
        source = '''
        .confirmationDialog("Cost?", isPresented: $show) {
          Button("Confirm") { Task { await save() } }
            .accessibilityIdentifier("refine.confirm")
          Button("Cancel", role: .cancel) { /* a } is not a delimiter */ }
            .accessibilityIdentifier(TrainingID.decline)
        } message: { Text("Cost {estimate}") }
        .alert("Cost?", isPresented: $show) {
          Button("Cancel", role: .cancel) {}.accessibilityIdentifier("alert.cancel")
        } message: { Text("Cost") }
        .confirmationDialog("Delete?", isPresented: $delete) {
          Button("Delete", role: .destructive) {}.accessibilityIdentifier("capture.confirmDelete")
          Button("Cancel", role: .cancel) {}
        }
        .confirmationDialog("Choice?", isPresented: $choice, actions: {
          Button(role: .cancel) {} label: { Text("Cancel") }
            .accessibilityIdentifier("choice.cancel")
        })
        '''
        self.assertEqual(confirmation_cancel_identifiers(source, {'decline': 'refine.cancel'}),
                         {'refine.cancel', 'choice.cancel'})

    def test_both_refine_cost_confirmations_use_alerts_with_cancel(self):
        for filename in ('TrainingScreen.swift', 'HomeScreen.swift'):
            source = (ROOT / 'apps/ios/App' / filename).read_text()
            alert = re.search(r'\.alert\(\s*"Refine in cloud\?",\s*isPresented: \$refineConfirmation\s*\)\s*\{', source)
            self.assertIsNotNone(alert, filename)
            actions = source[alert.end() - 1:swift_group_end(source, alert.end() - 1)]
            self.assertIn('.accessibilityIdentifier(TrainingID.confirm)', actions)
            self.assertIn('Button("Cancel", role: .cancel)', actions)
            self.assertIn('.accessibilityIdentifier(TrainingID.decline)', actions)

    def test_ui_actions_wait_for_state_and_scroll_to_capture_actions(self):
        source = (ROOT / 'apps/ios/UITests/HexUITests.swift').read_text()
        self.assertNotRegex(source, r'\b(?:sleep|usleep|Thread\.sleep)\s*\(')
        self.assertNotIn('Scan picker retry', source)
        self.assertIn('scrollTo("capture.delete")', source)
        self.assertIn('waitFor(NSPredicate(format: "label == %@", "Saved")', source)
        self.assertIn('waitFor(NSPredicate(format: "selected == true")', source)
        self.assertIn('"label CONTAINS %@", "Stored in cloud"', source)

    def test_fixture_transitions_do_not_start_spinners_or_training_animations(self):
        camera = (ROOT / 'apps/ios/App/CameraScreen.swift').read_text()
        training = (ROOT / 'apps/ios/App/TrainingScreen.swift').read_text()
        self.assertIn('private struct CameraActivityIndicator: View', camera)
        self.assertIn('if LaunchMode.current.isUITest {', camera.split('private struct CameraActivityIndicator:', 1)[1])
        self.assertIn('closing && !LaunchMode.current.isUITest', training)
        self.assertNotRegex(training, r'\.animation\(\.ease')
        self.assertEqual(training.count('.animation(LaunchMode.current.isUITest ? nil : .ease'), 3)

    def test_fixture_rendering_and_catalog_tasks_are_finite(self):
        ios = ROOT / 'apps/ios'
        camera = (ios / 'App/CameraScreen.swift').read_text()
        clock = camera.split('private struct CameraTimeline', 1)[1]
        self.assertRegex(clock, r'if LaunchMode.current.isUITest \{\s*content\(LaunchMode.current.date\(\)\)')
        training = (ios / 'App/TrainingScreen.swift').read_text()
        self.assertRegex(training, r'await model.loadExisting\(\)\s*if LaunchMode.current.isUITest \{ return \}')
        self.assertIn('if LaunchMode.current.isUITest || cloudPollDecision == .stop { return }', training)
        home = (ios / 'App/HomeScreen.swift').read_text()
        self.assertIn('if LaunchMode.current.isUITest || captures.contains', home)
        self.assertIn('if LaunchMode.current.isUITest || capture.serverUnavailable { return }', home)
        self.assertIn('if LaunchMode.current.isUITest { return }', camera)

    def test_queried_controls_keep_their_accessibility_constructs(self):
        ios = ROOT / 'apps/ios'
        camera = (ios / 'App/CameraScreen.swift').read_text()
        home = (ios / 'App/HomeScreen.swift').read_text()
        training = (ios / 'App/TrainingScreen.swift').read_text()
        # Source-level guards complement the identifier inventory: Text, Image and
        # Button are not interchangeable in XCUI queries (run 23 refine.progress).
        cases = [
            (home, r'SecureField\("Write token"[\s\S]*?\.accessibilityIdentifier\("settings.token"\)'),
            (home, r'Button\(saved \? "Saved" : "Sign in"\)[\s\S]*?\.accessibilityIdentifier\("settings.signIn"\)'),
            (home, r'Button\("Sign out"\)[\s\S]*?\.accessibilityIdentifier\("settings.signOut"\)'),
            (home, r'Text\("Providers"\)\s*\.accessibilityIdentifier\("settings.providers"\)\s*\.accessibilityValue\('),
            (home, r'Text\(registry.configuration.baseURL[^\n]*\)[\s\S]*?\.accessibilityIdentifier\("settings.server"\)'),
            (home, r'Image\(systemName: "iphone"\)\.accessibilityLabel\("Phone splat"\)\s*\.accessibilityIdentifier\(TrainingID.phoneBadge\)'),
            (home, r'Image\(systemName: "cloud.fill"\)\.accessibilityLabel\("Cloud splat"\)\s*\.accessibilityIdentifier\(TrainingID.cloudBadge\)'),
            (home, r'NavigationLink\s*\{[\s\S]*?\.accessibilityElement\(children: \.contain\)\s*\.accessibilityIdentifier\("capture.row.'),
            (camera, r'Button\(action: model.toggleRecording\)[\s\S]*?\.accessibilityIdentifier\(model.recording \? "scan.stop" : "home.record"\)'),
            (camera, r'Button\(action: train\)[\s\S]*?\.accessibilityIdentifier\(TrainingID.start\)'),
            (camera, r'Button\(action: done\)[\s\S]*?\.accessibilityIdentifier\("upload.done"\)'),
            (camera, r'\.accessibilityLabel\("Subject \\\(pin.id \+ 1\)"\)\s*\.accessibilityIdentifier\("answer.pin.'),
            (training, r'Button\("Refine in cloud"\)[\s\S]*?\.accessibilityIdentifier\(TrainingID.confirm\)'),
            (training, r'Button\("Cancel", role: \.cancel\)[^\n]*\.accessibilityIdentifier\(TrainingID.decline\)'),
            (training, r'ShareLink\(item: url\)[^\n]*\n\s*\.accessibilityLabel\("Share splat"\)\.accessibilityIdentifier\(TrainingID.share\)'),
        ]
        for source, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertIsNotNone(re.search(pattern, source))
        self.assertIn('app.secureTextFields[id].firstMatch : app.buttons[id].firstMatch',
                      (ios / 'UITests/HexUITests.swift').read_text())

    def test_unsigned_fixture_credentials_bypass_security_before_keychain_access(self):
        source = (ROOT / 'apps/ios/App/Support/APICredentials.swift').read_text()
        self.assertIn('#if DEBUG\n    private static let uiTestTokenStore', source)
        self.assertIn('mode.isUITest ? UITestTokenStore() : nil', source)
        read = source.split('static var writeToken: String {', 1)[1]
        write = source.split('static func setWriteToken(_ value: String) throws {', 1)[1]
        self.assertLess(read.index('if let uiTestTokenStore'), read.index('SecItemCopyMatching'))
        self.assertLess(write.index('uiTestTokenStore.setWriteToken(value)'), write.index('SecItemDelete'))

    def test_training_status_has_one_explicit_accessible_label(self):
        source = (ROOT / 'apps/ios/App/TrainingScreen.swift').read_text()
        status = source.split('Label(cloudLabel, systemImage:', 1)[1]
        status = status.split('.accessibilityIdentifier(TrainingID.refineProgress)', 1)[0]
        # Label's image must not also match refine.progress as the first element.
        self.assertIn('.accessibilityElement(children: .ignore)', status)
        self.assertIn('.accessibilityLabel(cloudLabel)', status)

    def test_fixture_preview_task_has_content_before_image_loads(self):
        source = (ROOT / 'apps/ios/App/CameraScreen.swift').read_text()
        preview = source.split('@ViewBuilder private var cameraSurface:', 1)[1]
        preview = preview.split('if model.recording,', 1)[0]
        # A task on Group { if let image { ... } } has no view to attach to
        # initially. It must run before the image or the first replay frame exists.
        self.assertRegex(preview, r'ZStack\s*\{\s*CameraStyle\.background\s+if let fixtureBackground')
        self.assertIn('.task(id: scenePhase)', preview)
        self.assertIn('model.beginUITestPreview(size: geometry.size)', preview)

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


if __name__ == '__main__':
    unittest.main()
