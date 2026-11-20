"""Build/test every local package with the iOS SDK, retaining all failures."""

import json
import os
from pathlib import Path
import subprocess
import sys

IOS = Path(__file__).resolve().parents[1]


def simulator_destination(devices):
    """Prefer iPhone 17 Pro, otherwise the newest available iPhone/runtime."""
    candidates = []
    for runtime, entries in devices.items():
        if '.iOS-' not in runtime:
            continue
        version = tuple(int(part) for part in runtime.split('.iOS-')[1].split('-'))
        for device in entries:
            name = device['name']
            if device.get('isAvailable') and name.startswith('iPhone'):
                # Model number first, with Pro before the base model on fallback.
                model = int(name.split()[1]) if name.split()[1].isdigit() else 0
                candidates.append(((name == 'iPhone 17 Pro', version, model, 'Pro' in name, name), device['udid']))
    if not candidates:
        raise RuntimeError('No available iPhone simulator; install an iOS simulator runtime.')
    return f'platform=iOS Simulator,id={max(candidates)[1]}'


def run_packages(mode, ios=IOS, destination=None):
    logs = ios / 'build' / ('package-logs' if mode == 'build' else 'test-logs')
    logs.mkdir(parents=True, exist_ok=True)
    rows = ['| Package | Result |', '| --- | --- |']
    failed = False
    packages = sorted((ios / 'Modules').glob('*/Package.swift')) + [ios / 'App/Package.swift']
    for manifest in packages:
        label = str(manifest.parent.relative_to(ios))
        log_path = logs / f'{manifest.parent.name}.log'
        with log_path.open('w') as log:
            try:
                result = subprocess.run(
                    ['swift', 'package', 'dump-package'], cwd=manifest.parent,
                    stdout=subprocess.PIPE, stderr=log, text=True, check=True,
                )
                package = json.loads(result.stdout)
                if mode == 'test' and not any(t['type'] == 'test' for t in package['targets']):
                    rows.append(f'| {label} | No test targets |')
                    continue
                scheme = package['name']
                command = [
                    'xcodebuild', mode, '-scheme', scheme,
                    '-destination', destination or 'generic/platform=iOS Simulator',
                    '-derivedDataPath', str(manifest.parent / '.build' / 'DerivedData'),
                    '-jobs', '4', '-skipMacroValidation', 'CODE_SIGNING_ALLOWED=NO',
                ]
                log.write(f'Command: {command!r}\n')
                log.flush()
                status = subprocess.run(command, cwd=manifest.parent, stdout=log, stderr=subprocess.STDOUT).returncode
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                log.write(f'{error}\n')
                status = 1
        rows.append(f'| {label} | {"PASS" if status == 0 else "FAIL"} |')
        failed = failed or status != 0
    summary = '\n'.join(rows) + '\n'
    print(summary, flush=True)
    (logs / 'summary.md').write_text(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
            output.write(summary)
    return int(failed)


def main():
    mode = sys.argv[1]
    if mode not in ('build', 'test'):
        raise ValueError('Expected build or test')
    destination = None
    if mode == 'test':
        result = subprocess.run(
            ['xcrun', 'simctl', 'list', 'devices', 'available', '--json'],
            capture_output=True, text=True, check=True,
        )
        destination = simulator_destination(json.loads(result.stdout)['devices'])
        print(f'Test destination: {destination}', flush=True)
    return run_packages(mode, destination=destination)


if __name__ == '__main__':
    sys.exit(main())
