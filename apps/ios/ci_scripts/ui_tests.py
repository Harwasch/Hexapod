"""Run app UI tests on the same available simulator policy as the package gate."""
import json
import os
import shutil
import subprocess
import sys

from packages import IOS, simulator_destination


def ui_command(destination, ios=IOS):
    return [
        'xcodebuild', 'test', '-project', 'Hex.xcodeproj', '-scheme', 'HexUITests',
        '-configuration', 'Debug', '-destination', destination,
        '-derivedDataPath', str(ios / 'build/UI-DerivedData'),
        '-resultBundlePath', str(ios / 'build/HexUITests.xcresult'),
        '-jobs', '4', '-parallel-testing-enabled', 'NO', '-skipMacroValidation',
        'CODE_SIGNING_ALLOWED=NO',
        'HEX_API_BASE_URL=' + os.environ.get('HEX_API_BASE_URL', 'https://twin-api.fly.dev'),
        'HEX_WEB_BASE_URL=' + os.environ.get('HEX_WEB_BASE_URL', 'https://twin-web-f57.pages.dev'),
    ]


def main():
    devices = subprocess.run(
        ['xcrun', 'simctl', 'list', 'devices', 'available', '--json'],
        capture_output=True, text=True, check=True,
    )
    destination = simulator_destination(json.loads(devices.stdout)['devices'])
    result = IOS / 'build/HexUITests.xcresult'
    if result.exists():
        shutil.rmtree(result)
    result.parent.mkdir(parents=True, exist_ok=True)
    command = ui_command(destination)
    print(f'UI test destination: {destination}', flush=True)
    with (IOS / 'build/ui-tests.log').open('w') as log:
        status = subprocess.run(command, cwd=IOS, stdout=log, stderr=subprocess.STDOUT).returncode
    print((IOS / 'build/ui-tests.log').read_text()[-12000:])
    return status


if __name__ == '__main__':
    sys.exit(main())
