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
        '-test-timeouts-enabled', 'YES', '-default-test-execution-time-allowance', '300',
        '-jobs', '4', '-parallel-testing-enabled', 'NO', '-skipMacroValidation',
        'CODE_SIGNING_ALLOWED=NO',
        'HEX_API_BASE_URL=' + os.environ.get('HEX_API_BASE_URL', 'https://twin-api.fly.dev'),
        'HEX_WEB_BASE_URL=' + os.environ.get('HEX_WEB_BASE_URL', 'https://twin-web-f57.pages.dev'),
    ]


def boot_simulator(destination):
    identifier = destination.split('id=', 1)[1]
    # simctl boot reports an error for an already booted device. Check state so a
    # real boot failure still fails the job, then wait for services to be ready.
    state = subprocess.run(
        ['xcrun', 'simctl', 'list', 'devices', 'booted', '--json'],
        capture_output=True, text=True, check=True,
    )
    booted = {d['udid'] for group in json.loads(state.stdout)['devices'].values() for d in group}
    if identifier not in booted:
        subprocess.run(['xcrun', 'simctl', 'boot', identifier], check=True)
    subprocess.run(['xcrun', 'simctl', 'bootstatus', identifier, '-b'], check=True)


def main():
    devices = subprocess.run(
        ['xcrun', 'simctl', 'list', 'devices', 'available', '--json'],
        capture_output=True, text=True, check=True,
    )
    destination = simulator_destination(json.loads(devices.stdout)['devices'])
    boot_simulator(destination)
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
