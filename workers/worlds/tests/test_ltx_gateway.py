"""CPU lifecycle fixtures, never substitutes for LTX GPU acceptance."""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway import ApiError, Gateway
from ltx25.adapter import CAPABILITIES, Adapter, LTXClient, validate_audio
from ltx25.controls import ExplorationControls


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Fixture condition did not complete')


class FixtureResident:
    process = None

    def __init__(self, block=False):
        self.closed = threading.Event()
        self.calls = []
        self.seeds = []
        self.block = block

    def close(self):
        self.closed.set()

    def generate(self, directory, prompt, image, action, seed, quality, stop, on_phase=None):
        self.calls.append((prompt, action))
        self.seeds.append(seed)
        if self.block:
            while not stop.wait(0.01) and not self.closed.is_set():
                pass
            raise InterruptedError()
        count = 12
        raw = directory / 'frames.rgb'
        raw.write_bytes(b'\x00' * (count * 2 * 2 * 3))
        continuation = directory / 'continuation.png'
        continuation.write_bytes(b'fixture')
        audio = directory / 'audio.pcm'
        samples = count * 2000
        audio.write_bytes(b'\x00' * samples * 4)
        return {'rawFrames': {'path': str(raw), 'width': 2, 'height': 2, 'count': count, 'pixelFormat': 'rgb24'},
                'audio': {'path': str(audio), 'sampleRate': 48000, 'channels': 2, 'sampleFormat': 's16le', 'samples': samples},
                'continuationPath': str(continuation), 'fps': 24, 'nativeKVContinuity': False,
                'generationSeconds': 0.01, 'appliedRevision': action['revision']}


class FixtureAdapter(Adapter):
    def __init__(self, block=False):
        self.client = FixtureResident(block)

    def ready(self):
        return True, None

    def open_session(self, *_args, **_kwargs):
        return self.client

    def release_session(self, resident, clean):
        resident.close()

    def close(self):
        self.client.close()


class ContinuousGatewayTests(unittest.TestCase):
    def start(self, block=False):
        adapter = FixtureAdapter(block)
        gateway = Gateway(adapter)
        self.addCleanup(gateway.close)
        created = gateway.create({'id': 'ltx-test-session', 'modelId': 'ltx-2.5', 'input': {'prompt': 'Original world'}, 'quality': 'quality'})
        return gateway, adapter.client, gateway.get(created['id'])

    def test_prefetch_is_bounded_pause_retains_and_resume_applies_prompt(self):
        gateway, client, session = self.start()
        wait_until(lambda: len(client.calls) == 2 and session.frame_index > 0)
        gateway.action(session.id, {'type': 'pause'})
        time.sleep(0.15)
        self.assertEqual(len(client.calls), 2, 'A queued future packet must block a third generation')
        frames = session.frame_index
        time.sleep(0.1)
        self.assertEqual(session.frame_index, frames)
        self.assertEqual(session.status, 'paused')
        self.assertEqual(client.calls[0][1]['exploration']['cadence'], 'smooth')
        ack = gateway.action(session.id, {'type': 'prompt', 'prompt': 'Make it rain'})
        self.assertEqual(ack['appliesAt'], 'next-chunk')
        resumed = gateway.action(session.id, {'type': 'resume'})
        wait_until(lambda: session.applied_revision == resumed['revision'])
        self.assertEqual(client.calls[2][0], 'Make it rain')
        self.assertEqual(set(client.seeds), {session.seed})
        self.assertEqual(session.public(CAPABILITIES)['queuedRevision'], resumed['revision'])
        self.assertLessEqual(len(list(session.directory.glob('chunk-*'))), 2)
        gateway.delete(session.id)
        self.assertTrue(client.closed.is_set())
        self.assertFalse(session.directory.exists())

    def test_delete_cancels_inflight_producer_and_cleans_private_spool(self):
        gateway, client, session = self.start(block=True)
        wait_until(lambda: len(client.calls) == 1)
        gateway.delete(session.id)
        self.assertTrue(client.closed.is_set())
        self.assertFalse(session.thread.is_alive())
        self.assertFalse(session.directory.exists())

    def test_exploration_validation_does_not_mutate_settings_on_failure(self):
        gateway, _client, session = self.start(block=True)
        ack = gateway.action(session.id, {'type': 'native', 'action': 'exploration', 'values': {'mode': 'cruise', 'speed': 0.8}})
        self.assertEqual(ack['revision'], 1)
        with self.assertRaises(ApiError):
            gateway.action(session.id, {'type': 'native', 'action': 'exploration', 'values': {'mode': 'walk', 'speed': float('nan')}})
        self.assertEqual(session.exploration_controls.settings['mode'], 'cruise')
        self.assertEqual(session.queued_revision, 1)

    def test_renewal_revisions_and_bounded_amendments_preserve_base(self):
        gateway, _client, session = self.start(block=True)
        body = {'type': 'native', 'action': 'forward', 'values': {'pressed': True}}
        first = gateway.action(session.id, body)
        self.assertEqual(gateway.action(session.id, body)['revision'], first['revision'])
        for index in range(8):
            gateway.action(session.id, {'type': 'prompt', 'prompt': 'Change ' + str(index)})
        state = gateway.adapter.control_state(session)
        self.assertEqual(state['basePrompt'], 'Original world')
        self.assertEqual(state['promptAmendments'], ['Change ' + str(index) for index in range(2, 8)])
        unchanged = gateway.action(session.id, {'type': 'prompt', 'prompt': 'Change 7'})
        self.assertEqual(unchanged['revision'], session.queued_revision)
        with self.assertRaises(ApiError):
            gateway.action(session.id, {'type': 'prompt', 'prompt': 'x' * 3001})
        gateway.action(session.id, {'type': 'prompt', 'prompt': 'a' * 2000})
        gateway.action(session.id, {'type': 'prompt', 'prompt': 'b' * 2000})
        self.assertEqual(session.prompt_amendments, ['b' * 2000])

    def test_bad_producer_packet_stops_and_cleans_spools(self):
        adapter = FixtureAdapter()
        generate = adapter.client.generate
        def invalid(*args, **kwargs):
            result = generate(*args, **kwargs)
            result['audio']['samples'] = 1
            return result
        adapter.client.generate = invalid
        gateway = Gateway(adapter)
        self.addCleanup(gateway.close)
        created = gateway.create({'id': 'ltx-bad-session', 'modelId': 'ltx-2.5', 'input': {'prompt': 'World'}})
        session = gateway.get(created['id'])
        wait_until(lambda: session.status == 'error')
        self.assertTrue(adapter.client.closed.is_set())
        self.assertFalse(list(session.directory.glob('chunk-*')))
        self.assertIn('invalid length', session.error)

    def test_explicit_resume_restores_cruise_but_not_explicit_stop(self):
        gateway, _client, session = self.start(block=True)
        gateway.action(session.id, {'type': 'native', 'action': 'exploration', 'values': {'mode': 'cruise'}})
        gateway.action(session.id, {'type': 'pause'})
        self.assertFalse(any(gateway.adapter.control_state(session)['motion'].values()))
        gateway.action(session.id, {'type': 'resume'})
        self.assertEqual(gateway.adapter.control_state(session)['motion']['forward'], 1)
        gateway.action(session.id, {'type': 'native', 'action': 'stop'})
        gateway.action(session.id, {'type': 'pause'})
        gateway.action(session.id, {'type': 'resume'})
        self.assertFalse(any(gateway.adapter.control_state(session)['motion'].values()))

    def test_ltx_escaped_prompt_payload_has_independent_bounded_limit(self):
        from astronex.resident_client import ResidentClient
        payloads = []
        def client(kind):
            value = object.__new__(kind)
            value.closed = threading.Event()
            value.request_lock = threading.Lock()
            value.timeout = 1
            value.close = lambda: None
            value._send = lambda payload, *_args: payloads.append(payload)
            value._receive = lambda *_args: {'type': 'result', 'result': {}}
            value._validate_output = lambda *_args: None
            return value
        action = {'basePrompt': '\0😀' * 4000, 'promptAmendments': ['\0' * 3000]}
        with tempfile.TemporaryDirectory() as directory:
            args = (directory, '\0' * 3000, None, action, 1, 'balanced', threading.Event())
            with self.assertRaises(ValueError):
                client(ResidentClient).generate(*args)
            client(LTXClient).generate(*args)
        self.assertEqual(len(payloads), 1)
        self.assertGreater(len(payloads[0]), 64 * 1024)
        self.assertLessEqual(len(payloads[0]), 128 * 1024)

    def test_audio_validation_enforces_private_path_and_exact_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / 'outside.pcm'
            outside.write_bytes(b'\0' * 4)
            private = root / 'private'
            private.mkdir()
            descriptor = {'path': str(outside), 'sampleRate': 48000, 'channels': 2, 'sampleFormat': 's16le', 'samples': 1}
            with self.assertRaises(RuntimeError):
                validate_audio(descriptor, private)
            inside = private / 'audio.pcm'
            inside.write_bytes(b'\0' * 4)
            self.assertEqual(validate_audio(dict(descriptor, path=str(inside)), private), inside)
            with self.assertRaises(RuntimeError):
                validate_audio(dict(descriptor, path=str(inside), samples=True), private)


class ControlTests(unittest.TestCase):
    def test_cruise_stop_and_combined_analog_controls(self):
        controls = ExplorationControls()
        controls.update('exploration', {'mode': 'cruise'})
        self.assertEqual(controls.snapshot(0)['motion']['forward'], 1)
        controls.update('look', {'dx': 30})
        steering = controls.snapshot(0)['motion']
        self.assertEqual(steering['forward'], 1)
        self.assertGreater(steering['yaw'], 0)
        controls.update('stop', {})
        self.assertFalse(any(controls.snapshot(1)['motion'].values()))
        controls.update('analog', {'forward': .5, 'right': .25, 'yaw': -.75})
        self.assertEqual(controls.snapshot(2)['motion']['yaw'], -.75)
        controls.clear()
        self.assertFalse(any(controls.snapshot(3)['motion'].values()))


if __name__ == '__main__':
    unittest.main()
