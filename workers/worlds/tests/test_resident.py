"""CPU protocol/lifecycle tests; no world-model inference is claimed."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astronex.resident import AstronexRuntime, ResidentEngine, generator_state, serve, validate_request
from astronex.resident_client import ResidentClient
from gateway import child_environment


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.closed = False
    def generate(self, request):
        self.calls.append(request)
        return {"generatedFrames": 29, "fps": 24, "generationSeconds": 2, "generatedFPS": 14.5}
    def close(self):
        self.closed = True


class ResidentEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.request = {"op": "generate", "prompt": "Private world", "seed": 42, "quality": "balanced", "action": "forward", "directory": self.temp.name, "image": None}
    def tearDown(self):
        self.temp.cleanup()
    def test_load_once_for_multiple_blocks_and_clear_on_close(self):
        loads = []
        def factory():
            runtime = FakeRuntime()
            loads.append(runtime)
            return runtime
        engine = ResidentEngine(factory)
        phases = []
        first = engine.generate(self.request, on_phase=phases.append)
        second = engine.generate({**self.request, "prompt": "Snow", "action": "look_left"}, on_phase=phases.append)
        self.assertEqual(phases, ["loading-model", "generating", "generating"])
        self.assertEqual(len(loads), 1)
        self.assertEqual(second["loadCount"], 1)
        self.assertEqual(second["blocks"], 2)
        self.assertFalse(second["nativeKVContinuity"])
        self.assertEqual(second["continuation"], "last-clean-latent")
        self.assertEqual(first["loadSeconds"], second["loadSeconds"])
        self.assertEqual(loads[0].calls[1]["action"], "look_left")
        engine.close()
        self.assertIsNone(engine.runtime)
        self.assertTrue(loads[0].closed)
    def test_cpu_host_fails_before_importing_cuda_initializing_upstream_module(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        original_directory, original_path = os.getcwd(), list(sys.path)
        try:
            with patch.dict(sys.modules, {"torch": fake_torch, "omegaconf": SimpleNamespace(OmegaConf=object())}), patch.dict(os.environ):
                with self.assertRaisesRegex(RuntimeError, "CUDA GPU is required"):
                    AstronexRuntime(Path(self.temp.name), Path(self.temp.name))
                # The CPU fixture has no upstream inference module. Reaching the
                # explicit hardware error proves imports did not initialize CUDA.
        finally:
            os.chdir(original_directory)
            sys.path[:] = original_path

    def test_session_runtime_is_never_shared(self):
        one, two = ResidentEngine(FakeRuntime), ResidentEngine(FakeRuntime)
        one.generate(self.request)
        two.generate(self.request)
        self.assertIsNot(one.runtime, two.runtime)
        one.close()
        self.assertFalse(two.runtime.closed)
        two.close()
    def test_protocol_scrubs_raw_exception_and_stops(self):
        class BrokenRuntime(FakeRuntime):
            def generate(self, request):
                raise RuntimeError("SECRET-PROMPT /private/secret.token")
        engine = ResidentEngine(BrokenRuntime)
        output = io.StringIO()
        serve(io.StringIO(json.dumps(self.request) + "\n"), output, engine)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies[0]["type"], "ready")
        self.assertEqual(replies[-1]["type"], "error")
        self.assertNotIn("SECRET", output.getvalue())
        self.assertIsNone(engine.runtime)
    def test_oversized_request_never_loads_model(self):
        output = io.StringIO()
        engine = ResidentEngine(lambda: self.fail("Model must not load"))
        serve(io.StringIO("x" * 65537), output, engine)
        self.assertIn("invalid-request", output.getvalue())
        self.assertEqual(engine.load_count, 0)
    def test_only_verified_unused_readout_is_filtered(self):
        state = {"generator": {"model.action_head.0.weight": 1, "model.action_head.9.weight": 2,
                               "model._fsdp_wrapped_module.action_embedder.proj.0.weight": 3}}
        normalized = generator_state(state)
        self.assertNotIn("model.action_head.0.weight", normalized)
        self.assertIn("model.action_head.9.weight", normalized)  # strict load must reject surprises
        self.assertEqual(normalized["model.action_embedder.proj.0.weight"], 3)
    def test_invalid_controls_and_escaping_paths_rejected_before_load(self):
        for changes in ({"action": "attack"}, {"seed": -1}, {"directory": "relative"}, {"quality": "max-fps"}, {"prompt": "x" * 8001}):
            with self.assertRaises(ValueError):
                validate_request({**self.request, **changes})


class ResidentClientTests(unittest.TestCase):
    def client(self, code, timeout=5):
        return ResidentClient("/unused-source", "/unused-weights", child_environment(), command=[sys.executable, "-u", "-c", code], timeout=timeout)
    def test_repeated_requests_use_same_subprocess(self):
        code = '''
import json,sys,pathlib
print(json.dumps({"type":"ready","protocolVersion":1}),flush=True)
count=0
for line in sys.stdin:
 request=json.loads(line);count+=1
 folder=pathlib.Path(request["directory"]); frame=folder/"frame.jpg"; frame.write_bytes(b"test")
 result={"framePaths":[str(frame)],"continuationPath":str(frame),"fps":24,"nativeKVContinuity":False,"loadCount":1,"blocks":count}
 print(json.dumps({"type":"result","result":result}),flush=True)
'''
        client = self.client(code)
        try:
            pid = client.process.pid
            with tempfile.TemporaryDirectory() as temp:
                first = client.generate(temp, "one", None, "forward", 42, "balanced", threading.Event())
                second = client.generate(temp, "two", None, "right", 43, "balanced", threading.Event())
                self.assertEqual(first["blocks"], 1)
                self.assertEqual(second["blocks"], 2)
                self.assertEqual(client.process.pid, pid)
                self.assertIsNone(client.process.poll())
        finally:
            client.close()
        self.assertIsNotNone(client.process.poll())
    def test_cancellation_terminates_process_group(self):
        client = self.client('import json,time; print(json.dumps({"type":"ready","protocolVersion":1}),flush=True); time.sleep(120)')
        stop = threading.Event()
        errors = []
        with tempfile.TemporaryDirectory() as temp:
            def generate():
                try:
                    client.generate(temp, "private", None, "stop", 1, "balanced", stop)
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=generate)
            thread.start()
            stop.set()
            thread.join(timeout=5)
            client.close()
            self.assertFalse(thread.is_alive())
            self.assertTrue(any(isinstance(error, InterruptedError) for error in errors))
            self.assertIsNotNone(client.process.poll())
    def test_cancellation_before_ready_terminates_startup(self):
        stop = threading.Event()
        failures = []
        timer = threading.Timer(0.1, stop.set)
        started = time.monotonic()
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                ResidentClient("/unused", "/unused", child_environment(), command=[sys.executable, "-c", "import time; time.sleep(120)"], stop_event=stop)
            self.assertLess(time.monotonic() - started, 3)
        finally:
            timer.cancel()
    def test_timeout_kills_stalled_resident(self):
        client = self.client('import json,time; print(json.dumps({"type":"ready","protocolVersion":1}),flush=True); time.sleep(120)', timeout=0.1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    client.generate(temp, "private", None, "stop", 1, "balanced", threading.Event())
            self.assertIsNotNone(client.process.poll())
        finally:
            client.close()
    def test_unicode_prompt_limit_uses_utf8_not_ascii_escape_expansion(self):
        code = '''
import json,sys,pathlib
print(json.dumps({"type":"ready","protocolVersion":1}),flush=True)
request=json.loads(sys.stdin.readline())
assert len(request["prompt"]) == 8000
folder=pathlib.Path(request["directory"]);frame=folder/"frame.jpg";frame.write_bytes(b"test")
print(json.dumps({"type":"result","result":{"framePaths":[str(frame)],"continuationPath":str(frame),"fps":24,"nativeKVContinuity":False}}),flush=True)
'''
        client = self.client(code)
        try:
            with tempfile.TemporaryDirectory() as temp:
                result = client.generate(temp, "🌲" * 8000, None, "stop", 42, "balanced", threading.Event())
                self.assertEqual(result["fps"], 24)
        finally:
            client.close()
    def test_large_prompt_timeout_is_effective_while_child_is_not_reading(self):
        client = self.client('import json,time; print(json.dumps({"type":"ready","protocolVersion":1}),flush=True); time.sleep(120)', timeout=0.1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    client.generate(temp, "🌲" * 8000, None, "stop", 42, "balanced", threading.Event())
                self.assertLess(time.monotonic() - started, 3)
                self.assertIsNotNone(client.process.poll())
        finally:
            client.close()

    def test_private_frame_directory_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "frame.jpg"
            frame.write_bytes(b"test")
            with self.assertRaisesRegex(RuntimeError, "outside"):
                ResidentClient._validate_output({"framePaths": [str(frame)], "continuationPath": str(frame), "fps": 24, "nativeKVContinuity": False}, root / "session")


if __name__ == "__main__":
    unittest.main()
