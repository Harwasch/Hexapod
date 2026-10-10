"""CPU-only contract and lifecycle tests; fixture media is never production output."""
import base64
import json
import io
import os
from pathlib import Path
import sys
import struct
import tempfile
import threading
import time
import unittest
import zlib
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway import ApiError, Gateway, Session, child_environment, handler_for, image_bytes
from astronex.adapter import AstronexAdapter, CAPABILITIES


class FixtureAdapter:
    model_id = "astronex-world"
    capabilities = CAPABILITIES
    def ready(self):
        return True, None
    def metadata(self):
        return {"id": self.model_id, "status": "ready", "capabilities": self.capabilities}
    def command(self, directory, *_):
        return ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=128x72:r=12", "-t", "0.25", "-c:v", "libx264", "-y", str(directory / "chunk.mp4")], None


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.gateway = Gateway(FixtureAdapter())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.gateway, "t" * 40))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.gateway.close()
    def request(self, method, path, body=None, token="t" * 40):
        request = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                         headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            return response.status, json.loads(data) if data and response.headers.get_content_type() == "application/json" else data
    def create(self, **updates):
        body = {"id": "anonymous-test-001", "modelId": "astronex-world", "input": {"prompt": "Private scene"}, "seed": 42}
        body.update(updates)
        return self.request("POST", "/sessions", body)
    def test_auth_and_health(self):
        self.assertEqual(self.request("GET", "/health", token="wrong")[0], 401)
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["models"][0]["id"], "astronex-world")
        self.assertFalse(health["models"][0]["capabilities"]["runtime"]["realtime"])
    def test_invalid_startup_configuration_fails_before_creating_worker(self):
        from gateway import main
        for updates in ({"WORLD_GATEWAY_TOKEN": "private token " * 4}, {"WORLD_IDLE_SECONDS": "0"}, {"WORLD_IDLE_SECONDS": "invalid"}):
            with self.subTest(updates=updates), patch.dict(os.environ, {"WORLD_GATEWAY_TOKEN": "t" * 40, **updates}), patch("gateway.Gateway") as create:
                with self.assertRaises(SystemExit):
                    main()
                create.assert_not_called()
    def test_shutdown_rejects_new_sessions_and_closes_transport_on_cleanup_failure(self):
        gateway = Gateway(FixtureAdapter())
        gateway.sessions["pending-session"] = object()
        try:
            with patch.object(gateway, "delete", side_effect=ApiError(503, "Still stopping")), patch.object(gateway.streaming, "close", wraps=gateway.streaming.close) as close:
                with self.assertRaises(ApiError):
                    gateway.close()
                close.assert_called_once()
            self.assertTrue(gateway.root.exists())
            with self.assertRaises(ApiError) as error:
                gateway.create({"id": "shutdown-test-001", "modelId": "astronex-world", "input": {"prompt": "Private"}})
            self.assertEqual(error.exception.status, 503)
        finally:
            gateway.sessions.clear()
            gateway.close()
        self.assertFalse(gateway.root.exists())
    def test_real_decode_snapshot_and_deletion(self):
        status, created = self.create()
        self.assertEqual(status, 201)
        identifier = created["id"]
        deadline = time.monotonic() + 10
        frame = b""
        while time.monotonic() < deadline:
            code, frame = self.request("GET", f"/sessions/{identifier}/frame")
            if code == 200:
                break
            time.sleep(0.05)
        self.assertTrue(frame.startswith(b"\xff\xd8"), "Fixture video must be decoded to an actual JPEG")
        code, snapshot = self.request("POST", f"/sessions/{identifier}/snapshot", {})
        self.assertEqual(code, 200)
        self.assertEqual(snapshot["resumeKind"], "visual-checkpoint")
        self.assertIsNone(snapshot["state"])
        self.assertTrue(snapshot["frame"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(self.request("POST", f"/sessions/{identifier}/offer", {})[0], 400)
        self.assertEqual(self.request("DELETE", f"/sessions/{identifier}")[0], 204)
        self.assertFalse((self.gateway.root / identifier).exists())
        self.assertEqual(self.request("GET", f"/sessions/{identifier}")[0], 404)
    def test_reject_unsupported_or_sensitive_inputs(self):
        self.assertEqual(self.create(id="../../outside")[0], 400)
        self.assertEqual(self.create(modelId="imagined-model")[0], 422)
        self.assertEqual(self.create(input={"prompt": "Private", "images": ["http://169.254.169.254/"]})[0], 400)
        self.assertEqual(self.create(input={"prompt": "Private", "images": ["x", "y"]})[0], 422)
        self.assertEqual(self.create(seed=-1)[0], 400)
        self.assertEqual(self.create(resolution="1280x704")[0], 422)
    def test_native_and_prompt_actions_are_explicitly_next_clip(self):
        self.assertEqual(self.create()[0], 201)
        path = "/sessions/anonymous-test-001/actions"
        code, result = self.request("POST", path, {"type": "native", "action": "forward"})
        self.assertEqual(code, 200)
        self.assertEqual(result["appliesAt"], "next-clip")
        self.assertEqual(self.request("POST", path, {"type": "semantic", "prompt": "attack"})[0], 422)
        self.assertEqual(self.request("POST", path, {"type": "native", "action": "attack"})[0], 422)
        self.assertEqual(self.request("POST", path, {"type": "pause"})[0], 200)
        self.assertEqual(self.request("GET", "/sessions/anonymous-test-001")[1]["status"], "paused")
        self.assertEqual(self.request("POST", path, {"type": "prompt", "prompt": "Storm"})[0], 200)
    def test_null_seed_gets_a_reported_effective_seed(self):
        status, created = self.create(seed=None)
        self.assertEqual(status, 201)
        self.assertIsInstance(created["seed"], int)
        self.assertGreaterEqual(created["seed"], 0)
        self.assertLess(created["seed"], 2**32)
    def test_key_release_never_requeues_motion_or_cancels_newer_key(self):
        # Avoid the generation thread consuming the queue while checking ordering.
        with patch.object(self.gateway, "_generate"):
            self.assertEqual(self.create()[0], 201)
        session = self.gateway.get("anonymous-test-001")
        path = "/sessions/anonymous-test-001/actions"
        self.request("POST", path, {"type": "native", "action": "forward", "values": {"pressed": True}})
        self.assertEqual(session.action, "forward")
        self.request("POST", path, {"type": "native", "action": "right", "values": {"pressed": True}})
        self.request("POST", path, {"type": "native", "action": "forward", "values": {"pressed": False}})
        self.assertEqual(session.action, "right")
        self.request("POST", path, {"type": "native", "action": "right", "values": {"pressed": False}})
        self.assertEqual(session.action, "stop")
    def test_single_gpu_session_limit(self):
        self.assertEqual(self.create()[0], 201)
        self.assertEqual(self.create(id="anonymous-test-002")[0], 409)
    def test_held_actions_refresh_and_expire_without_a_release(self):
        with patch.object(self.gateway, "_generate"):
            self.assertEqual(self.create()[0], 201)
        session = self.gateway.get("anonymous-test-001")
        self.gateway.action(session.id, {"type": "native", "action": "forward", "values": {"pressed": True}})
        expiry = session.action_expires_at
        self.assertEqual(session.current_action(expiry - 0.5), "forward")
        self.assertEqual(session.current_action(expiry + 0.1), "stop")
        self.gateway.action(session.id, {"type": "native", "action": "forward", "values": {"pressed": True, "held": True}})
        self.assertGreaterEqual(session.action_expires_at, expiry)
        self.assertEqual(session.current_action(session.action_expires_at - 0.1), "forward")
        self.gateway.action(session.id, {"type": "native", "action": "forward", "values": {"pressed": False}})
        self.assertEqual(session.current_action(), "stop")


class AdapterTests(unittest.TestCase):
    def test_prompt_is_file_and_chunk_shape_matches_upstream(self):
        adapter = AstronexAdapter()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            command, _ = adapter.command(directory, "../../secret\nsecond job", None, "forward", 42, "balanced")
            self.assertEqual(command[command.index("--frames") + 1], "8")
            self.assertEqual(command[command.index("--trajectory") + 1], "w*8")
            self.assertNotIn("../../secret", command)
            self.assertEqual((directory / "prompt.txt").read_text(), "../../secret second job\n")
            command, _ = adapter.command(directory, "Image", directory / "image.png", "look_left", 1, "low-latency")
            self.assertEqual(command[command.index("--frames") + 1], "7")
            self.assertEqual(command[command.index("--trajectory") + 1], "j*7")
            self.assertEqual(command[command.index("--steps") + 1], "4")
    def test_no_cloud_credentials_or_telemetry_reach_model_process(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "secret", "WORLD_GATEWAY_TOKEN": "secret", "AWS_SECRET_ACCESS_KEY": "secret", "HF_TOKEN": "secret"}):
            env = child_environment()
            for name in ("RUNPOD_API_KEY", "WORLD_GATEWAY_TOKEN", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"):
                self.assertNotIn(name, env)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
    def test_not_ready_without_source_and_weights(self):
        with patch.dict(os.environ, {"ASTRONEX_SOURCE": "/nonexistent", "ASTRONEX_WEIGHTS": "/nonexistent"}):
            self.assertFalse(AstronexAdapter().ready()[0])
    def test_image_validation(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(buffer, "PNG")
        data = buffer.getvalue()
        self.assertEqual(image_bytes("data:image/png;base64," + base64.b64encode(data).decode()), data)
        with self.assertRaises(ApiError):
            image_bytes("data:image/png;base64," + base64.b64encode(b"#EXTM3U\nfile:///private").decode())
        with self.assertRaises(ApiError):
            image_bytes("file:///etc/passwd")
        # Oversized IHDR is rejected before pixel decompression, even with a tiny upload.
        large_header = bytearray(data)
        large_header[16:24] = struct.pack('>II', 5000, 5000)
        large_header[29:33] = struct.pack('>I', zlib.crc32(large_header[12:29]))
        with self.assertRaises(ApiError) as error:
            image_bytes("data:image/png;base64," + base64.b64encode(large_header).decode())
        self.assertEqual(error.exception.status, 413)
    def test_process_group_cancellation_removes_files(self):
        class BlockingAdapter(FixtureAdapter):
            def command(self, *_):
                return [sys.executable, "-c", "import time; time.sleep(120)"], None
        gateway = Gateway(BlockingAdapter())
        try:
            gateway.create({"id": "cancel-test-001", "modelId": "astronex-world", "input": {"prompt": "Private"}})
            session = gateway.get("cancel-test-001")
            deadline = time.monotonic() + 3
            while session.process is None and time.monotonic() < deadline:
                time.sleep(0.01)
            process = session.process
            self.assertIsNotNone(process)
            gateway.delete(session.id)
            self.assertIsNotNone(process.poll())
            self.assertFalse(session.directory.exists())
        finally:
            gateway.close()


if __name__ == "__main__":
    unittest.main()
