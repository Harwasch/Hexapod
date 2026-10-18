"""CPU contracts and real software video tests; no GPU inference benchmark."""
from fractions import Fraction
import io
import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astronex.adapter import AstronexAdapter
from astronex.controls import CameraControls, camera_poses
from astronex.resident import AstronexRuntime, ResidentEngine, serve
from astronex.resident_client import ResidentClient
from encoding import H264PacketEncoder, encoder_settings
from gateway import Session, Gateway, ApiError, child_environment
from media import raw_video_frame, validate_raw_frames


class CameraControlTests(unittest.TestCase):
    def test_combined_keys_cancel_opposites_and_release_only_one_axis(self):
        control = CameraControls()
        for action in ("forward", "right", "look_left"):
            control.update(action, {"pressed": True}, now=10)
        self.assertEqual(control.snapshot(now=11), {"forward": 1, "right": 1, "yaw": -1, "pitch": 0})
        control.update("backward", {}, now=11)
        self.assertEqual(control.snapshot(now=11)["forward"], 0)
        control.update("right", {"pressed": False}, now=11)
        self.assertNotIn("right", control.snapshot(now=11))
        self.assertEqual(control.snapshot(now=14), {"yaw": 0, "pitch": 0})

    def test_analog_replace_expire_and_mouse_accumulate_once(self):
        control = CameraControls()
        control.update("move", {"forward": 0.4, "yaw": 0.2}, now=10)
        control.update("look", {"x": 30, "y": -75}, now=10)
        first = control.snapshot(now=11)
        self.assertAlmostEqual(first["yaw"], 0.4)
        self.assertAlmostEqual(first["pitch"], 0.5)
        self.assertAlmostEqual(control.snapshot(now=11)["yaw"], 0.2)
        control.update("analog", {"up": 0.5}, now=11)
        self.assertNotIn("forward", control.snapshot(now=11))
        control.update("move", {"pressed": False}, now=11)
        self.assertNotIn("up", control.snapshot(now=11))

    def test_invalid_unbounded_or_nonfinite_controls_rejected(self):
        for action, values in (("move", {"forward": float("nan")}), ("move", {"yaw": 2}), ("look", {"dx": 501}), ("forward", {"value": True}), ("move", {"unknown": 0})):
            with self.subTest(action=action, values=values), self.assertRaises(ValueError):
                CameraControls().update(action, values)

    def test_camera_pose_matches_pinned_translation_and_rotation_convention(self):
        import numpy as np
        forward = camera_poses({"forward": 1}, 2)
        np.testing.assert_allclose(forward[-1, :3, 3], [0, 0, -0.16], atol=1e-6)
        combined = camera_poses({"forward": 0.5, "right": 0.25, "up": 1}, 1)
        np.testing.assert_allclose(combined[-1, :3, 3], [-0.02, 0.08, -0.04], atol=1e-6)
        yaw = camera_poses({"yaw": 1}, 1)
        self.assertAlmostEqual(float(yaw[-1, 0, 2]), -0.052335956, places=6)


class RawMediaTests(unittest.TestCase):
    def test_raw_spool_bounds_and_lazy_jpeg_cache(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "frames.rgb"
            pixels = bytes([10, 30, 240]) * 8 * 4
            path.write_bytes(pixels)
            raw = {"path": str(path), "width": 8, "height": 4, "count": 1, "pixelFormat": "rgb24"}
            self.assertEqual(validate_raw_frames(raw, root), path)
            session = Session("raw-test-session", "fixture", 1, "balanced", root)
            session.raw_frame, session.raw_size, session.frame_index = pixels, (8, 4), 1
            self.assertIsNone(session.frame)
            jpeg, index = session.latest_jpeg()
            self.assertEqual(index, 1)
            self.assertIs(session.latest_jpeg()[0], jpeg)
            with Image.open(io.BytesIO(jpeg)) as image:
                self.assertGreater(image.getpixel((4, 2))[2], 200)
            for bad in ({**raw, "count": 2}, {**raw, "width": 10000}, {**raw, "path": "/etc/passwd"}):
                with self.assertRaises(RuntimeError):
                    validate_raw_frames(bad, root)

    def test_raw_frame_handles_rgb_stride_padding(self):
        frame = raw_video_frame(bytes([10, 30, 240]) * 7 * 4, (7, 4))
        self.assertEqual(frame.to_image().getpixel((6, 3)), (10, 30, 240))


class PacketEncodingTests(unittest.TestCase):
    def test_nvenc_failure_falls_back_once_to_real_decodable_software_h264(self):
        import av
        attempted = []
        def create(name, mode):
            attempted.append(name)
            if name == "h264_nvenc":
                raise RuntimeError("Synthetic missing GPU fixture")
            return av.CodecContext.create(name, mode)
        encoder = H264PacketEncoder("nvenc", 1000000, codec_factory=create)
        decoder = av.CodecContext.create("h264", "r")
        decoded = []
        for index in range(3):
            frame = raw_video_frame(bytes([10, 30, 240]) * 320 * 180, (320, 180))
            frame.pts, frame.time_base = 3000 * index, Fraction(1, 90000)
            packet = encoder.encode(frame)
            self.assertIsNotNone(packet)
            decoded.extend(decoder.decode(packet))
        self.assertEqual(attempted, ["h264_nvenc", "libx264"])
        self.assertEqual(encoder.encoder, "software-h264")
        self.assertIn("NVENC", encoder.fallback_reason)
        self.assertEqual((decoded[0].width, decoded[0].height), (320, 180))
        self.assertGreater(decoded[0].to_image().getpixel((160, 90))[2], 200)
        encoder.close()
        self.assertIsNone(encoder.codec)
        self.assertIsNone(encoder.encode(frame))
        self.assertEqual(attempted, ["h264_nvenc", "libx264"])

    def test_encoder_config_is_bounded_and_opt_in(self):
        self.assertEqual(encoder_settings({}), ("software", 3000000))
        for env in ({"WORLD_VIDEO_ENCODER": "magic"}, {"WORLD_VIDEO_BITRATE": "0"}, {"WORLD_VIDEO_BITRATE": "12000001"}):
            with self.assertRaises(ValueError):
                encoder_settings(env)


class WarmIsolationTests(unittest.TestCase):
    def client(self):
        return SimpleNamespace(process=SimpleNamespace(poll=lambda: None), closed=threading.Event(), reset=Mock(), close=Mock())

    def test_only_reset_acknowledged_clean_process_is_reused(self):
        adapter = AstronexAdapter()
        client = self.client()
        try:
            adapter.release_session(client, clean=True)
            client.reset.assert_called_once()
            self.assertIs(adapter.open_session({}), client)
            adapter.release_session(client, clean=False)
            client.close.assert_called_once()
            self.assertIsNone(adapter.pool.warm)
        finally:
            adapter.close()

    def test_failed_reset_is_destroyed_and_idle_expiry_closes_resident(self):
        adapter = AstronexAdapter()
        failed, expired = self.client(), self.client()
        failed.reset.side_effect = RuntimeError("Missing acknowledgement")
        try:
            adapter.release_session(failed, clean=True)
            failed.close.assert_called()
            self.assertIsNone(adapter.pool.warm)
            adapter.release_session(expired, clean=True)
            adapter.pool.expire(expired)
            expired.close.assert_called_once()
            self.assertIsNone(adapter.pool.warm)
        finally:
            adapter.close()

    def test_runtime_reset_erases_latent_text_vae_and_all_attention_caches(self):
        runtime = AstronexRuntime.__new__(AstronexRuntime)
        runtime.last_latent = object()
        runtime.device = "fixture-no-gpu"
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock()))
        runtime.pipeline = SimpleNamespace(text_encoder=SimpleNamespace(_cache={"private": object()}), vae=SimpleNamespace(model=SimpleNamespace(clear_cache=Mock())),
                                           _CACHES=("kv_cache_pos", "kv_cache_neg", "crossattn_cache_pos"), kv_cache_pos=object(), kv_cache_neg=object(), crossattn_cache_pos=object(), _prope_mirror=[object()], _prev_retrieved=[object()], _cache_keep=object())
        runtime.reset()
        self.assertIsNone(runtime.last_latent)
        self.assertEqual(runtime.pipeline.text_encoder._cache, {})
        self.assertTrue(all(getattr(runtime.pipeline, name) is None for name in runtime.pipeline._CACHES))
        self.assertEqual(runtime.pipeline._prope_mirror, [])
        runtime.pipeline.vae.model.clear_cache.assert_called_once()

    def test_engine_reset_protocol_preserves_weights_and_resets_block_counter(self):
        runtime = SimpleNamespace(reset=Mock(), close=Mock())
        engine = ResidentEngine(lambda: runtime)
        engine.runtime, engine.blocks, engine.load_count = runtime, 4, 1
        output = io.StringIO()
        serve(io.StringIO('{"op":"reset"}\n{"op":"shutdown"}\n'), output, engine)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies[1], {"type": "reset", "sessionStateCleared": True})
        runtime.reset.assert_called_once()
        self.assertEqual(engine.blocks, 0)
        self.assertEqual(engine.load_count, 1)

    def test_client_reset_round_trip_keeps_process_alive_and_rejects_bad_ack(self):
        for accepted in (True, False):
            code = 'import json,sys; print(json.dumps({"type":"ready","protocolVersion":1}),flush=True)\nfor line in sys.stdin:\n print(json.dumps({"type":"reset","sessionStateCleared":' + str(accepted) + '}),flush=True)\n'
            client = ResidentClient("/fixture", "/fixture", child_environment(), command=[sys.executable, "-u", "-c", code])
            try:
                if accepted:
                    client.reset()
                    self.assertIsNone(client.process.poll())
                else:
                    with self.assertRaisesRegex(RuntimeError, "isolation"):
                        client.reset()
                    self.assertIsNotNone(client.process.poll())
            finally:
                client.close()


class GatewayPluginTests(unittest.TestCase):
    def test_image_only_adapter_enforces_model_resolution_and_quality_before_open(self):
        from PIL import Image
        output = io.BytesIO()
        Image.new("RGB", (4, 4)).save(output, "PNG")
        reference = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        adapter = SimpleNamespace(model_id="fixture-image-model", ready=lambda: (True, None), capabilities={
            "input": {"text": False, "image": True, "requiredImage": True}, "runtime": {"resolutionOptions": ["640x352"], "qualityOptions": ["balanced"]}, "nativeActions": []})
        gateway = Gateway(adapter)
        body = {"modelId": adapter.model_id, "input": {"prompt": "", "images": [reference]}, "resolution": "640x352"}
        try:
            for changes in ({"input": {"prompt": "ignored text", "images": [reference]}}, {"input": {"prompt": ""}}, {"resolution": "832x480"}, {"quality": "quality"}):
                with self.assertRaises(ApiError):
                    gateway.create({**body, **changes})
                self.assertEqual(gateway.sessions, {})
            with patch.object(gateway, "_generate"):
                created = gateway.create(body)
            self.assertEqual(created["modelId"], adapter.model_id)
            with self.assertRaises(ApiError):
                gateway.action(created["id"], {"type": "native", "action": "forward"})
            with self.assertRaises(ApiError):
                gateway.action(created["id"], {"type": "prompt", "prompt": "Not supported"})
        finally:
            gateway.close()

    def test_raw_resident_output_reaches_lazy_http_snapshot_and_cleans_up(self):
        from PIL import Image
        class Client:
            process = None
            closed = False
            def generate(self, directory, *args, **kwargs):
                path, last = directory / "frames.rgb", directory / "last.png"
                path.write_bytes(bytes([10, 30, 240]) * 8 * 4 * 24)
                Image.new("RGB", (8, 4), (10, 30, 240)).save(last)
                return {"rawFrames": {"path": str(path), "width": 8, "height": 4, "count": 24, "pixelFormat": "rgb24"}, "continuationPath": str(last), "fps": 24, "generationSeconds": 0.2, "loadCount": 1}
            def close(self):
                self.closed = True
        client = Client()
        adapter = SimpleNamespace(model_id="fixture-raw-model", ready=lambda: (True, None), capabilities={"input": {"text": True}}, open_session=lambda *args, **kwargs: client)
        gateway = Gateway(adapter)
        try:
            created = gateway.create({"modelId": adapter.model_id, "input": {"prompt": "Synthetic no inference"}})
            session = gateway.get(created["id"])
            deadline = time.monotonic() + 3
            while session.raw_frame is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(session.raw_frame)
            self.assertIsNone(session.frame)
            self.assertTrue(session.latest_jpeg()[0].startswith(b"\xff\xd8"))
            self.assertEqual(session.total_generated_frames, 24)
            self.assertEqual(session.generated_fps, 120)
            gateway.delete(session.id)
            self.assertTrue(client.closed)
            self.assertFalse(session.directory.exists())
            self.assertIsNone(session.raw_frame)
        finally:
            gateway.close()


if __name__ == "__main__":
    unittest.main()
