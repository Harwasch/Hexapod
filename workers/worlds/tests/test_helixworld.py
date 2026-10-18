"""CPU-only Helix CLI/media contracts; sine/color fixtures are never AI output."""
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helixworld.adapter import Adapter, CAPABILITIES, MANIFEST
from helixworld.runner import completed_clip
from gateway import Gateway, Session, ApiError


class HelixContractTests(unittest.TestCase):
    def test_offline_action_is_an_explicit_one_clip_plan_not_a_held_motion(self):
        adapter, session = Adapter(), SimpleNamespace()
        adapter.apply_native(session, "forward", {"pressed": True})
        adapter.apply_native(session, "forward", {"pressed": False})
        self.assertEqual(adapter.control_state(session), "forward")
        self.assertEqual(adapter.control_state(session), "stop")
        adapter.apply_native(session, "left", {})
        adapter.clear_controls(session)
        self.assertEqual(adapter.control_state(session), "stop")

    def test_real_cli_receives_text_descriptions_and_cannot_claim_file_conditioning(self):
        adapter = Adapter()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = directory / "reference.png"
            image.write_bytes(b"fixture")
            command, cwd = adapter.command(directory, "Waves and seabirds", image, "forward", 42, "balanced")
            self.assertIn("runner.py", command[1])
            self.assertEqual(command[command.index("--action") + 1], "W")
            self.assertEqual(json.loads((directory / "prompts.json").read_text()), {"video": "Waves and seabirds", "audio": "Waves and seabirds", "av": "Waves and seabirds"})
            self.assertFalse(CAPABILITIES["input"]["audio"])
            self.assertFalse(CAPABILITIES["input"]["video"])
            self.assertTrue(CAPABILITIES["output"]["audio"])
            self.assertEqual(CAPABILITIES["runtime"]["interactionMode"], "offline-clip")
            with self.assertRaises(ValueError):
                adapter.command(directory, "Private", None, "forward", 42, "balanced")

    def test_completed_clip_requires_real_muxed_audio_and_private_output_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "release/native"
            folder.mkdir(parents=True)
            clip = folder / "clip.mp4"
            clip.write_bytes(b"fixture-not-a-real-video")
            marker = folder / "INFERENCE_COMPLETE.json"
            for outputs in ({"audio_muxed": False, "clean_av_mp4": str(clip)}, {"audio_muxed": True, "clean_av_mp4": "/etc/passwd"}):
                marker.write_text(json.dumps({"outputs": outputs}))
                with self.assertRaises(RuntimeError):
                    completed_clip(root)
            marker.write_text(json.dumps({"outputs": {"audio_muxed": True, "clean_av_mp4": str(clip)}}))
            self.assertEqual(completed_clip(root), clip)

    def test_readiness_is_artifact_only_and_pins_both_checkpoint_and_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Adapter()
            adapter.source, adapter.weights = Path(temp), Path(temp) / "models"
            self.assertFalse(adapter.ready()[0])
            paths = ("infer.py", "scripts/infer_model.py", "configs/model_manifest.json", "code/LTX-2.3/projects/helixworld_runtime/scripts/inference_runtime.py")
            for name in paths:
                path = adapter.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            (adapter.source / ".worlds-helix-source").write_text(MANIFEST["commit"])
            (adapter.source / "configs/model_manifest.json").write_text(json.dumps({"revision": MANIFEST["checkpointRevision"], "text_encoder": {"revision": MANIFEST["textEncoderRevision"]}}))
            checkpoint = adapter.weights / "weights/model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test")
            encoder = adapter.weights / "text_encoder/gemma-3-12b"
            encoder.mkdir(parents=True)
            for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "shard.safetensors"):
                (encoder / name).write_text("fixture")
            (encoder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "shard.safetensors"}}))
            with patch.dict(MANIFEST, {"checkpointBytes": 4}):
                self.assertTrue(adapter.ready()[0])
                self.assertFalse(adapter.metadata()["gpuInferenceVerified"])
                (encoder / "shard.safetensors").unlink()
                self.assertFalse(adapter.ready()[0])


class OfflineAudioGatewayTests(unittest.TestCase):
    def test_planned_action_survives_paused_wait_and_is_consumed_once(self):
        adapter = Adapter()
        gateway = Gateway(adapter)
        session = Session("planned-clip-test", "Fixture", 1, "balanced", gateway.root / "planned")
        session.directory.mkdir()
        gateway.sessions[session.id] = session
        try:
            gateway.action(session.id, {"type": "native", "action": "right", "values": {"planned": True}})
            with patch("time.monotonic", return_value=time.monotonic() + 60):
                self.assertEqual(session.current_action(), "stop")
                self.assertEqual(adapter.control_state(session), "right")
                self.assertEqual(adapter.control_state(session), "stop")
            adapter.offline_clip = False
            with self.assertRaises(ApiError) as error:
                gateway.action(session.id, {"type": "native", "action": "right", "values": {"planned": True}})
            self.assertEqual(error.exception.status, 422)
        finally:
            gateway.close()

    def test_real_ffmpeg_audio_is_preserved_and_clip_pauses_without_auto_generation(self):
        class Fixture:
            model_id = "synthetic-audio-fixture"
            offline_clip = True
            output_fps = 24
            output_frames = 24
            capabilities = {"input": {"text": True}, "runtime": {"resolutionOptions": ["320x180"], "interactionMode": "offline-clip"}, "output": {"audio": True}}
            def ready(self):
                return True, None
            def command(self, directory, *args):
                return ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(directory / "chunk.mp4")], None
        gateway = Gateway(Fixture())
        try:
            created = gateway.create({"modelId": "synthetic-audio-fixture", "input": {"prompt": "Explicit synthetic audio test"}})
            session = gateway.get(created["id"])
            deadline = time.monotonic() + 10
            while session.chunks < 1 and not session.error and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNone(session.error)
            self.assertEqual(session.chunks, 1)
            self.assertEqual(session.status, "paused")
            self.assertFalse(session.resumed.is_set())
            self.assertGreater(len(session.audio), 100000)
            self.assertNotEqual(session.audio, bytes(len(session.audio)))
            self.assertIsNotNone(session.raw_frame)
            self.assertEqual(session.public(Fixture.capabilities)["interactionMode"], "offline-clip")
            time.sleep(0.1)
            self.assertEqual(session.chunks, 1)
            gateway.delete(session.id)
            self.assertIsNone(session.audio)
        finally:
            gateway.close()


if __name__ == "__main__":
    unittest.main()
