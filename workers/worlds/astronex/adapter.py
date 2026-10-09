"""Pinned upstream adapter with session-resident, one-block inference."""
import json
import os
from pathlib import Path
import sys
from residency import WarmResidentPool
from .controls import CameraControls, NATIVE_ACTIONS

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())
ACTIONS = {"forward": "w", "backward": "s", "left": "a", "right": "d", "look_left": "j", "look_right": "l", "look_up": "i", "look_down": "k", "up": "u", "down": "dn", "stop": "h"}
CAPABILITIES = {
    "input": {"text": True, "image": True, "multiImage": False, "video": False, "audio": False},
    "control": {"wasd": True, "mouseLook": True, "camera6DoF": False, "gamepad": True, "discreteActions": True, "continuousActions": True, "semanticActions": False, "promptDuringRollout": False, "promptSwitching": True, "timedEvents": False, "characterReference": False},
    "output": {"video": True, "audio": False, "depth": False, "cameraPose": False},
    "runtime": {"resolutionOptions": ["832x480"], "realtime": False, "interactionMode": "next-clip"},
    "persistence": {"nativeMemory": False, "snapshotRestore": False, "deterministicSeed": True},
    "customization": {"lora": False, "fineTune": False, "adapters": False},
    "nativeActions": NATIVE_ACTIONS, "resumeKind": "visual-checkpoint",
}


class AstronexAdapter:
    model_id = "astronex-world"
    capabilities = CAPABILITIES

    def __init__(self):
        self.source = Path(os.environ.get("ASTRONEX_SOURCE", "/opt/astronex")).resolve()
        self.weights = Path(os.environ.get("ASTRONEX_WEIGHTS", "/models/astronex")).resolve()
        self.pool = WarmResidentPool()

    def apply_native(self, session, action, values):
        if not hasattr(session, "camera_controls"):
            session.camera_controls = CameraControls()
        session.camera_controls.update(action, values)

    def control_state(self, session):
        return session.camera_controls.snapshot() if hasattr(session, "camera_controls") else {}

    def clear_controls(self, session):
        if hasattr(session, "camera_controls"):
            session.camera_controls.clear()

    def ready(self):
        from .artifacts import check_artifacts
        return check_artifacts(self.source, self.weights, MANIFEST)

    def command(self, directory: Path, prompt: str, image: Path | None, action: str, seed: int, quality: str):
        # The upstream CLI treats a prompt matching a filesystem path as a prompt file.
        # Always pass our own anonymous UTF-8 file; newlines must not create extra jobs.
        prompt_file = directory / "prompt.txt"
        prompt_file.write_text(" ".join(prompt.splitlines()) + "\n")
        frames = 7 if image else 8  # one complete 8-latent-frame block
        command = [os.environ.get("INFERENCE_PYTHON", sys.executable), str(self.source / "inference/generate.py"),
                   "--mode", "causal", "--config", str(self.source / "inference/causal_consumer.yaml"),
                   "--weights", str(self.weights), "--prompt", str(prompt_file),
                   "--frames", str(frames), "--trajectory", f"{ACTIONS[action]}*{frames}",
                   "--seed", str(seed), "--steps", "4" if quality in ("low-latency", "speed") else "8",
                   "--fps", "24", "--out", str(directory)]
        if image:
            command += ["--image", str(image)]
        return command, self.source

    def open_session(self, env, stop_event=None):
        from .resident_client import ResidentClient
        return self.pool.acquire(lambda: ResidentClient(self.source, self.weights, env, stop_event=stop_event))

    def release_session(self, resident, clean):
        self.pool.release(resident, clean)

    def close(self):
        self.pool.close()

    def metadata(self):
        ready, reason = self.ready()
        return {"id": self.model_id, "status": "ready" if ready else "unavailable", "reason": reason,
                "capabilities": self.capabilities, "version": MANIFEST["commit"], "checkpointRevision": MANIFEST["checkpointRevision"],
                "readiness": "artifacts-verified-gpu-unverified" if ready else "artifacts-incomplete", "servingMode": "resident-chunks",
                "warmIdleSeconds": self.pool.idle_seconds, "warmReuse": "reset-acknowledged-clean-boundaries"}
