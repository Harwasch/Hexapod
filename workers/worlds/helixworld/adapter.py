"""Real public preview CLI, preserving its generated audio and offline semantics."""
import json
import os
from pathlib import Path
import sys
from astronex.artifacts import complete_shards

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())
ACTIONS = {"forward": "W", "backward": "S", "left": "A", "right": "D", "look_left": "left", "look_right": "right", "look_up": "up", "look_down": "down", "stop": "stop"}
CAPABILITIES = {
    "input": {"text": True, "image": True, "requiredImage": True, "multiImage": False, "video": False, "audio": False},
    "control": {"wasd": True, "mouseLook": False, "camera6DoF": False, "gamepad": False, "discreteActions": True, "continuousActions": False, "semanticActions": False, "promptDuringRollout": False, "promptSwitching": True, "timedEvents": False, "characterReference": False},
    "output": {"video": True, "audio": True, "depth": False, "cameraPose": False},
    "runtime": {"resolutionOptions": ["768x512"], "qualityOptions": ["balanced"], "realtime": False, "interactionMode": "offline-clip"},
    "persistence": {"nativeMemory": False, "snapshotRestore": False, "deterministicSeed": True},
    "customization": {"lora": False, "fineTune": False, "adapters": False},
    "nativeActions": list(ACTIONS), "resumeKind": "visual-checkpoint",
}


class Adapter:
    model_id = "helix-world"
    capabilities = CAPABILITIES
    offline_clip = True
    output_fps = 24
    output_frames = 121

    def __init__(self):
        self.source = Path(os.environ.get("HELIXWORLD_SOURCE", "/opt/helixworld")).resolve()
        # The released CLI resolves assets relative to its source package. Mount
        # the model volume here instead of silently patching upstream paths.
        self.weights = self.source / "models"

    def ready(self):
        try:
            if (self.source / ".worlds-helix-source").read_text().strip() != MANIFEST["commit"]:
                return False, "Prepare the pinned HelixWorld source first"
            if any(not (self.source / path).is_file() for path in ("infer.py", "scripts/infer_model.py", "configs/model_manifest.json", "code/LTX-2.3/projects/helixworld_runtime/scripts/inference_runtime.py")):
                return False, "HelixWorld inference source is incomplete"
            upstream = json.loads((self.source / "configs/model_manifest.json").read_text())
            if upstream.get("revision") != MANIFEST["checkpointRevision"] or upstream.get("text_encoder", {}).get("revision") != MANIFEST["textEncoderRevision"]:
                return False, "HelixWorld artifact revisions do not match the reviewed release"
            checkpoint = self.weights / "weights/model.safetensors"
            if not checkpoint.is_file() or checkpoint.stat().st_size != MANIFEST["checkpointBytes"]:
                return False, "The pinned HelixWorld checkpoint is missing or incomplete"
            encoder = self.weights / "text_encoder/gemma-3-12b"
            if not complete_shards(encoder, "model.safetensors.index.json") or any(not (encoder / name).is_file() for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
                return False, "The licensed pinned Gemma text encoder is missing or incomplete"
            return True, None
        except (OSError, ValueError, TypeError):
            return False, "Prepare the pinned HelixWorld source and licensed model artifacts"

    def metadata(self):
        ready, reason = self.ready()
        return {"id": self.model_id, "status": "ready" if ready else "unavailable", "reason": reason,
                "capabilities": self.capabilities, "version": MANIFEST["commit"], "checkpointRevision": MANIFEST["checkpointRevision"],
                "servingMode": "offline-clips", "gpuInferenceVerified": False,
                "readiness": "artifacts-verified-gpu-unverified" if ready else "artifacts-incomplete",
                "continuation": "explicit-next-clip-visual-only", "audioContinuity": False}

    def apply_native(self, session, action, values):
        # Offline commands are one-shot plans. Releasing a key must not erase a
        # choice made while waiting for the next explicit generation.
        if action not in ACTIONS:
            raise ValueError("Unsupported HelixWorld camera action")
        if values.get("pressed") is not False:
            session.pending_native_action = action

    def control_state(self, session):
        action = getattr(session, "pending_native_action", "stop")
        session.pending_native_action = "stop"
        return action

    def clear_controls(self, session):
        session.pending_native_action = "stop"

    def command(self, directory, prompt, image, action, seed, quality):
        if image is None or action not in ACTIONS or quality != "balanced":
            raise ValueError("HelixWorld requires an image, supported camera action and balanced quality")
        prompts = directory / "prompts.json"
        # These are descriptions, not audio/video file conditioning. The caller's
        # single world prompt supplies all three required public CLI descriptions.
        prompts.write_text(json.dumps({"video": prompt, "audio": prompt, "av": prompt}, ensure_ascii=False))
        return [os.environ.get("HELIXWORLD_PYTHON", os.environ.get("INFERENCE_PYTHON", sys.executable)), str(Path(__file__).with_name("runner.py")),
                "--source", str(self.source), "--output", str(directory), "--image", str(image), "--prompts", str(prompts),
                "--action", ACTIONS[action], "--seed", str(seed)], self.source
