# HelixWorld Preview adapter

This adapter invokes the actual pinned public inference CLI. **GPU inference,
memory use, performance and installation of its GPU environment are unverified.**
No checkpoints have been downloaded and no GPU test or deployment has been run.
Publisher guidance is an 80 GB NVIDIA GPU (about 70 GB for its reviewed BF16 run).

The public preview accepts an image, camera actions and three text descriptions
of video, audio and their relationship. It does **not** accept audio files or
video clips for conditioning. Names such as `--audio-prompt` refer to text.
The application's world prompt supplies all three descriptions. This is exposed
accurately through `input.audio=false`, `input.video=false` and `output.audio=true`.

The result is a genuine jointly generated 121-frame, 768×512, 24 fps MP4 with
audio. The worker decodes RGB frames and stereo 48 kHz PCM, then delivers video
and generated audio through WebRTC/Opus. The browser initially mutes audio until
the user enables it. HTTP frame fallback cannot deliver audio. Missing model
audio is an error; no soundtrack or silent substitute is generated.

This is an **offline clip** integration. Each request reloads the public CLI's
model, creates one clip, plays it, and pauses automatically. Explicitly resuming
generates another clip from the last visual frame with the current prompt/action.
This continuation is visual only: audio continuity, persistent KV state, exact
resumption and real-time inference are not claimed. A command cannot change a
clip already being sampled.
Camera commands queue a one-clip plan; key release does not discard that plan,
and it is consumed exactly once at the next explicit generation.

## Local preparation by the operator

Prepare source without weights or a GPU:

```sh
python workers/worlds/helixworld/bootstrap.py --source /opt/helixworld
python3.11 -m venv /opt/helixworld-venv
/opt/helixworld-venv/bin/pip install -r workers/worlds/helixworld/requirements.lock
```

The requirements file pins direct dependencies exactly as the reviewed upstream
release; it is not a transitive hash lock. Installation is separate from the
gateway's aiortc environment. System `ffmpeg` and `ffprobe` are required.

Separately prepare authorized model artifacts using upstream's pinned
`download_models.py` and mount them read-only at `/opt/helixworld/models`. The
preview checkpoint is Apache-2.0; its separate Google Gemma text encoder is
subject to Gemma access and license terms. Source `manifest.json` records both
immutable revisions and the preview checkpoint SHA256. Do not substitute the
latest checkpoint revision: upstream pins an earlier release in its own manifest.
The gateway never downloads artifacts or authenticates with Hugging Face.

To verify already prepared artifacts without downloading them:

```sh
/opt/helixworld-venv/bin/python workers/worlds/helixworld/bootstrap.py \
  --source /opt/helixworld --verify-assets
```

Run the existing authenticated worker entrypoint with:

```text
WORLD_MODEL_ID=helix-world
HELIXWORLD_SOURCE=/opt/helixworld
HELIXWORLD_PYTHON=/opt/helixworld-venv/bin/python
```

Its approved provider profile must select an image/environment containing these
dependencies and mounted artifacts. Existing gateway token, lifecycle, budget,
TURN and transport controls apply. The worker uses one GPU session and enforces
image input, balanced quality and 768×512 resolution before launching inference.

An operator-only container recipe is provided at `helixworld/Dockerfile`, with a
digest-pinned Python 3.11 base, separate inference/transport environments, official
dependency versions and a non-root runtime. Its PyTorch wheels supply CUDA/cuDNN;
the NVIDIA container runtime supplies host drivers. Suggested local image name:
`worlds-helix:local`. Build context is the repository root. Mount model
artifacts read-only at `/opt/helixworld/models`. The recipe has **not been built**
and does not download weights. CPU tests do not establish container readiness.

## Evidence and tests

Source revision: `5bc3e0e219aadac6c153748347ccfd6c4bf0e777`. Reviewed files:
`infer.py`, `scripts/infer_model.py`, `scripts/_common.py`, `scripts/controls.py`,
`configs/model_manifest.json`, and the bundled `inference_runtime.py`. The last
file fixes 24 fps / 768×512 and writes `audio_muxed` plus the clean MP4 path in
`release/native/INFERENCE_COMPLETE.json`. The adapter requires that completion
record and a valid contained clip path.

`test_helixworld.py` tests CLI arguments, artifact completeness, containment,
required generated audio, real FFmpeg sine/color fixture decoding and automatic
pause. `test_streaming.py` verifies real PCM-to-Opus transport over local WebRTC.
These fixtures validate software contracts and transport, not model output.
