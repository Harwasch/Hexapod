# Runtime controls, encoding, and residency

All model inference and NVENC performance remain **GPU-unverified**. Development
tests use CPU fixtures, real software encoders, real WebRTC peers and Chromium.
No GPU is provisioned by the tests. The synthetic transport fixture is never used
by production inference.

## Model boundary

`WORLD_MODEL_ID` selects one adapter for each worker through `adapters.py`.
The gateway reads its input requirements, native actions, resolution and quality
options before opening the inference subprocess. It does not substitute another
model when artifacts are missing. Native/prompt controls remain capability gated.
The existing public `open_session`/`generate` contract supports either JPEG frame
paths or a `rawFrames` result. This allows independently installed model runtimes
to share the authenticated gateway and transport environment.

Raw results contain a private file path, `pixelFormat: rgb24`, width, height and
count. Validation limits a block to 240 frames, 1920×1080 and 192 MiB, including
Matrix's 57-frame first block and HelixWorld's 121-frame offline clip. Playback holds only the latest immutable RGB frame
in memory, and WebRTC constructs an AV frame directly. HTTP frame fallback and
visual snapshots encode the latest frame to JPEG only when requested. One PNG
continuation is retained between blocks. This removes a lossy JPEG round-trip;
it is not zero-copy GPU encoding, and the raw spool requires adequate local disk
or a suitably sized tmpfs. Generated throughput excludes model loading and
includes frame transfer/spooling. Offline CLI adapters include their model load
in total generation time, because that boundary is not separately measured.
Playback remains paced separately.

## Encoder modes

| Setting                                  | Behavior                                                                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `WORLD_VIDEO_ENCODER=software` (default) | aiortc negotiated software VP8/H264 encoder, with its normal bitrate/keyframe feedback.                                              |
| `WORLD_VIDEO_ENCODER=h264-software`      | Explicit libx264 packet encoder for local testing or fixed-bitrate H264.                                                             |
| `WORLD_VIDEO_ENCODER=nvenc`              | Try PyAV `h264_nvenc` on the first real frame. On initialization or encoding failure, permanently switch that connection to libx264. |
| `WORLD_VIDEO_BITRATE=3000000`            | Fixed bitrate for packet modes, bounded to 250,000–12,000,000 bits/second.                                                           |

Packet modes use aiortc's public `av.Packet` track interface and select H264 in
SDP. They do not patch aiortc private encoder internals. NVENC requires a PyAV
FFmpeg build containing that encoder and NVIDIA video driver libraries exposed
to the worker; codec presence alone does not prove the driver works. Configuration
reports `pending-probe` until real encoding succeeds; DataChannel telemetry
reports `nvenc`, `software-h264`, or the default `software`, with a sanitized
fallback reason. No successful hardware probe is claimed by CPU tests.

Preencoded packet tracks cannot use aiortc's private PLI/REMB encoder state. They
send an IDR at least once per second of delivered video and expose `bitrateMode:
fixed`. This is a deliberate tradeoff, not adaptive bitrate support. The default
software path remains available where congestion feedback is more important.
TURN configuration and authenticated offer/control semantics are unchanged.

## Astronex control protocol

These controls affect the **next generation block**. They do not make model
inference real-time or alter a block already being sampled.

- Existing directional actions accept `{pressed: true, value?: 0..1}` and
  `{pressed: false}`. Multiple held actions compose, opposing keys cancel, and
  releasing one action does not cancel another. Clients refresh held state;
  each action expires after two seconds without renewal.
- `move` (alias `analog`) accepts a complete snapshot of normalized
  `{forward, right, up, yaw, pitch}` axes in `[-1,1]`. The snapshot replaces the
  prior analog state and expires after two seconds. `pressed: false` clears it.
- `look` accepts mouse pixel deltas `{dx,dy}` (aliases `{x,y}`), each bounded to
  `[-500,500]`. Deltas accumulate with bounded magnitude and are consumed once at
  the next block. Positive X turns right; negative Y looks up.
- `stop` and pause clear all held, analog and accumulated mouse state.

Camera transforms mirror pinned `utils/camera_trajectory.py`: 0.08 translation
units and three degrees of rotation per latent frame, in OpenCV camera space.
The pinned camera API accepts combined poses. Its inference API resets KV and
cross-attention caches on each call (source commit
`27584bf1a89da01ef35a03b48d83bc818d2adf2b`), and its consumer profile may free them
before decoding. This adapter therefore retains only the final clean latent for
continuation. `nativeKVContinuity`, native memory and exact snapshot restoration
remain false. Persistent cross-call KV would require a separately validated
upstream API change.

## Isolated warm weights

`WORLD_WARM_MODEL_SECONDS` defaults to 180 and accepts 0–900. Zero disables reuse.
After a clean Astronex session ends at a block/playback boundary, the gateway
requests a reset and requires its acknowledgement before lending the process to
another session. Reset clears the last latent, text embeddings, VAE cache, all
positive/negative/camera KV and cross-attention caches, and retrieval metadata;
only model weights remain. Prompts/media and session directories are removed.
The next request reseeds the model. This is logical session isolation, not a
claim of forensic GPU memory erasure.

Cancellation during loading or inference, timeouts, protocol errors and model
failures kill the subprocess instead of pooling uncertain state. Warm idle
expiry, worker shutdown and reset failure also destroy it. At most one idle
process is retained and one session may own a worker. Forge and Matrix opt into
the shared pool with their own reset routines. SANA destroys its process because
its cache isolation has not been audited; Helix reloads the upstream CLI for each
explicit offline clip. Warm weights do not renew the
API's session lease or worker billing deadline.

## Generated audio

HelixWorld's public CLI generates audio with each video clip; see
[its adapter notes](helixworld/README.md). A bounded stereo 48 kHz PCM buffer is
paced into 20 ms audio frames and encoded as WebRTC Opus only when the selected
model advertises audio output. Audio pauses with video, and session teardown
stops the audio track and clears its samples. No microphone is captured, no
audio is synthesized by the gateway, and HTTP frame fallback is video only.
Its public preview accepts image/text conditioning; audio/video _text prompts_
must not be confused with support for conditioning on uploaded audio/video files.

## CPU verification

`python -m pytest workers/worlds/tests -q` exercises controls,
boundaries, reset isolation, cancellation, real codec encode/decode and real
loopback WebRTC. `tests/browser_transport.mjs` uses the actual browser transport
with a clearly labeled synthetic RGB fixture. Set `WORLD_VIDEO_ENCODER=h264-software`
for H264; omit it for normal software negotiation. Managed browsers that forbid
direct UDP can use an isolated loopback TURN server through
`WORLD_TEST_TURN_BINARY`; the test does not modify browser policy.
