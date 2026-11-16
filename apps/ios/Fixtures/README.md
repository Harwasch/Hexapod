# Synthetic capture replays

These fixtures are **SYNTHETIC**, with no recorded people, property, or device data.
Each directory under `bundles/` contains a valid flat `bundle/`, the same bytes
packed as uncompressed USTAR `capture.hexcap` for upload/normalize, and a sibling
`ground_truth.json`. Ground truth is deliberately outside the bundle inventory.

- `yard-small`: 24 RGB/depth frames, first two `limited:initializing`, then 22
  `normal` frames. Suitable for parser, upload, normalize, and analysis replays.
- `yard-limited`: the same scene and camera path, with two initializing frames,
  three normal frames, and 19 `limited:excessiveMotion` frames. Pose rejects it
  with the default minimum of ten tracked frames.

The scene has a bounded ground plane, one box shed, and one tree made of a capped
cylinder trunk and spherical crown, all in metres with flat colours. Ground truth
lists component centres and full extents, object counts, tree height, and visible
surface probes for every frame. Cameras move sideways while looking at the scene;
poses are row-major ARKit camera-to-world matrices (+Y up, looking down −Z).
RGB is HEVC/hvc1, 192×144 at 12 fps for two seconds. Raw depth is 48×36 little-endian
float16 optical-axis depth, with intrinsics scaled by 1/4 and integer pixel centres.
Confidence is 2 for hits, 0 for sky; sky depth is positive infinity. There is no
smoothed depth. Two seeded synthetic GPS fixes are near 37.25, −122.1, with fixed
UTC timestamps and the first fix selected as the manifest location.

Regenerate **only after changing the generator**, from `tools/pipeline` (create
`.tmp-test` at the repository root first):

```bash
TMPDIR="$PWD/../../.tmp-test" UV_OFFLINE=1 uv run --no-sync python scripts/make_fixture_bundles.py
```

Delete `.tmp-test` afterward. No network, GPU, or external renderer is used.
The generator uses the installed `imageio_ffmpeg` binary's libx265, a fixed seed,
one encoder thread, disabled encoder information, fixed JSON ordering, and zeroed
USTAR timestamps/ownership. Outputs are byte-stable with the same toolchain;
HEVC bytes may differ across ffmpeg/libx265 versions or platforms. Tests replay
committed bytes without regeneration, compare packed/folder payloads exactly,
decode and count HEVC frames, check rendered colour classes, and project analytic
surface probes into depth with exact float16 rounding. Codec byte identity across
toolchains is not a test requirement. These fixtures verify format and synthetic
geometry consistency; they do not establish reconstruction quality on real captures.

Python replay checks live in `tools/pipeline/tests/test_fixture_bundles.py`;
Swift decoding checks live in Capture's `FixtureTests.swift`, resolving this folder
from `#filePath` without copying resources into the package.
