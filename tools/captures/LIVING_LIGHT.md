# Living Light: real-photo prototype

`light_photos.py` adds real capture teachers to the original `light_bake.py`
experiment. It does not change scan geometry, captured colours or the viewer shader.
It reads the pipeline's `poses.json` and exact `placement.json`, rectifies
`SIMPLE_RADIAL` photos in memory, and fits one constant albedo per splat.

The capture inputs are private. Read them in place. Never commit or serve photos,
COLMAP files, teacher arrays or fitted sidecars. Keep the `--work` directory under
the approved private cache. Only scan renders and numerical summaries belong in
the results folder. The local playback server must point at a separate directory
containing only the published scan and fitted sidecars.

Set `TMPDIR`, `UV_CACHE_DIR` and `HF_HOME` to the approved cache locations. Run
heavy stages sequentially through `heavy timeout 3500`, checking active jobs first.
The inference uses eight PyTorch threads. Install the optional `light` dependency
group only for inference; `uv sync` restores the base environment before gates.

The CLI takes `--inputs`, `--work`, and `--scan` for every stage:

1. `prepare --count 36 --width 1024`: select uniformly spaced registered images,
   transform cameras into the packaged scan's frame, and render depth/coverage.
   Photo pixels are never saved. The work directory holds derived camera metadata,
   depth arrays and numerical photo/render agreement.
2. `alignment`: compare the correct projection to fixed image-offset controls on
   three views. This is an alignment diagnostic, not a quality pass bar.
3. `teacher --limit 1`, using `uv run --group light`: measure the CPU cost first.
   Reduce frame count or resolution if the full batch would exceed an hour.
   Run `teacher` again to continue; completed albedos are skipped. Use a fresh work
   directory when changing views, resolution or model settings.
4. `bake --previous <round2-sidecars> --output <playback-sidecars>`: fit a weighted
   component median, reject residual outliers with a weighted MAD threshold, then
   average retained samples. Depth tests reject occluded samples; weights account
   for footprint and incidence. Unobserved splats retain captured colour with zero
   confidence. Normal signs are fixed from capture observers, never the live camera.

The 48-byte sidecar format stays unchanged. The prior sidecar is matched by position
checksum, count and format marker; its sun visibility and sky bytes are reused exactly
to isolate the material change. Coverage is reported for both full-detail leaves and
all LODs. This is a teacher-driven material fit, not a reproduction of GaRe's training.
More observations do not guarantee that the teacher removes captured shadows.

The analytic tests in `tests/test_light_photos.py` cover similarity projection,
robust outlier rejection, missing observations and occlusion. They require no capture,
network, model or GPU. Use the existing fixed-camera recorder and ten transects for
comparisons, without changing their locations to suit the new result.

## Spool round 3 result

The run used 28 real capture views at 1024×576 (native frames: 1600×900), reduced
from 36 after a 108-second CPU pilot projected more than an hour. Total teacher
time was 3338.66 seconds, with no paid compute. The input JSON matched all 179
binary COLMAP poses and their intrinsics exactly; three image-offset controls
favoured the unshifted scan projection. No private images are included here.

| Measurement                                     | Round 2 | Round 3 |
| ----------------------------------------------- | ------: | ------: |
| Median old-shadow contrast reduction, A         |  10.45% |  72.70% |
| Median old-shadow contrast reduction, B diffuse |   4.26% |  66.49% |
| Teacher coverage, all LODs                      |  68.17% |  77.21% |

The same ten transects cover two boundaries: six on the wooden flange and four
on ground. Ground-only median reductions were 44.17% / 40.00% for A/B. Both arms
miss the 80% overall target; only one transect per arm meets both 80% reduction
and residual contrast ≤0.05. The residual ground pattern prevents a clean
no-double-shadow result. This improves materially on rendered teachers but remains
below the GaRe reference. Full-detail leaf coverage is 77.76%; unseen splats retain
captured lighting, and 37.65% of observed splats retain only one teacher sample.

The visibility and sky bytes are identical to round 2. Results, scan-only clips,
stills, timing batches, exact transect coordinates and command logs live in the
experiment's external `living-light/round3` notes folder. Private photos, pose files,
teacher arrays and sidecars are excluded from the repository.
