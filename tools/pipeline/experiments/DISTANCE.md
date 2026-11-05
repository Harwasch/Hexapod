# Splats by distance from the camera path

Run locally from `tools/pipeline`, with no Modal or torch:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run python -m experiments.bench distance \
  --scene-dir /path/to/frozen-scene \
  --ply /path/to/A/trained.ply --ply /path/to/B/trained.ply \
  --output .tmp/distance
```

Input PLYs must be finished binary Gaussian PLYs in the frozen scene's original
COLMAP coordinate frame. The reader streams 32,768 rows at a time; it reads only
XYZ, opacity and the optional `far_field` marker into columns. The algorithm uses
one process and exact point-to-segment distances. It neither renders nor trains.

The path includes **all registered cameras**, including held-out cameras, in
capture order. A single capture's `frames_meta` candidate indices (then timestamps)
establish the order; numbered frame names are the explicit fallback. COLMAP image
IDs and registration order do not establish capture order.

`distance.json` records the camera sequence, scale fit and each PLY's measurements;
`distance.md` is its compact table. Approximate metre bands are `[0,2)`, `[2,5)`,
`[5,10)`, `[10,20)`, `[20,50)`, and `[50,infinity)`. `beyond` uses strictly greater
than 10 or 20 metres. Distance is three-dimensional, including camera height.

- `countShare`: band count divided by **all** splats.
- `opacityShare`: band's summed sigmoid(logit opacity) divided by that sum over
  **all** splats. This does not measure visibility or screen coverage.
- `farField`: separately reports the marked subset with those same denominators.
  It is not added to the total. An absent marker is unknown (`null`), not zero.
- Projected pixel area is omitted: projecting millions of ellipsoids over every
  training camera is a separate, substantially more expensive diagnostic.

The scale assumes a median phone height of **1.4 m** (`--camera-height-m` overrides
it). RANSAC fits a dominant plane to local SfM points below their nearest camera,
constrained within 45 degrees of estimated camera-up. SVD refines the inliers.
The output records support, residuals, camera-height spread and sensitivity to
halving/doubling the fitting threshold. These are diagnostics, not confidence
intervals. Camera tilt, sloped ground, rocks and missing ground texture can bias
scale. Distances scale linearly with the assumed height. No credible plane means
`status: unavailable`, with a reason and no invented metre-band measurements.

Future `bench rescore` results include the same object in each `rows[].distance`
and the shared fit/path in `distanceGeometry`; `rescore-table` JSON retains it.
This is additive: existing render metrics and live-console field names are unchanged.
A missing ground fit or unknown capture order does not discard photo scores.
Neither distance bands nor the far-field marker identify sea, sky or cliff
semantics. Scene-specific image-region evidence must be reported separately.
