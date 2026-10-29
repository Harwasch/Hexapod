# Living weather wind research

Research prototype, **Simulated**, not proposed for merge. Nothing changes the shipped
`WindSettings` or its dimensionless strength mapping. Weather inputs are 10 m m/s and
meteorological **from** bearings; the older rig UI uses **to** bearings.

## Model and provenance

`packages/world/src/weatherWind.ts` holds the pure maths because speed, spatial phase,
height reduction and bounded drag need no renderer, browser, request or store. It reuses
the shipped wind's distinction between a shared frozen field and local branch response,
and its separation of mean lean, along-wind buffeting and lateral response. It deliberately
has its own types so weather speeds cannot silently replace arbitrary shipped strength.

- [Taylor's frozen turbulence hypothesis](https://doi.org/10.1098/rspa.1938.0032): translate
  one spatial field at the mean speed. This gives exact delay `distance / mean speed` along
  the wind for **both** velocity components. Zero mean freezes advection; gust forcing may
  still be nonzero if a zero-mean input declares gusts. This is a limitation of the frozen
  assumption in calm, not a minimum hidden advection speed.
- [Kolbe et al. 2022](https://doi.org/10.3390/f13071026) reports field loads on plantation
  Scots pine and discusses approximately 20 s organized canopy turbulence. The reference
  field has wavelength 80 m at 4 m/s with gust factor 1.5 (20 s). The continuous rule
  `L = clamp((40 + 20 U)/(1 + (G-U)/max(1,U)),20,160)` metres is a **judgement**, not a
  fitted meteorological law. Increasing gustiness shortens structures and increases their
  passage frequency. Cross-wind length `0.4 L`, mode shares 0.6/0.4, wavelength ratio 1.618,
  cross modulation .27/-.43, lateral speed .25 of gust excess, lateral phase .9, and phase
  constants .173/.119/1.7 are deterministic **judgements**. These sparse modes illustrate
  correlated advection; they do not reproduce a Kaimal spectrum or measured gust durations.
- [Raupach, Finnigan & Brunet 1996](https://doi.org/10.1007/BF00120941) relates canopy
  coherent motions to canopy shear scales. It supports anisotropic, shared coherent
  structures; it does **not** supply the exact scale-vs-speed rule above. The 10 m hourly
  mean plus maximum gust does not identify turbulence spectra, eddy lengths or phase.
- [Cionco 1965](<https://doi.org/10.1175/1520-0450(1965)004%3C0517:AMMFAF%3E2.0.CO;2>)
  motivates exponential within-canopy attenuation. A neutral roughness log profile with
  `z0=.3 m` (terrain III, EN 1991-1-4 Table 4.1; also the shipped `turbulence.ts`) is normalized
  at 10 m. It uses `ln((z+z0)/z0)` to reach exact zero at the ground. Within the assumed
  6 m canopy it multiplies by `exp(2 min(0,z/6-1))`. Canopy height 6 m and attenuation 2
  are **assumptions**, not a fitted Minnetonka wind profile. Roughness displacement and terrain flow are absent. Dense-stand shelter is described below. Open-Meteo's grid wind is not the
  wind observed inside this crown.
- [Cardona & Dabiri 2022](https://doi.org/10.1017/flo.2021.15), a field study of a
  **5 m Magnolia grandiflora**, supplies the amplitude constraint. A 10 m anemometer
  and 20 one-minute, 900-frame optical treetop records span 5.322–15.79 m/s. The
  supporting MAT data at [Stanford](https://purl.stanford.edu/mg505cc6103) gives OLS
  `U = 5.42642597 sqrt(sigma_px)`, reproducing the paper's R²=.81. The raw pairs
  and fit are in `weatherCalibration.ts`. Metric transfer uses approximate 5 m height
  divided by **265±10 source-image pixels**, then divides by cos(50°), the reported
  wind angle from the image plane. Thus `sigma_m ≈ .000997 U²` at the study's
  10–12% turbulence intensity. Height accuracy, image perspective, assumed along-wind
  motion and transfer to this unknown species are limitations; this is **not a
  Minnetonka field calibration**. We allow 20% reference-curve error, specified before
  the final fit. The actual pose probe is within 6% at 6, 9.17, 12 and 15 m/s.
- Weather angular forcing is now `.0115 U²/sqrt(1+(U/25)^4)`, fitted through the
  **hierarchical crown-top pose**, including height reduction and geometric compliance.
  It uses the complete fluctuating load, with a quarter of the mean load retained as
  static lean. The .25 lean fraction and 25 m/s roll-off are **judgements**: the study
  constrains dynamic standard deviation, not static bend or survival in a storm.
  The finite high-speed bound is a software/visual guardrail, not a failure threshold.
  The four archive hours have different gust excess; they therefore do not all lie on
  the constant-11%-turbulence curve. Matching that curve under reference forcing and
  then applying their own gusts gives the following **600 s, 10 Hz** probe results.
  The probe sits above the highest fitted joint at captured top height 6.2246 m;
  centimetres are horizontal displacement, not a leaf-resolved motion measurement.

  | Hour                | Before σ cm | After σ cm | Before peak cm | After peak cm |
  | ------------------- | ----------: | ---------: | -------------: | ------------: |
  | Calm .40 / 1.10     |        .048 |       .156 |           .183 |          .540 |
  | Light 2.46 / 4.00   |        .494 |      1.908 |          2.140 |         5.925 |
  | Fresh 8.51 / 15.40  |       3.577 |     30.684 |         14.497 |        98.267 |
  | Storm 11.49 / 22.90 |       5.688 |     60.396 |         22.201 |       179.544 |

  Calm/light are extrapolations below the field data; storm gusts exceed it. The
  gust maximum does not identify turbulence intensity or spectra. The generated
  motion is weather-conditioned **Simulated**, never a replay of the observed tree.

- Branch modal frequencies reuse round 4's 0.24..1.1 Hz band, informed by
  [Campiformio 2012](https://digitalcommons.lib.uconn.edu/gs_theses/253/) (white ash 0.2..1.1 Hz)
  and [Jackson et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597771/) (architecture-dependent
  modes). Segment compliance `clamp(length/.7,.15,1)/sqrt(max(1,stiffness))`, depth divisor
  `1+.45(order-1)`, phase seed .61803398875, frequency `.24+.11 depth+.12 seed`, and modal
  modulation .22, cross modulation .14, phase offset 1.3 and frequency ratio .83 are **round-4/visual priors**.
  The extracted stiffness is not EI. Parent rotations propagate rigidly. Root/trunk bindings
  and ground are exact static pins by the brief even though real trunks bend.
- Only a passing Tier C scan with piece-specific bindings can enable flutter. Its existing
  3..5 Hz band is informed by [Roden & Pearcy 1993](https://pubmed.ncbi.nlm.nih.gov/28313608/);
  [Tadrist et al. 2018](https://doi.org/10.1098/rsif.2018.0010) measured cherry leaf torsion
  6±1 Hz and lamina length 99.1±29.7 mm. Species dependence matters. The .08 normalization
  for weather flutter and preexisting .035..0.10 rad amplitudes remain **judgements**.

The field's speed excursion is bounded by the supplied gust above the hourly mean;
clipping negative along-wind speeds biases the temporal mean upwards when gust excess
exceeds the mean. The cross-wind component means vector magnitude can exceed that
along-wind maximum slightly. The archive gust is not claimed to be a replayed event.

## Feed and controls

`apps/web/src/lib/weather/openMeteo.ts` uses archive for hours older than five days and
forecast (including recent past) otherwise, UTC, `wind_speed_unit=ms`, with hourly
`wind_speed_10m,wind_gusts_10m,wind_direction_10m`.
[Open-Meteo documentation](https://open-meteo.com/en/docs/historical-weather-api) describes
reanalysis rather than a site anemometer. Its hourly archive is spatially gridded;
requested location and returned grid-cell location are retained in the fixture.

Public `VITE_LIVING_WEATHER=1` opts into requests; off is the default. Manual/recorded
inputs work with it off. Cache: at most 64 daily responses, 24 h archive TTL, 15 min
forecast TTL, in-flight deduplication, failed requests evicted, 12 s timeout. Those are
**engineering choices**. No API key, persistence, background poll or production write.
Missing values, HTTP/units/time errors and absent hours return **No weather, wind off**.
The map panel's Live weather reads the active site's centroid and Cesium scene clock;
changing site/hour invalidates the displayed response and suppresses stale completions.
The standalone lab has explicit UTC map time and fixed Minnetonka coordinates.

Recorded fixtures are four real archive hours in 2024, all UTC:

| Example   | Hour              | Mean m/s | Gust m/s | FROM ° |
| --------- | ----------------- | -------: | -------: | -----: |
| Near calm | May 2 03:00       |      .40 |     1.10 |    360 |
| Light     | July 31 12:00     |     2.46 |     4.00 |    117 |
| Fresh     | November 19 04:00 |     8.51 |    15.40 |     97 |
| Storm     | December 4 21:00  |    11.49 |    22.90 |    316 |

The leaf-on capture stays the same in winter comparisons: this prototype tests wind,
not seasonal phenology. Presets are example weather inputs; response is continuous.

## Detail check

`tools/captures/living_weather_detail.py` adapts the structure probe's bounded 12-NN graph:
2.5 cm radius, RGB distance .12, unoriented normal cosine .8 (flat splats), alpha ≥.15,
maximum covariance radius ≤.12 m. Piece support includes conservative 2σ bounds.
Plausible pieces have ≥8 splats and 3..30 cm support diameter. Skeleton-contact pieces
have a centre within 4 cm of the fitted skeleton (2 cm skeleton samples overestimate
nearest distance by at most 1 cm). A nominal 10 cm lamina from Tadrist provides scale
context; `4 max(sigma) <= .10 m` is the splat-size criterion. These are **probe priors**,
not botanical identification or universal leaf dimensions.

Score = minimum of separable-piece crown share, contact-supported crown share, and
small-enough-splat crown share. Threshold **.65** is a deliberately conservative
**judgement**: a majority of the crown must clear every test, preventing a handful of
leaf-sized specks from enabling crown-wide flutter. Piece-specific bindings and a rig
are additionally required. Round 4's 12 cm packets never count as such bindings.
A passing scan needs most crown mass in distinguishable multi-splat leaf-scale pieces,
resolved twig/skeleton attachment and splats smaller than a leaf, with actual per-piece
bindings. No result depends on a passing scan existing.

Tier A: vegetation without rig, smooth height-weighted shared-field sway. Tier B: fitted
branch rig, parent-inherited two-axis response. Tier C: B plus verified leaf bindings
and passing score. No vegetation: off. The browser's segmented Tier A path converts published `instances.json` run bindings into checksum-bound
clump bindings at load time (`livingTierA.ts`). Static and unlabelled splats get exact zero
bindings. Vegetation categories (trees, shrubs, grass, crops, flowers, ferns, moss) or
vegetation property ≥.5 (only when the class is absent) nominate clumps; parents supply
missing classes. Known non-vegetation classes always stay static. A static behaviour,
rigidity score ≥.5 or static score ≥.8 excludes the segment even when vegetation is high.
The same veto applies recursively to **every ancestor**, including a known ground/table
class above a finer shrub label. This pins all seven unsupported Spool shrub fragments;
Spool selects off. A fine child label cannot establish a plant inside a rigid ground segment.
These are conservative **classification judgements**, informed by the shipped object's
rigid/static exclusions, not botanical truth or measured mechanical stiffness. The real
Spool check exposed table fragments moving under vegetation scores alone; ground/wood
regression tests and the final browser control prevent that. A segmented scan containing
only ground/rigid produce can select off despite green/organic appearance. A 2 m spatial
bin is doubled until at most 500 clumps remain (at most 1,000 nodes). Those are **budget
choices**, not measured plant boundaries. Root is the clump's minimum bound; a smooth
3.5 cm..15 cm height weight suppresses the base (**visual prior**). This is whole-clump
sway, not an inferred woody skeleton. Tier A uses the clump’s own height for canopy
attenuation and a compliance multiplier `min(3,6/max(.15,height))` so grass is not
treated as a stiff branch under an imaginary 6 m canopy. The multiplier and 15 cm
minimum are **visual choices**, not measured grass biomechanics. `instances.json?tier-a=1` is accepted as a lab
articulation address; the panel finds it automatically for a loaded segmented asset.
Without segmentation it falls back to a green/height candidate mask, as round 1 did. Audits distinguish
published segmentation presence from geometry/colour evidence. The public-scan check
streams terminal tiles only and uses ≤400k evenly sampled native splats; sample-based
scores are conservative diagnostics and cannot alone justify Tier C.

## Forest, lab and benchmark

Bake once, then serve (from repository root):

```sh
TMPDIR=~/.cache/hexapod-living-weather/tmp heavy uv run --project tools/captures python tools/captures/living_weather_forest.py \
  ~/.cache/hexapod-living-breeze/source ~/.cache/hexapod-living-weather/forest
TMPDIR=~/.cache/hexapod-living-weather/tmp heavy node apps/web/scripts/living-lab/serve-bench.mjs \
  --host 127.0.0.1 --port 5720
```

Use the captures project's environment for Python. `--source`, `--bake`, `--forest`,
`--detail` override asset paths. `--host` accepts a specific Tailscale interface; wildcard
binds are refused. Ports 5720..5729 only. Open `/` or `/?living-forest=1`. No API server.
All paths in committed code derive from the repo or home directory.

The deterministic **80-copy jittered hex grove** has nominal **3.50013 m** spacing,
.9 of the measured **3.88903 m** crown diameter. Crown width is the mean of east–west
3.67919 m and north–south 4.09888 m central-98% splat-centre extents above 1.5 m,
measured from **all 2,221,311 terminal splats** (no replacement-parent duplication).
The 1.5 m crown base and 1% tail trimming are measurement conventions. Raw horizontal
bounds span 4.28418×4.69214 m, including sparse tails. The seed remains 281026.
Jitter is ±.035 diameters per axis, yaw is uniform, scale .9–1.1, random copies are
mirrored in local x with covariance orientations reflected correctly, and brightness
varies .95–1.05 in DC RGB. These are **layout/appearance judgements**, applied only to
labelled **Simulated** forest derivatives. The original scan GLBs stay byte-identical.
As captured compares to each scene's immutable derivative baseline; it does not undo
the simulated forest layout, LOD or colour choices.

**30,000 splats/copy = 2,400,000 drawn**, 2,430,000 cached including the replaced
single-copy overview. **12 connected nodes/copy = 960 nodes**, below the existing
1024-node texture limit. **29,985/30,000 splats per copy** retain moving crown bindings;
15 measured static pins remain zero. An ancestor-only LOD remap had incorrectly
pinned 23,358 crown splats to the last trunk joint. A failing regression test now
guards reskinning to the retained branch segments, preserving original static pins.
The coarse rig uses **weatherResponseScale=.31**, fitted through an exposed copy
against the same field curve (within 5% at all four reference speeds); shelter is
applied afterwards. Geometry is a deterministic sample of existing coarse tiles;
DC colour only. The low LOD and reduced rig are budget choices. No leaf flutter is enabled.

Shelter closure is computed **once per copy when weather changes**. The Cionco
height profile then gives each retained joint a fixed ratio:
`S(z) = exp(-2.5 min(1, fetch/(6 D)) max(0,1-z/H))`, with **H=6 m**.
[Kim et al. 2014](https://doi.org/10.1016/j.agrformet.2013.11.013) uses **alpha=2.5**,
explicitly assumed there too, for mixed broadleaf canopy. Wind recovers toward the
canopy top; lower interior and lee joints stay sheltered. The first look-pass
incorrectly applied the trunk-space ratio to upper crowns as well and suppressed
visible weather motion. Its recordings are superseded. Exposed-copy calibration
is unchanged because windward closure is zero.

The exponential height form and coefficient are cited; fetch closure over six
crown diameters, the .75 D cross-wind corridor with linear taper, and a uniform
6 m canopy are **judgements**, not a fitted forest-edge law. Closure is per copy,
height ratios are prepared once per retained joint; no per-splat or per-frame
shelter work. The existing roughness/own-crown height profile still applies. At
full closure the extra ratios are .082 at ground, .287 at 3 m, .535 at 4.5 m and
1 at/above 6 m. This is still a simple canopy-flow approximation, not a fluid solve.

Cameras: oblique target [0,0,3], heading25°, pitch−25°, range38 m; low target [0,−7,3.4],
heading10°, pitch+6°, range13.4 m gives **2.00 m eye height** near the edge; overhead
target [0,0,2], pitch−89.9°, range48 m. `?living-forest=1&view=oblique|edge|overhead`
selects them. The overhead inset is north-up, **45 m** wide, at 10 m, fixed 0..25 m/s
colour scale. It shows the shared advected field, not sheltered canopy velocity.
Ground is still a flat simulated sheet, reduced to 55×45 m.

The laptop page defaults to **8 rounds, eight warmups, 32 retained samples/block**,
with a recorded random seed and a fresh Fisher–Yates arm order per round. Off blocks
bracket each arm. Added time is the arm median minus the average of its two adjacent
off medians; JSON retains each round's pair, ratio, off medians and drift. The page
shows median added time and p10–p90 across rounds. If max/min off medians exceeds 1.3,
the verdict is **inconclusive, rerun with the laptop plugged in and nothing else open**.
A clear **Run forest benchmark next (80 trees)** button starts the forest run. Buffers
stay 1920×1080, DPR1, with the existing one-pixel readPixels fence. Software rendering
cannot establish hardware-GPU overhead. Harrison's old tree-only laptop run has 4.08×
off-block drift and remains inconclusive; the forest was not measured on that laptop.

Detailed scan scores, measured times, fixed-clock 24 fps media, immutable controls,
gate tails and limitations belong in the task's `living-weather-look/results.md` gallery.

For offline/browser validation and media, set `LAB_OUT` to the task notes folder and run
`heavy node apps/web/scripts/living-lab/weather-run.mjs record-hours crown` (or forest-oblique / forest-edge; `record forest-overhead storm` keeps the inset). `controls tree`, `controls forest`, `bench tree`, `bench forest` and
`device tree` / `device forest` exercise restoration and the real download action.
The `device` task also tests an actual weather read and an intercepted HTTP 503; build
it with `VITE_LIVING_WEATHER=1`. `tier-a tree` checks published Spool segmentation and exact rigid pixels,
using the explicitly labelled manual transfer of Minnetonka weather. It requires the
cached public Spool tiles; the default tree/forest bench does not. The two-sample
device run is a smoke check, not a p95 estimate. Use the default page settings on the laptop.

`feed tree` / `feed forest` reruns the weather-read, failure and stale-request checks
without repeating a completed benchmark. It initializes a known archive hour and
observes either completion or failure of the deliberately held older request.

Recordings stop the browser's automatic render loops after the assets and camera have
settled, then call `widget.render()` once at each `i/24`. The existing one-pixel fence
completes the draw before its PNG is read. Normal versus manually stepped rendering is
checked at the identical fixed time for crown and forest. This changes capture cadence,
not the motion clock or interpolation. The initial storm crown reference used the normal
loop at the same fixed clock. Timings use the normal loop.

Run `heavy node apps/web/scripts/living-lab/weather-amplitude.mjs calibrated` for the
600 s crown-top amplitude probe and its explicit 20% reference-curve assertions.
`record-hours` records four hours in one browser; each source frame is still rendered
at i/24, 192 frames, with no interpolation. `weather-gallery.mjs` encodes the three
four-panel comparisons, overhead storm, contact sheet and captioned gallery.

Add `forest` after `calibrated` to run the exposed coarse-stand pose probe with the same assertions.

The amplitude probe also checks **all 20 actual field records** after metric transfer.
The source's inverse-curve relative σ residual reaches **58.47%** (at 15.79 m/s),
so this check uses a recorded **60% field-scatter envelope**, independently of the
tighter 20% fitted-curve transfer bound. The source-fit test records that scatter;
it is not an arithmetic tolerance or a claim of 20% agreement with every noisy record.

For the four hours, the source's force/sway relation also supplies a **gust-adjusted
comparison curve**. For this simulated field's `.6 sin(a)+.4 sin(b)`, the second
moment is .26 and `Var(w²)=.077`. With `d=G-U`, quadratic-load standard deviation
is `sqrt(1.04 U² d² + .077 d⁴)`. Normalize it by
`sqrt(1.04(.216)²+.077(.216)⁴)` at the reference ~11% turbulence, then multiply by
`.000997`. This gives target σ **.143, 1.77, 27.6 and 62.5 cm** for calm through
storm, versus the pose probe's **.156, 1.91, 30.7 and 60.4 cm**. All four are checked
against the same **20%** bound, with no per-hour gain. These targets follow the
**imposed sinusoidal field statistics**, not an observed hourly turbulence intensity.
Clipping, saturation, branch modes and finite observation windows make the relation
approximate. A mean and maximum gust alone cannot identify the true distribution.
