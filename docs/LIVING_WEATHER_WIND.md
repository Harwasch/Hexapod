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
  are **assumptions**, not a fitted Minnetonka wind profile. Real urban/forest shelter,
  roughness displacement and terrain flow are absent. Open-Meteo's grid wind is not the
  wind observed inside this crown.
- [Vogel 1989](https://doi.org/10.1093/jxb/40.8.941) and
  [de Langre, Gutierrez & Cossé 2012](https://comptes-rendus.academie-sciences.fr/mecanique/item/10.1016/j.crme.2011.11.005.pdf)
  motivate subquadratic loading as plants reconfigure. The proxy load is
  `U²/sqrt(1+(U/4)²)`: quadratic at low speed and linear at high speed (Vogel exponent -1).
  The 4 m/s transition is a **judgement**. Angle `0.10 load/(28+load)` has a strict .10 rad
  ceiling; .10 and 28 are visual guardrails, not measured drag, wood modulus or failure loads.
  This caps motion, not weather speed. Inputs are sanitized to 0..60 mean and 0..90 gust
  m/s as defensive **software bounds**, not physical thresholds.
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

The deterministic 6×6 stand has 24 m spacing with ±3 m jitter, random yaw and .9..1.1
scale (seed 281026, **layout choices**). Per-copy LOD is a 40k systematic sample from
existing depth-one replacement tiles, DC colour only, with 25 spatially spread fitted
nodes reconnected to retained ancestors. Total **1,440,000 splats**, **900 nodes**.
Canonical GLBs remain untouched. A temporary root overview reuses one copy; it is fully
replaced by the 36 leaf tiles before a recording or timing sample. The inset shows the same field, coordinates and clock
at 10 m, with a fixed 0..25 m/s colour scale and velocity direction arrows. The stand
uses height-reduced wind; the inset is not canopy velocity or a weather raster.

The device bench interleaves plain/calm/plain/light/plain/fresh/plain/storm, three
rounds, four warmups and 32 retained samples/block by default, 1920×1080 buffers at
DPR 1. The same page supports forest. One-pixel readPixels includes CPU pose evaluation,
upload and completed draw. SwiftShader cannot verify a laptop's +1 ms target.

Detailed scan scores, measured times, fixed-clock 24 fps media, immutable controls,
gate tails and limitations belong in the task's `living-weather/results.md` gallery.

For offline/browser validation and media, set `LAB_OUT` to the task notes folder and run
`heavy node apps/web/scripts/living-lab/weather-run.mjs record crown storm` (or forest,
calm/light/fresh). `controls tree`, `controls forest`, `bench tree`, `bench forest` and
`device tree` / `device forest` exercise restoration and the real download action.
The `device` task also tests an actual weather read and an intercepted HTTP 503; build
it with `VITE_LIVING_WEATHER=1`. `tier-a tree` checks published Spool segmentation,
using the explicitly labelled manual transfer of Minnetonka weather. It requires the
cached public Spool tiles; the default tree/forest bench does not. The two-sample
device run is a smoke check, not a p95 estimate. Use the default page settings on the laptop.

Recordings stop the browser's automatic render loops after the assets and camera have
settled, then call `widget.render()` once at each `i/24`. The existing one-pixel fence
completes the draw before its PNG is read. Normal versus manually stepped rendering is
checked at the identical fixed time for crown and forest. This changes capture cadence,
not the motion clock or interpolation. The initial storm crown reference used the normal
loop at the same fixed clock. Timings use the normal loop.
