# 0018: Geometry-first change detection

Status: proposed

## Pipeline

Date 2026-10-20 · M5 of the splat-pipeline build (design M4–M6 §1 C1, C2, C8; §3 P5), branch
`claude/feat-splat-height-surface`.

### Context

Milestone 5 is "compose, compare, detect change": once a site has two epochs, the map should say
which parts of it changed between them, and how sure it is. A splat is a poor witness for that on
its own. Two scans of an unchanged field never train the same gaussians, so comparing splats, or
their renders, compares two optimisers' answers as much as two days. What both epochs do share is
the frame they are placed in (ADR 0013) and, since M4, a lattice of cells in it with a measure of
how well each cell was seen (ADR 0016).

### Decision

1. **Change is measured in geometry, never in splat pixels.** Each epoch measures, per cell, the
   height of its top surface, of its bare ground, and how far its own samples disagree. A change
   map compares those heights between two epochs. No rendered image, and no colour, enters it.
2. **The `surface` stage** (`height_surface`, `tools/pipeline/surface.py`, after `confidence` and
   before `package`, no GPU) writes `surface/surface.{json,bin}`, a `hexapod.cellgrid` of kind
   `surface` (C2):
   - **samples**: the pose model's sparse SfM points (`poses/points3D.bin`), placed by the
     similarity `place` applied to the splat (`placement.json`), and `canonical.ply`'s centres at
     least half opaque;
   - **plane**: exactly the epoch's confidence grid's finest plane (same level, same `i0`, `j0`,
     `ni`, `nj`), so a surface cell is a confidence cell, and two epochs' surfaces nest as their
     confidence grids do (C1). One plane; no pyramid;
   - **channels**: `dsm` and `ground` (`i2`, centimetres above the frame origin's height; `-32768`
     where the cell has no surface), `sigma` (`u1` centimetres, saturating: 255 is "at least
     2.55 m"; 0 only where there is no surface), `support` (`u1`: bit 1 SfM, bit 2 splat, the
     floor of log2 of the samples in the high nibble) and `groundSigma` (`u1` centimetres,
     `ground`'s own sigma, saturating as `sigma` does; 0 where `ground` is `-32768`).
3. **The estimators, from first principles and uncalibrated.** Per cell, streaming, the eight
   highest and eight lowest sample heights (`z1 ≥ z2 ≥ … ≥ z8` from the top) and the count `n`:
   - `dsm = z2`, the 2nd highest: one floater cannot raise it;
   - `ground` = the 2nd lowest, then the least of it over the cells whose centres are within 2.5 m
     along each axis (a 5 m square): bare ground under a crown or a roof up to 5 m across. That
     holds at levels 0 to 3 (0.25 to 2 m cells: 21, 11, 5 and 3 cells a side). At level 4 and
     coarser (4 m cells and up) no other cell's centre is within 2.5 m, the window is the cell
     alone, and `ground` is just the cell's own 2nd-lowest sample. The descriptor records the
     window it used (`groundWindowCells`). `groundSigma = max(1 cm, s_low·RMS2(N))`: the
     least 2nd-lowest over a window of `N` samples is never below the 2nd lowest of those `N`
     together, so that order statistic bounds its distance below the ground; `s_low` is the
     bottom-eight noise scale of the cell that set the minimum — never an average over the
     window (round 2: a sparse lattice of 10 cm cells among 1 cm ones, 2,000 samples a cell,
     read 1,862 of 2,304 unchanged cells as lowered with the average; 0 with this). At 1 cm noise and 100
     samples a cell it is about 5 cm, against dsm's 2 cm: a lower envelope over 441 cells is the
     more biased estimator, and its sigma says so (review B5a's control, for the ground: 0 of
     400 unchanged cells called changed at 20 vs 2,000 samples or 1 vs 10 cm). Where the window saw only canopy or roof, the ground
     was **never observed** and says so: `ground` is `-32768` under a `dsm` that is there, when
     the window's least is higher than any slope up to 1 in 2 (`GROUND_MAX_GRADE`) from a lower
     2nd-lowest sample anywhere allows (`summary.groundUnobserved` counts them; review B5a: a
     10 m canopy at 4 m with ground seen only around it gave its middle cells 4 m of "ground");
     and deeper inside a crown than that test reaches (round 2: a 20 m crown at 4 m kept 397 cm
     of "ground" at its centre; a 40 m crown growing from 4 to 6 m read as 4,356 cells of ground
     raised), the canopy's edge gives it away: a cell standing above its window's ground by
     more than any slope of 1 in 2 could explain (1.77 m at the 5 m window, plus 3σ) carries
     that ground across the canopy's top, never down a drop a slope could not make, and every
     cell it reaches whose window ground stands that far above it has none. Both round-2 cases
     now read 0 cells of ground changed. Not caught: a canopy or roof lower than about 1.8 m and
     wider than the window;
   - `sigma = max(1 cm, (z1 − zm) · RMS2(n) / SPAN(n))`, `m = min(n, 8)`: the root-mean-square
     distance of `z2` from the surface, **its bias included**, under Gaussian noise of the scale
     the top eight's spread shows. `RMS2(n) = √E[Z(2:n)²]` and `SPAN(n) = E[Z(1:n)] − E[Z(m:n)]`
     for `Z(k:n)` the k-th highest of `n` standard normals, tabulated exactly to 64 samples and
     per half-octave above (each bucket at its top, the larger sigma). The 2nd highest of `n`
     noisy samples reads high by `s·E[Z(2:n)]` (about 1.4 noise scales at 20 samples, 3.2 at
     2,000), so two scans of unchanged ground with different counts or noise differ by that
     bias; both biases have the same sign, so their difference is never more than the larger,
     and the threshold below covers it (review B5a). From three to six samples the spread is a
     poor estimate of the noise, and an under-read sigma calls noise a change (round 2: two
     three-sample scans at 1 and 10 cm, 51 of 5,120 unchanged cells changed); the scale is
     widened there (`FEW_SAMPLES`: 5.1, 2.25, 1.45, 1.15 at 3 to 6 samples, the least keeping a
     cell at the rule's 0.27% against a noiseless other side, by Monte Carlo, with 3% to spare);
   - fewer than 3 samples is no surface: with two, the 2nd lowest is the higher of them and the
     ground would stand above the top. From three up, `ground ≤ dsm` by construction.

   The descriptor says `status: uncalibrated` and names each estimator, so no reader mistakes them
   for measured accuracies.

4. **The threshold.** Between epochs `a` and `b`, at the coarser of their levels (dsm by maximum,
   ground by minimum, sigma by maximum over the children that have a surface; a coarse cell is
   compared only when the finer epoch measured every one of its children, else it is
   `insufficient`: its top would be the measured children's alone against the whole cell):
   `Δ = dsm_b − dsm_a`, and the cell is **changed** when `|Δ| > 3·√(σa² + σb² + ua² + ub²)`, where
   `u` is each epoch's uncertainty **from the site frame** (its confidence grid's
   `alignmentCap.uncertaintyM`, ADR 0016 §5: 0 for the epoch that made the frame) or its
   alignment's own claim, whichever is larger, and at least 1 m for a coarse epoch (an unaligned
   one is refused; review B5a round 2: the demo took the cap alone, so a coarse epoch 3 mm from
   the frame called a 5 cm rise raised). Classes:
   `raised`, `lowered`, `unchanged`, `insufficient` (either epoch has no surface in the cell, or
   the finer measured only part of it, or either sigma saturates: "at least 2.55 m" makes the
   threshold only a lower bound), and
   `appearance-only`, reserved and never emitted (below). Three sigma is a first-principles choice
   (about one false call in 370 cells of Gaussian noise), not a calibrated one. A5 computes the
   map; the demo computes it the same way to show it works. Both refuse surfaces whose heights
   are not metres (`summary.metric` false: a run whose scale was never resolved) or are above
   different origin heights (`vertical.originHeightM`): no change in metres can be read there.
5. **Shipped beside the tiles, opt-in (C8).** `package` copies `surface.bin` beside the tiles and
   appends the descriptor to the root extras as `extras.surface`, after `confidence` (after
   `collision` when there is none). Nothing per tile. `catalog` writes it at the top level of
   `registration.json` as `surface` (null: not measured). `convert(surface=None)` writes exactly
   the bytes it always did. A grid the packer refuses is left out alone; the confidence grid and the
   tiles still ship. `photo-reconstruct` v15.

### Why

- **Why not splat pixels, or renders.** The research behind this build recommends comparing
  renders of the old model with the new photos, in the manner of arXiv:2506.21117 (features of the
  rendered past model against new captures, votes in 3D, local re-optimisation) and
  arXiv:2508.08867 (lighting change separated from geometry change). That is deliberately **not**
  used here:
  - it measures appearance, and an outdoor site's appearance changes with sun, cloud, season and
    exposure on every visit; the geometry of an unchanged rock does not;
  - a render of the old model at a new view is generated content, not an observation, and the
    living map keeps generated content out of anything it calls observed;
  - neither method has been measured on leaf-on against leaf-off vegetation, which is most of what
    a land-management site holds; the research names that gap itself;
  - it needs a GPU per comparison; heights per cell are a few passes on the worker.
- **Why SfM points and splat centres.** The SfM points are triangulated from matched features, so
  they are where the photos agree; the splat's opaque centres are dense where the SfM cloud is
  sparse. Each sample says which it came from (`support`), so a reader can tell a surface only the
  splat vouches for.
- **Why the confidence grid's plane.** Alignment is free when it is the same cells: no resampling
  between an epoch's confidence and its surface, and every reader that can read one grid can read
  the other.
- **Why `u` from the frame.** Two epochs of a site are compared in its frame; an error they share
  (where the first epoch's GPS put the frame) moves both alike and is no change between them. An
  epoch placed only by GPS carries its own metres into the threshold, so it cannot call anything
  smaller changed: honest, and the reason a phone epoch must be aligned visually to say anything.

### Consequences

- **Seasonal epochs read as geometry change until measured.** A canopy in leaf is a higher top
  surface than the same canopy bare; until a pilot site scanned in both seasons measures it, a
  leaf-on/leaf-off pair is reported as raised or lowered, exactly as it measures, and the UI must
  say the map is derived from heights, not observed change.
- **Appearance-only change is stubbed.** The class is reserved and never emitted: telling a repaint
  or a dried lawn from no change needs the new photos compared with the old epoch at fixed geometry,
  which is a photo comparison, never a splat-pixel one. Not built.
- **The ground is biased low, never high, or unobserved.** A lower envelope reads low on a slope
  (up to 2.5·g m along each axis on grade `g`), and a low outlier that survives the 2nd-lowest
  rule spreads over its 5 m square. The unobserved test (a slope of 1 in 2 from lower ground)
  catches a canopy or roof of height `H` wherever ground is within `2H` metres of the cell; deeper
  inside a plateau than that nothing tells canopy from ground, and ground steeper than 1 in 2 beside
  a drop reads as unobserved (lost, never wrong).
- **Under noise both surfaces are biased, and sigma says by how much.** With height noise σ the
  2nd highest of `n` samples reads `σ·E[Z(2:n)]` high (1.4σ at 20, 3.2σ at 2,000), and the
  ground, a lower envelope over 441 cells' 2nd-lowest values at 0.25 m, about 3σ low (the demo:
  1 cm noise, flat ground at −3 cm, the tree's 4 m top at 4.01 m). The first version's sigma,
  half the top four's spread, held none of that: two scans of the same flat ground at 20 and
  2,000 samples a cell called 204 of 400 cells raised, and at 1 cm against 10 cm of noise 329 of
  400 (the review's control, `test_no_change_control_…`). With the bias in sigma, at most one of
  400 (the 3σ rule's own rate). The cost is a larger sigma: at 1 cm noise and ~100 samples a
  cell about 2 cm, not the 1 cm floor. Two limits: the noise model is Gaussian, and heavier
  tails read further high; and a cell of 3 to 7 samples reads its noise from that few, so it is
  the weakest. Calibrating either needs a site surveyed twice.
- **Floaters.** One floater in a cell moves neither surface but widens `sigma` (to "at least
  2.55 m" when it is high enough), which makes the cell slower to call changed rather than wrongly
  changed. Two or more floaters in one cell are a surface as far as this rule can tell.
- **Heights are a tangent plane.** `z` of the tileset frame, along the origin's up: a kilometre out
  it is about `d²/2R` (8 cm) above the ellipsoid there, and the origin's own height is in whatever
  datum the site frame names (`msl-unverified` for backfilled frames). Differences between epochs
  of one site are unaffected; absolute heights must not be quoted as elevations. `i2` centimetres
  hold ±327.67 m from the origin; a cell past that has no surface and is counted (`outOfRange`).
- **Weak cells.** The change rule above does not consult the confidence score. In the demo every
  changed cell that one epoch's grid calls weak is the crate's own side, so excluding weak cells
  would have hidden real change and caught nothing false; A5 classifies them and flags them
  `lowConfidence` (§ API 9).
- **Memory and determinism.** One band of extremes at a time (137 bytes a cell, 2^19 cells a band)
  and the plane (10 bytes a cell); order statistics then roundings, so the bytes depend on neither
  chunk nor band. `points3D.bin` is streamed.
- Until A5 adds a `surface` sidecar kind (§ API 1, done), a re-publish that carries sidecars from a
  previous generation reports `extras.surface` as an unrecognised sidecar if the new run has none,
  as `extras.confidence` does until A4.

### Demo (synthetic)

`tools/pipeline/experiments/confidence_demo.py --change`: two scans of the M4 demo's field, each
its own sample with 1 cm of height noise and a sparse SfM cloud, and a 1.5 m crate in the second on
ground both saw well, the pair taken as survey grade. At 0.25 m cells, `u` 0.03 and 0.062 m (each
epoch's placement): all 36 cells on the crate raised by 1.48–1.51 m, the 13 its sides stand in
raised too, no cell that does not touch it changed, nothing lowered; unchanged cells reach at most
0.09 m, under 0.17 of their threshold (at least 0.21 m). Every capture in it is synthetic; nothing
here was calibrated on it.

### Not done here (stubs and later PRs)

- The change map itself, its API and its classes: A5 (§ API; computed per request, not stored).
- The change drape and the "Derived from height surfaces, not observed" label: W5c.
- Appearance-only change; any recorded measurement behind the estimators or the 3σ rule; a
  measured leaf-on/leaf-off pair; a cross-language surface fixture for the web decoder (the API's
  is A5's `synthetic-yard-surfaces`).

## API

Date 2026-10-20 · M5 of the splat-pipeline build (design M4–M6 §3 A5, DECISIONS D7), branch
`claude/feat-splat-change-api`. Revised after review B5a (its rulings are marked so).

### Decision

1. **A surface is an epoch revision's map, like its confidence grid.** Migration 0036 widens
   `epoch_maps.kind` to `confidence` · `surface` (the downgrade deletes the surface rows first).
   `register` records `registration.json`'s `surface` on the revision it published, through the
   same record / carry / repoint path as the confidence grid (ADR 0016 § API 2). A descriptor is
   checked first — the container, a file beside the tiles, a cell on the lattice, the five
   channels at the types and units the arithmetic assumes (`dsm`, `ground` i2 cm; `sigma`,
   `groundSigma` u1 cm; `support` u1), `noData` −32768, the summary counts, a `status`, and
   `vertical.originHeightM` — and one that fails is logged and not recorded. `surface.bin` is a
   sidecar of kind `surface`, bound to the tiles and not attachable, like `confidence.bin`. Every
   epoch says `surface: {status: measured, assetId, url, cellM, level, policy, calibration,
cells, surfacedCells, medianSigmaM, originHeightM, groundWindowM}`, or null — not measured.
2. **`GET /sites/{id}/change?from=&to=`** (`change-v1`) compares the current revisions of two
   published epochs of one site in **two layers** (B5a ruling), each classified on its own:
   - **`ground`**, the bare earth (`ground`, `groundSigma`): the primary product;
   - **`vegetation`**, the top surface (`dsm`, `sigma`): canopy, crops and anything standing on
     the ground, labelled "never ground change", at its own higher threshold (6). A change of
     the ground wider than P5's ground window shows in both; a crate, a spoil heap or a crop
     only in vegetation.

   Both are compared at the coarser of the surfaces' levels, each coarse cell taken over only
   the children both epochs measured — the top the highest, the ground the lowest, each sigma
   the largest (§ Pipeline 4's rule; `coarsen_surface` matches P5's demo byte for byte on the
   committed fixture `data/tiles/synthetic-yard-surfaces`). The finer epoch is first brought up
   over the children whose coarse cell the coarser one measured, then both go up together.

3. **What is compared** (B5a). Where both epochs have the layer's height over every child the
   finer measured, `Δ = h_to − h_from` (in 32 bits) against
   `t = 3·√(s_from² + s_to² + u_from² + u_to² + 2w² + (g·h)²) + drift`
   (`g·h = 0` under the original absolute-placement rule). A cell is **`insufficient`**
   where either measured the layer but the two cannot be compared: no shared child; **fewer
   shared children than either measured** (a coarse cell against part of it — it was compared
   and flagged; the review's crate-against-one-ground-child read "lowered 2 m"); **either sigma
   saturated** (255 cm, "at least": the threshold would be a lower bound — two 30 m floaters read
   as raised against 7.65 m); for the ground, **ground never observed** (P5's `NO_HEIGHT` under a
   surface). No class where neither measured anything.
4. **`u`, each epoch's placement, takes its alignment class** (B5a ruling). `u` is the larger of
   its uncertainty from the site frame (its confidence grid's `alignmentCap.uncertaintyM`; 0 for
   the epoch that made the frame) and its alignment's own claim (`alignment.uncertaintyM`, 10 m
   when none is recorded). A `survey` epoch (an RTK drone with RTK evidence in its frames) claims
   centimetres; a `coarse` one — any phone, a drone without RTK evidence, anything aligned into
   a coarse reference — metres, so no change under metres is claimed with it, however well it was
   aligned visually: a phone epoch is never presented as survey grade. **A coarse epoch's `u` is
   at least `max(claim, 1 m)`** (round-2 ruling): coarse is a class, not a number — a phone may
   claim 3 cm, and a survey claimed without its RTK evidence is recorded coarse with its
   centimetre claim kept; against a 3 cm survey either read a half-metre rise as 1,024 ground
   cells raised at a 0.21 m floor. An epoch whose verdict is `unaligned`, or that recorded none,
   is refused (409 `epoch-unaligned`). The answer states each epoch's verdict, its `u` and its
   uncertainty from the frame.

   **Measured relative fit (Harrison, 2026-10-28: “Use the measured fit”).** When one
   epoch's accepted visual alignment references the other's **current job**, use that
   alignment's positive, finite `frameRelativeUncertaintyM` once for the pair: `u = 0`
   on the reference and `u = relative` on the aligned epoch. When both visually reference
   the same epoch **and reference job**, use both recorded relative terms in quadrature.
   The record must say `method: visual`, with `frameRelativeBasis: visual` (including
   the older conservative “reference at its absolute uncertainty” basis); a fit refuted by
   check targets, missing measurement or missing/mismatched reference revision keeps the
   original rule. These fields are recorded by alignment already. No chain inference or
   variance subtraction is made: frame-relative terms can include earlier reference error,
   and siblings can share it, so retaining both is conservative.

   The **1 m coarse floor does not apply to these measured relative terms**. Absolute GPS
   error is shared when one reconstruction inherits the other's frame; it does not measure
   their separation. This changes B5a round 2 for this case only: that review caught a
   coarse label plus a centimetre claim/cap without a measured relationship. Those cases
   still take exactly the old floor, and unaligned epochs are still refused. Neither
   epoch's verdict changes: relative precision does not make a drone or phone survey-grade.
   The answer adds nullable `relativeFit {referenceEpochId, uncertaintyM, basis}`; its
   `uncertainty` contributions show the terms actually used, while the verdicts and
   frame uncertainties remain explicit. The legend names “Relative to <date> (visual fit
   ±0.40 m)” and still says “Not survey-grade” for coarse dates.

   **Slope propagation for these relative pairs.** A horizontal displacement `dx` on
   `z(x,y)` causes `dz ≈ ∇z·dx`. With scalar horizontal uncertainty
   `h = √(u_from² + u_to²)`, its height-error variance is bounded by `(g·h)²`, where
   `g = max(|∇z_from|, |∇z_to|)` for this layer at the compared cell size. Add that variance
   under the same 3σ root, with coefficient **one**, derived from the differential, and
   retain the neighbourhood guard (5) and ground drift (7). The two guards overlap
   conservatively: the neighbourhood explains discrete shifts; the slope term raises the
   reported detectable bar at steep edges. Motion's existing `2w²` and neighbourhood
   allowance remain separate; `h` here is placement uncertainty alone.

   Estimate each gradient component by the **larger absolute one-sided difference** to
   the adjacent cells on that axis, divided by cell size, then take the Euclidean norm.
   Central differences halve a step and cancel a ridge; this rule does neither. Take the
   larger norm from either surface, never a fitted/tuned edge constant. Missing neighbours
   on an axis leave slope unbounded: that cell is insufficient. Recompute at every drawn
   level. This is a grid-scale first-order bound, not a guarantee against unresolved
   sub-cell geometry, and remains uncalibrated. Other pairs keep today's arithmetic and
   classification exactly.

5. **Horizontal uncertainty** (B5a). Placement error moves an epoch sideways too: cell for cell,
   an unchanged 4 m roof one 25 cm cell off, at 25 cm declared, read as a column lowered 4 m and
   one raised. Each cell is compared with the other epoch's neighbourhood within the horizontal
   radius — the least whole number of cells covering `3·√(u_from² + u_to² + 2w²)` — raised when
   the later height is above everything the earlier had nearby by more than `t`, or the earlier
   below everything the later has nearby (a hole filled); lowered the mirror. A shift within the
   radius explains every cell; a post put up or taken away is still a change.
6. **Vegetation's threshold, derived** (B5a ruling). Between two visits a canopy's top moves
   without anything having grown: a crown sways by decimetres at its top in the wind, and the
   leaves the 2nd-highest sample lands on flutter by their own size. `w = 0.25 m` per epoch — the
   lattice's finest cell — is taken as one standard deviation of that, vertically and
   horizontally, in each epoch independently: `2w²` under the root, so with perfectly placed,
   noiseless epochs the least vegetation change seen is `3·√2·w`, about **1.06 m**, and the
   horizontal radius about as much. The ground has `w = 0`: what sways is above it.
   Uncalibrated, like the 3σ rule; a leaf-on/leaf-off pair is real vegetation change.
7. **The ground's drift, derived.** P5's ground is a lower envelope over a window of
   half-diagonal `R` (from the descriptor's `groundWindowCells`), so on grade `g` it reads up to
   `g·R` low, and which cell of the window it reads moves with whatever stands in it: a bush grown
   at a window's downhill corner would move the ground of every cell within its reach. So the
   ground's `t` adds `g·R`, `g` the steeper of the two epochs' local grades over the window's
   half-width (one-sided differences of the ground, whole centimetres; 1 in 2 where no cell that
   far has a ground). Nothing on flat ground; a real ground change is seen where it is wider than
   the window's span, and its rim reads its own step as drift.
8. **Refused rather than guessed** (B5a ruling): 409 **`not-metric`** when a surface's heights
   are not metres (`summary.metric` not true: a run whose scale was never resolved) — a ratio of
   heights cannot be put against `u`, which is in metres, so no relative-only answer is offered;
   409 **`scale-corrected`** when an epoch is drawn at a runtime scale other than 1
   (`renderConfig.scale`): its surface was measured on the files as registered, and honouring the
   scale would need the heights scaled and the cells resampled about the frame origin, which is
   not done. The committed fixtures are a scene built in metres, and say `metric: true`.
9. **Low confidence is flagged, not hidden** (D7). A compared cell is `lowConfidence` when either
   epoch's confidence grid does not vouch for it at the compared level — weak-v1 weak, unseen, or
   no evidence, through the grid's own min-of-children pyramid. Counted per class.
10. **`minDetectableM`**, per layer, is the least threshold over its compared cells: no change of
    that layer smaller can be seen anywhere between those dates.
11. **Areas**, per layer, are the 4-connected regions of one changed or insufficient class —
    raised and lowered split further by confidence — outlined on their cells' exact edges and
    placed on the globe through the site frame's tangent plane, each with its cells, area,
    low-confidence cells and, for raised and lowered, the mean and largest change of the layer's
    height. **Unchanged cells are counted, never outlined.** Past weak areas' outline bounds the
    layer is classified again a level up and drawn there (`areaLevel`).
12. **What it says it is.** Every answer is `derived: true`, labelled "Derived from height
    surfaces, not observed", and `calibration: uncalibrated`. It is computed per request and not
    stored. The answer: `{…, level, nativeLevel, cellM, thresholdSigma, uncertainty {fromM, toM,
fromVerdict, toVerdict, fromFrameM, toFrameM}, ground, vegetation, appearance}`, each layer
    `{layer, label, heights, minDetectableM, detectable, detectableGrid, motionM,
horizontalRadiusM, driftReachM, counts, lowConfidence, areaLevel, areaCellM, areas}`. **Each
    layer says what it could see** (round-2 ruling): `detectable` sums up its compared cells'
    thresholds (least, median, 90th percentile, largest), and `detectableGrid` gives every cell's
    at the area level (u2 centimetres, base64; at most 2^20 cells, 2 MB a layer), so a cell the
    map calls unchanged reads as "nothing this large", never "nothing".
13. **Other refusals**: 404 unknown site or an epoch not of it; 422 the same epoch twice; 409
    `epoch-not-published`, `no-site-frame`, `surface-not-measured`, `confidence-not-measured`,
    `frames-differ` (another origin height than the site frame's, to the millimetre),
    `surface-unreadable` / `confidence-unreadable` — among them a grid whose finest plane is over
    2^22 cells, refused from its descriptor before any byte is fetched. Both epochs are found
    (404) before either one's status is judged (409).
14. **Behind the write token, rate-limited** (decided in review): a call decodes four grids, the
    same cost and pattern as weak areas and fill targets, and spends the cell-map bucket. The
    comparison holds per layer a few tens of bytes a cell and classifies a band of rows at a
    time: measured with tracemalloc at 1024 × 1024 cells, 36 bytes a cell for vegetation (its
    5-cell neighbourhoods) and 51 for the ground (its neighbourhoods and local rise), so the
    ground layer peaks at about **210 MiB at the 2^22-cell ceiling**; round 2 measured a whole
    call's process at 341 MiB peak RSS for two 2^22-cell surfaces, both layers kept, even at a
    170-cell horizontal radius (the neighbourhoods are log-time doublings). The API machine moves
    to 2 GB with the Fly change; grids over 2^22 cells are refused (13), never compared coarser.

### Limits (measured after round 2; synthetic scenes, uncalibrated)

What the map can see, measured on P5's own surfaces of flat 0.25 m-cell scans at 8, 50 and 2,000
samples a cell, both epochs at 3 cm, with this code (its least detectable change over the scan):

- **Ground: at least 0.23 m, over areas wider than the 5 m window**, at 1 cm of height noise
  (0.23, 0.30 and 0.43 m at 8, 50 and 2,000 samples: the lower envelope's bias grows with the
  samples it is the least of), 0.74–1.48 m at 5 cm and 1.34–2.94 m at 10 cm; plus the drift bound
  on slopes and at a change's own rim. A change narrower than the window is not ground change:
  the window's least reads the ground beside it.
- **Vegetation: at least 1.06 m** (3·√2·w; 1.07–1.19 m measured), with a horizontal radius of
  about a metre.
- **Small objects under about 1 m are not detected yet**: a 0.5 m crate of 3 × 3 cells is
  unchanged in both layers in every configuration above (and in all round 2's: 0 of 45 cells),
  its cells' `detectable` values over half a metre — said, not hidden. A canopy or roof lower
  than about 1.8 m and wider than the window still reads as ground (§ Pipeline 3), and a cell of
  fewer than three samples has no surface at all.

### Evidence (synthetic)

- **The no-change controls** (B5a ruling and round 2), each scan P5's own, committed as
  `control/a..l`: 20 against 2,000 samples a cell, and 1 against 10 cm of noise: 0 of 1,024
  cells changed in either layer (P5's own control before its sigma carried its bias: 204 and
  329 of 400 raised); a sparse lattice of 10 cm cells among 1 cm ones at 2,000 samples: 0 of
  2,304 (it was 28.7% of ground lowered when the ground's sigma averaged the window's noise); a
  4 m roof one cell east at 25 cm each: 0 of 2,304; a 40 m crown growing from 4 to 6 m: its
  25,600 cells raised in vegetation and 0 ground changes (it was 4,356 cells of ground raised).
- **Coarse claiming centimetres** (round 2): a phone epoch claiming 3 cm, or a survey claimed
  without evidence, against a 3 cm survey: `u` 1 m, least detectable change over 3 m, nothing
  claimed (it was 1,024 cells raised at 0.21 m).
- **Coarse against survey** (B5a ruling): the yard re-scan as a phone epoch aligned coarsely
  (±5 m), its grid 6 cm from the frame: least detectable change 15 m in both layers, nothing
  claimed; it had been the 2.5 m shed raised against 0.23 m.
- **Horizontal shift**: an unchanged 4 m roof one 25 cm cell off at 25 cm declared: nothing
  changed (cell for cell: 10 cells lowered, 10 raised).
- **The yard fixture** (`synthetic-yard-surfaces`, both epochs surveyed): ground — the pond dug,
  48 cells lowered by 1.0 m (its 10 × 10 m less the rim, where its step reads as drift), nothing
  raised; vegetation — the shed 24 cells raised 2.5 m, the stack 23 raised 1.6 m (low
  confidence: e1 weak there), the spoil heap 24 lowered 1.8 m, the pond's metre under its 1.09 m
  threshold. Every capture is synthetic; nothing was calibrated on it.

### Not done here

- The drape and its legend: § Web. A stored change map per epoch pair: not needed while a call
  costs about a second at the grid ceiling.
- Appearance-only change; any measurement behind the 3σ rule, the surface estimators, `w` or the
  drift bound; honouring a runtime scale by resampling; per-cell classes (areas, counts and each
  cell's least detectable change are answered); detecting objects under about a metre.
- The surface descriptor names the origin's height, not its datum or the frame's full identity:
  only the height is checked against the site frame (round 2 found it; not yet done).
- P5's demo compares the top surface cell for cell, with each epoch's placement as the change map
  takes it (the cap or the claim, the larger; a metre for coarse): it is the reference the
  fixture's coarsening and cell rule are held to, not the API's layers.

## Web

Date 2026-10-20 · M5 of the splat-pipeline build (design M4–M6 §3 W5c), branch
`claude/feat-splat-change-drape`. Follows the two-layer answer of § API.

### Decision

1. **Offered from Compare › Dates, for two dates that both measured a surface.** A Change map
   switch below the date swipe asks for the map from the older of the two chosen dates to the
   newer, whichever side each is on, so raised always means higher later. A date with no
   `surface` is named instead ("Not offered: … has no height surface measured"); a date chosen
   while it is on moves it to the new pair, or ends it with that reason. The request keeps the
   two dates as they stood either side of the divider, so the panel shows them again.
2. **The ground first; vegetation apart** (coordinator ruling). The ground layer, the primary
   product, is drawn and keyed whenever the drape is on. Vegetation is a separate layer labelled
   "Vegetation: canopy height change, higher threshold, wind and growth not ground change",
   behind its own switch (off each time the map is drawn), drawn in its own colours (yellow up,
   violet down; the ground orange up, blue down) and keyed in its own section, so its counts are
   never read as the ground's. Each section's "changes under ±X m cannot be seen between these
   dates" uses that layer's own `minDetectableM`, rounded up to two figures (a floor is never
   understated).
3. **Ground geometry, and outlines over the scan, under every renderer.** Each layer is drawn
   as `GroundPrimitive`s classified onto terrain and 3D tiles, one per class and confidence,
   never picked, low confidence hatched in its class's colour in screen space. Splats cover the
   ground where they are dense, so each layer is also outlined over the scan
   (`cesium/changeOverlay.ts`): an SVG laid above every renderer's canvas, never in a pointer's
   way, each class outlined with a light fill (low confidence hatched, not comparable dashed
   grey), projected through the camera at the globe's height (the answer carries no absolute
   heights), a polygon with a point behind the camera left out, redrawn after a frame only when
   the view changed, at most every 50 ms. The scan's own colours are not touched.
4. **Beside the swipe and a per-cell rule, not instead of them.** Those choose which splats are
   drawn; the drape marks change under and over them and names its own two dates. It outlives
   the Layers panel (a slot in the details dock loads it on demand), and ends from its own Hide
   or when another site takes the globe.
5. **What the legend says.** Each layer's classes, their low-confidence cells (hatched) and
   unchanged counted, not drawn; the label and that layer's floor; the 3σ rule and the ground
   window (for vegetation, its wind sway) uncalibrated; appearance-only change not computed;
   and, when either verdict is coarse, "Not survey-grade" with both dates' verdicts and the `u`
   they were compared with, so a phone date never reads as a survey. For a measured visual
   pair (§ API 4), name the reference date and relative fit instead of presenting the
   absolute placement as the comparison error, retaining the coarse verdict warning. Each layer also says what it
   could see (round 2): its least detectable change from `detectable`, the least and the median
   (one figure when they round alike), "Small objects under ~1 m are not detected yet", and the
   best that layer was measured to see (§ API Limits: ground 0.23 m over areas wider than the
   5 m window at 1 cm of noise; vegetation 1.06 m), named synthetic and uncalibrated.
6. **The token is the client's rule.** The auth middleware sends the write token on this one read
   as on a write, and a 401 opens the token field; the legend and Compare say the map "needs the
   pilot's write token", open Settings in one click, and ask again once a token is entered
   (the query is keyed by a fingerprint of the token, never the token). Every 409 code,
   `epoch-unaligned`, `not-metric` and `scale-corrected` among them, has its sentence; nothing is
   drawn on a refusal.

### Evidence (synthetic)

`e2e/changeDrape.spec.ts` mocks `/change` with the API's own answer for the committed yard fixture
(`e2e/fixtures/changeYard.py` runs § API's analysis and response code over
`synthetic-yard-surfaces` and `synthetic-yard-epochs`, both dates surveyed, placed in the yard's
frame): under CesiumJS, PlayCanvas and Spark, the ground drawn alone first (the pond lowered, 48
cells, under ±0.49 m), on the ground and outlined over the scan, its colours on screen from above;
vegetation apart behind its switch (the shed and the stack raised, the stack hatched, the spoil
heap lowered, under ±1.1 m) and gone again when it is off; a coarse later date (its claimed 0.062 m held to the 1 m floor) claiming nothing under a metre (both floors 3.2 m) and saying "Not survey-grade" with both verdicts; the token asked for and then
sent; no offer without surfaces. Before the outlines over the scan, the ground drape alone showed
only 0.24–0.48 % of the view in its raised colour from 70 m over the yard (the grass splats
covered the rest); with them, 1.13–1.31 % (one-layer answer, SwiftShader, one run each).

### Not done here

- Outlines stand at the globe's height, not each area's own: the answer carries no absolute
  heights, so over a raised area they sit at its foot.
- Per-cell change (areas and counts only) and appearance-only change.
- A "least detectable" overlay drawn from `detectableGrid`: the legend states the layer's
  summary; drawing each cell's threshold is a later decision.
