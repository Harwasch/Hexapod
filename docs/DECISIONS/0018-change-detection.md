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
  `lowConfidence` (§ API 3).
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
`claude/feat-splat-change-api`.

### Decision

1. **A surface is an epoch revision's map, like its confidence grid.** Migration 0036 widens
   `epoch_maps.kind` to `confidence` · `surface` (the downgrade deletes the surface rows first).
   `register` records `registration.json`'s `surface` on the revision it published, through the
   same record / carry / repoint path as the confidence grid (ADR 0016 § API 2). A descriptor is
   checked first — the container, a file beside the tiles, a cell on the lattice, the four
   channels at the types and units the arithmetic assumes (`dsm`, `ground` i2 cm; `sigma` u1 cm;
   `support` u1), `noData` −32768, the summary counts, a `status`, and
   `vertical.originHeightM` — and one that fails is logged and not recorded. `surface.bin` is a
   sidecar of kind `surface`, bound to the tiles and not attachable, like `confidence.bin`. Every
   epoch says `surface: {status: measured, assetId, url, cellM, level, policy, calibration,
cells, surfacedCells, medianSigmaM, originHeightM, groundWindowM}`, or null — not measured.
2. **`GET /sites/{id}/change?from=&to=`** (`change-v1`) compares the current revisions of two
   published epochs of one site at the coarser of their surfaces' levels, each coarse cell taken
   over **only the children both epochs measured** — of those, dsm the highest, ground the
   lowest, sigma the largest (§ Pipeline 4's rule; `coarsen_surface` matches P5's demo byte for
   byte on the committed fixture `data/tiles/synthetic-yard-surfaces`). Coarsening each epoch
   on its own would let one side's single ground child stand for a whole cell against the
   other's crate in a sibling (a 2 m "lowered" with nothing flagged, found in review), or set
   one side's uphill children against the other's downhill ones. So the finer epoch is first
   brought up over the children whose coarse cell the coarser one measured, then both go up
   together over the children both measured. Where both have that, `Δ = dsm_to − dsm_from` (in
   32 bits: two i2 heights can differ by more than i2 holds) against
   `t = 3·√(σ_from² + σ_to² + u_from² + u_to²)`, `u` each epoch's confidence grid's
   `alignmentCap.uncertaintyM`: `raised` above `t`, `lowered` below `−t`, else `unchanged`;
   `insufficient` where either measured something but no child is shared; no class where
   neither measured anything (the windows' bounding box is not a place either epoch saw — P5's
   demo report counts those cells as insufficient too; the API does not, kept in review). `appearance-only`
   stays reserved: `appearance: {status: not-computed, reason}`. Two windows together over 2^22
   cells are compared a level up, and the answer says so (`level` above `nativeLevel`).
3. **Weak cells are classified and flagged, not hidden** (D7). A compared cell is `lowConfidence`
   when either epoch's confidence grid does not vouch for it at the compared level — weak-v1 weak,
   unseen, or no evidence at all, through the grid's own min-of-children pyramid, so a coarse cell
   is as weak as its weakest child — when it was compared over fewer children than either side
   measured (the coarser epoch counting as having measured its whole cell), or when either sigma
   is saturated (255 cm: the threshold is then only a lower bound). The answer counts them per compared class beside the class counts.
4. **`minDetectableM`** is the least threshold over the compared cells: no change smaller can be
   seen anywhere between those dates (null when none was compared). An epoch placed by GPS alone
   at ±10 m makes it about 30 m, and so finds no change rather than false ones.
5. **Areas** are the 4-connected regions of one changed or insufficient class — raised and
   lowered split further by confidence — outlined on their cells' exact edges (rasterio `shapes`,
   as weak areas are) and placed on the globe through the site frame's own tangent plane (§ API 5
   of ADR 0016), each a MultiPolygon with its cells, area, low-confidence cells (and
   `lowConfidence` when all are), and for raised and lowered the mean and largest dsm change and
   the mean ground change. Raised and lowered first, then insufficient. **Unchanged cells are
   counted, never outlined** (decided in review): they are most of a map and of its outline's
   cost, and an area of them would read as confident wherever the comparison could not see — an
   epoch placed by GPS alone is unchanged everywhere. The drape draws changes and insufficient
   only. The outline is held to weak areas' bounds (2^20 cells, 2^18 cell edges): past them the
   map is classified again a level up, by the same rule, and drawn there (`areaLevel`); the
   counts stay the compared level's.
6. **What it says it is.** Every answer is `derived: true`, labelled "Derived from height
   surfaces, not observed", and `calibration: uncalibrated`: the 3σ rule (and the surfaces' own
   estimators, the 5 m ground window among them) is first principles until a pilot site scanned
   twice measures it. It is computed per request from the stored maps and not stored itself.
7. **Refusals, each with a code**: 404 unknown site or an epoch not of it; 422 the same epoch
   twice; 409 `epoch-not-published`, `no-site-frame`, `surface-not-measured` (a revision with no
   surface: one side missing everywhere is no map, not a map of insufficient cells),
   `confidence-not-measured` (its `u` and its vouching are unknown), `frames-differ` (a surface
   above another origin height than the site frame's, to the millimetre), `surface-unreadable` /
   `confidence-unreadable`. Both epochs are found (404) before either one's status is judged
   (409). The origin-height check is kept (review): a site frame is fixed once created and
   `register` holds every run to its lat, lon and height exactly, so the descriptor's
   `vertical.originHeightM` is the one place a surface could still disagree with its frame; lat
   and lon are not in the descriptor.
8. **Behind the write token, rate-limited** (decided in review). It stores nothing, but a call
   decodes four grids, and the 96 MB grid cache cannot hold four of the largest: the same cost and
   the same pattern as weak areas (POST, token) and fill targets (GET, token). It also spends the
   per-client cell-map bucket they spend. The drape (W5c) sends the console's token.
   The comparison holds about 30 bytes a cell for the window (both surfaces over their shared
   cells, the outputs, what each side measured) and classifies a band of rows at a time; a
   2048 × 2048 map with confidence pyramids peaked at about 150 MiB here.

### Evidence (synthetic)

The P5 demo's two epochs (`confidence_demo.py --change`) registered through `register` on one site
and asked `GET /change`: at 0.25 m cells, `u` 0 and 0.062 m, `minDetectableM` 0.191 m; 49 raised
(the crate's 36 cells and the 13 its sides stand in, mean Δ 1.50 m), 0 lowered, 1,423 unchanged,
1,997 insufficient (the unchanged ones counted, not outlined); 11 of the raised cells low confidence (the crate's sides, as the demo found).
From and to swapped: 49 lowered. The same re-scan as if unaligned (±10 m): `minDetectableM` 30.0 m
and nothing raised or lowered. Every capture is synthetic; nothing was calibrated on it.

### Not done here

- The drape and its legend (W5c). A stored change map per epoch pair: not needed while a call costs
  about a second at the grid ceiling.
- Appearance-only change; any measurement behind the 3σ rule or the surface estimators; per-cell
  output (only areas and counts are answered; a per-cell raster for the web is a later decision).
