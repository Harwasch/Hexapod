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
     2.55 m"; 0 only where there is no surface) and `support` (`u1`: bit 1 SfM, bit 2 splat, the
     floor of log2 of the samples in the high nibble).
3. **The estimators, from first principles and uncalibrated.** Per cell, streaming, the four
   highest and four lowest sample heights (`z1 ≥ z2 ≥ z3 ≥ z4` from the top):
   - `dsm = z2`, the 2nd highest: one floater cannot raise it;
   - `ground` = the 2nd lowest, then the least of it over the cells whose centres are within 2.5 m
     along each axis (a 5 m square): bare ground under a crown or a roof up to 5 m across;
   - `sigma = max(1 cm, (z1 − z4) / 2)`, half the spread of the top four;
   - fewer than 3 samples is no surface: with two, the 2nd lowest is the higher of them and the
     ground would stand above the top. From three up, `ground ≤ dsm` by construction.

   The descriptor says `status: uncalibrated` and names each estimator, so no reader mistakes them
   for measured accuracies.

4. **The threshold.** Between epochs `a` and `b`, at the coarser of their levels (dsm by maximum,
   ground by minimum, sigma by maximum over the children that have a surface):
   `Δ = dsm_b − dsm_a`, and the cell is **changed** when `|Δ| > 3·√(σa² + σb² + ua² + ub²)`, where
   `u` is each epoch's uncertainty **from the site frame** (its confidence grid's
   `alignmentCap.uncertaintyM`, ADR 0016 §5: 0 for the epoch that made the frame). Classes:
   `raised`, `lowered`, `unchanged`, `insufficient` (either epoch has no surface in the cell), and
   `appearance-only`, reserved and never emitted (below). Three sigma is a first-principles choice
   (about one false call in 370 cells of Gaussian noise), not a calibrated one. A5 computes the
   map; the demo computes it the same way to show it works.
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
- **The ground is biased low, never high.** A lower envelope reads low on a slope (up to 2.5·g m
  along each axis on grade `g`), and a low outlier that survives the 2nd-lowest rule spreads over
  its 5 m square. `ground` does not enter the change threshold; it is for the bare-earth view.
- **Under noise both surfaces are biased, alike in every epoch.** With height noise σ the 2nd
  highest of a cell's samples reads about σ to 2σ high, and the ground, a lower envelope over 441
  cells' 2nd-lowest values at 0.25 m, about 3σ low (the demo: 1 cm noise, flat ground at −3 cm,
  the tree's 4 m top at 4.01 m). Epochs with the same noise share the bias, so `Δ` does not; an
  epoch much noisier than another does not. `sigma` is the spread of the top four, not a noise
  estimate: under σ with ~20 samples a cell it is about σ/2, so the 1 cm floor is what it mostly
  reads. Calibrating either needs a site surveyed twice.
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
  would have hidden real change and caught nothing false; whether A5 should exclude them is open.
- **Memory and determinism.** One band of extremes at a time (73 bytes a cell, 2^20 cells a band)
  and the plane (8 bytes a cell); order statistics then roundings, so the bytes depend on neither
  chunk nor band. `points3D.bin` is streamed.
- Until A5 adds a `surface` sidecar kind, a re-publish that carries sidecars from a previous
  generation reports `extras.surface` as an unrecognised sidecar if the new run has none, as
  `extras.confidence` does until A4.

### Demo (synthetic)

`tools/pipeline/experiments/confidence_demo.py --change`: two scans of the M4 demo's field, each
its own sample with 1 cm of height noise and a sparse SfM cloud, and a 1.5 m crate in the second on
ground both saw well. At 0.25 m cells, `u` 0 and 0.062 m: every cell on the crate raised by
1.48–1.51 m, the cells its sides stand in raised too, no cell that does not touch it changed,
nothing lowered; unchanged cells reach at most 0.09 m, under 0.4 of their threshold (about
0.19 m). Every capture in it is synthetic; nothing here was calibrated on it.

### Not done here (stubs and later PRs)

- The change map itself, stored per epoch pair, its API and its classes: A5 (migration 0036).
- The change drape and the "Derived from height surfaces, not observed" label: W5c.
- Appearance-only change; any recorded measurement behind the estimators or the 3σ rule; a
  measured leaf-on/leaf-off pair; a cross-language surface fixture for the API and web decoders.

## API

Filled by `claude/feat-splat-change-api`.
