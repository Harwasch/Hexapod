# 0016: Per-cell confidence per epoch, and the weak-v1 policy

Status: proposed

## Pipeline

Date 2026-10-20 · M4 of the splat-pipeline build (design M4–M6 §1 C1–C4, C8; §2 P4), branch
`claude/feat-splat-confidence-map`.

### Context

Milestone 4 is "confidence drives re-scans": after each capture the site should know which parts of
it were seen well, so the next flight (or the next phone walk) can be aimed at the rest, and so a
later epoch can be judged against an earlier one cell by cell. The quality stage already measures,
per trained gaussian, how many frames saw it, from how wide a spread of directions, at what ground
sampling distance and — where held-out frames exist — how accurately (`tools/pipeline/quality.py`). It then threw those numbers away, kept a tier per gaussian only long
enough to gate the splat, and summarised the rest as one `keepPct`. Nothing said _where_ an epoch was
weak, in a frame another epoch could be compared in.

### Decision

1. **One site lattice (C1).** Every per-cell map is axis-aligned 2.5D columns in the **tileset
   frame** — the site frame whenever the run had one (ADR 0013), which `root.transform` places on
   the globe. A cell is `0.25 m × 2^level`; `i = floor(x / cell)`, `j = floor(y / cell)` (x east, y
   north) from the frame origin. Levels nest exactly (`x / cell` divides by a power of two, exact
   in float64), so two epochs compare at the coarser of their two levels with no resampling.
2. **One container (C2)**, `hexapod.cellgrid` v1 (`tools/captures/cell_grid.py`): a gzip (level 9,
   `mtime = 0`) of the level planes, finest first, each `nj × ni` cells row-major with its channels
   interleaved little-endian; a JSON descriptor — the tileset's root `extras.confidence` — names the
   frame, lattice, channels, each level's window and byte range in the uncompressed planes, the
   kind's own keys, and the sha256 of the planes. Confidence's channels are `score` (u1),
   `reasons` (u1, bits `views 1, spread 2, gsd 4, accuracy 8, unseen 16, alignment 32`) and `relief`
   (u1, 0.25 m). Above the finest level, up to eight coarser levels, **min-of-children**: score the
   minimum of the children that have one, reasons their OR, relief their maximum — a coarse cell is
   never shown stronger than its weakest child, and an unseen child carries its bit up.
3. **The evidence is kept (`support/`).** The quality stage writes every trained gaussian's centre,
   alpha, final tier, which keep criteria it failed (the negation of each comparison
   `assign_tiers` makes, plus `accuracy` when held-out frames demoted it), views, spread and GSD —
   about 20 bytes a gaussian — before its scratch columns go. All rows, not `gated.ply`: under
   `bar: strict` that holds keep only, and a map of it would call every published cell strong.
4. **weak-v1 (C3), from first principles, not calibrated.** Per cell, over the gaussians at least
   0.05 opaque: `keepShare = Σα[keep] / Σα`, `score = 1 + round(254 · keepShare · f)`, 0 for no
   evidence. **Weak** is `score < 128` — fewer than half of what is visible there, by alpha, passed
   keep: a majority rule — or **unseen**: the cell's centre is inside the disc the cameras were
   pointed at (the quality stage's pointed ROI, placed like the gaussians) and nothing is in it. A
   reason bit is set when more than half of the cell's non-keep alpha failed that criterion. The
   level is the finest at which the median occupied cell holds at least 16 visible gaussians (a
   binomial share from 16 has σ ≤ 0.125, enough to tell under a half from three quarters) within
   2^22 cells.

   **The weak share** (`summary.weakShare`, and what every later reader — A4's weak areas, the
   web's legend — must mean by it) is `weak / (scored + unseen)` at the finest level: the weak
   cells over the cells this epoch speaks for, those with evidence and those it should have seen
   and did not. Cells with no evidence outside the pointed disc are not this epoch's and count in
   neither. `summary.reasons` counts, per bit, the weak cells that carry it.

5. **The alignment cap (C4), relative to the site frame.** `f = min(1, cell / (2 u_rel))`: a cell
   can only claim evidence when the epoch's placement error is under half a cell. `u_rel` is the
   epoch's uncertainty **from the site frame, not from the globe** — every epoch of a site is
   compared in that frame, so an error they all share is no error between them. `align` writes it
   as `frameRelativeUncertaintyM` (`alignment.frame_relative_uncertainty`):
   - the epoch that makes the frame (no earlier published epoch: `site_context.json`
     `priorEpochs` 0): **0**;
   - aligned visually to a reference: **`sqrt(ref_rel² + (3 rms)²)`**, where `ref_rel` is the
     reference's own frame-relative figure — 0 when the reference made the frame (ordinal 1), else
     its alignment's `frameRelativeUncertaintyM`, else (an older alignment) its absolute
     `uncertaintyM`, which is never smaller; the worker passes it in the reference block;
   - not aligned to the reference it had (a failed fit, or an RTK epoch that kept its own GNSS
     frame over a coarser reference): its own placement and the reference's absolute one in
     quadrature — the frame is where an earlier epoch was placed, so it is never better than
     either;
   - not aligned and no reference, but the site has earlier epochs: **its own georeference
     uncertainty** (5 m EXIF, 10 m a located point or a pin).

   So a phone aligned visually to the consumer-GPS epoch that made the frame (±5 m on the globe) is
   capped by its fit alone, and a later GPS-only epoch is weak everywhere with the `alignment`
   reason (±5 m on 0.5 m cells: `f = 0.05`, a score of at most 14) — honest, since nobody knows
   where its evidence is to a metre. The confidence stage reads `frameRelativeUncertaintyM`; an
   alignment from before it existed is read at its absolute `uncertaintyM`, else `georef.json`'s,
   else 10 m. `priorEpochs` is the site's published epochs other than this capture's, and for a
   re-run only those with a lower ordinal, so a re-run of the frame's own epoch stays the frame. The
   cap is applied at the finest level; the coarser levels inherit it through the minimum, so they
   never relax it.

6. **Shipped beside the tiles, opt-in (C8).** The `confidence` stage (`confidence_grid`, between
   `place` and `package`, no GPU) writes `confidence/confidence.{json,bin}`. `package` copies
   the file beside the tiles, appends the descriptor to the root extras after `collision`, and gives
   every tile `extras.confidenceMin` right after `gaussians` — a leaf the lowest score of the scored
   cells its gaussians fall in, a parent the lowest of its children's. `catalog` writes the
   descriptor at the top level of `registration.json` (never inside `quality`; `quality.json` is
   unchanged). `convert(confidence=None)` writes exactly the bytes it always did, so every fixture
   stays byte-identical. `photo-reconstruct` v14; `splat-ingest` has no evidence, so no grid:
   "not measured".

### Why

- **2.5D, not voxels.** Every consumer in sight — the re-scan planner, the mission zones, the
  compositor's owner per cell, the change map — is a 2D thing on the ground. A column per cell with
  its relief is what they read; 3D confidence is deferred until something needs it.
- **Not GLB attributes.** A per-gaussian confidence in the tiles would need an SPZ slot it does not
  have, a shader patch in two copies, and would tie the map to the tiles' level of detail. A
  separate grid is loaded once, compared across epochs without touching a tile, and decoded by a
  gunzip in Python, the API and TypeScript alike.
- **Why the cap.** Without it a GPS-placed epoch would mark cells strong that it may have seen ten
  metres away; a re-scan planned from it would skip the very cells it never looked at. With it, an
  unaligned epoch can never make a cell look better than an aligned one did.
- **Determinism.** The stage sums integers (alpha in 1/255ths) and writes the placement out per
  component, so its bytes do not depend on chunking, on banding or on a BLAS kernel; it reads
  `support/` a chunk at a time in five passes and sums in bands of 2^20 cells, so its memory does
  not grow with the splat (`tests/test_bounded_memory.py`).

### Consequences

- `support/` adds ~20 bytes a trained gaussian to what `quality` (on Modal's CPU box) sends back:
  ~160 MB at 8M gaussians. If that transfer is measured too slow, a fallback is to drop the
  positions and re-read `trained.ply` on the worker.
- **The ROI disc is coarse.** "Should have been seen" is where the cameras pointed (half the median
  camera distance about the point nearest every optical axis), not each camera's frustum; a
  lawnmower survey's disc covers its middle, not its corners.
- **What the frame-relative figure cannot see.** An epoch with no reference on a site that has
  earlier epochs is taken at its own placement alone: the frame's own absolute error (where the
  first epoch's GPS put it) is not known to the run, so an RTK epoch placed there by its GNSS
  could be further from the frame than it claims. With a reference, the reference's absolute
  uncertainty is added in quadrature, which covers it.
- A tile's `confidenceMin` is computed from the leaf positions as packed and the grid from
  `support/` placed by `placement.json`'s single similarity; the two placements agree to float32
  rounding, so a gaussian exactly on a cell edge may be looked up in its neighbour. It is a
  summary for finding weak tiles, not a per-gaussian score.
- Until A4 adds a `confidence` sidecar kind, a re-publish that carries sidecars from a previous
  generation reports `extras.confidence` as an unrecognised sidecar if the new run has none.
- New trip-wires: `docs/HANDOFF.md` § 6.

### Not done here (stubs and later PRs)

- Storing the map per epoch revision, weak-area polygons and the re-scan flow: A4, W4b.
- The fill gate: F4. The web views: W4a (a CesiumJS colour part, no engine patch).
- 3D confidence; per-camera-frustum "unseen"; any recorded measurement behind weak-v1's 128 or the
  cap's `cell / 2u` — both are first principles until a pilot site scanned twice measures them.

## API

Filled by `claude/feat-splat-confidence-api`.

## Web

Filled by `claude/feat-splat-confidence-view`.
