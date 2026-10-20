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
   0.05 opaque: `keepShare = Σα[keep] / Σα`, `score = 1 + floor(254 · keepShare · f)`, 0 for no
   evidence. **Weak** is `score < 128` — fewer than half of what is visible there, by alpha, passed
   keep: a majority rule, exact because the score floors (`254 · keepShare · f ≥ 127` is
   `keepShare · f ≥ ½`; rounding, as first written, put a 49.9 % share at 128, strong) — or
   **unseen**: the cell's centre is inside the disc the cameras were
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
   - aligned visually to a reference: **`v = sqrt(ref_rel² + (3 rms)²)`**, where `ref_rel` is
     the reference's own frame-relative figure — 0 when the reference made the frame (ordinal 1),
     else its alignment's `frameRelativeUncertaintyM`, else (an older alignment) its absolute
     `uncertaintyM`, which is never smaller; the worker passes it in the reference block —
     **unless the epoch's check targets refute the fit** (below);
   - not aligned to the reference it had (a failed fit, a refuted one, or an RTK epoch that kept
     its own GNSS frame over a coarser reference): its own placement and the reference's
     absolute one in quadrature — the frame is where an earlier epoch was placed, so it is never
     better than either;
   - not aligned and no reference, but the site has earlier epochs: **its own georeference
     uncertainty** (5 m EXIF, 10 m a located point or a pin).

   **Check targets, from first principles.** A check target measures the epoch **on the
   globe**: its RMSE `d` (`checkTargets.rmseM.total`, from however many were evaluated) is how
   far this epoch's content is from where the targets were surveyed. The frame is not the
   globe: it is where the frame-defining epoch was placed, which is at most `ref_abs + ref_rel`
   from the globe (the reference's absolute placement, then the reference from the frame). So,
   by the triangle inequality, the epoch is **at least `d − ref_abs − ref_rel` from the frame**.
   - When that lower bound exceeds the fit's claim `v`, the claim is proven wrong — the
     targets see an error the fit and the chain of placements cannot hold — and the epoch is
     taken as not aligned to its reference (the quadrature rule above). Its own placement
     `uncertaintyM` already carries `d` when three or more targets were evaluated; with fewer,
     `d` is still the floor, since a measured disagreement is never less than the placement.
   - When the chain explains it (a phone fitted to a ±5 m GPS frame is 2 m off its targets,
     and so is the frame), the targets say nothing about the fit, and `v` stands: an error
     every epoch of the site shares is no error between them.
   - Past the frame-defining epoch, `d` is a floor on every branch; the frame-defining epoch
     stays 0 — it is the frame, wherever the globe says it is.

   The alignment's own `confidence` (0–1) does not enter the cap: it is a heuristic, not a
   probability. A tight visual fit scores 0 there when its check targets' RMSE reaches the
   length of their tolerance; those metres now enter `u_rel` themselves, through the rule above.

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
   every tile `extras.confidenceMin` right after `gaussians`: the lowest score, at the grid's
   finest level, of the cells the tile's bounding box covers in plan. A cell **with evidence
   counts at its score**; an **unseen** cell (empty, inside the pointed disc) counts at **0** —
   expected and not seen, as weak as a cell can be; an empty cell outside the disc is not this
   epoch's and does not count; a tile over none that count has no `confidenceMin`. The box, not
   the tile's gaussians, because strict gating removes exactly a weak cell's gaussians: the
   tile drawn over that cell holds none of them, and a minimum over what it holds would call it
   strong. A parent's box holds its children's, so a parent is never more confident than any
   child. `catalog` writes the
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
- A tile's `confidenceMin` is a summary for finding weak tiles, not a per-gaussian score. Its box
  is padded by its gaussians' reach and is axis-aligned, so it can take in a weak neighbour's
  cell the tile barely draws over: the summary errs to the weak side, never the strong one.
- `confidenceMin` is the grid's, not the tiles': the API's tile fingerprint
  (`sidecars.tiles_fingerprint`) leaves it out on every tile, so a re-pack that changes only the
  grid keeps what is bound to the tiles (`nativeLod`, the rig, objects).
- A re-publish of new tiles without a grid drops the old one with "Confidence needs
  re-measuring" (the `confidence` sidecar kind, A4; until A4 it was an unrecognised sidecar).
- New trip-wires: `docs/HANDOFF.md` § 6.

### Not done here (stubs and later PRs)

- Storing the map per epoch revision, weak-area polygons and the re-scan flow: A4, W4b.
- The fill gate: F4. The web views: W4a (a CesiumJS colour part, no engine patch).
- 3D confidence; per-camera-frustum "unseen"; any recorded measurement behind weak-v1's 128 or the
  cap's `cell / 2u` — both are first principles until a pilot site scanned twice measures them.

## API

Date 2026-10-20 · M4 of the splat-pipeline build (design M4–M6 §2 A4), branch
`claude/feat-splat-confidence-api`.

### Decision

1. **A map per epoch revision, as rows** (`epoch_maps`, migration 0035; `docs/DATA_MODEL.md`).
   One row per **asset** and kind: the descriptor verbatim, its summary, the file's URL, the
   finest cell and level. A grid is measured on one reconstruction, so a re-run (a new asset)
   has its own map and the superseded revision keeps the one it was measured with, as it keeps
   its tiles. The planes stay in the bucket.
2. **`register` records it** from `registration.json`'s `confidence`: on a first run's asset,
   on a re-run's new revision, and — for the very URL the epoch already shows — whatever the
   run that last published those tiles says (one bucket rewrites a run's keys in place). A
   republish of the same tiles by a run that measured nothing, whose publish carried the grid
   into the new generation, carries the map; an attach moves the map's URL with its
   generation. A descriptor is checked first (the container and version, a `uri` that is a
   file beside the tiles and nothing else, a cell on the lattice, the summary keys every
   reader uses); one that fails is logged and **not recorded**, so the epoch reads "not
   measured" rather than pointing at a grid nobody can read. `priorEpochs` in
   `site_context.json` was seeded by the pipeline PR and is unchanged here.
3. **Every epoch says `confidence`**: its current revision's map summarised
   (`EpochConfidenceRead`), or null — not measured. Null is said plainly: the run that made the
   revision measured none (a splat upload has no capture evidence). Read with the epochs in one
   eager load.
4. **`confidence.bin` is a sidecar of kind `confidence`**, bound to the tiles (`TILES`, like
   collision and view cones): carried onto the same tiles, replaced by a run's own, dropped from
   new tiles with "Confidence needs re-measuring". Not attachable: only a run ships one, because
   only `register` records the row that describes it.
5. **Weak areas** — `POST /sites/{id}/epochs/{epoch}/weak-areas` `{budgetM2?, focus?,
maxAreas 1..8}`. The finest level's weak cells (weak-v1, where the summary is stated), closed
   on the raster by whole cells (≥ 0.5 m each way — exactly a mitred buffer out and back in, so
   a block keeps its area), outlined region by region with holes under the smallest zone
   filled, simplified by half a cell, parts under 1 m² dropped, largest first; over 2^20
   cells, or over 2^18 cell edges of outline (a dense noisy plane is one region with a hole
   per strong cell, which took minutes), drawn a level up where at least half of the
   children the epoch speaks for are weak (weak-v1's own majority; the min-of-children pyramid
   would mark a whole coarse cell weak for one child). Counts are of the finest cells inside.
   To the globe through the **site frame's own tangent plane** (PROJ `topocentric`, inverted):
   the exact inverse of how the pipeline placed the tiles. The design's azimuthal-equidistant
   projection is not that plane — it measures on the ellipsoid, and at a 1,500 m origin a point
   1 km east lands 23 cm short. A budget takes whole zones in rank order — under `focus`, then largest, then nearest —
   while they fit (within one zone cell), and clips the first that does not to a disc about
   the focus or its weakest cell, grown until it holds what is left (a disc follows a
   ring-shaped zone where a square would cover the strong ground beside it): the budget goes on the zones that rank
   first, never on a speck behind a zone too large for it, and any budget that can hold a
   zone gets one. (Trimming cells and outlining again, as the fog scout does, left nothing on
   scattered weakness: closing grows the cells faster.) 409 `confidence-not-measured`, `confidence-unreadable`, `no-site-frame`.
6. **Patterns are UX heuristics, labelled so in every answer** (`heuristic`, each `hint`):
   mostly `views`/`spread` → `orbit` when the patch stands ≥ 3 m and is ≤ 40 m across, else
   crossing `oblique` passes; `gsd` → `nadir`, lower, `targetGsdM` = the capture's median (null
   without a metric scale); `unseen`/`accuracy` → `nadir`; `alignment` → `nadir` overlapping
   the reference area. Ties go to `alignment` first (it caps every score). Suggested pitch:
   −90, −45, −30.
7. **A scout capture can be a re-scan**: `purpose: re-scan-weak`, `pattern`, `gimbalPitchDeg`
   (−90..−30) and `rescanOf {siteId, epochId, policy, cells, reasons}` — a reference, **not a
   foreign key**: plans are a workspace's and sites are public (the privacy split), and the
   re-scan of a cell is any later epoch of the site that covers it (C7). The planner sizes
   oblique as two crossing sets of the nadir lines and an orbit as two rings round the convex
   hull at the stand-off where the pitched camera meets the ground (altitude / tan pitch,
   ≥ 5 m) — rules of thumb too.
8. **Fill targets** — `GET /sites/{id}/fill-targets?epochId=` (`rescan-gate-v1`): cells of
   epoch B's **own finest level** (default B: the latest published) that are weak in B and
   were weak in the newest earlier measured epoch whose footprint covers the cell. An earlier
   epoch counts for a cell only when it covers it; one that saw it well fixed it, so B's
   weakness there is not a re-scan's. A coarser earlier grid is read by the cell holding B's;
   a finer one is weak in B's cell only when none of the children it saw there was strong.
   The gate never coarsens B: one weak child never opens its parent (the first build compared
   at the coarser level and did; review R3). Each target records the earlier epoch it
   re-scanned (`satisfiedBy`, one byte per target, an index into `comparedWith`). 409
   `no-rescan-yet` when nothing earlier was measured. A bit-packed window in the tileset frame
   (rows south to north, least significant bit first). Compared with at most the **eight**
   most recent earlier measured epochs (`comparedWith`; `capped` says there were more): each
   may be a ~17 MB grid decoded, on a 1 GB machine. F4's fill gate fills only these.
9. **Reading grids**: from the public bucket by the URL's own key (the published, immutable
   copy — not the run's private key a Refine rewrites; the design said the unpublished key), a
   HEAD refusing anything over 32 MB, a bounded ranged GET, inflated no further than the
   descriptor declares (≤ 64 MB) and checked against its sha256. Decoded grids sit in a
   per-process LRU bounded by bytes (96 MB), keyed by URL **and** sha256 so a rewritten grid is
   never a stale hit. Neither route stores anything, but both take the write token: weak
   areas because every POST does (`test_no_mutating_route_is_left_ungated`) and it is the
   first step of planning a flight, gated anyway; fill targets because a call may decode
   nine grids, and its one consumer, the fill workflow, holds the token. Both also spend a
   per-client token bucket (20, then 60 a minute) like the other routes whose cost is the
   attack.

### Why not fog-v2

The Fog of Knowing stays "how fresh" (C6; no `POLICY_VERSION` bump). Crediting a capture by its
confidence would mix two questions — when was this seen, and how well — in one layer whose
policy readers already rely on. Confidence is its own layer and its own endpoints; open question
2 of the design stays at its default.

### Not done here

- The re-scan flow in the console (W4b), the DJI export of obliques and orbits (W4c; the export
  still flies nadir whatever the pattern), the fill gate itself (F4).
- Any measurement behind the pattern rules, the 0.5 m closing or the 1 m² smallest zone: UX
  heuristics until a pilot site re-scanned by them says otherwise.

## Web

Filled by `claude/feat-splat-confidence-view`.
