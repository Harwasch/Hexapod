# 0017: Compare two dates

Status: proposed

## Two dates side by side (web)

Date 2026-10-20 · design M4–M6 §1 C5, W5a; branch `claude/feat-splat-compare-dates`.

### Context

A site can hold several dated captures (epochs, ADR 0013), all drawn with one placement: the
frame anchor's model matrix (`SiteManager.followAnchor`). The capture-date control shows one at a
time. Seeing what changed means flicking between them; a swipe shows both at once.

### Decision

1. **Two primitives, the stock split.** `SiteManager.setCompareDates({ left, right } | null)`
   shows both epochs' current models and sets their tilesets' `splitDirection` to `LEFT` and
   `RIGHT`. The engine's `GaussianSplatPrimitive` copies `tileset.splitDirection` every update and
   its fragment shader discards at `czm_splitPosition` (`GaussianSplatPrimitive.js` `update`,
   `PrimitiveGaussianSplatFS`); `scene.splitPosition` comes from the Compare panel's existing
   divider (`SplitSlider`). No engine-patch change. The halves never overlap on screen, so the
   two sorts never have to agree.
2. **An overlay on the date shown, not a new choice of date.** The single date chosen stays, the
   `epoch` event keeps naming it, and it is drawn again when the comparison ends. The comparison
   is the scene's (`compare-dates` event into `useSites.compareDates`); the controls follow it.
3. **It ends rather than going dormant.** Whenever the two can no longer both be drawn by
   CesiumJS (one date chosen, another splat renderer, a representation one date lacks, a date
   withdrawn) or the site stops being the primary one or unloads, the comparison ends and says
   so. A pair kept but not drawn would leave split tilesets nobody is looking at. A date run
   again is compared as its new model. Whatever is not drawn is reset to `SplitDirection.NONE`.
4. **What travels with a scan is cut with it.** Inferred layers drawing themselves and split
   objects copy their scan's `splitDirection` each frame; companions drawn by the scan's
   primitive are cut with it already.
5. **CesiumJS only (C5).** Under PlayCanvas or Spark, Compare › Dates is disabled with "Needs
   the CesiumJS renderer (Settings › Advanced › Splat renderer)" and "Draw with CesiumJS".
6. **Picks follow the side clicked.** A ray through a point on screen (Measure, selection, the
   orbit pivot, the wheel zoom) is answered by `SplatCollider` only from tilesets drawn at that
   point: a tileset split to the other side of `scene.splitPosition` is skipped. The site card
   names the date whose tileset was hit (`SiteManager.drawnAssetOf`). Object selection
   (`SceneSelectController`) does the same: each pick source says its scan's side, a click
   selects only from the date drawn where it is, and a brush stroke paints the date drawn where
   it begins, only that date's splats on its side. Queries with no point on screen (camera
   collision, the floor) count both dates. What was selected or painted on a date stops being
   selected once that date is no longer drawn, and an open site card names the date shown.
7. **Said where a date is named.** The capture-date control checks no date and marks each "left
   of divider" / "right of divider"; the Inspector's Date shown becomes "Left of the divider" and
   "Right of the divider", each with its own facts. The Inferred control and its legend speak
   for both compared dates: either side drawing fill puts them up.
8. **Cost.** The Compare mode of the Layers panel (the layer swipe and Dates) now loads on
   demand; the scene's part cannot. Index entry 193,054 B → 193,893 B gzip (budget 197,000 B);
   on its rebased base 194,131 B, and 194,479 B after the review fixes (Node's gzip, as
   `check-bundle.mjs` counts).

### Consequences

- Resident splats roughly double while two dates are compared; the splat budget counts both.
  A date hidden for switching back (the one shown before a comparison, a date compared once)
  is kept only while the splats resident across the scene fit what the budget keeps
  (`SPLAT_CACHE_FACTOR` times the ceiling); past that, checked each second, the hidden model
  drawn longest ago is disposed and loaded again if asked for. The frame's anchor stays while
  other dates copy its placement. A hidden tileset is never updated, so its own cache would
  never trim it.
- The load pill is ready once the last drawn date is in, whichever of the two streamed last.
- The right-hand date is the one the clip under the site and the far view speak for; the two
  dates share the site's footprint, so the clip is the same in practice. The load pill waits for
  both dates' first tiles.
- The Living Survey still reads the single date chosen before the comparison (`tilesetFor`).
- A Living Survey scan, which CesiumJS draws under any renderer, could be compared under
  PlayCanvas, but the control disables Dates for every splat scan off CesiumJS: one rule to say.
- Undo of the switch is scoped to the site (`scope: "site"`), like the site's other steps.

## Per-cell view rules

Date 2026-10-20 · design M4–M6 §1 C1–C5, W5b; branch `claude/feat-splat-compositor`.

### Context

A swipe shows two dates side by side. Often the question is another one: what is the newest
thing we know, everywhere? One capture rarely covers a whole site, so the answer is a patchwork:
this corner from June, that one from March where June never looked. Drawing two dates as two
primitives cannot make that patchwork: each primitive is sorted on its own and drawn after the
other, so wherever both reach one is painted over the other.

### Decision

1. **One primitive draws every date.** `SiteManager.setViewRule(rule | null)` makes the frame
   anchor's splat primitive (the date that made the site's frame, whose placement every other
   date copies) the host: the other dates' tilesets are its companions
   (`cesium/companions.ts`, the engine patch's `companions` array, now one list with several
   owners), in its one texture and one sort, and each date's own primitive draws nothing
   (`drawnBy`). Their inferred layers follow their scan onto the host. Only dates that share the
   anchor's placement (`followAnchor`) take part; one placed on its own is left out, and the
   key names it and why (still loading, or placed in its own frame), the As of date included.
2. **Each slot knows its source** (`cesium/splatSources.ts`): an R8 texture of our own, one
   byte a slot (0 the host, then up to 15 companions; 255 none), rewritten from `_tileSlots`
   whenever the patch replaces that map, only the rows touched uploaded. A reset of the slots
   (model matrix, texture growth) is every tile having moved. R8 read back as `round(r·255)`, not
   R8UI: this CesiumJS's `Texture` has no internal format for `RED_INTEGER` with unsigned bytes.
3. **A rule picks one owner per cell** (`lib/compositor.ts`, Cesium-free). The owner plane is one
   byte a cell of the site lattice (C1) at the coarsest of the drawn dates' finest levels, coarser
   while over 2^22 cells, so a cell never has two owners. A date covers a cell where its
   confidence grid scored it; a date without a grid, and an inferred layer, cover the cells under
   their leaf tiles' boxes, which the key flags as "tile bounds, overstates what it saw". Only
   dates observed up to **As of** (the capture-date control's choice, which reads "As of" while a
   rule is on) take part.
   - **Latest**: the newest observed date that covers the cell; else an older one; else a scan
     dated only by its upload (`observedAtSource: "created"`: when it was captured is unknown),
     which never outranks a dated date however new its upload, and which the key names
     "Uploaded <date>, capture time unknown"; else, where inferred layers are drawn at all, the
     newest inferred layer.
   - **Ground truth**: Latest, never an inferred layer.
   - **Highlight invented**: Latest, with the inferred owners tinted as the inferred highlight.
     The viewer's own Highlight (Layers › Inferred) still applies: the host's own layers keep
     their scan's highlight part, the other dates' layers are tinted by the rule's colour part,
     so no layer is tinted twice.
   - **Data age**: Latest, each cell coloured by its owner's age (W4a's colour part on a
     one-level grid of the plane), striped where a date after As of saw it. A cell an
     upload-dated scan owns has no age (an upload is not when the ground looked so).
4. **The shader keeps the owner's splats** (`cesium/splatCompositor.ts`): a part of the host's
   visibility chain weighs a splat 0 unless its source owns its cell. Near a seam -- a cell edge
   across which another date owns the cell; a cell nobody owns is no seam -- a splat is weighed
   by its own spread across it: `smoothstep(reach / 2, reach, d)`, `d` the distance to the edge
   in the splat's standard deviations across it (its covariance, read from the engine's
   attribute texture and turned into the plane's frame), `reach` = `SEAM_REACH` = 2. A slot of
   no source or a cell of no owner draws nothing: the rule fails closed rather than blending.
   **Measured** on the synthetic yard (`e2e/compositorScene.spec.ts`, Latest: March west, June
   east, each debug-coloured, red and blue; SwiftShader, 960 x 640): of the covered pixels within
   one cell (1 m) of the seam on screen, the share that blend both dates -- after the
   background's own red and blue are taken out, the lesser of red and blue is at least a tenth
   of their sum. With splats kept by their centres and the earlier 0.5 feather, 6.3% (11,491
   pixels; the same with a 0.1 m feather or none: the feather was never the cause, each splat's
   footprint reaching past its cell was). With the reach, **0 of 9,459** (under 1%, the target);
   a reach of 1.5 measured 0.16%. The price is a darker line where neither date's splats reach:
   the band's covered pixels fall from 11,519 (centres alone) to 9,459. One duplicated synthetic
   scene, where both dates are the same splats; to be measured again on a real two-date scan when
   one exists. The spec asserts under 1% at the seam, and over 2% with centres alone, so the
   measure is seen to bite.
5. **Off by default** (open question Q5): resident splats roughly double while a rule is on.
   The splat budget counts a date the host draws once.
6. **One or the other with Compare › Dates.** Turning either on ends the other, and the control
   says so (before, and after: "Per-cell dates ended: Compare › Dates was turned on").
7. **CesiumJS only (C5).** Under PlayCanvas or Spark the control is disabled with the views'
   notice and one-click switch; choosing another renderer ends a rule, and says so.
8. **What it turns off while on, and says.** A date's own Confidence and Data age views (they read
   one date's grid) are held at Photo. The host's view cones are off: the anchor's cones would
   fade the other dates' splats where the anchor never looked. A Living Survey rig takes its scan
   out of incremental mode, which drawing companions needs: the control is disabled while a rig
   moves the site's scan, and a rig that attaches during a rule ends it, saying why. A fill's
   swap (hiding the measured splats its `supersedes` file names) is off: the rule draws each
   cell from one date, measured or inferred, and a fill it does not draw there (never under
   Ground truth) must not leave a hole.
9. **Split objects** are tilesets of their own beside their scan, outside the host's sort: the
   rule says per scan which are drawn (`companions.ts` `ruledObjects`), each whole, by the date
   that owns the cell under its centre, so one date's copy of an object is drawn, never two. They
   keep their own colours under Data age.
10. **Cost.** The rule's parts load on demand (`compositorSession.ts`, 7,314 B gzip; 7,967 B
    after the review fixes); the globe's entry grows by what the scene keeps of it. Index entry
    193,893 B → 194,589 B gzip (budget 197,000 B); on its rebased base 195,199 B, and 195,217 B
    after the review fixes (Node's gzip, as `check-bundle.mjs` counts).

### Consequences

- Per-scan hooks on a date's own primitive (hidden or highlighted objects, skins, telemetry) do
  not act on it while it is the host's companion: its own primitive draws nothing. The host's own
  hooks act on the host's slots as before.
- A site under a rule keeps every date loaded, hidden where it is not composed (each is loaded to
  learn whether it shares the anchor's placement): none is let go under the splat budget then.
- A companion is drawn with the host's spherical-harmonics degree: one with more is drawn in its
  base colour, one with fewer has its range zeroed (the patch's `zeroShRange`).
- A pick (Measure, selection) is answered by the collider from every drawn date's solids, not by
  the rule's owner of the cell hit.
- Sixteen sources at most on one primitive: past them a layer is not drawn, and the key says so.

### Not built: GPU seam stitch

A seam between two owners is kept clean, never fixed: the two dates' splats near it were
optimised against different photographs. Stitching would re-optimise a band of splats either
side of each seam against the newest images, on a GPU. It is not built here (no GPU runs from
this work), and when it is, its output is a **derived layer of its own** -- a tileset beside the
dates, declared as derived, with its inputs named -- never written into either epoch: an epoch is
what one capture measured, and a stitched band is not.
