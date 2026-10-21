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
7. **CesiumJS only (C5)** when first built. PlayCanvas draws the rules too since
   (§ "Per-cell view rules under PlayCanvas" below); under Spark or the WebGPU trial the control
   is disabled with the views' notice and one-click switch, and choosing one ends a rule, and
   says so.
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
   that owns the cell under its anchor, so one date's copy of an object is drawn, never two. They
   keep their own colours under Data age. The anchor is the origin the package declares
   (`extras.objects[].origin`, the centre of its splats' bounds on the ground,
   `split_objects.py`) where its pose puts it -- its own frame's translation
   (`lib/compositor.ts` `objectAnchor`), the same under PlayCanvas. (It was the middle of the
   object tileset's bounding volume, which the package pads by three scales a splat; near a seam
   that could name another date than PlayCanvas did.)
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

## Per-cell view rules under PlayCanvas

Date 2026-10-20 · PlayCanvas port 3; branch `claude/feat-splat-pc-view-rules`.

### Context

PlayCanvas is the app's default renderer, and the rules above were CesiumJS's alone: most
visitors met Per cell disabled, with a switch to another renderer. A rule needs one sort for
every date (two sorts paint one date over the other wherever both reach). PlayCanvas sorts
every entity of its app together, and its copy pass (`modifySplatColor` of the work buffer's
modifier) already draws Confidence, Data age and the objects' rules per tile.

### Decision

1. **One app draws every date.** The scan PlayCanvas draws stays the date chosen
   (SiteManager `scanTarget`, the date the rule is As of), so turning a rule on restarts nothing.
   The rule's session hands that renderer the owner plane (`cesium/scanView/scanRule.ts`) and the
   other dates, which it streams into the same app beside its scan
   (`cesium/scanView/ruleDates.ts`, loaded with the first rule that names one): each date's tiles
   placed where the date is in the scan's frame (`P = C_scan⁻¹ · C_date`, the two roots' computed
   transforms), its inferred layers and split objects with it. A date's tile is drawn as a scan
   tile (the cell views, the sun) but bound to none of the scan's objects, motion or picks, as a
   CesiumJS companion is not.
2. **The plane, in the scan's frame.** `lib/compositor.ts` is unchanged: the session works the
   plane out once and draws it one of two ways. Under PlayCanvas the plane is in the frame of
   the scan PlayCanvas draws, and every tileset is named by the URL the renderer streams it from:
   each date, each one's inferred layers, each one's split objects. A renderer switch between
   CesiumJS and PlayCanvas carries the rule over.
3. **Each tile is weighed in the copy pass.** The weight depends on no eye (the splat's cell,
   its source, its own spread), so it is applied once per copy of a tile, on the opacity, after
   the motion has moved the splat (`RULE_WEIGHT_GLSL`, `compositorWeight` in GLSL). A tile's
   source is one uniform, from the tileset it came from: no per-splat stream and no entity ids
   (`enableIds`) are needed. A tile of no named source draws nothing: the rule fails closed. A
   split object is weighed whole by the cell under its centre (moved with its pose), and keeps
   its own colours under Data age. Highlight invented sets the layers' own Highlight look (each
   layer already carries the viewer's Highlight, so none is tinted twice); Data age is the
   plane's owner-age grid as a cell view, drawn in place of the scan's own while the rule is on.
   **A colour-only copy.** PlayCanvas also copies an SH splat's colour alone as the camera moves,
   and for a natively streamed SOG that copy reads the geometry back from the work buffer
   (`GSPLAT_WORKBUFFER_GEOMETRY`) and never calls `modifySplatRotationScale`; there the weight
   reads the rotation and scales back from the work buffer too (`workBufferWorldRotation`,
   `workBufferWorldScale`, as the full copy wrote them). Without it each refresh weighed every
   splat as a metre-wide one: measured on the SH tree's streamed package
   (`data/tiles/synthetic-tree-sh-sog`) under a seam down the tree, 19,316 covered pixels before a
   camera move away and back, 8,066 after; with it, 19,316 (`e2e/compositorScene.spec.ts`).
   **Fail closed, in pixels.** Where nobody owns a cell (Ground truth's gap), a resident tile the
   rule names no source for draws nothing: 15 of the fill's tiles resident and unnamed, 0
   pixels; with the unnamed case mutated to draw, the gap is all the fill's and the spec fails.
4. **The seam in PlayCanvas's footprints.** Both renderers fall off as `exp(-4 u²)` across a quad
   sized from the projected spread, but PlayCanvas's quad reaches `√(2λ)` pixels from the centre
   and CesiumJS's half that: PlayCanvas draws every splat twice as wide for the same covariance.
   The rule above (no splat's drawn footprint carries its date over a seam) is then twice the
   sigmas: the reach handed over is CesiumJS's 2, scaled by `PLAYCANVAS_FOOTPRINT` = 2.
   **Measured** by the method above (`e2e/compositorScene.spec.ts`, the same scene and counts,
   SwiftShader, 960 x 640), seam band, pixels blending both dates:

   | reach (σ) | 0 (centres) | 2      | 2.5    | 3      | **4**          |
   | --------- | ----------- | ------ | ------ | ------ | -------------- |
   | mixed     | 5.59%       | 1.29%  | 0.54%  | 0.01%  | **0 of 9,276** |
   | covered   | 11,416      | 10,847 | 10,544 | 10,054 | 9,276          |

   The band keeps 81% of its covered pixels at 4 σ (CesiumJS at 2 σ: 82%). The other figures:
   both dates unruled 93.5% mixed; Latest left 99.7% red, right 100% blue; the gap south-west
   100% green under Latest, nothing under Ground truth. The spec asserts the same figures under
   both renderers.

5. **One budget for every date.** Every date a rule draws is resident at once in PlayCanvas's one
   work buffer, so they share the one budget the host draws to: split by each one's share of the
   cells in view it owns, its inferred layers' cells counted with it
   (`cesium/scanView/ruleDates.ts` `viewShares`, the plane sampled at 16k cells at most).
   Together they hold what one scan alone would. A date that owns no cell in view streams
   nothing and lets its tiles go until it does. The scan PlayCanvas draws takes its share through
   the host (PlayCanvas's `loadFactor`; the host hands its streamer what the cut may hold
   whenever any input of it moves, checked each frame), or for a natively streamed scan through
   PlayCanvas's own budget. A budget or share that moves with the camera still is planned for at
   once. (Each date first took the whole budget: N dates held N + 1 budgets, up to about 18M
   splats against the 16.8M one texture holds.) **Four dates at most**: a rule names the dates up
   to As of, four at most (`lib/viewModes.ts` `MAX_RULE_DATES`), under either renderer; past
   four a date's share is too thin to draw it well. The control is offered disabled with the
   reason and the way out (an earlier As of), the scene refuses such a rule, and a running rule
   whose As of moves past four ends, saying why. Still off by default.
6. **Not Spark, not the WebGPU trial.** Spark's world modifier has no port of the weight, and
   the trial's modifiers are GLSL only: there the control stays disabled, with the notice naming
   Per cell and its one-click switch.
7. **Cost.** The rule's PlayCanvas parts load on demand: the channel (`scanRule.ts`, 7.2 kB raw)
   with the rule's session or the renderer, the dates (`ruleDates.ts`, 3.8 kB raw) with the
   first rule that names one. They borrow the globe's own streaming classes (the tile streamer,
   a scan's layers and objects, the parser, the layers' pick) **lent at runtime, not imported**
   (`cesium/scanView/ruleKit.ts`): SiteManager hands the rule's session a tuple of them as a rule
   starts, and the session lends it to the renderer. Imported, those modules were shared by the
   entry and an on-demand chunk, and the bundler cut them out of the entry into three chunks of
   their own that the page still loads up front: the entry fell 196,148 → 190,489 B gzip but the
   page's JavaScript before its first frame grew 1,602,079 → 1,603,880 B gzip (+976 B raw, two
   more files), which is the measure that matters. Lent, no module moves (83 files before and
   after): before its first frame the page loads 1,602,079 → 1,602,391 B gzip (+71 B raw), its
   entry 196,148 → 196,349 B gzip (budget 197,000; Node's gzip, as `check-bundle.mjs` counts).
   The raw growth is the session's context and the tuple. Gzip grows more than raw because any
   new reference to those bindings has the minifier re-letter the export aliases of the shared
   chunks (`ci as f` becomes `li as f`), which compress a little worse: the tuple in another
   module, without the helper in a shared chunk, or of fewer parts measured +280 to +330 B gzip
   each; without it, +37. After the cross review's changes (the colour-only copy, one budget,
   the cap, the anchor), measured against the branch before them (369bde60): before its first
   frame the page loads 5,623,570 → 5,623,554 B raw, 1,602,398 → 1,602,371 B gzip, 83 files;
   entry 196,356 → 196,329 B gzip. The cap lives in a module the control and the session already
   share, and the dates' module imports nothing the channel does not, so neither adds a chunk
   to the preload map the entry carries.

### Consequences

- The colour-only copy is reached only by a natively streamed scan with SH bands. The pipeline
  ships SH degree 0 by default (`ship_sh_degree`, tools/pipeline/harmonics.py); the streamed
  package is an opt-in backfill (`.github/workflows/streamed-lod-backfill.yml`, never run by
  default) that keeps whatever bands the tiles carry; and the app streams it only for a scan
  that has one and declares no objects (`ScanRendererHost` `runNative`). So no scan published
  with the defaults meets it.
- Compare › Dates stays CesiumJS's: the stock split is CesiumJS's, and a rule and a comparison
  remain one or the other. As captured draws a rule as Ground truth, and At capture is off while
  the rule draws more than one date, under either renderer.
- The other dates' split objects are placed at their poses as frames are drawn; a pose set on
  one while nothing else changes waits for the next frame.
- A pick is answered from the dates' solids (the collider) as under CesiumJS; scene selection
  picks the scan's own tiles, never another date's.
- Sixteen sources at most, as under CesiumJS: past them a tileset is named by no source and not
  drawn, and the key says so.

### Not built: GPU seam stitch

A seam between two owners is kept clean, never fixed: the two dates' splats near it were
optimised against different photographs. Stitching would re-optimise a band of splats either
side of each seam against the newest images, on a GPU. It is not built here (no GPU runs from
this work), and when it is, its output is a **derived layer of its own** -- a tileset beside the
dates, declared as derived, with its inputs named -- never written into either epoch: an epoch is
what one capture measured, and a stitched band is not.
