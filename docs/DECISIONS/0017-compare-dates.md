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
   names the date whose tileset was hit (`SiteManager.drawnAssetOf`). Queries with no point on
   screen (camera collision, the floor) count both dates.
7. **Said where a date is named.** The capture-date control checks no date and marks each "left
   of divider" / "right of divider"; the Inspector's Date shown becomes "Left of the divider" and
   "Right of the divider", each with its own facts.
8. **Cost.** The Compare mode of the Layers panel (the layer swipe and Dates) now loads on
   demand; the scene's part cannot. Index entry 193,054 B → 193,893 B gzip (budget 197,000 B).

### Consequences

- Resident splats roughly double while two dates are compared; the splat budget counts both.
- The right-hand date is the one the clip under the site and the far view speak for; the two
  dates share the site's footprint, so the clip is the same in practice. The load pill waits for
  both dates' first tiles.
- The Living Survey still reads the single date chosen before the comparison (`tilesetFor`).
- A Living Survey scan, which CesiumJS draws under any renderer, could be compared under
  PlayCanvas, but the control disables Dates for every splat scan off CesiumJS: one rule to say.
- Undo of the switch is scoped to the site (`scope: "site"`), like the site's other steps.

## Per-cell view rules

Filled by `claude/feat-splat-compositor`.
