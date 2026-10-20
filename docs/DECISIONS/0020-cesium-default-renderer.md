# ADR 0020 — CesiumJS draws splat scans by default

**Status:** proposed (Harrison merges or drops it) · **Date:** 2026-10-20 · **Decision record:**
D5 of the splat-pipeline build, decision (a) of 2026-10-19

## Context

Since settings version 3 (commit 124d8adf, 2026-10-01) PlayCanvas has drawn splat scans unless a
person chose otherwise: it streams a scan's own level of detail the way superspl.at does, and it was
the renderer that felt right in use. CesiumJS kept the globe, navigation, collision and tools, and
every Living Survey scan (its motion is CesiumJS's splat shader).

On 2026-10-19 the splat pipeline settled where its view features are built: on CesiumJS's own splat
primitive, as patched in `patches/@cesium__engine@26.3.0.patch` (decision (a)), so that each feature
is written once and not three times. The epoch slider works under every renderer. The features after
it do not: the confidence and data-age tints, the per-cell compositor, compare-two-dates (two
primitives and the stock `splitDirection`) and capture lighting are CesiumJS-only. Under another
renderer they are disabled with "Needs the CesiumJS renderer" and a one-click switch, never a silent
no-op. With PlayCanvas the default, a person who never opens Settings would meet every one of them
disabled.

## Decision

1. **`DEFAULT_SPLAT_RENDERER` is `"cesium"`** (`apps/web/src/state/settings.ts`).
2. **Settings version 4 moves a stored `"playcanvas"` to `"cesium"` once.** Stored `"spark"` and
   `"playcanvas-webgpu"` stay: neither was ever a default, so each is a person's choice.
3. **A stored PlayCanvas choice cannot be told from the old default, so it moves too.** The settings
   store writes every field, so the v3 default, the v3 migration (which overwrote whatever was
   stored before it, Spark included) and a click on PlayCanvas in Settings all store the same
   `"playcanvas"`. Someone who really chose it chooses it once more. After v4 a PlayCanvas choice is
   kept.
4. **PlayCanvas, its WebGPU trial and Spark stay selectable**, in Settings › Advanced and for one
   visit by `?renderer=`, which still wins over the saved setting. Nothing about how they draw
   changes.

## Consequences

- A fresh device, and every device that was on PlayCanvas, draws scans with CesiumJS: the renderer
  the CesiumJS-only features need, and the one a Living Survey scan always used.
- What PlayCanvas did better is now opt-in: its streamed level of detail for the largest scans, and
  the overlay rules of CESIUM.md, _Dedicated splat renderers_. A large scan is drawn by CesiumJS's
  tileset level of detail instead. The local measurement under software GL (SwiftShader, not a real
  GPU) is in the PR body; it says nothing reliable about a phone, so the large-scan feel is to be
  judged on a real device before this is merged.
- The e2e fixtures seed settings version 1, so every full-app spec now runs CesiumJS; specs about
  PlayCanvas name it (`?renderer=` or the harness's `renderer`) rather than relying on the default.
- The Settings switcher keeps its order (PlayCanvas first); only the default moved.
- Dropping this ADR's commit restores the PlayCanvas default with nothing else to undo: no other
  change depends on it, and no later number is taken by it.
