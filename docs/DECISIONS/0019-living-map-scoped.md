# 0019: Living map, scoped

Status: proposed

Milestone 6 of the splat pipeline (design M4–M6 §5) makes the map "living" only in two ways it
can be honest about: a switch that shows the data alone (As captured, below) and, later, a
relight by the sun (W6b, its own section). Everything else that moves or fills in on the map
already exists and is labelled; this ADR is about keeping it from being mistaken for data.

## As captured (web)

Date 2026-10-20 · design M4–M6 §1 C5, W6a; branch `claude/feat-splat-as-captured`.

### Context

Over a measured scan the app can draw things nobody measured: inferred layers (an image model's
fill of what the capture missed, ADR 0009's view cones fading them), a fill method that hides the
measured splats it replaces (`supersedes`), the hole a split object left, filled
(`fills/<id>/`), modelled motion (the Living Survey's wind, the wind on skins, the poke tool's
ring), objects moved by telemetry or by a person, a dress rehearsal's machines, a simulated
fleet. Each is labelled where it is chosen, and the Simulated motion badge says so while
anything moves. What was missing is one way to see the capture alone, and a link that opens on
it.

### Decision

1. **One switch, not persisted** (`state/asCaptured.ts`): `useAsCaptured {on, set}`. A reload
   forgets it, so nobody is left looking at a scan with its fills hidden without having asked.
   `?asCaptured=1` turns it on for the visit; nothing else in the address does. The switch is in
   the Splat view control (`ViewsControl`), beside Photo · Confidence · Data age; while it is on,
   a chip in the HUD strip says **"As captured — only what was measured is drawn; plans stay as
   annotations"**, on any site, with **Turn off** (a link may land someone on it who never saw
   the switch). It claims exactly what is drawn: an earlier "nothing simulated or inferred is
   drawn" was not true while the plan's zones stayed on the map.
2. **It never writes a viewer's choice.** Each effect reads the switch where it decides; the
   Inferred style, the wind, the fill picked, a view rule, a pose set by a person stay as they
   were, so turning it off draws exactly what was drawn before. `effectiveInferredStyle()` is the
   viewer's style, or Hide while on; `onInferredStyle` tells its readers when that changes.
3. **What it hides, each in one place:**
   - **Inferred layers** under CesiumJS (`cesium/inferredLayers.ts`: off the scan's companion
     list and their own primitives hidden) and under PlayCanvas or Spark
     (`cesium/scanView/scanLayers.ts`: their tiles off screen, nothing streamed or fetched while
     on). Split objects' hole fills are inferred layers (`split_objects.py` declares them in
     `extras.inferredLayers`), so they go the same way.
   - **Supersedes swaps** end (`state/supersedes.ts`): the measured splats a fill replaced are
     drawn again.
   - **Per-cell view rules** are drawn as Ground truth (`cesium/compositorSession.ts`): no
     inferred owner, no tint, no Data age colouring; the rule chosen comes back after. The
     control says "Drawn as Ground truth while As captured is on".
   - **Motion**: the scene's wind is calm (`SceneBridge.tsx`: `reducedMotion || asCaptured ?
WIND_CALM : wind`, the reduced-motion rule extended), which stills the Living Survey's
     deformers and the skins' wind; the poke tool is turned off (and straight off again if asked
     meanwhile), and a ring in progress stops at once rather than decaying (`cesium/skinPoke.ts`);
     telemetry writes every bound object at rest, its measured frame, while the tracks keep
     reading, so turning it off shows the pose of now (`cesium/telemetry.ts`); split objects are
     drawn at rest, where they were measured, under every renderer (`state/sceneObjects.ts`
     `objectPose`).
   - **Agents, and what they did**: a dress rehearsal draws only the plan's lanes, all pending --
     no ghost, no lane done, no hand raised, nor the hand open in the card (`RehearsalManager`
     `hideAgents`; the run's own clock, hands and card are untouched and are drawn as they stand
     once it is off). A simulated project's fleet draws no machine marker, no track, and none of
     its zones' progress: each zone chip keeps its name and drops its status ("mowing 58%"), a
     zone the fleet "treated" is drawn as the plan's (dashed, faint), and the site's pin counts
     no machine (`MissionOverlays`; `MissionManager.hideAgents`, told by the rehearsal's). A
     project whose fleet is real keeps all of it: what its machines reported is data.
4. **The badge is the test.** The Simulated motion badge is not told about the switch. It goes
   because `status.animating` becomes false once the wind is calm: if it stayed, something still
   moved, and the e2e spec would fail.
5. **What it does not change.** W4a's analytic views (Confidence, Data age of one date) are
   readings of the capture, not additions to it: they stay. Plans, zones' outlines and names,
   the plan preview's lanes and the site's own footprint are annotations, not depictions of the
   place: they stay drawn, which is what the chip says. HUD text about a simulated fleet (the
   status line, the agent's line, the plan card) is not drawn on the map and stays, labelled
   simulated where it is. The Inferred style and Compare methods' Fill row
   are disabled while on, each saying "Hidden while As captured is on".
6. **Renderer-independent** (C5): every effect acts under CesiumJS, PlayCanvas and Spark; the
   motion drivers are shared (they write the parts every renderer reads, `scanMotion.ts`).
7. **The sunlight relight subscribes** (W6b): it is to read `useAsCaptured` and draw nothing
   while it is on.
8. **Cost.** The switch and the chip live in the views control's chunk, loaded on demand (the
   chip from the same slot, so a site without a splat scan loads it only while As captured is
   on). The globe's entry grows by the reads in the scene and the slot: index entry 194,589 B →
   195,112 B gzip (+523 B; budget 197,000 B). Of that, hiding what simulated machines did is
   about 240 B: the rehearsal's ghosts and the fleet's markers and tracks (about 100 B), then,
   after review, the rehearsal's progress and the fleet's zone progress (143 B). The store is a
   chunk of its own the page preloads (`asCaptured-*.js`, 453 B gzip), as the settings and
   view-rule stores are.

### Consequences

- A fill picked while it is on is kept for after: Compare methods' Fill row waits, disabled.
- An object a person drags while it is on is drawn at rest; the pose they set is drawn once it
  is off.
- Turning it on during a rehearsal leaves the clock running and the card counting: the map draws
  the plan's lanes only, until it is off.
- Not covered: a whole asset of `sourceKind: "inferred"`, which the app never draws as a site's
  model already (`lib/epochs.ts`), and the Time Lantern's dated imagery, which is observed.

Evidence: `e2e/asCaptured.spec.ts` (the yard with Today's inferred layer, fill methods, a
supersedes swap and the forest rig in the wind, under CesiumJS and PlayCanvas; `?asCaptured=1`;
the demo ranch's simulated fleet and its zones' progress), `e2e/poke.spec.ts` (a ring stops), and the unit tests beside
each effect.
