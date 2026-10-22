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

1. **One switch, kept only in the page address** (`state/asCaptured.ts`):
   `useAsCaptured {on, set}`. `?asCaptured=1` turns it on; nothing else in the address does.
   The switch keeps the address in step (`history.replaceState`, so no history entry), so a
   reload or a copied link shows what is on screen. Nothing is stored on the device: a later
   visit without the address has it off, so nobody is left looking at a scan with its fills
   hidden without having asked. The switch is in the Splat view control (`ViewsControl`),
   beside Photo · Confidence · Data age; while it is on,
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
     `objectPose`). The poke tool is the viewer's choice, so it is given back once it is off:
     on if it was on, or was asked for meanwhile.
   - **Agents, and what they did**: a dress rehearsal draws only the plan's lanes, all pending --
     no ghost, no lane done, no hand raised, nor the hand open in the card (`RehearsalManager`
     `hideAgents`; the run's own clock, hands and card are untouched and are drawn as they stand
     once it is off). A simulated project's fleet draws no machine marker, no track, and none of
     its zones' progress: each zone chip keeps its name and drops its status ("mowing 58%"), a
     zone the fleet "treated" is drawn as the plan's (dashed, faint), and the site's pin counts
     no machine (`MissionOverlays`; `MissionManager.hideAgents`, told by the rehearsal's). A
     project whose fleet is real keeps all of it: what its machines reported is data. A hand
     marker counts the hands up by the time it is drawn again, not those up when it was hidden.
   - **The Land surface's inferred and agent-made overlays**: Show once's heatmap, a similarity
     inferred from satellite signatures (its card says "Inferred"), is lifted with its outline
     and taps, as a hidden card lifts it; the session and its answer wait, and the card says why
     (`ShowOnceBridge`, `ShowOnceCard`). The Land agent's work is not drawn: its findings' pins,
     the outlines of what it found and where it is looking (layers marked `agent`), a research
     map it produced (a layer with a `researchArtifactId`; its card says why), and the outline
     and leader line of a card in the agent's tone (`agentMade` in `state/landContext.ts`;
     `LandContextBridge`, `LandOverlays`, `LandFocusBridge`). Each bridge filters as it draws and
     the stores keep everything, so turning it off draws the same again.
   - **Procedural fixtures**: the synthetic tree ("procedural; no capture, no reconstruction",
     `lib/procedural.ts`) is generated, not captured. Its model is withheld
     (`SiteManager.hideFixtures`): tileset hidden, world unclipped, no renderer handed it.
4. **The badge is the test.** The Simulated motion badge is not told about the switch. It goes
   because `status.animating` becomes false once the wind is calm: if it stayed, something still
   moved, and the e2e spec would fail.
5. **What it does not change.** W4a's analytic views (Confidence, Data age of one date) are
   readings of the capture, not additions to it: they stay, as does the Fog of Knowing (where
   the Land's evidence is, and how old: a reading of the record). Plans, zones' outlines and
   names, the plan preview's lanes and the site's own footprint are annotations, not depictions
   of the place: they stay drawn, which is what the chip says. On the Land surface the same
   holds: plans (the Plans lens' work areas and steps, a planned solar array and its modelled
   size, a boundary split's pieces) are annotations and stay; what records and public datasets
   say stays (parcels, old map sheets, soil units and samples, flood zones, the Overview's
   terrain, land cover and greenness rasters and spot readings, mapped assets, survey plots, a
   mapped parcel found on descent). Measurements and drawings a person makes are theirs. HUD text about a simulated fleet (the
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
   after review, the rehearsal's progress and the fleet's zone progress (143 B). The second
   review's fixes (Land overlays, the fixture, the poke tool given back, the address kept in
   step, the hand count) add 71 B to the entry (195,359 B → 195,430 B), the scene's fixture
   read; the Land bridges and cards are lazy chunks. The store is a chunk of its own the page
   preloads (`asCaptured-*.js`, 453 B gzip, 572 B with the poke tool given back and the address
   kept), as the settings and view-rule stores are.

### Consequences

- A fill picked while it is on is kept for after: Compare methods' Fill row waits, disabled.
- An object a person drags while it is on is drawn at rest; the pose they set is drawn once it
  is off.
- Turning it on during a rehearsal leaves the clock running and the card counting: the map draws
  the plan's lanes only, until it is off.
- A Show once session waits while it is on: taps on the map do nothing, and the card says why.
- The agent's findings and research maps are still listed in the Land panel (text, labelled as
  the agent's), as the simulated fleet's HUD text stays; only the map leaves them off.
- Not covered: a whole asset of `sourceKind: "inferred"`, which the app never draws as a site's
  model already (`lib/epochs.ts`), and the Time Lantern's dated imagery, which is observed.

Evidence: `e2e/asCaptured.spec.ts` (the yard with Today's inferred layer, fill methods, a
supersedes swap and the forest rig in the wind, under CesiumJS and PlayCanvas; `?asCaptured=1`
and the address following the switch; the demo ranch's simulated fleet and its zones'
progress), `e2e/poke.spec.ts` (a ring stops), and the unit tests beside each effect
(`showOnceBridge`, `showOnceCard`, `landAgentAsCaptured`, `researchMap`,
`siteFixtureAsCaptured`, `rehearsalManager`, `asCaptured`, `asCapturedUrl`).

## Sunlight (web)

Date 2026-10-20 · design M4–M6 §1 C5, W6b; branch `claude/feat-splat-sunlight`.

### Context

A splat scan is lit once, by the sun and sky of the moment it was captured: its colours are the
photographs'. Seeing it under another sun (now, or the capture's own, beside the sky the globe
draws for that time) needs a light model the capture never measured, so it is a depiction, not
data, and it must say so. A Gaussian splat also has no normal; only a flat one has one in all but
name, the axis of its smallest variance.

### Decision

1. **A colour part, no engine-patch edit** (`cesium/splatSunlight.ts`, `splatSunlightColor`,
   `colorOrder` 3: after the objects' highlight, before Confidence / Data age (5), which tint by
   the relit luminance, the inferred layers' Highlight (10) and the compositor's (11)). It
   refetches the splat's covariance texel (the odd column of `u_splatAttributeTexture`, at the
   engine's own `u_splatRowMask`/`u_splatRowShift`, as the vertex shader does) and, when the motion
   chain declares a Jacobian, draws the moved covariance as the engine does.
2. **A normal only for a clearly flat splat.** The closed-form eigenvalues of the symmetric 3×3
   (Smith 1961); the smallest axis is the normal only when `λmin/λmid < 0.25`, from the largest
   cross product of two rows of `A − λmin·I`. Round splats, needles and empty ones have none.
   `lib/splatNormals.ts` restates the arithmetic in TypeScript, and the GLSL transcribes it; its
   tests are the test vectors (rotated discs, the 0.25 boundary, a 2 cm splat after the engine's
   half-float storage, the lambert sign).
3. **The engine's sun.** The normal is turned toward the eye, into eye coordinates by
   `mat3(czm_modelView)`, and lit by `czm_sunDirectionEC`, which follows `clock.currentTime`, so the
   relight's sun is the sky's. The engine declares an automatic uniform only for the `czm_` names
   a shader's text uses; this text uses it, and it compiles and lights under SwiftShader
   (`e2e/sunlightScene.spec.ts`). `c.rgb *= mix(1, 0.35 + 0.65 · lambert · sunUp, strength)`,
   strength 1. `sunUp` is a uniform worked out once a frame on the CPU at the scan's centre by the
   engine's own sun arithmetic (`sunSine`), faded in over the first 5° of elevation (a rendering
   choice, not a measurement), so the shader and the badge agree on night. Before the engine's
   Earth-orientation data loads there is no ICRF matrix; the engine turns by TEME then, and so
   does `sunSine`. A sun that is still not a number (a scan not placed, a matrix of NaN) is
   never drawn or claimed: the uniform's strength goes to 0, the colours are the photograph's,
   and the frame withdraws its report.
4. **Deviation: a splat without a normal counts as facing the sun** (lambert 1), not as untouched:
   by day it is unchanged, as designed; at night it goes to the same ambient floor as everything
   else, instead of the round splats (foliage) staying daylit beside a darkened ground.
5. **Sun: Off · Now · At capture** (`state/lighting.ts`, not persisted), in the Splat view control
   (`features/views/SunControl.tsx`, in its lazy chunk; no tool-rail button). At capture is
   offered only for a date whose capture time was recorded (`observedAtSource` not `created`);
   a seeded model without a date uses its own `observedAt`. It is labelled **Simulated
   lighting**, as the repo's rule asks of anything simulated: the control, its hint and the
   badge say so.
6. **At capture holds the scene clock** at the shown date's capture time, `shouldAnimate` off,
   and gives back the clock it found, exactly (`cesium/sunClock.ts`): the time it would read had
   it never been held (the held span at its own multiplier when it was running, nothing when it
   was still), its animate flag and its step; a time someone else set meanwhile stands. The hold
   lives with the badge, so whatever ends the relight ends it, and it needs the scene's report
   that the shown model is drawn relit, so a moved sky is never unlabelled. **Not under
   Compare or a per-cell view rule**: more than one date is drawn, each with its own capture
   time, and the scene has one clock, so holding either would relight the other at a moment
   that is not its own. At capture is disabled while more than one date is drawn, with the note
   "At capture is off while more than one date is drawn: one scene clock holds one capture's
   sun, so the sun of now is drawn"; the choice is kept and holds again with one date.
7. **The badge is the label** (`features/views/RelitBadge.tsx`, in the HUD's messages):
   "Simulated lighting: the sun (now) over shadows baked in at capture, {capture date and time}
   UTC" (or "(at capture)"; "each date's own" while more than one date is drawn), and "Sun
   below horizon" when the frame drawn had the sun down. It shows only what the scene reports
   drawing (`state/sunReport.ts`), never the choice: a model reports only while its tileset is
   shown, so a date kept loaded but hidden, or a site left behind, claims nothing and holds no
   clock. With one date it reads the shown model's report; with more, any of the site's.
8. **Off while As captured is on** (the slot reads `useAsCaptured`, as § As captured asked): the
   part leaves the chain, the badge goes, the clock is given back; the control shows Off, disabled,
   saying "Off while As captured is on", and the choice comes back after.
9. **CesiumJS and PlayCanvas** (C5, then port 2 below): under Spark or the WebGPU trial the
   control is disabled, saying "Needs the CesiumJS or PlayCanvas renderer (Settings › Advanced ›
   Splat renderer)", with a one-click "Draw with PlayCanvas"; the views control's shared notice
   no longer names Sunlight. The relight requires the chosen splat renderer to draw it
   (`lib/viewModes.ts` `sunlightDrawable`), even for a scan CesiumJS draws under another one (a
   Living Survey rig's), which nothing there would label.
10. **Cost.** Off: nothing in the shader; `SiteManager.attachTileset` holds a slot
    (`cesium/sunlightSlot.ts`) that loads the relight on the first switch. On: one more fetch of a
    texel the engine has just read, a closed-form eigen solution (a determinant, an `acos`, two
    `cos`, three cross products) and a few dozen flops a splat; with a motion Jacobian, that
    Jacobian once more. The globe's entry: index 194,969 → 195,331 B gzip (+362 B; budget
    197,000 B), the badge's slot and the store; the relight, its part, the badge and the words are
    chunks loaded on demand. After the second review, with § As captured's fixes beneath it, the
    stack's entry is 195,742 B → 195,811 B (+69 B): the Simulated lighting label in the slot's
    error boundary; Compare, the shown-tileset guard and the NaN guard are in lazy chunks.

### Under PlayCanvas (port 2)

Date 2026-10-20 · branch `claude/feat-splat-pc-sunlight`, on PlayCanvas port 1 (#73); proven by
the spike `claude/research-pc-sunlight-spike`.

11. **PlayCanvas's per-frame hook, not its copy pass** (`cesium/scanView/sunHook.ts`). The
    relight depends on the eye (a flat splat is lit on the side it is seen from) and on the sun,
    which moves with the clock; in the copy pass that draws Confidence and Data age either would
    copy every tile again. PlayCanvas 2.22.6's user material `app.scene.gsplat.material` takes a
    `gsplatModifyVS` chunk that it copies onto the unified renderer's draw material whenever the
    material's version moves; that vertex shader hands the chunk the work buffer's rotation and
    scales, then `modifySplatColor(centre, colour)`, in the scan's frame (the manager's identity
    node), with the eye in `view_position`. The chunk transcribes CesiumJS's arithmetic: the
    smallest scale's axis as the normal when the splat is clearly flat
    (`lib/splatNormals.ts` `scaleNormal`: the eigenvalues of `R·diag(s²)·Rᵀ` are the `s²`, so
    it is the covariance's own `λmin/λmid < 0.25`, held to `splatNormal` on 20 000 random
    splats), flipped toward the eye, lambert against the sun, the same `mix`. In the user
    material while there is a sun and out when there is none.
12. **One sun, worked out once.** `cesium/sunlight.ts` still works the sun out on the globe's
    `preRender` at the frame's scene time by the engine's arithmetic (`sunSine`, and
    `sunInFrame` for its direction in the scan's frame from the root's inverse rotation); under
    PlayCanvas the tileset is hidden, so instead of the colour part's uniform it hands
    `{dir, sunUp, strength}` and the two constants to the renderer drawing that asset
    (`state/sunReport.ts` `publishScanSun`), which sets them as uniforms: a uniform copies
    nothing into the work buffer. Handed on only when it moved by 1e-4 (the overlay draws a
    frame for each). The renderer's watching is what says the relight is on screen, so the
    badge's report, and with it At capture's held clock, follows a renderer drawing the scan;
    NaN sun: strength 0, no report, as under CesiumJS. As captured, the badge, At capture and
    its "more than one date" rule are the renderer-independent code above, unchanged.
13. **One word per splat** (`hexapodSun`, an R32U stream of PlayCanvas's work buffer, added the
    first time the sun is on in a session and kept: nothing while it has never been on). Every
    tile's copy writes it once it is there (a word nobody writes would be undefined), so a tile
    with no objects and no view gets a modifier then too, with an empty colour step. It carries:
    - **Whether a copy has written it** (bit 31, in every word a copy writes). The stream starts
      at zero, and so does a work buffer PlayCanvas makes again: until a tile's copy lands
      (about a frame) its splats are drawn as they are, neither lit with no look nor, a
      layer's, lit.
    - **Inferred layers skipped.** A layer's tiles are drawn and sorted with the scan's but are
      not the capture, and under CesiumJS (a tileset of their own) are never relit: their
      modifier sets bit 30 and the hook leaves them as drawn.
    - **CesiumJS's order, sun before look.** CesiumJS lights (part 3) before Confidence / Data
      age (5), which tint by the relit luminance; under PlayCanvas the look is drawn in the copy
      pass and the hook after it. Every branch of the look is affine in the colour's scale,
      `look(s·c) = A + s·(look(c) − A)` with `A = look(black)`, so under a cell view the copy
      pass also draws the look of a black splat of the same alpha into the word (ten bits a
      channel) and the hook relights `A + (drawn − A)·factor`: the look of the relit splat
      within half a step of 1/1023 (tested on every branch of `cellViewColor`). Relit after the
      look instead, a shadowed cell's tint would fall to the ambient floor (0.35 of the unlit
      look at the evening sun) where CesiumJS keeps its floor (0.73). A colour-only copy
      (spherical harmonics as the eye moves) leaves the word as it was, which still holds: `A`
      does not depend on the photograph.
14. **Cost.** Off: nothing (no stream, no chunk). On: the stream, 4 B a splat of the work buffer
    for the session; in the final draw a stream read, a few comparisons, a quaternion turn and
    a dot product a splat (the spike: about +10 % frame time on SwiftShader, relative only);
    under a cell view, the look evaluated twice per splat in the copies it already makes. The
    globe's entry 196,147 (#73 at de79fc1d) → 196,146 B gzip at the branch's head (a few bytes either way as chunk names change), 83 files up front as before: the hand-off lives in
    the relight's shared-state chunk the entry already names, and the hook's constants travel
    with the sun, so the PlayCanvas and relight chunks share no new module.
15. **Spark: not ported; measured.** Spark's per-frame equivalent is a world modifier with
    `enableWorldToView`, which regenerates a mesh whenever the view moves. Turned on for every
    tile of the synthetic yard (43 337 splats, 15 meshes, SwiftShader, 40 moving frames), the
    moving frame was 420 ms either way (on 420.3 / 419.5, off 421.9 / 424.3 ms; PlayCanvas
    28–30 ms on the same view): Spark already regenerates while the view moves away from its
    sort, and SwiftShader's frames are sort-bound, so the measurement cannot see the cost it
    is meant to show -- a pass over every splat of the scan each moving frame, which grows with
    the scan where PlayCanvas's uniforms do not. With that unmeasured and a third relight path
    (a dyno, its harness) to build and keep, Spark keeps its honest notice. **WebGPU trial:**
    not drawn (the hook is GLSL, as the motion and the cell views).

### Stubbed

- **A per-splat normal sidecar**: a surface fit from the pipeline's SfM points (keyed by tile
  checksum, as instances are), which would give round and noisy splats on a surface the normal of
  the surface. Deferred: SPZ has no attribute slot to carry it, so it needs a sidecar file, its
  loader and a texture per tile.
- **Shadows cast by today's sun**: no shadow map of splats; only the facing term is modelled.
- **Removing the capture's own shading**: not attempted. The badge says the shadows are baked in.

### Consequences

- While At capture holds the clock, everything on the scene clock stands still: the Living
  Survey's wind stands mid-bend (the Simulated motion badge stays, the bend is still modelled, and
  the survey's own guard logs that the clock stopped) and the sky is the capture's.
- A flat surface relit is never brighter than its photograph, only darker where it faces away.
- Splats whose cut-off is close to 0.25 may flip between relit and not from one splat to the next
  on the same surface; the threshold is the design's, not a measurement.

Evidence: `e2e/sunlightScene.spec.ts`, once a renderer (an east-facing slab under SwiftShader;
the discs' brightness lit over unlit against the mirror's factor). CesiumJS: discs 137.6 off,
126.0 at 07:15 local solar time, 0.916 of the photograph against 0.917; 50.3 at 16:45, 0.366
against 0.35. PlayCanvas: 149.5 off, 136.6 in the morning, 0.914 against 0.917; 52.1 in the
evening, 0.349 against 0.35; night 0.348 (flat) and 0.349 (round). Both: round splats unchanged by
day, everything at the floor at night, the sun off and As captured on drawing exactly the
photograph. With Confidence over the wall (every cell strong), lit over unlit: CesiumJS 0.965
(morning) and 0.738 (evening), PlayCanvas 0.964 and 0.727, against 0.965 and 0.728 for the look of
the relit grey (0.35 for the look relit after). `e2e/sunlight.spec.ts` (the app under CesiumJS,
PlayCanvas and Spark), and the `splatNormals`, `sunHook`, `playcanvasSunlight`, `splatSunlight`,
`sunClock` and `sunlightControl` unit tests.
