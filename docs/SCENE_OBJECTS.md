# Scene objects: segment once, move by skin, search by meaning

Target architecture, agreed 2026-10-01 on branch `living-models`. It replaces the per-class
rules of `scene_plants.py` / `skeleton.py` and the per-limb oscillator fit (LIVING_ENGINE.md
§4) as the direction for living, live and dream. Checklist: [LIVING_PLAN.md](LIVING_PLAN.md).

**Principle: nothing boutique.** No fixed class list and no hand rules per object type. Every
stage uses general, off-the-shelf learned models and stores general data that later questions
can be asked of.

## 1. The pipeline

```
dense scan
  │
  ▼
SEGMENT ── per splat: instance id (hierarchy: area → object → part)
  │        per instance: embedding (open vocabulary), tags, properties, bounds
  │        └─► viewer: hide / highlight / search, with no class list
  ▼
STORE by behaviour (from properties, not class names)
  │  static, deforms in place ─► spatial tiles + per-splat instance id (+ skin weights)
  │  moves as a whole         ─► its own object tileset, own frame (+ fill the hole it leaves)
  ▼
SKIN each moving instance: Simplicits / FreeForm (Kaolin, Apache-2.0) ─► m weight fields;
  │  each field's transform = a handle (offline, per instance, no training data)
  ▼
DRIVE the handles
  │  living: wind forces + materials (class prior, refined by the video teacher)
  │  live:   telemetry (a car is 1 rigid handle; a robot its joints)
  │  dream:  world model → handle trajectories
  ▼
GPU: x' = x + Σ_j w_j(x) · Z_j · [x; 1]   (one skinning path, the existing vertex hook;
     signed weights, never normalised; a constant field carries rigid motion)
```

## 2. Terms

- **Handle**: a transform (affine: rotation, scale, shear, translation) that changes over
  time. It is a degree of freedom, not necessarily a point in space.
- **Skin**: per splat, a weight per handle. A splat moves by its rest position plus the
  weighted sum of its handles' displacements (`Z_j`, rest-relative affine). Weights are
  signed and do not sum to one; the shader must not normalise or clamp them.
- **Simplicits / FreeForm** (NVIDIA Kaolin, Apache-2.0): computes a shape's skin offline,
  per instance, from the shape alone. The FreeForm/RKPM basis needs no network and no
  training data; MLP Simplicits is trained per object. It gives _how a shape can bend_;
  materials, forces and non-elastic behaviour come from data or drivers.
- **PhysSkin** (zju3dv, CVPR 2026) predicts the same kind of skin in one pass, but has no
  licence (research only), depends on GPL/ShapeNet-trained Michelangelo, and on the
  synthetic tree folded and either toppled or locked when pinned. Not used (§7).
- **Video teacher**: fits a few material/forcing numbers per instance (stiffness, damping,
  drag) rather than per-limb frequencies. Per-limb frequencies proved unidentifiable (Wind on
  Trees, arXiv 2609.17810).

## 3. Segmentation: method

All of these are general models; none knows our scenes.

1. **Views.** Render the scan from many viewpoints (`splat_render.py`): observer-near views
   plus a ring, each with per-pixel splat ids (which splats dominate each pixel). A scan wider
   than one view's footprint (a 0.24 m cell spanning ~4 px: ~30 m on the camp) also gets
   local views that scale with its area -- obliques per footprint anchor, placed by line of
   sight, and more eye-height views -- each with a far plane, up to a cap (252 on the camp).
   They render in forked workers while the GPU masks the ones already done -- as many as
   the Modal call's reservation holds (8 CPUs and 32 GiB on an L4: the camp in 1,416 s for
   $0.56; a scan past 500 tiles gets 16 and 64 GiB; `infra/modal/segment.py`), not a fixed
   count, since the masks rather than the renders set the pace.
2. **Masks.** Class-free automatic masks at several scales per view (SAM 2 family). A mask
   over (nearly) the whole view is no evidence of what belongs together and is left out.
3. **Lift.** Each mask votes for the splats it covers. Splats that co-occur in masks across
   views are merged into instances (a graph whose edge weights are co-occurrence over
   visibility). Scales give the hierarchy: a coarse mask is the parent of the finer ones
   inside it. Neighbouring cells first join only on strong evidence (single linkage over a
   million cells otherwise chains across a scan), then regions join as wholes in rounds;
   specks and slivers between masks take their neighbours' instance.
4. **Meaning.** For each instance, crop its best views and embed them with an
   image-text model (SigLIP/CLIP family). Text search is cosine similarity at query time.
   An instance too small in every view for a useful crop (`DESCRIBE_MIN_PX`) keeps no
   embedding and no tags (its row in `instances.emb` is zero) and its nearest described
   ancestor's properties.
5. **Tags and properties.** Zero-shot against a large open vocabulary (tags) and a short,
   fixed list of _attribute_ prompts (not classes): movable, rigid, elastic/flexible,
   static structure, vegetation, water, vehicle, person/animal. Each property is a
   score, so physics and the UI read numbers rather than matching class names. Later a
   vision-language model can fill the same record with free-text descriptions.
6. **Behaviour.** `static` | `in-place` | `movable`, derived from the property scores by
   rule. It decides storage (§1) and which driver applies.

## 4. Data contract (v1)

Written beside the measured tiles; read by the viewer, the skinning step and the engine.

### `instances.json`

```jsonc
{
  "format": "hexapod.instances", "version": 1,
  "frame": "tileset local ENU (the root transform's frame), metres",
  "embedding": { "file": "instances.emb", "model": "<model id>", "dim": 768, "dtype": "float16" },
  "vocabulary": { "model": "<model id>", "size": 1203 },  // what tags were scored against
  "instances": [
    {
      "id": 1,                       // 1-based; 0 means none / mixed
      "parent": null,                // id of the coarser instance containing it, or null
      "level": 0,                    // 0 = coarsest
      "splats": 123456,              // how many gaussians carry this id (leaf level)
      "bounds": { "min": [x, y, z], "max": [x, y, z] },
      "centroid": [x, y, z],
      "tags": [ { "label": "pickup truck", "score": 0.31 }, ... ],   // top-k, descending
      "properties": { "movable": 0.82, "rigid": 0.77, "elastic": 0.05, "static": 0.10,
                      "vegetation": 0.02, "water": 0.0, "vehicle": 0.91, "creature": 0.01 },
      "behaviour": "movable",        // "static" | "in-place" | "movable"
      "views": 14                    // how many views it was seen in
    }
  ],
  "tiles": { "<tile checksum>": [id, count, id, count, ...] },  // per tile, RLE, tile order
  "tilesEncoding": "rle; a merged parent splat takes an id only if all its children share it"
}
```

- `tiles` uses the same checksum keys and run-length encoding as `plants.json`
  (`scene_plants.plant_binding`), so the viewer's existing per-tile binding code applies.
- Ids are leaf-level (the finest instance). The hierarchy is walked through `parent`.
- `instances.emb`: `float16`, `count × dim`, row `k` is instance id `k + 1`, L2-normalised; an
  instance that was not described (no `tags`) has a zero row.

### Root extras

The measured tileset's `root.extras.instances = { "uri": "instances.json", "count": n }`, so
the viewer finds it without probing (the same pattern as `viewCones` and `inferredLayers`).

**In the viewer**, hide and highlight work under every splat renderer, from the same per-tile
binding: CesiumJS's own primitive (`cesium/splatInstances.ts`: the visibility chain and the
colour hook), and the dedicated renderers laid over the globe (`cesium/scanView/scanInstances.ts`).
PlayCanvas digests each tile's positions in its decode worker, keeps the ids beside the tile's
splats as one more resource stream (in PlayCanvas's Morton order), and applies the rule in a
work-buffer modifier; Spark digests the tile's SPZ centres and applies it in an object
modifier (a dyno). The rule is CesiumJS's: a hidden splat has no opacity, a highlighted one
is pulled toward the tint, and while anything is highlighted the rest are dimmed. A scan that
PlayCanvas streams from its own package (`sog/lod-meta.json`) has no tile checksums, so its
objects cannot be hidden there; the objects panel says so and offers the CesiumJS renderer.
The panel's "Hide all N matches" and "Show only matches" act on every match of the query, not
only the fifty it lists.

### `skin.json` + `skin.bin` (step B2)

Written by `tools/captures/skin_scene.py TILES_DIR instances.json [--out DIR] [--link]`, one
skin per object whose behaviour is `in-place` or `movable`: the **coarsest** such instance on a
splat's ancestor chain owns it (a tree, not each branch), so the skin is smooth across parts.

```jsonc
{
  "format": "hexapod.skin", "version": 1,
  "frame": "tileset local ENU (the root transform's frame), metres",
  "formula": "x' = x + sum_j w_j(x) * Z_j * [x - origin; 1] ...",
  "method": { "name": "simplicits-rkpm", "source": "NVIDIA Kaolin (Apache-2.0) ...",
              "material": { "uniform": true, "poisson": 0.45 } },
  "weights": { "file": "skin.bin", "dtype": "int8", "rowBytes": 16, "scale": 0.007874, "rows": 12568 },
  "skins": [
    {
      "id": 1,                     // 1-based; 0 means none
      "instance": 1,               // the instances.json id it moves (with its descendants)
      "handles": 14,               // m, the constant handle included (8..16)
      "origin": [x, y, z],         // rest frame origin: the object's base (bounds' base centre)
      "scale": 4.84,               // half its largest extent, metres
      "splats": 8980, "nodes": 600,// fit points and RKPM kernels
      "eigenvalues": [ ... ],      // per learned handle 1..m-1: stiffness (unit box, E = 1)
      "support": [ { "centre": [x, y, z], "radius": 1.9 }, ... ], // per learned handle, rest frame
      "dynamics": {                // optional (C1): what a modal driver needs
        "mass": [ ... ],           // M_ij = mean over its splats of w_i w_j (w_0 = 1), m×m upper triangle
        "anchor": { "source": "base", "splats": 103, "band": 0.968,
                    "gram": [ ... ] } // G_ij, the same over its anchor splats
      }
    }
  ],
  "tiles": { "<tile checksum>": { "skins": [skin, count, ...], "row": 0 } },
  "tilesEncoding": "rle [skin, count, ...] in the tile's own order (0: no skin); ..."
}
```

- **Motion.** `x' = x + Σ_j w_j(x) · Z_j · [x − origin; 1]`, `Z_j` a 3×4 affine (row-major)
  per handle per frame, in the rest frame (tileset axes, about `origin`). Handle 0 is the
  constant field, `w_0 ≡ 1`, not stored: `Z_0 = [R − I | t]` moves the object rigidly and
  exactly. Handles `1..m−1` are the smallest elastic eigenmodes (`H c = λ M c`), each scaled to
  `max |w| = 1` (positive) over the object's splats, so `Z_j` is "the displacement where handle
  j acts fully". Signed, never normalised to sum to one.
- **Handles per object**: `m = clamp(round(8 + 2·log2(d / 2 m)), 8, 16)` for a bounds diagonal
  `d` (8 at 2 m, 12 at 8 m, 16 from 32 m), and at most its node count allows. Nodes: an eighth
  of its splats, 48 to 600; at most 4000 integration points.
- **`skin.bin`**: one 16-byte row per **skinned** splat (unskinned splats take none), in each
  tile's order from `row`, tiles in checksum order. Byte `k` is handle `k + 1`'s weight,
  `int8 = round(127·w)` (weight = byte × `weights.scale`), unused bytes 0. 16 bytes is one
  RGBA32UI texel, the viewer's upload unit. Merged level-of-detail parents are evaluated at
  their own position (the RKPM basis is defined everywhere), not fitted.
- **Dense, not top-k** (measured, `skin_scene.sparsity_report`, synthetic tree, 13 handles,
  random handles whose largest displacement is 5% of its half-height; error as a share of the
  rms displacement):

  | weights          | bytes/splat | rms error | max error | kNN stretch p99 | stretch max |
  | ---------------- | ----------- | --------- | --------- | --------------- | ----------- |
  | float32 (dense)  | 48          | 0         | 0         | 1.046           | 1.16        |
  | **int8 (dense)** | **16**      | **0.65%** | **1.8%**  | **1.050**       | **1.17**    |
  | top-8, int8      | 12          | 4.2%      | 14%       | 1.057           | 1.39        |
  | top-4, int8      | 6           | 22%       | 102%      | 1.088           | 3.03        |
  | top-2, int8      | 3           | 53%       | 147%      | 1.135           | 4.61        |

  The eigenmodes are global (non-zero almost everywhere), so keeping the k largest per splat
  switches modes on and off between neighbours and tears; on the yard's objects top-4 is 22–55%
  rms error. Dense int8 is within 1% everywhere measured (yard: 0.5–0.8% rms, ≤ 2.2% max).

- **Size**: 16 B per skinned splat, nothing for static ones; `skin.json` ~0.4 KB per tile plus
  ~1.3 KB per skin, and its `dynamics` 0.7–2.5 KB (two `m×m` triangles). The yard fixture: 4
  skins, 12,568 rows, `skin.bin` 201 KB, `skin.json` 12 KB. Everything skinned in the yard would be 727 KB (45k splats; gzip 446 KB) against
  688 KB of splat tiles: the cost is only worth paying for what moves.
- **Smoothness** (tests on the synthetic tree, int8 weights): random handles at 2% of its
  half-height stretch kNN edges by p99 1.022 (max 1.09), and the deformation's Jacobian stays
  positive (min det 0.88); the constant handle as a rotation keeps every edge to 1e-12. One
  handle pushed a full metre on the 6.5 m tree (15% of its height) does fold (min det ≈ 0): a
  driver keeps a handle's displacement below its `support` radius.
- **Fit cost** (NumPy port, 4 shared CPUs): 3–8 s for a 7–12k-splat object at 600 nodes, under
  a second below 2k splats; the torch path the evaluation used took 389 s for the tree.
- **Root extras**: `root.extras.skin = { "uri": "skin.json", "count": n }` (`--link`), like
  `instances`.

**Viewer** (step B3, `apps/web/src/cesium/splatSkin.ts`, `lib/skin.ts`): a part of the
splat primitive's motion chain (`splatMotionChain.ts`; the Living Survey's rig is another), so
motion composes with the visibility chain (hide, view cones) and the colour hook (highlight),
which all see the displaced position. Per splat, uploaded per tile as the instance ids are
(un-bake, checksum, decode): its skin id (RGBA32UI, four a texel) and its row (RGBA32UI, one a
texel). Per skin, 64 RGBA32F texels: `(moving, m)` then `Z_j`'s rows folded into the baked
frame (`A_b = L·A·L⁻¹`, `t_b = L·(t − A·o) − A_b·b`), uploaded when a driver sets them, one
row per 16 skins. A skin at rest costs one fetch; nothing moving costs none. **Covariances**
follow `J = I + Σ_j w_j A_j` through the engine patch's optional `splatVertexJacobian`
(`J·Σ·Jᵀ`); dropped is the weights' gradient term `Σ_j Z_j[x;1]∇w_jᵀ` -- exact for the
constant handle, 0.14 at most (against 1 on the diagonal) for the tree's 2% random handles.
The sorter still orders by rest positions. **Drivers** (C1 wind, C3 telemetry) call
`skinningOf(assetId).setInstanceHandles(instanceId, Z)` with `12·m` numbers (rest frame) per
frame, or `null` for rest; `skin.json`'s `eigenvalues` and `support` are what a modal wind
model needs (a handle's stiffness, and where it acts).

### Wind on skins (step C1)

`packages/world/src/skinWind.ts` (the model), `apps/web/src/cesium/skinWind.ts` (the driver,
ticked by `LivingSurveyManager` on the scene clock with the Living Survey's wind control).

- **Coordinates.** Each handle `j` (the constant one included) carries a horizontal
  translation `q_j`: `Z_j = [0 | q_j]`, so a splat moves by `Σ_j w_j q_j`. Linear parts stay 0
  (covariances drawn as measured); no vertical motion.
- **Mass** `M` = `dynamics.mass`. **Stiffness** `K = (c/scale)² diag(0, λ_1 M_11, …)`: the
  handles are eigenmodes of `H c = λ M c` on the unit box with `E = 1`, so a material of wave
  speed `c = √(E/ρ)` gives **`ω_j = c·√λ_j / scale`**. `c` is the material's `stiffness`
  (prior 3.5 m/s for vegetation: the yard's 9.7 m tree, `λ_1` 12.2, `scale` 4.84, has its first
  handle at 0.40 Hz and its first anchored mode at 0.22 Hz; the 1.9 m shrub 3.7 Hz and 2.7 Hz).
- **Load.** The Living Survey's wind (`speedFromStrength`, the frozen EN 1991-1-4 turbulence
  field of `turbulence.ts`, 64 modes, one field for the whole scene so gusts cross from object
  to object), sampled at each handle's support centre: `v = U((1 + I a)ŵ + 0.75 I b ĉ)`, `I`
  at the centre's height. Handle `j` is driven by `(D/scale)(M_0j a_0 + M_jj (a_j − a_0))`,
  `a_j = |v_j|v_j`, `a_0` at mid-height: the mean drag through each handle's mean weight, the
  gust's variation across the object through each handle's own. `D` is the material's `drag`
  (dimensionless: drag per unit mass falls as `1/size`, so with `ω ∝ 1/size` an object sways
  a fixed share of its size; prior 0.025 for foliage, a quarter of it bare).
- **Anchor.** The eigenmodes are free: alone they move an object's base too. `G y = μ M y`
  ranks handle-space directions by how much of their motion reaches the anchor splats (`√μ` =
  rms there over rms of the object); those with `√μ ≤ 0.05` (`ANCHOR_TOLERANCE`) span what the
  object may do, and `M`, `K` projected onto them and diagonalised give the anchored modes (the
  constant handle cancels a learned mode at the base). Anchor splats are the **lowest tenth of
  the object's height** (at least 3 median spacings; `skin_scene.anchor_mask`). Contact with
  static neighbours was tried first and rejected: a shrub touching the next shrub was anchored
  up to its top and kept no direction at all. The yard: the tree keeps 12 of its 14 directions
  (worst `√μ` 0.006), the snag 9 of 12, the 1.9 m shrub 3 of 9, the 1 m shrub 2 of 8. A skin
  without `dynamics` (written before C1) is not swayed: re-run `skin_scene.py`.
- **Integration.** Each anchored mode `s̈ + 2ζΩṡ + Ω²s = Φᵀ F` advances by its exact
  discrete-time solution on a fixed 1/60 s grid (force held over a step, sampled at its middle):
  unconditionally stable, and the state at a grid time is identical at any frame rate (frames in
  between interpolate). A jump in scene time over 0.5 s (or back) restarts from the equilibrium;
  wind turned on starts from rest. Calm hands every skin `null` in the same tick: the measured
  frame, pixel for pixel. Deterministic given the field's seed (`SKIN_WIND_SEED`).
- **Bounded.** `ρ = max_j |q_j| / (0.25 r_j)` (`HANDLE_REACH`, `r_j` the support radius; the
  object's `scale` for the constant handle) and every `q` scaled by `tanh(ρ)/ρ`, one factor
  for the object so the anchor holds: no handle moves beyond a quarter of its support radius.
- **Numbers** (yard, default strength 0.1 = 6.3 m/s): the tree sways ~0.11 m rms over its
  splats, its base's rms motion ≤ 0.4% of that; the snag 2 cm; the shrubs millimetres (their
  anchored modes are stiff: 2.7–8 Hz). **Cost**: 0.34 ms a frame on the main thread for 30
  fourteen-handle objects (`skinWind.test.ts`).

### `materials.json` (C1 reads; C2 writes)

Per-instance materials, beside the tiles, declared by `root.extras.materials = { "uri",
"count" }` (the pattern of `instances` and `skin`). A separate file, not a block of `skin.json`:
`skin.json` is geometry rewritten by `skin_scene.py`; materials are fitted later by another
producer (the video teacher) and must survive a skin refit.

```jsonc
{
  "format": "hexapod.materials",
  "version": 1,
  "materials": [
    {
      "instance": 1, // instances.json id (the skin's owner)
      "stiffness": 3.5, // c, m/s: ω_j = c·√λ_j / scale
      "damping": 0.1, // ζ of every anchored mode, 0..0.95
      "drag": 0.025, // D, dimensionless: a handle's acceleration D·|v|v / scale
      "wind": true, // whether the wind drives it at all
      "evidence": "fitted-real", // the motion evidence ladder, or "prior"
    },
  ],
}
```

Every field but `instance` is optional; what a record leaves out comes from the **prior**
(`materialPrior`, from property scores, never class names): `wind` iff behaviour is
`in-place` (movable and static objects are not swayed); softness `σ = clamp(max(vegetation,
elastic) − rigid/2, 0, 1)`, `stiffness = 3.5 · 4^(1−σ)` m/s; `damping = 0.05 + 0.05 ·
vegetation`; `drag = 0.025 · (0.25 + 0.75 · vegetation)`; `evidence: "prior"`. The yard's
`skin/materials.json` turns the wind on for 1, 9 and 10 (its stand-in segmentation reads every
instance `movable`) and leaves 12 on its prior, still.

A fitted record may also carry a `fit` block (source clip, frames, fps, tracked points,
signal to noise, loss) for provenance; readers ignore it.

### Video teacher (step C2)

`tools/captures/teacher_materials.py` fits `stiffness`, `damping` and `drag` per instance to a
clip of the object moving in the wind, by matching what the C1 model would show the camera to
what the camera saw. The model is `skin_wind.py`, a line-by-line port of `skinWind.ts` (and of
the turbulence field it reads), held to it by `data/tiles/synthetic-yard/skin/wind_parity.json`:
numbers `skinWind.test.ts` writes (`UPDATE_SKIN_WIND_PARITY=1`) and both test suites check
(handles to 1e-9 of their peak).

- **Observe.** The scan rendered from the clip's camera at rest labels the object's pixels
  (the skin owner per splat). Textured points there are tracked by Lucas-Kanade against frame
  0 (no drift over minutes), forward-backward checked; the static background's median motion
  (the camera's) is taken out. Each point is lifted onto the scan by the rest render's depth:
  its skin weights (nearest skinned splats) and `J`, how a horizontal metre there moves on
  screen.
- **Predict.** A point moves on screen by `J Σ_j w_j q_j`, so the mean of the points'
  spectra is a quadratic form in the handles: a few channels carry it exactly. The wind's
  realisation is unknown, so the model is driven by an ensemble of the same EN 1991-1-4 field
  at the clip's mean speed and bearing (other seeds, 320 modes each: the browser's 64 make a
  line spectrum, a real wind does not); `Φ` and the modal loads do not depend on the
  material, and the response to any `(c, ζ)` is the integrator's own exact transfer function
  (one FFT).
- **Fit.** Welch spectra in log-spaced bands, `mean (log(S(c, ζ, D) + b) − log P)²` with a
  white tracking floor `b`: a grid over `(c, ζ)` (the prior's `c` / 6 to × 6) with `D`, `b`
  profiled, then Nelder-Mead with the output bound on. Where the resonances sit gives `c`,
  their width and the share of motion below them `ζ`, the level `D` (relative to the wind
  speed given: `--speed` / `--strength`; a clip of an unknown wind fits `D` for the assumed
  one). Written only when the fitted motion stands 3× above the floor.
- **Evidence.** `--source real` writes `fitted-real`, `--source generated` (a world-model
  clip, or this tool's synthetic ones) `fitted-generated`; other records are kept.

**Validation** (the yard, strength 0.5 = 14.1 m/s, bearing 60°, 180 s at 15 fps, 320 × 240;
the truth 0.8 × the prior's `c`, `ζ` 0.07, 1.3 × its `D`; fitted once from the property prior
and once from a prior 4× too stiff, a quarter of the drag and `ζ` 0.3 -- both land in the
same place):

| instance          | true c / ζ / D        | fitted (from the prior) | error c / ζ / D    |
| ----------------- | --------------------- | ----------------------- | ------------------ |
| 9, snag (7 m)     | 4.87 / 0.070 / 0.0325 | 4.92 / 0.068 / 0.0319   | +0.9% / −3% / −2%  |
| 1, tree (9.7 m)   | 2.80 / 0.070 / 0.0325 | 2.70 / 0.081 / 0.0339   | −3.6% / +16% / +4% |
| 10, shrub (1.9 m) | 5.43 / 0.070 / 0.0323 | 5.48 / 0.072 / 0.0277   | +0.9% / +3% / −14% |

The fitted snag replays its clip (same wind realisation) at a correlation of 0.998, the tree
0.93 (its sway reaches the output bound at this strength), the shrub 0.998. Damping needs a
continuous wind: a clip under the browser's own 64-mode field still gives `c` (−0.4%) and `D`
(+4%) but `ζ` 2.6× off, its spectrum being a few lines on the resonance. The tree's 0.17 Hz
mode wants minutes of footage; a world-model clip of ~5 s (`teacher_materials.py world`, Wan
or Cosmos on Modal, `huggingface` secret needed) resolves only fast objects.
`tests/test_teacher_materials.py` runs the snag at 90 s (c within 5%, ζ 35%, D 20%).

### Telemetry on instances (step C3)

Live mode's first driver: a pose stream bound to an instance moves that object rigidly.
`packages/world/src/telemetry.ts` (frames, rigid motion, the playout track, synthetic paths),
`apps/web/src/lib/telemetry.ts` (the bindings file, the wire format),
`lib/telemetrySources.ts` (sources), `cesium/telemetry.ts` (the driver, attached by
`SiteManager` beside the skin), `cesium/splatRigid.ts` (rigid motion without a skin).

**Bindings: `telemetry.json`**, beside the tiles, declared by `root.extras.telemetry = { "uri",
"count" }` (the pattern of `instances`, `skin` and `materials`). A separate file for the reason
`materials.json` is one: a binding is deployment data (which robot is which object) and must
survive a re-segmentation or skin refit; it is keyed by instance id, so a re-segmentation that
renumbers instances needs its bindings re-pointed. The loader takes any URL, so the API can
serve the same document per site when bindings move out of the tiles.

```jsonc
{
  "format": "hexapod.telemetry", "version": 1,
  "sources": {                       // by id
    "r1":   { "kind": "sse", "url": "https://bridge/r1", "frame": "geodetic" },
    "yard": { "kind": "websocket", "url": "wss://bridge/fleet" },  // frame defaults to "scan"
    "loop": { "kind": "synthetic", "frame": "ecef", "rateHz": 2, "delayMs": 40, "jitterMs": 60,
              "dropouts": [[from, to], ...],                       // source ms, optional
              "path": { "kind": "circle", "centre": [x, y, z], "radius": 2.5, "periodS": 24,
                        "phaseDeg": -90 } }  // or { "kind": "polyline", "points", "speedMps" }
  },
  "bindings": [
    {
      "instance": 8,                 // instances.json id: it and everything below it move
      "source": "loop",
      "stream": "R1",                // optional: only readings whose "stream" is this
      "rest": { "frame": "scan", "position": [x, y, z], "orientation": [x, y, z, w] },
                                     // the body frame at capture; omitted: the instance's
                                     // base centre (bounds), level, scan axes
      "latencyMs": 250,              // playout delay
      "extrapolateMs": 1000,         // dead reckoning past the newest reading
      "staleMs": 3000,               // newest reading older than this: stale
      "stale": "rest",               // "rest": fade back over fadeMs; "freeze": hold
      "fadeMs": 2000
    }
  ]
}
```

A synthetic path is written in the scan frame whatever the source's `frame`; the source emits
its readings in that frame, so the conversions run as they would for a real stream.

**Readings** (the wire format of an SSE or WebSocket source, and what a robot or fleet bridge
publishes): a JSON message is a reading, an array of them, or `{ "samples": [...] }`; a reading
is `{ "t", "position", "orientation"?, "headingDeg"?, "pitchDeg"?, "rollDeg"?, "frame"?,
"stream"? }`. `t` is milliseconds (epoch, the source's clock) or ISO 8601. `position` and
`orientation` (`[x, y, z, w]`, body → frame) are the body frame's pose in one of three frames:
`scan` (the tileset's local ENU, what `instances.json` is written in), `ecef` (WGS84 metres),
`geodetic` (`[lon°, lat°, h]`, orientation body → east-north-up there). Without an
`orientation`, heading (clockwise from north), pitch (nose up) and roll (right side down) are
read in the level frame at the position, for a body with x forward, y left, z up. Everything is
brought into the scan frame through the root's computed transform (its linear part taken as a
rotation).

**Motion.** The pose shown `P` moves the object by `M = P · rest⁻¹` (`x' = R x + t`). A
**skinned** instance takes it through its skin's constant handle, `Z_0 = [R − I | t + (R − I)o]`
about the skin's origin `o`, the elastic handles at rest (exact: `w_0 ≡ 1`, and the covariance
follows `R`). Any **other** instance takes it through the rigid part of the motion chain
(`splatRigid.ts`): the scan's per-splat instance ids (shared with hide and highlight), a slot
per driven instance written at it and every instance below it, and per slot the motion folded
into the baked frame as a skin handle is (three RGBA32F texels); it gives the chain its linear
part, so covariances turn with the object. No skin, no new data: any segmented object can be
driven. A skin that arrives after the readings takes the object over from the rigid part.

**Playout** (`PoseTrack`). The source's clock need not match ours: the offset is the smallest
`arrival − t` over the last 32 readings (the fastest transport seen), so the time shown is
`now − offset − latencyMs` in source time. Between readings: linear position, slerp. Past the
newest: constant velocity and turn rate from the last two (not across a gap over 5 s) for up to
`extrapolateMs`, then held there. Past `staleMs`: `freeze` holds; `rest` blends the pose back to
the rest pose over `fadeMs` and then hands the object `null`, the measured frame pixel for
pixel. Before the first reading the object is at rest; a reading after a fade is live again at
once. `driver.statuses` gives each binding's state (`none`, `live`, `extrapolated`, `held`,
`stale`, `rest`), its age and the pose, what a "Observed · n s old" badge reads
(LIVING_ENGINE.md §1).

**With the wind.** A bound instance, its ancestors and its descendants are claimed
(`motionClaims.ts`); the wind (`skinWind.ts`) neither sways nor writes a claimed skin, so a
driven shrub keeps its elastic handles at rest while the tree beside it sways. Overlap between
two bindings (one instance inside another) gives the deeper one its own slot.

**Sources.** `synthetic` (deterministic, sampled on the driver's clock: the same clock gives the
same readings, with delay, hashed jitter and dropouts; dev and e2e), `sse` and `websocket` (JSON
messages, stamped on arrival). All three implement one interface (`TelemetrySource`:
`subscribe`, optional `pump(now)`); the Fleet tab's machines (`missions/types.ts`, a demo
provider today) reach a scan the same way once a robot bridge publishes their poses.

**Sorted where drawn.** The splat sorter orders by rest positions; a driven object moves metres
and turns, so the rigid part tells the sorter (`splatSorter.ts`, `setSortMotion`) each splat's
slot and each slot's motion. Every sort carries, per slot, the eye moved back by the slot's
motion (`M⁻¹(eye − t)`): a rigid motion keeps distances, so the rest positions sorted against
that eye give the moved splats' order exactly, with no positions re-sent. A slot's eye moving as
far as the camera would have to move calls for a new sort. A skinned object moved by its constant
handle still sorts at rest (small objects; the same groups could come from the skin ids).

**Renderers.** CesiumJS (its own splat primitive) only, as the skin is. The dedicated renderers
would need the same rule over the instance ids they already stream for hide and highlight
(`cesium/scanView/scanInstances.ts`): in PlayCanvas, a per-instance 3×4 motion (a small uniform
array or texture indexed by a slot per id) applied to centres, and rotations applied to the
covariances, in the work-buffer modifier that already reads the ids; in Spark, the same in the
object modifier (dyno), with a motion uniform per driven object. Neither has the skin path yet,
so a skinned instance would take the rigid path there. Nothing else changes: the driver hands
each renderer motions, not pixels.

### Split objects (step C4)

`tools/captures/split_objects.py split TILES_DIR OUT_DIR [--ids 3,7] [--absorb]
[--filler ...] [--renderer cpu|gsplat] [--distill N]` takes chosen instances out of the
spatial tiles into tilesets of their own and fills the holes they leave, into a **new**
directory (the input is never written; an output it did not write is refused; the same
arguments write the same bytes). Default choice: the coarsest `movable` instances with at
least 500 gaussians and bounds no larger than 8 m; or explicit `--ids` (an id inside another
chosen one is dropped). `candidates` lists the choice without writing.

```
OUT_DIR/
  tileset.json            the scan without the objects; root.extras.objects, .split, and each
                          fill in .inferredLayers
  instances.json (+.emb)  re-bound: rewritten tiles under their new checksums, object tiles added
  <tile>.<digest>.glb     each tile that held any of an object's gaussians, without them
  <tile>.glb, sidecars    everything else, copied unchanged
  objects/<id>/           tileset.json + object.glb (+ viewcones.bin): the object, its own frame
  fills/<id>/             tileset.json + tiles + viewcones.bin: the inferred layer under it
```

**Root extras** of the split scan:

```jsonc
"objects": [
  {
    "uri": "objects/3/tileset.json",   // relative to the scan's tileset.json
    "instance": 3,                     // instances.json id (the object is it and its descendants)
    "origin": [2.0828, 1.8201, -0.0776], // its frame's origin in the scan's local ENU (m), on
                                       // the 1/4096 m SPZ grid: its bounds' base centre
    "pose": { "translation": [0, 0, 0], "rotation": [0, 0, 0, 1] }, // where it is drawn
    "splats": 27319,
    "fill": "fills/3/tileset.json"     // the inferred layer under it, when one was made
  }
],
"split": { "format": "hexapod.split", "version": 1, "source": "...",
           "removed": { "leaves": 27319, "parents": 2124 }, "rule": "..." }
```

- **The object's tileset**: one tile (leaf gaussians only; small objects need no LOD), its
  positions relative to `origin` -- the same SPZ records shifted by an integer number of grid
  steps, so the shift is exact -- and its root transform the scan's times `T(origin)`: loaded
  alone it draws where it was measured. `root.extras.object = { format: "hexapod.object",
version, instance, ids, origin, frame, scene, fromTiles }` (`ids`: every id its gaussians
  carry), `root.extras.instances` points at the scan's `instances.json` (`../../`), and its
  view cones are rebuilt from the scan's observers in its own frame, so they turn with it.
- **Pose**: a rigid motion about `origin` in the scan's frame, `p -> origin + t + R (p -
origin)`, `rotation` a unit quaternion `x, y, z, w`. The rest pose is the identity; the
  viewer (or a driver) sets another at runtime.
- **The scan's tiles**: a tile that held any of an object's gaussians is rewritten without
  them, byte for byte what was there minus the removed records (the SPZ is sliced, never
  re-quantised; `unpack -> pack` would move a rotation byte here and there), under
  `<stem>.<new checksum digest>.glb` so no cache serves the old tile for the new. A leaf's
  gaussian goes with the object by its id; a merged parent's by its id when bound, else
  (merged across ids) when most of its 8 nearest leaves are the object's. Bounding volumes
  stay as they were (conservative); a tile left empty loses its content.
- **Bindings**: `instances.json` keeps every instance and drops the replaced tiles' keys; a
  rewritten tile's runs are its old runs without the removed gaussians, under its new
  checksum, and each object tile's runs are added -- the same ids, so hide, highlight and
  search act on the object wherever it is drawn. `skin.json` (when linked) is re-bound the
  same way (rows of removed gaussians dropped, rows re-packed in checksum order); an object's
  own skin is not carried into its tileset yet.
- **The fill** (`teacher_fill.fill_hole`, the drop test for a region gone for good): the
  surface the object stood on is a plane fitted to the scan around its footprint (2 half
  sizes out, no higher than its lower quarter; the farthest fifth dropped per round); views
  look down on where it stood from the observers' side, at least 35° steep, the object half
  the frame; the mask is the pixels of its silhouette that now **see through** that plane
  over its footprint (nothing there, or what is there lies 5% beyond it); the empty pixels
  around (a scan's edge) are painted from the covered ones before the filler sees them;
  filled, gated as every fill is, lifted onto the plane, opacity by distance from measured
  pixels scaled to the hole (its middle is not left transparent for being wide), optionally
  distilled; a held-out view scores how much of what sees through the scan covers before and
  after. Packaged as an inferred layer (`extras.evidence.hole` = the instance id) and
  declared in `inferredLayers`, so the viewer labels it inferred like any other.
- **`--absorb`**: segmentation leaves pieces of an object under other ids (the pumpkin: 40
  small instances, most of them top-level). With it, every other id with 80% of its leaf
  gaussians inside the object's box (its 3rd-97th percentiles, padded 5%) and at most a fifth
  of its size goes with it. Geometry only.

**In the viewer** (`cesium/splitObjects.ts`, `lib/sceneObjects.ts`, `state/sceneObjects.ts`):
each declared object is loaded beside the scan (as inferred layers are), shown while the scan
is shown, faded by its own view cones, and drawn under the model matrix `P · S · L · S⁻¹`
(`P` the scan's model matrix, `S` its root transform, `L = T(origin + t) R T(−origin)`), so
at rest it is exactly where it was measured and a pose moves it in the scan's frame
wherever the scan itself was placed. A pose set in the store (`useSceneObjects.setPose(asset,
instance, pose)`; `null` for the declared one) overrides the declared pose: what C3's
telemetry driver will call. Hide and highlight: `attachInstances(..., { follower: true })`
installs the same hooks on the object's primitive from the scan's `instances.json` and the
scan's store entry, so an object hides, highlights and dims with the ids it carries; search
and the table stay the scan's; flying to a moved instance follows its pose
(`setInstanceOffset`). Not yet: the dedicated renderers (PlayCanvas, Spark) draw the scan
without its split objects, and collision still has the object at rest.

**Validation** (`tests/test_split_objects.py`, the yard with the lawn under one shrub taken
away, packed in 6000-gaussian tiles, instances from its labels; CPU, Telea): every leaf
gaussian is in the scan or the object exactly once (the SPZ records compare equal as a
multiset), each id's count is conserved, every drawn tile and the object tile are bound with
runs of their length and nothing stale is left, untouched tiles are byte-identical, scene and
object at rest render as the scan (≤ 1/255), the shrub's footprint, which the held-out view
sees through (6% covered), is 95% covered after the fill and lifted to within 0.25 m of the
ground; a second run writes the same bytes, the input is untouched, and a split scan splits
again. The committed yard's skins re-bind (rows of what stayed equal, the shrub's left).

**On a real scan** (the pumpkin, `fill.py --jobs split:pumpkin`, WORLD_MODEL_RUNBOOK.md §7):
`--ids 3 --absorb` takes the red pumpkin (27,319 gaussians, 40 fragment ids absorbed) out
cleanly; the hole, 0.4% covered from a held-out view, is 99% covered after the fill with
every filler tried, but NVIDIA Fixer (t50-t250) only cleans what it is shown: inside the
hole it keeps the rough Telea fill's flat colour. A generative inpainter is the next filler
to try for holes.

### Fixture and browser checks

`data/tiles/synthetic-yard/skin/` is the yard's tree (instance 1), a snag (9) and two shrubs
(10, 12) skinned (`skin_scene.py ... --only 1,9,10,12`), beside `instances/`;
`test_skin_scene.py` rebuilds it and compares. `apps/web/e2e/skin.spec.ts` drives it in a real
CesiumJS (`src/dev/skinHarness.ts`): the driven tree's pixels move while an unskinned tree and
an undriven skinned shrub do not, rest is the measured frame pixel for pixel, the constant
handle lifts a shrub whole, a hidden object stays hidden while it moves, and a shrub scaled up
stays filled only with the covariance following. `apps/web/e2e/wind.spec.ts` blows the wind
over it (with `skin/materials.json`): the tree's pixels keep changing while its base, an
unskinned tree and the undriven shrub stay still; the same clock steps give the same frame;
calm is the measured frame, pixel for pixel.

`data/tiles/synthetic-yard/telemetry/telemetry.json` binds two synthetic loops: the box
building (8, unskinned: the rigid part) round a 2.5 m circle, readings in ECEF at 2 Hz with
delay and jitter; the shrub 10 (skinned and wind-swayed by its material) round a 0.6 m circle,
geodetic at 5 Hz, set to freeze. `apps/web/e2e/telemetry.spec.ts` drives them in a real
CesiumJS (`skinHarness.ts`, a harness clock): the pose shown is the path at the playout time
(within 2 cm), the building's pixels leave its place and arrive where the path is while
unbound objects stay still, a silent source holds, fades and returns the measured frame
pixel for pixel, the frozen shrub holds, and with the wind on the bound shrub keeps its elastic
handles at rest while the tree sways; the same clock gives the same frame.

`data/tiles/synthetic-yard/instances/` is the committed yard segmented against its own labels
(`segment_scene.py ... --truth labels.json --tile-gaussians 6000`). It sits beside `splat/`,
not in it, so the yard tiles stay byte-identical to what the packer writes; the e2e links it
from the root's extras at request time. `apps/web/e2e/instances.spec.ts` drives the hooks in a
real CesiumJS (`src/dev/instancesHarness.ts`): hide, hide everything, highlight with and
without dimming, both primitive modes, and composition with the view cones.
`e2e/instancesScan.spec.ts` runs the same steps on any segmented scan
(`INSTANCES_SCAN_DIR=...`) and saves screenshots.

## 5. Storage by behaviour

| Behaviour                                   | Storage                             | Why                                                                                                                                                     |
| ------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| static                                      | spatial tiles                       | most splats; LOD stays efficient                                                                                                                        |
| in-place (plants, flags, water)             | spatial tiles + id (+ skin weights) | stays inside its tile; pad bounds by maximum displacement                                                                                               |
| movable (vehicles, robots, people, animals) | own object tileset, own frame       | a moving object leaves its spatial tile's bounds (culling breaks); in its own tileset it moves by one matrix. The hole it leaves is filled by Teacher B |

Segmentation writes the ids and the table; `split_objects.py` (§4, "Split objects") moves
the movable ones into object tilesets and fills their holes, as a separate step on a copy.

**The behaviour rule needs a size term (A6).** On the pumpkin scan both pumpkins scored
vegetation 0.85-0.96, movable 0.40-0.47 against static 0.43-0.52, so `in-place` took them,
though nothing roots them; only fragments (`movable` 0.53-0.65) read movable. Proposed
(`split_objects.py --select loose`, not yet in `segment_scene.BEHAVIOUR_RULE`): movable also
when `movable >= 0.4`, `movable >= static - 0.1` and the object is compact (bounds' largest
side at most 0.4 of the scan's), whatever its vegetation. On the pumpkin it picks both
pumpkins (2, 3) and four fragments, and not the ground patches (5: 3.8 × 5.2 m) or the dirt
(`movable` 0.11). Folding it into the rule rewrites every published `instances.json`'s
behaviours, so it waits for a re-segmentation.

## 6. Order

See [LIVING_PLAN.md](LIVING_PLAN.md).

## 7. Open questions

- **Skin method, decided 2026-10-01 (step B1):** Kaolin Simplicits with the FreeForm/RKPM
  basis. On the synthetic tree (CPU) it gave a plausible pinned sway (crown 0.51 m peak,
  swings back) and the least tearing (edge stretch p99 1.11 vs PhysSkin's 1.25, max 1.35 vs
  4.30). PhysSkin: no licence; Michelangelo encoder GPL-3.0, weights trained on non-commercial
  ShapeNet; folded in 2 of 5 deformation trials. Pip Kaolin 0.18 lacks RKPM (vendor it from
  master until a release has it).
- Browser runtime: for wind (small strain), linear/modal dynamics with a prefactored
  `(M/h² + K)`, about 40k flops per object per frame: tens of objects are trivial in JS.
  Full Neo-Hookean Newton only with few cubature points (Q ≈ 200–300, m ≈ 8–10).
- Covariances follow the skin's Jacobian, which includes the weights' gradients. **Decided
  (B3): drop that term** and draw covariances through `I + Σ w_j A_j`; on the tree it is at
  most 0.14 (against 1) for strong-wind amplitudes and exactly 0 for the constant handle.
  Storing ∇w would cost 3·(m−1) more numbers per splat. Revisit if a driver pushes handles
  past their `support` radius.
- SAM 3 / some lifting methods carry their own licences; SAM 2 is Apache-2.0.
