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
   the Modal call's request holds (sized from the scan: "What a run reserves", below), not
   a fixed count, since the masks rather than the renders set the pace. On a GPU the image
   the mask model sees is rasterized by gsplat
   (`--renderer gsplat`, as a viewer draws it; the per-pixel splat ids still come from the
   CPU's samples of the same camera), without the floaters larger than `--max-scale-m`
   (0.5 m). Then **coverage rounds**
   (`coverage_views`, two by default): after a lift, views are aimed at what is still
   without an instance -- unassigned splats binned in 3D, each target seen by two obliques
   and once from eye height (into a canopy) -- and everything is lifted again.
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
   The crops (`DESCRIBE_KINDS`) are square, from the CPU's point samples: uncut (box
   padded), and alone (the rest black, and grey); each kind is embedded on its own and the
   kinds averaged. Measured on the pumpkin and the camp (`describe_variants`, 2026-10-03):
   gsplat's images, dimmed context, wide crops and portraits of an instance's own splats all
   read worse to SigLIP (a conifer's parts as "bush", crops on grey as "plume" or "map").
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
7. **Categories.** The 1,300 labels are too fine to act on (the camp lists 650 "forest floor"
   pieces and 238 "bush"es), so each label also belongs to one of 27 broad scene categories
   (Trees, Shrubs & bushes, Grass & ground cover, Ground & soil, Water, Buildings, Walls &
   fences, Paths & roads, Vehicles, People, Animals, Furniture, ..., Other). The mapping is
   data, `tools/captures/data/categories.json`, made by the same SigLIP 2 text encoder: each
   label's tag prompt goes to the nearest category (a few phrasings each, averaged) by cosine,
   and the outliers inspection found are corrected in `scene_categories.OVERRIDES`
   (`python scene_categories.py --review` lists every label with its nearest three). It is
   general, not per scan; the viewer bundles the same file.

   An instance's category is the one its tags vote for (each tag's score added to its label's
   category); an instance without tags takes what most of its tagged siblings are (by splats:
   a coarse parent is often a mixed region -- on the pumpkin scan a 6 m "pumpkin" instance holds
   the hay around the pumpkins, and its untagged parts are hay like their tagged siblings, not
   pumpkin), else its nearest tagged ancestor's (a part is what it is part of), else the
   category most of the splats below it are in, else (a fragment no crop showed, with no
   tagged relative) the category of the smallest categorised instance whose box holds its
   centre, else Other. An **object** is an instance whose parent is in another category (or that has none), with
   every descendant reached through its own category: a category is the union of its objects,
   so hiding Ground & soil hides the ground and its untagged bits, not the trees a ground
   region contains. An object is named by its best tag of its own category, else by its
   category ("Trees 3"), never by an id.

   What categories cannot fix is an instance that mixes two things. Measured on the published
   pumpkin (leaf splats coloured orange, by the instance they carry): 78% are in "Fruit,
   vegetables & crops", 21% in instances SigLIP described as dirt or bush -- parts of a
   pumpkin's crop that are half pumpkin, half hay (instance 279: 43% orange, tags dirt 0.06,
   bush 0.06, pumpkin 0.03) -- and the red pumpkin is instance 4, whose crop SigLIP read as
   dirt / nest / moss (0.10, 0.08, 0.05), plus 3,900 splats the segmentation left without an
   instance. Classifying each instance's embedding against the category prompts directly
   (instead of through its top five labels) is no better there (instance 4: ground 0.20,
   produce 0.19). Hiding "pumpkins" exactly needs finer segmentation, not another rule here.

   **v2 (published 2026-10-03, segment.yml run 37100026825).** The category of a described
   instance is now `segment_scene`'s own: per crop kind, its 10 best labels' probabilities
   summed per category; averaged over kinds; then mixed 1:1 with its parent's (a part is
   seen with what it is part of). The zero-shot head over the category prompts was measured
   and is weighed 0 (it pulled crowns to Shrubs and hay to Sky). Before -> after:
   splats without an instance 8.3% -> 1.7% on the camp (rim 54% -> 12%), 2.4% -> 0.2% on
   the pumpkin, 3.1% -> 0.3% on the spool; the red pumpkin 57% -> 98% in Fruit, vegetables
   & crops, orange splats 75% -> 89%, hay 65% -> 22% in it; the camp's canopy (over 3 m)
   21% -> 28% Trees, 13% -> 12% Shrubs. Worse: the camp now has 6% Household and 4% Sky
   (labels such as "map" and "plume" on crowns), and only 4.3k of 28.4k instances are
   described.

**What a run reserves** (`infra/modal/segment.py`, `sizing`). Each scan is sized before it
is spawned, from its tileset.json (every tile names its gaussians). An L4; 6 cores whatever
the scan -- 4 render processes, enough to keep SAM 2.1's masks fed (v1 on the camp: 2.8 s
of masks against ~8 s of rendering a view, so 32 cores were only 14% faster than 8: 1,217 s
for $1.04 against 1,416 s for $0.56), and 2 for the process that rasterizes, masks and
votes; and memory for what that process holds at its peak, the end of the last coverage
round: 4 GiB of process and models, 256 bytes a gaussian (the scan in float64 and the copy
its views are drawn from), ~3.8 MB for each of up to 672 views kept for describing, and
2.5 GB for each render process. Both go to Modal as (request, limit), the limit 1.5x the
cores and 2x the memory: Modal bills max(request, used), so an estimate that is low costs a
little more, or throttles, instead of failing; and the render processes follow the request,
never the limit. Every run writes its peaks into its summary.json (`usage`: the main
process's and a render process's peak memory, the container's, CPU seconds, cores busy, the
GPU's busy share) beside the estimate (`sizing`), and prints them in one line, so the
estimate's constants (in segment.py, with the runs they come from) can be tuned.
**Estimated, not measured** -- v2's memory has not been measured yet, and the times are
guesses (its camp took 2,351 s on 32 cores and 96 GiB, about $2.00):

| scan    | leaf gaussians | request (limit)          | ~time  | ~$ a run |
| ------- | -------------- | ------------------------ | ------ | -------- |
| spool   | 153,566        | 6 cores (9), 16 GiB (32) | 15 min | 0.30     |
| pumpkin | 387,813        | 6 cores (9), 16 GiB (32) | 15 min | 0.30     |
| camp    | 22,577,243     | 6 cores (9), 22 GiB (44) | 45 min | 0.94     |

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
      "views": 14,                   // how many views it was seen in
      "category": "vehicles"         // optional (§3 step 7); the viewer derives it if absent
    }
  ],
  "tiles": { "<tile checksum>": [id, count, id, count, ...] },  // per tile, RLE, tile order
  "tilesEncoding": "rle; a leaf splat carries its instance; a merged splat the instance most of the 8 leaf splats nearest to it carry"
}
```

- `tiles` uses the same checksum keys and run-length encoding as `plants.json`
  (`scene_plants.plant_binding`), so the viewer's existing per-tile binding code applies.
- Ids are leaf-level (the finest instance). The hierarchy is walked through `parent`.
- A coarse tile's merged splat takes the id most of the leaf splats nearest to it carry
  (`rebind_instances.py`). The first rule gave it an id only when all the leaf splats merged
  into it shared one; leaf instances are an object's parts, so few merged splats qualified --
  on the published camp 46% of the non-leaf splats of the fourth level of detail and 22% of
  the fifth carried 0, now 32% and 11% (8.7% to 0.9% at the sixth); the pumpkin's unassigned
  share halves (2.4% to 1.2%). Scans published under the old rule are fixed by
  `python rebind_instances.py TILES_DIR`, which rewrites only their instances.json from the
  published tiles.
- What the segmentation never saw keeps 0 and can be neither hidden nor highlighted: on the
  camp that is the sparse rim of the capture (its shallow leaf tiles, 2.7 M splats, are 36-94%
  unassigned), which a view from above the whole camp is mostly made of. (Before v2's
  coverage rounds; now 12% of the rim and 1.7% of the camp.)
- `instances.emb`: `float16`, `count × dim`, row `k` is instance id `k + 1`, L2-normalised; an
  instance that was not described (no `tags`) has a zero row.

### Root extras

The measured tileset's `root.extras.instances = { "uri": "instances.json", "count": n }`, so
the viewer finds it without probing (the same pattern as `viewCones` and `inferredLayers`).
Other methods' objects, fills and skins for the same scan are declared beside them in
`extras.variants` ("Variants", below).

**In the viewer**, hide and highlight work under every splat renderer, from the same per-tile
binding: CesiumJS's own primitive (`cesium/splatInstances.ts`: the visibility chain and the
colour hook), and the dedicated renderers laid over the globe (`cesium/scanView/scanInstances.ts`).
PlayCanvas digests each tile's positions in its decode worker, keeps the ids beside the tile's
splats as one more resource stream (in PlayCanvas's Morton order), and applies the rule in a
work-buffer modifier; Spark digests the tile's SPZ centres and applies it in an object
modifier (a dyno). The rule is CesiumJS's: a hidden splat has no opacity, a highlighted one
is pulled toward the tint, and while anything is highlighted the rest are dimmed. The store
(`state/instances.ts`) holds exact id sets -- a category's or an object's members -- and every
renderer applies them id for id, so hiding one category never takes another's instances with
it.

A scan whose package also has PlayCanvas's own streamed format (`sog/lod-meta.json`) is drawn
from its 3D Tiles instead when it has objects (`extras.instances`): the native package carries
no tile checksums, so its splats have no object ids: before this, the published camp (which
has both) streamed natively under the default renderer, and hide and highlight did nothing
there but a note offering CesiumJS. A scan with a native package and no objects still
streams natively. The rule is one switch, `NATIVE_SOG_FOR_SCANS_WITH_OBJECTS`
(`cesium/scanView/ScanRendererHost.ts`, off), to be flipped to measure what the native package
would gain such a scan.

**The objects panel** (`features/sites/InstanceSearch.tsx`), from the "Objects" button beside
the representation switcher:

- a search box ("Search objects");
- the scan's categories, largest share of the scan first, each with its colour, name, number
  of objects and an eye that hides or shows the whole category; clicking a row highlights the
  category (the rest dims), clicking it again clears it;
- a chevron (or the right arrow key) opens a category onto its objects, fifty at a time, each
  with its own eye; clicking an object highlights it and flies to it;
- while searching, the matching objects are listed the same way, grouped by category, with
  "Hide all" and "Show only" for every match. Words match tags and category names ("trees",
  "water"); a typed property filter (`vegetation > 0.5`, `behaviour:movable`) still works but
  has no buttons;
- one "Reset" whenever anything is hidden or highlighted, with what is hidden in words.
- every change of what is hidden (an eye, Hide all, Show only, Reset, and the selection card's
  Hide and Show only) is one step of the app's undo (`Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y`;
  `state/history.ts`, docs/MISSION_CONTROL.md "Undo and redo"): undo puts back the scan's
  hidden set exactly, a category partly hidden included, and says what it took back
  ("Undid: Hide Pumpkin 3"). The highlight and the search's words are not undone: they are a
  selection, not a change to the scan. The steps are the site's: they are dropped when another
  site becomes active, and a step of a scan whose table was loaded again no longer applies.
- selecting in the scene (a click, the cycle keys or the brush; the HUD's selection card,
  `ObjectCard.tsx`) opens that object's category and marks it; clicking an object in the panel
  selects it in the scene, so the card offers its actions, and flies to it as the card's Fly to
  does. The card names a selection as the panel does (its top tag, else its category, never an
  id).
- the panel is a popover from the strip, marked `data-hud-popover` like the HUD's other
  popovers; Escape closes it (and only it).

The property scores and behaviours are not shown: they drive physics, not browsing (SigLIP's
"vegetation" scored the pumpkins 0.88 -- true of a gourd, and confusing in a list).

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
The sorter still orders by rest positions. PlayCanvas and Spark apply the same skins (C3 "Renderers"). **Drivers** (C1 wind, C3 telemetry) call
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

**Renderers.** All three. The drivers are shared: wind and telemetry write the skin part
(`setInstanceHandles`) and the rigid part (`setInstanceMotion`) whatever renderer draws the
scan (CesiumJS keeps the tileset loaded, hidden, under every renderer), and those parts keep
what was set (`drivenSkins`, `instanceMotions`, each with a `motionVersion`). Under PlayCanvas
and Spark a `ScanMotionLink` (`cesium/scanView/scanMotion.ts`) reads them once a frame and hands
the back-end one `ScanMotion` when anything changed: the skin handles folded into the scan's
frame (`[A | t − A·o]`, 64 texels a skin, the layout CesiumJS uploads), a slot per instance
id and `[R − I | t]` per slot, and which skins and instances changed. Per splat, the skin id
and `skin.bin` row are bound by tile checksum exactly as the instance ids are (PlayCanvas: the
`splatSkin` R32U and `splatWeights` RGBA32U resource streams, in its Morton order, only on
tiles with skinned splats; Spark: per-tile R32UI / RGBA32UI textures). One shader
(`SCAN_MOTION_GLSL`) serves both: PlayCanvas's work-buffer modifier, Spark's world modifier
(a dyno). Neither takes a covariance, only a rotation and scales, so `J·Σ·Jᵀ` with `J = I + Σ
w_j A_j` (+ the rigid part's) is decomposed again (Jacobi, exact up to float precision; a
rotation `J` just turns the splat), the same Jacobian CesiumJS draws through. Only the tiles
holding a changed skin or instance are recopied (PlayCanvas) or regenerated (Spark). Sorting:
Spark sorts its generated, moved splats; PlayCanvas sorts on the CPU from each resource's
`centers`, so a tile holding a rigidly moved object has its centres moved the same way and
`centersVersion` bumped (at most every 100 ms while it moves, at once at rest); a skin's
sway sorts at rest, as under CesiumJS. Skins and telemetry bind to objects, and a scan with
objects is always streamed from its 3D Tiles (C4 above), so PlayCanvas's own package never
has motion to lose. A back-end that cannot move objects (no `setMotion`, or a scan streamed
without checksums) still reports it: one line in the objects panel and under the wind
control, "Wind and telemetry need the Cesium renderer", with the reason as its tooltip and a
"Use Cesium" button (`motionGaps` in `state/instances.ts`, `MotionRendererNote`). Under the
PlayCanvas WebGPU trial a scan with objects or motion is drawn with WebGL2: these modifiers are
GLSL only for now (docs/WEBGPU_TRIAL.md). The overlay draws only when something changes
(`overlayFrames.ts`): a new motion handed over, or a split object moved, is such a change, and
the drivers ask the globe for a frame as they move something, so a still scene costs no
frames. Scene selection picks a split object where its pose puts it (`pickTiles` returns the
placed positions, made when a pick asks); a rigidly or skin-moved object is picked at rest.
PlayCanvas's tiles arrive in Morton order (its tile worker), and `pickTiles` puts each tile
back in its own order, the order `instances.json` lists ids by.

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
  A generative filler (`world_model_client:GenerativeFiller`, e.g. `?model=qwen&chain=1`)
  is told what is around the hole (`hole_context`: instances.json tags around the footprint
  as the prompt, the object's own labels as the negative), repaints the void with the hole
  so only measured pixels are its context, and with `chain` fills the views in turn, each
  shown the earlier views' fill (WORLD_MODEL_RUNBOOK.md §8). The held-out strip's last
  panel shows the object moved aside, the fill showing.
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
(`setInstanceOffset`). Under PlayCanvas and Spark (`cesium/scanView/scanObjects.ts`) each
object's tile is loaded by the back-end like a scan tile (so its ids bind by checksum and hide,
highlight and rigid motion act on it), drawn beside the scan's tiles under `L · S⁻¹ · O` (its
pose about `origin`, times its root transform `O` in the scan's frame), placed again whenever
the store's pose changes. Not yet: collision
still has the object at rest.

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

`apps/web/e2e/motionRenderers.spec.ts` runs the same checks under PlayCanvas and Spark
(`skinHarness.ts` with `renderer`, the scan drawn by the dedicated renderer over a hidden
CesiumJS tileset, frames composited): skins move their objects and nothing else, hidden stays
hidden, the constant handle lifts a shrub, a scaled shrub stays filled only with the covariance
following, the wind sways the tree but not its base, replays exactly and calms to the measured
frame, telemetry moves the building along its path and fades it back to the measured frame
exactly, and a split object (made at request time from a leaf tile) is drawn at its pose and
back; with PlayCanvas's own package beside the tiles, a scan with objects is still streamed
from its tiles and moves (no motion gap).

`data/tiles/synthetic-yard/instances/` is the committed yard segmented against its own labels
(`segment_scene.py ... --truth labels.json --tile-gaussians 6000`). It sits beside `splat/`,
not in it, so the yard tiles stay byte-identical to what the packer writes; the e2e links it
from the root's extras at request time. `apps/web/e2e/instances.spec.ts` drives the hooks in a
real CesiumJS (`src/dev/instancesHarness.ts`): hide, hide everything, highlight with and
without dimming, both primitive modes, and composition with the view cones.
`e2e/instancesScan.spec.ts` runs the same steps on any segmented scan
(`INSTANCES_SCAN_DIR=...`) and saves screenshots.

### Selecting in the scene

The viewer selects objects where they are drawn, not only from the panel. It works the same way
under every splat renderer (PlayCanvas, Spark, CesiumJS) because picking runs on the CPU over
the tiles the renderer draws now:

- **Pick sources** (`cesium/sceneSelect/pickSources.ts`). Per asset, the renderer drawing the
  scan provides its drawn tiles as `PickTile`s. A `PickTile` holds each tile's own positions in
  the scan's frame (the order and frame its checksum, and so its ids, are keyed by), each
  splat's largest axis and its opacity.
  - PlayCanvas keeps these from the worker's decode, before its Morton reorder.
  - Spark reads them from the SPZ (`spzPickData`).
  - CesiumJS un-bakes the committed snapshot's tiles (`cesiumPickSource.ts`).
  - A dedicated renderer registers above CesiumJS, so whichever draws the scan is the one
    picked from.
- **Click** (`lib/splatPick.ts`). A ray through the cursor tests the splats as soft spheres. It
  uses a per-tile index: Morton-ordered blocks of 64 splats, one box each. The hits are
  composited front to back (`T · α`), so splats behind a solid surface count for almost
  nothing. Hidden objects' splats let the ray through. The hit splats' leaf ids give the
  candidates (`lib/sceneSelect.ts`):
  - the strongest leaf's chain, from the leaf up to the top level;
  - then the other instances hit near the front.

  A click chooses the whole object first, and each click again on it one level finer, toward
  the part hit (`drillIndex`): a cable spool, then a plank of it. With nothing selected, or a
  selection the hit's chain does not hold (another object, or one met nearby), the click
  chooses the top of the chain. A top-level instance with more than half the scan's splats
  (with everything below it) is the scene, not an object, and is passed over for its child on
  the chain. With the selection in the chain, the click chooses the level below it; at the leaf
  it stays. Drilling goes on only within the scan already selected. A click within 400 ms and
  6 px of the last is the same click, so a double-click selects one level, not two.

- **Cycling**. `[` / `]`, Alt+wheel, the wheel over the card's arrows, or the arrows
  themselves move between candidates: `[` one level finer, `]` one level up, then on to the
  instances met nearby (and both round). The card counts the chain's levels from the top
  (`levelText`): "1 of 3" is the whole object, "3 of 3" its finest part, "+2 nearby" says
  other instances were met, and "Nearby 1 of 2" is one of them. Tab / Shift+Tab cycle too,
  but only while the map (the canvas) or the card itself has focus: a hit gives the map the
  keyboard, so Tab cycles right after a click; from anywhere else, the page's body included,
  Tab moves focus as it always does. Esc steps back: the brush away first, then the
  selection.
- **The card** (`features/sites/ObjectCard.tsx`). The HUD's one selection card
  (`features/mission/SelectionCard`) shows an object as it shows a machine or a zone, in the
  right dock (a bottom sheet on a phone): the name (top tag, else the category, never an id)
  and the category, "◀ 1 of 3 ▶", **Hide**, **Show only**, **Fly to**, the brush and **Clear**
  (its close button). A combination is named by its members and what holds them, "Top flange +
  drum + Bottom flange (of Spool)" (`combinationLabel`; with more than three members, a name
  said twice or too long a name, "4 parts of Spool" or "4 objects"), says its overlap with the
  painted area, and offers **Save as object**. One selection at a time: picking an object clears a machine or zone, and
  the reverse (`state/oneSelection.ts`). The selection is the objects store's highlight: the
  controller writes it through `useInstances.highlight`, expanded to descendants. **Fly to**
  goes through the app's camera controller (`CameraController.flyToObject`): the pace and
  range of every fly-to, the dedicated renderer's destination prefetch, and a site flight
  still settling gives the camera up to it.
- **Brush** (`B`, bound in the app's hotkey registry, or the card's brush;
  `lib/splatPaint.ts`). The camera holds still while you paint. Every drawn splat is projected
  once, and 3 px cells keep the nearest depth of their fairly solid splats (an approximation
  of the rendered depth). A splat counts as painted when it is near its cell's front and the
  cell is under a stroke. Shift adds to the painted area, Alt takes away, and a plain stroke
  starts again. A touch screen has none of those keys: there the card offers **New / Add /
  Remove** for what a stroke does, and a brush size in place of Alt+wheel. The match is the
  combination of instances, at whatever levels fit, whose union has the best intersection over
  union with the painted area (see **Combinations** below): one instance, or several. The IoU
  is weighted by opacity and counts only visible splats, so an object's hidden back does not
  count against it. While the stroke is painted, its best match so far is highlighted (at most
  every 100 ms) and the card says its overlap, so you can stop once the right objects light
  up; it is selected when the stroke ends. Matching every visible splat at every move would be
  too slow (the camp has 22.6 M), so the view is indexed once when it is projected
  (`paintIndex`): per 3 px cell, the visible splats' leaf ids and weights, and per instance its
  visible weight rolled up its chain and its parent. A match then walks only the painted cells
  and the instances they hold (`paintSumsIndexed`), and gives the same sums, so the same
  answer, as matching every splat (`paintSums`).
- **Combinations** (`bestSet`). Painting is for selecting the right combination or level of
  the hierarchy: painted over a spool's bottom flange and its top flange and drum, it selects
  both, not one then the other. The members are instances from disjoint subtrees (none holds
  another), so the union's painted and visible weights are the members' sums and a set's IoU
  is `Σinter / (painted + Σvisible − Σinter)`, from the per-instance sums a match gathers
  anyway.
  - The best set is found exactly, not greedily. A greedy search starts from the best single
    instance and can never trade a parent for its children (the spool, which also holds ground
    nobody painted, against the two flanges that were). Dinkelbach's method from the best
    single instance's IoU λ reads the antichain maximising `Σ (inter − λ·(visible − inter))`
    off the hierarchy bottom-up (an instance, or the best of its children, whichever is more),
    takes its IoU as the next λ, and stops when that no longer rises: three or four passes
    over the instances met.
  - Then the answer is made simple at little cost. Members that add little go, the least
    first, while the set's IoU stays within 2% of the best's (`PAINT_SET_GAIN`): a sliver of a
    neighbour under the brush's edge is not a part. Then members that share an ancestor become
    that ancestor, the deepest first, while the IoU stays within 3% (`PAINT_PARENT_SLACK`): the
    whole spool when it is as good as its parts, its parts when the spool also holds unpainted
    ground. One instance stays one instance. Both are shares of the IoU, not differences: in a
    loosely painted area (IoU 0.1) a member that is half of it adds only 0.01, and stays.
  - The members are listed largest on screen first (ascending ids among equals), the same for
    the same sums, so the same painted area gives the same set however its splats were
    gathered.
  - While a stroke is painted, the match shown is kept until another's IoU is better by more
    than 3% (`PAINT_STEADY`, `steadySet`), so two near-equal answers do not take turns at every
    preview; the stroke's end goes by the same rule, so what is lit is what is selected.
  - Cost, on a camp-sized synthetic view (2.16 M visible splats, 48,000 instances, 1440 × 900
    px): an 18 px stroke across a third of the screen meets 1,003 instances, and the best set
    takes 1.3–2.4 ms against 0.9–1.1 ms for the best single instance; the largest brush (120 px)
    scrubbed over half the screen meets 25,488, and takes about 13 ms against 8 ms.
  - A combination is selected as one (`state/sceneSelect.ts` `selectSet`): its first member is
    candidate 0 and stands for the combination, and what its members are parts of together
    (`commonChain`) are the coarser candidates, so `]` goes up to the spool and `[` back.
    `selectedIds` is what is selected, whichever it is, and the highlight, **Hide**, **Show
    only** and **Fly to** (the sphere around the members') act on all of it. A click is
    unchanged: the whole object first, again for its parts; a click from a combination starts
    at the whole object.
- **Painted objects** (`lib/customSets.ts`). When the best set's IoU is below 0.5, the card
  offers **Use painted area**. This keeps the exact splats as an object of the viewer's own:
  - **Save as object** keeps a combination the same way (`setFromInstances`): as every splat
    its members carry in every tile of the scan, at every level of detail, not only those drawn
    or painted. It is kept as splats, not as the members' ids: the format draws a set by its
    splats, and splats stay the same object if the scan is segmented again, where ids would
    name others.
  - It is stored per scan in this browser (`localStorage`,
    `hexapod.customObjects.<asset>`) as `{ key, name, tiles: { checksum: [start, length, …] },
splats, bounds }`.
  - It is drawn through the same pipeline: `withCustomSets` gives each set an id past the
    file's (`maxId + 1`, …), relabels its splats in the tile runs, and appends it as a
    top-level instance. Every renderer reads ids by checksum from that document
    (`paintedDocOf`, `SplatInstances.setDoc`), so it hides and highlights like any instance.
  - While the set exists, its splats no longer carry their segmented id.
  - Making one and deleting one are steps of the app's undo (`state/sceneSelect.ts`): undo
    puts the scan's painted objects back as they were, stored again, so a deleted one returns
    at its place and the later ones keep their ids; a selection of a painted object is cleared
    when they change under it.

The controller is `cesium/sceneSelect/SceneSelectController.ts`, and its state is in
`state/sceneSelect.ts`. Unit tests are in `__tests__/sceneSelect.test.ts` and
`__tests__/selectionCard.test.tsx` (the card, its keys, one selection at a time), and
`e2e/sceneSelect.spec.ts` runs on the yard (`src/dev/sceneSelectHarness.ts`, which mounts the
card where the app's dock puts it) under PlayCanvas, Spark and CesiumJS. The e2e checks that:

- clicking the tree's crown selects the tree or a part of it, and gives the map the keyboard;
- `]` goes to the parent and `[` comes back; Tab and Shift+Tab do the same from the map, and
  from the page's body Tab moves focus instead;
- painting over shrub 10 selects that shrub;
- **Hide** in the card removes it from the frame;
- one stroke across two walls of the shed (8) selects both walls together, not the shed (whose
  roof was not painted) and not one wall; the card names the combination, **Show only** leaves
  both walls (the roof goes from the frame) and **Hide** hides both.

In the app, `e2e/app.spec.ts` ("a scan object in the selection card") checks the keys and the
card around a selection: `B` and `V` once each, one Escape one step, Tab on the body, an object
replacing a machine's card, and the touch screen's brush.

### Variants: other methods for the same scan (bake-offs)

A bake-off publishes other methods' objects, fills and skins for the **same measured splats**,
for the owner to switch between in the live app and judge by eye. Labels stay visible: it is
not blind. Each candidate's output goes beside the measured tiles under
`variants/<system>/<name>/` (`<system>` is `objects`, `fill` or `skins`), in today's formats --
an `instances.json` (above), a `skin.json` with its `skin.bin`, an inferred tileset whose root
carries `extras.evidence` -- and the measured tileset's root declares them all in
`extras.variants`:

```json
"variants": {
  "objects": [{"name": "ground-first", "label": "A · Ground first", "about": "One plain sentence on what this method does.", "instances": "variants/objects/ground-first/instances.json"}],
  "fill":    [{"name": "vace-14b", "label": "Wan2.1-VACE 14B", "about": "…", "inferredLayers": [{"uri": "variants/fill/vace-14b/tileset.json", "evidence": {"kind": "inferred", "filler": "wan2.1-vace-14b", "views": 0, "gaussians": 0, "meanConfidence": 0.0}}]}],
  "skins":   [{"name": "freeform", "label": "FreeForm (eigenmodes)", "about": "…", "skin": "variants/skins/freeform/skin.json"}]
}
```

- **Paths** are relative to the measured `tileset.json`, as `extras.instances`, `extras.skin`
  and `extras.inferredLayers` are, and resolve the same way (a signed URL's query is kept).
- **`name`** is unique within its system (a repeat keeps the first); **`label`** is what the
  viewer shows (the name when absent); **`about`** is one plain sentence, shown under the pick.
- **Today stays the default.** The scan's own `extras.instances`, `extras.skin` and
  `extras.inferredLayers` are "Today": a viewer who never picks a variant sees exactly what it
  saw before, and a system with no Today (no `extras.skin`, say) is "nothing" until a variant
  is picked.
- **Registering** adds or replaces the entry of the same `name` within its system, and never
  removes another system's entries or other variants.
- **Read defensively** (`apps/web/src/lib/variants.ts` `variantsOf`): an entry without a name,
  without its system's file, or with an `inferredLayers` entry that does not read is skipped;
  the rest of the list stands. An `inferredLayers: []` is a legitimate "no fill" method.
- **Ids.** A skins variant names instance ids (`skins[].instance`), and the wind reads each
  skin's object's properties from whichever objects are shown: a skins variant fitted on
  Today's objects moves the same objects under an objects variant only where the two files
  agree on those ids. Telemetry and `materials.json` name Today's ids likewise.
- **The API** (`apps/api/app/services/sidecars.py` `KINDS`) does not know `variants` yet: an
  attach carries it onto the same tiles as any unrecognised key, and a republish of new tiles
  drops it and flags the asset ("Unrecognised sidecar extras.variants needs re-attaching").

**In the viewer.** A scan that declares variants gets a **Methods** button beside the
representation switcher (`features/sites/CompareMethods.tsx`): a panel with one row per system
it offers -- Objects, Fill, Motion -- each with Today and the variants by label (a segmented
control while they fit one line, a list beyond), and the pick's `about` beneath. A pick is
kept per scan for the session (`state/variants.ts`, `sessionStorage`), and swaps what is drawn
in place, with no reload and the camera where it is, under every renderer:

- **Objects** (`cesium/splatInstances.ts` `attachInstances`): the variant's `instances.json`
  is loaded and swapped in where the old one was drawn -- the same hooks, a new table -- the
  scan's selection cleared (its ids were the old file's) and its painted objects
  (`lib/customSets.ts`, keyed by splat) drawn over the new file. The objects panel lists the
  new file's categories; PlayCanvas and Spark rebind their tiles' ids through the store
  (`scanView/scanInstances.ts`).
- **Fill** (`cesium/inferredLayers.ts`): the drawn layers are unloaded and the variant's
  loaded. Inferred layers are CesiumJS's under every renderer (below).
- **Motion** (`cesium/splatSkin.ts` `attachSkin`): the variant's skin replaces the skin part;
  the wind makes a driver for the new part and keeps blowing (`LivingSurveyManager`), and the
  overlay rebinds its tiles' skin weights (`scanView/scanMotion.ts`).

A pick whose files do not load says so in its row ("Did not load: …") and draws nothing for
that system rather than the previous pick. A scan with objects or skins variants counts as a
scan with objects or motion: the WebGPU trial draws it with WebGL2, and the overlay streams its
3D Tiles rather than a native package (`ScanRendererHost.declaresInstances`).

**Inferred style.** How inferred layers are drawn is a viewer's setting
(`inferredStyle`, kept on the device; hidden until chosen), beside the switcher as
**Show · Highlight · Hide**, with the one-line legend "Inferred: generated where no camera
saw. Not measured." while they are drawn. Highlight is the engine patch's colour hook on each
layer's own splat primitive (never the measured scan's): pulled toward purple
(`INFERRED_PURPLE`), hatched in 0.35 m bands across the layer, and a little see-through
(`INFERRED_HIGHLIGHT`). Inferred layers are drawn by CesiumJS whichever renderer draws the
measured splats: under PlayCanvas or Spark the scan's own tileset is hidden and drawn on the
overlay, while its layers stay on the globe's canvas under it, so where both cover a pixel the
measured splats are in front. Picking a fill while the style is Hide switches it to Show.

**Fixture and checks.** The synthetic yard has two variants per system
(`data/tiles/synthetic-yard/variants/`, written by `tools/captures/yard_variants.py`, declared
in its `variants.json` with paths relative to `splat/tileset.json`): objects `whole` (the 24
top-level objects, every part folded in, with categories) and `parts` (all 103, the trees' and
the shed's parts in categories of their own); fills `hedge` (beyond the west edge) and `mound`
(beyond the east edge); skins `tree` (the big tree only) and `small` (the snag and two
shrubs). `e2e/variants.spec.ts` (`src/dev/variantsHarness.ts`, the app's panels mounted beside
the scan) runs under PlayCanvas, Spark and CesiumJS and checks that picking an objects variant
changes what the objects panel lists, picking a fill draws its layer and not the other's,
Highlight turns the layer's pixels purple and changes no other pixel, Hide leaves the frame
the scan's own, and a skins pick replaces the skin; and that the panel fits a 400 px phone and
works from the keyboard. Unit tests: `__tests__/variants.test.ts` (the parser, the store, the
swaps), `__tests__/inferred.test.ts`, `__tests__/compareMethods.test.tsx`.

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

## 8. Publishing beside the tiles: one publisher

Everything in §4 is published beside a scan's tiles, and since the one-publisher change only
the API writes it there: `POST /api/v1/assets/{id}/sidecars` cuts a new generation holding
the current tiles, every sidecar they already have and the staged files, with the merged
root extras in a `tileset.json` written last, and repoints the asset under a row lock
(docs/DEPLOYMENT.md, "Sidecars: one publisher"; `apps/api/app/services/attach.py`). A
worker republish carries each sidecar of the live generation into the run's new one only
where it still holds for the new splats (`apps/api/app/worker/carry.py`), and flags the
asset for each kind it drops (`sidecarFlags` on the asset: "Objects need re-segmenting").

### What each kind depends on

Read off the code that writes each one; `apps/api/app/services/sidecars.py` (`KINDS`) is
the table the API and the worker use, and `tests/test_sidecar_attach.py` holds it to this.

| Kind             | Beside `tileset.json`                    | Declared by                                   | Written by                                              | Bound to                                                                                                                                                                                                                                                                  | Class             | On a republish of new tiles                                                                               | Flag when dropped                                      |
| ---------------- | ---------------------------------------- | --------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `instances`      | `instances.json`, `instances.emb`        | `extras.instances`                            | `segment_scene.py` (segment.yml, publish-instances.yml) | `tiles`: each tile's FNV-1a checksum of its decoded float32 positions (`synthetic_tree.checksum_positions`) to run-length ids in that tile's gaussian order; coarse tiles' ids by kNN from the leaf splats (`rebind_instances.py`), still keyed by those tiles' checksums | positions         | carried when every new tile's position checksum is in `instances.json`                                    | Objects need re-segmenting                             |
| `skin`           | `skin.json`, `skin.bin`                  | `extras.skin`                                 | `skin_scene.py`                                         | tile checksums to skin runs and rows of `skin.bin` in each tile's order; each skin moves an instance id                                                                                                                                                                   | positions         | carried when every new tile's position checksum is in `skin.json`                                         | Skins need refitting                                   |
| `materials`      | `materials.json`                         | `extras.materials`                            | the video teacher (C2)                                  | one record per `instances.json` id                                                                                                                                                                                                                                        | follows instances | carried exactly when `instances` is                                                                       | Materials need re-pointing at the new objects          |
| `telemetry`      | `telemetry.json`                         | `extras.telemetry`                            | deployment data (C3)                                    | bindings by `instances.json` id                                                                                                                                                                                                                                           | follows instances | carried exactly when `instances` is                                                                       | Telemetry bindings need re-pointing                    |
| `objects`        | `objects/<id>/…`, `fills/<id>/…`         | `extras.objects`, `extras.split`              | `split_objects.py` (C4)                                 | rewrites the scan's own tiles without the objects and binds the object tiles in `instances.json`. **Not attachable**: a split is a new tileset, not files beside one                                                                                                      | tile bytes        | carried only onto the same tiles                                                                          | Split objects need re-splitting                        |
| `collision`      | `collision.bin`                          | `extras.collision`                            | the packer (`convert`), or collision-backfill.yml       | `splat_tiles.collision_grid`: the solid cells of the kept splats (`opacity_min`) in the tileset's local ENU frame; the backfill refuses unless the leaves hold exactly the PLY's kept gaussians                                                                           | tile bytes        | the run's own replaces it; else dropped                                                                   | Collision needs a backfill                             |
| `viewCones`      | `viewcones.bin`                          | `extras.viewCones`                            | the packer, or `splat_tiles.py viewcones`               | per cell of the splats' grid, the directions it was seen from                                                                                                                                                                                                             | tile bytes        | the run's own replaces it; else dropped                                                                   | View cones need rebuilding                             |
| `inferredLayers` | `inferred/<name>/…`                      | `extras.inferredLayers` (`[{uri, evidence}]`) | `teacher_fill.py` (fill.yml, publish-fill.yml)          | nothing of the splats: a tileset of its own whose root transform is the scan's (publish-fill.yml checks it), drawn with the scan's model matrix. World-space in the scan's frame, no splat indices, no checksums                                                          | independent       | carried                                                                                                   | (only when it cannot be carried at all)                |
| `nativeLod`      | `sog/lod-meta.json`, `sog/*.webp`        | `extras.nativeLod` (optional: the web probes) | streamed-lod-backfill.yml (splat-transform)             | every leaf tile's SPZ merged, Morton-ordered and decimated: the splats themselves, re-encoded                                                                                                                                                                             | tile bytes        | carried only onto the same tiles                                                                          | Streamed LOD needs a backfill                          |
| `rig`            | `rig.json`, `motion.json`, `plants.json` | the asset's `renderConfig.rigUrl`             | living-plants.yml (`scene_plants.py`)                   | the rig is stamped with the published tiles' checksums (`rig_tiles.stamp`) and `plants.json` binds per tile checksum; the viewer refuses a tile the binding does not list                                                                                                 | positions         | carried when every new tile's position checksum is in `rig.json` and `plants.json`; else `rigUrl` cleared | Plants need re-rigging                                 |
| anything else    | —                                        | an extras key none of the above               | —                                                       | unknown                                                                                                                                                                                                                                                                   | —                 | carried onto the same tiles; else dropped                                                                 | Unrecognised sidecar extras.`<key>` needs re-attaching |

**"The same splats"** is checked, not assumed, and how depends on the class. A kind bound to
**positions** (objects, skins, the rig) holds on new tiles exactly when the viewer would draw
them with it: every tile of the new tileset must have a position checksum its binding lists
(`instances.json` and `skin.json` key their `tiles` by it, `rig.json` lists `tileChecksums`
and `plants.json` keys its `tiles`). The worker has the new tiles, so it computes each tile's
checksum with the function the bindings were written with (`apps/api/app/worker/positions.py`,
a transcription of `rig_tiles.tile_positions` and `synthetic_tree.checksum_positions`, held to
`checksum_vectors.json` and the fixture tree's stamped `rig.json`) and stops at the first tile
no binding lists: a re-pack that left every position where it was — another `--sh-degree`,
`ship_sh_degree` — keeps them (`data/tiles/synthetic-tree-sh` against `synthetic-tree-lod`
is the test), a new reconstruction or a Refine does not. A kind bound to the **splats' bytes**
(collision, view cones, `sog/`, a split) is carried only when the tileset without its root
extras (the tree, its bounds and errors, every content uri) and every tile's size and ETag
are equal (`sidecars.tiles_fingerprint`): nothing records what a grid was computed from, and
`sog/` is the old encoding itself, harmonics and all. Both err on the safe side — a kind
dropped by mistake is flagged and rebuilt, one kept by mistake hides and moves the wrong
splats. The attach's `basedOn` check is the byte test (the API does not download tiles):
sidecars computed on the legacy prefix may be attached to a generation an attach cut from
it, never to one a republish wrote.

**An attach replaces what it sends, and what was keyed by it** (`attach._plan`). A staged
file replaces the same path; under `inferred/<name>/` or `sog/` it replaces that whole
directory, and of any other kind it replaces the kind's whole file set: a new
`instances.json` without an `instances.emb` leaves the generation with no `instances.emb`,
since the old one's rows were the old ids. A sibling meant to stay is staged again; there is
no "keep" — one tool writes a kind's files together, and the API cannot tell an old one still
matches a new one. A kind's key set to `null` with none of its files staged removes the kind,
files and key. And the class "follows instances" holds on an attach as on a republish: an
attach that replaces `instances` (stages one of its files, or sets or removes
`extras.instances`) drops the `skin`, `materials` and `telemetry`, files and keys, and flags
each on the asset ("Materials need re-pointing at the new objects", "Skins need refitting"; reason "it names instances ids, and
an attach replaced instances…", no `jobId`) — unless the same request sends them too, which
is the caller's word that they name the new ids. The response lists them in `dropped`, and
every path of the previous generation the new one lacks in `removed`. An attach of anything
else (a fill, a grid, the streamed LOD, a rig) leaves objects, the skin, materials and
telemetry as they were.

### What each workflow sends

All five go through one script, `tools/captures/attach_sidecars.py` (tested against a stub
bucket and a stub API in `tools/captures/tests/test_attach_sidecars.py`). The `build` job
finds the asset and its **current** tileset URL (`GET /api/v1/assets/{id}`, `source.url`),
checks its files against that tileset, and writes the request beside them as `attach.json`
(`attach_sidecars.py manifest`: `assetId`, `basedOn`, `files`, `extras`, `rigUrl`), so the
review artifact is exactly what will be sent. The `publish` job (only when asked) runs
`attach_sidecars.py attach` on that directory: it uploads the files to
`staging/assets/<asset id>/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT/` in the private bucket
(`vars.R2_BUCKET || 'twin-assets'`) with the R2 pair, laid out as beside `tileset.json`,
and POSTs `{stagingPrefix, basedOn, files, extras, rigUrl?}` with `API_WRITE_TOKEN` to
`TWIN_API_URL`. The API says why it refused in the 409's `code`: `busy` (another attach
holds the asset) is retried; `tiles_changed` (the tiles changed under the run) ends it with
exit status 3, so run it again on the asset's current tiles; `not_attachable` and every other
refusal end it with exit status 1 and the API's words. No workflow reads, diffs or uploads a `tileset.json` any more, and none writes to the
public bucket. docs/DEPLOYMENT.md ("Sidecars: one publisher") lists the secrets and
variables.

- **publish-instances.yml.** A scan is named as in `infra/modal/segment.py` `SCANS`, whose
  URLs are the legacy prefixes segment.yml read. The asset comes from the repository
  variable `SCAN_ASSET_IDS` (`{"camp": "<asset id>", ...}`), or, where that has no entry,
  from the one gaussian-splat asset whose tileset is the scan's run (`runs/<job>/…`, at the
  legacy prefix or in a generation cut from it); anything else is a refusal naming the
  variable. `build` fetches the asset's current tileset and every tile and keeps the binding
  check (every tile's checksum a key, runs covering its gaussians) against **those** tiles;
  `link_instances` and the tileset diff are gone. It sends
  `files: ["instances.emb", "instances.json"]`,
  `extras: {"instances": {"uri": "instances.json", "count": <instances>}}`, `basedOn` the URL
  it bound against.
- **publish-fill.yml.** The same asset lookup for `scan`. `build` keeps the layer checks and
  the frame check, now against the **current** tileset's root transform, and stages
  `inferred/<name>/…` (the layer's `tileset.json`, tiles and sidecars), sending only its own
  entry, `extras: {"inferredLayers": [{"uri": "inferred/<name>/tileset.json", "evidence":
{…}}]}`: the API merges the list by uri, so another fill's entry stays, and a staged
  `inferred/<name>/` replaces that layer's old files whole.
- **collision-backfill.yml.** The asset is the capture's splat asset (`fetch_capture.py`'s
  `capture.json`). `build` keeps `splat_tiles.py collision`, its identity check, and the
  check that the command changed the tileset by `extras.collision` alone; it sends
  `files: ["collision.bin"]`, `extras: {"collision": {…}}` as the command declared it, and
  `basedOn` the asset's URL the tiles were fetched from.
- **streamed-lod-backfill.yml.** Its input is the asset id (`[streamedlod|asset=<uuid>]`),
  the tileset read from the API. It stages `sog/lod-meta.json` and every chunk under `sog/`
  and sends them in `files`, with `extras: {"nativeLod": "sog/lod-meta.json"}` so the viewer
  need not probe, and `basedOn` the tileset whose leaves it merged. A staged `sog/` replaces
  the old one whole.
- **living-plants.yml.** It stages `rig.json`, `motion.json` and `plants.json` and sends
  them in `files` with `rigUrl: "rig.json"` and `basedOn` the asset's URL the tiles were
  fetched from, in place of the in-place upload **and** the separate `PATCH /assets/{id}` of
  `renderConfig.rigUrl` (which replaced the whole render config, racing every other writer
  of it).

Each publish job is in a concurrency group of its workflow and target (streamed-lod-backfill,
which publishes from its build job, puts that whole job in one), so two runs on one scan queue
rather than interleave; the API's row lock is what makes interleaving safe in any case.

**Split objects are not published, and stay out of this.** `split_objects.py` (C4) rewrites
the scan's own tiles without the chosen objects and writes the objects as tilesets of their
own under `objects/` and `fills/`, binding them in `instances.json`. fill.yml's `split:<scan>`
jobs run it on Modal and keep the result (`split.tar.gz`) in the run's `fill` artifact for
review; no workflow publishes it, and the attach refuses `objects/`, `fills/`,
`extras.objects` and `extras.split` (`attachable=False`). Publishing one is a different
operation from an attach — a **replace-tiles** publish: a new generation whose tiles are the
split's, with `instances.json` re-bound to those tiles' checksums, and everything else bound
to the old tiles (collision, view cones, `sog/`, a rig, skins) dropped and flagged as a
republish would. That belongs beside the worker's publish (`carry.py` already decides what
survives new tiles), and is left until something needs a split on the live site.

`segment.yml`, `fill.yml` and the Modal apps read tiles by URL and are unchanged: they read
`SCANS`' legacy URLs, which keep serving the same tiles after an attach (an attach copies
tiles into a new generation and never deletes the legacy prefix). Once a scan is republished
with new tiles, a segmentation of the legacy URL no longer binds the asset's tiles, and
publish-instances refuses it at the binding check.
