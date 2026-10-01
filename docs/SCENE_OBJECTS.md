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
   plus a ring, each with per-pixel splat ids (which splats dominate each pixel).
2. **Masks.** Class-free automatic masks at several scales per view (SAM 2 family).
3. **Lift.** Each mask votes for the splats it covers. Splats that co-occur in masks across
   views are merged into instances (a graph whose edge weights are co-occurrence over
   visibility). Scales give the hierarchy: a coarse mask is the parent of the finer ones
   inside it.
4. **Meaning.** For each instance, crop its best views and embed them with an
   image-text model (SigLIP/CLIP family). Text search is cosine similarity at query time.
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
- `instances.emb`: `float16`, `count × dim`, row `k` is instance id `k + 1`, L2-normalised.

### Root extras

The measured tileset's `root.extras.instances = { "uri": "instances.json", "count": n }`, so
the viewer finds it without probing (the same pattern as `viewCones` and `inferredLayers`).

### Fixture and browser checks

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

v1 writes ids and the table only; the split into object tilesets is a later step (§6).

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
- Covariances follow the skin's Jacobian, which includes the weights' gradients: store
  ∇w per splat or drop that term (to measure).
- SAM 3 / some lifting methods carry their own licences; SAM 2 is Apache-2.0.
