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
SKIN each moving instance: PhysSkin(shape) ─► m weight fields; each field's transform = a handle
  ▼
DRIVE the handles
  │  living: wind forces + materials (class prior, refined by the video teacher)
  │  live:   telemetry (a car is 1 rigid handle; a robot its joints)
  │  dream:  world model → handle trajectories
  ▼
GPU: x' = Σ_j w_j · T_j · x   (one skinning path, the existing vertex hook)
```

## 2. Terms

- **Handle**: a transform (affine: rotation, scale, shear, translation) that changes over
  time. It is a degree of freedom, not necessarily a point in space.
- **Skin**: per splat, a weight per handle. A splat's new position is the weighted sum of its
  handles' transforms applied to its rest position.
- **PhysSkin** (zju3dv, CVPR 2026; built on NVIDIA Simplicits): predicts the skin of a shape
  in one pass, trained against elastic energy. It gives _how a shape can bend_. Materials,
  forces and non-elastic behaviour (walking, flowing) come from data or drivers.
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

- PhysSkin's licence is not stated in its repository; Simplicits (Kaolin) is the fallback.
- Its runtime cost for hundreds of instances in a browser is unmeasured.
- Trees with thin branches are outside PhysSkin's training data; the synthetic tree is the test.
- SAM 3 / some lifting methods carry their own licences; SAM 2 is Apache-2.0.
