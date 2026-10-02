# Living models — world, video and state models for the three views

Research note, written 2026-09-30 on branch `living-models`. It extends
[LIVING_WORLD.md](LIVING_WORLD.md) and [ADR 0008](DECISIONS/0008-living-mode.md). Every external
claim was read on that date; anything not confirmed is marked **UNVERIFIED**. Nothing here is
built yet.

## The product idea

The globe shows a user's scanned land in three views:

1. **Static** (default): detailed and still. Regions no camera saw may be filled in, and the
   fill is labelled as inferred.
2. **Living survey**: gentle motion as though you were there (wind, water, grass), with or
   without a grounding video of the scene.
3. **Live render**: a best-effort, real-time scene that agrees with sparse live telemetry. A
   **scenario ("dream") mode** plays out a prompted action, such as a robot mission or a storm,
   over time.

Everything stays grounded in known data, and much of that data is enduring.

## The one change: split what changes from what it looks like

No released world model is a source of truth for a measured map. Genie 3, Cosmos, Wan and
Matrix-Game output pixels. None takes a georeferenced state or preserves metric geometry, and
all drift over minutes. _Wind on Trees_ (arXiv 2609.17810) finds that video-learned deformation
"optimize[s] photometric consistency rather than recover the motion".

So each view is built from two parts:

- **What changes**: measured data plus physical or state models. These are deterministic,
  replayable and georeferenced.
- **What it looks like**: our splats, rendered by Cesium. Generative models add appearance only,
  offline, into separately labelled layers.

This keeps ADR 0008's line, which says generative models produce parameters or a labelled
separate layer and never runtime frames.

## Layers by how long they endure

| Layer        | Contents                          | Static                               | Living survey                                                                | Live render                                                                                 |
| ------------ | --------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **enduring** | terrain, trunks, rocks, buildings | measured splats, frozen              | same bytes, never written                                                    | same; a re-capture supersedes a tile by `validFrom`                                         |
| **slow**     | vegetation, season, water level   | as of the capture date               | modal rig; parameters from `allometric` → `fitted-generated` → `fitted-real` | filtered sensor state drives water height and tints; change detection patches tiles         |
| **fast**     | wind, flow, robots, people        | off                                  | synthetic wind spectrum, flow maps                                           | live wind drives the same shader; robot pose drives glTF; scenarios run through a simulator |
| **inferred** | regions no camera saw             | separate tileset, tinted, toggleable | same, moving with its plant                                                  | optional generated scenario clip (video only)                                               |

## Per view

### Static: gap fill, offline, labelled

1. Count each Gaussian's supporting real views (its `_SUPPORT`, as in LIVING_WORLD §3).
2. Render virtual cameras aimed at low-support regions, near real paths.
3. Repaint those views with **Cosmos-Predict2.5** (or **Cosmos 3**), then clean them up with
   **NVIDIA Fixer**. Several seeds give a variance, which becomes the confidence.
4. Distil the result into new Gaussians with gsplat while the measured Gaussians stay frozen by
   checksum. The output is a separate `inferred` tileset.
5. **Lyra 1.0** (Apache + OML) can lift generated video straight into 3DGS, as an alternative
   to distillation.

No model provides per-region confidence, so we compute it from support and seed variance. The
gates in LIVING_WORLD §7 apply: a held-out view check, bit-identical measured splats, and a
canary.

### Living survey: the world model teaches parameters, not motion

This is ADR 0008's Teacher A. Generate clips of the object with **Wan 2.2 TI2V-5B** or
**Cosmos-Predict2.5**, track rig nodes, fit per-limb frequency, gain and coherence from spectra,
and write `motion.json` tagged `fitted-generated`. A real grounding video takes the stronger
rungs (`fitted-real`, `recorded`). Learned whole-scene animation (AniGS, arXiv 2607.18539: about
7 h per scene, no code) and DynamicTree (no code) are worth watching. Neither is commercially
usable yet.

### Live render: state estimation, not a real-time world model

- **State layer (server, CPU):** a per-asset Kalman or particle filter over robot pose, wind,
  water level and soil moisture. Each field carries an uncertainty and an age (how stale it is),
  streamed to the browser at 1–10 Hz (WebSocket or CZML).
- **Drivers (browser, GPU, 60 fps):** wind drives the existing modal shader, water level sets a
  water mesh clipped to the basin, robot pose drives a glTF model through
  `SampledPositionProperty`, and soil and rain drive tints. The cost is about $0 per viewer.
- **Map maintenance (server GPU, minutes):** render the baseline from each robot or drone
  frame's pose, compare, and re-optimise only the changed tile with gsplat. The patched tile is
  republished with a date.
- **Robot-local live view (later):** real-time GS-SLAM exists (RMGS-SLAM 2604.12942, GLAM-SLAM
  2607.21416), but most of it is research code or non-commercial (MonoGS is CC BY-NC-SA). A
  shippable version would be rebuilt on gsplat with our own RTK/VIO tracking.

A real-time world model can't hold a georeferenced map. Matrix-Game 3.0 (Apache, 720p, up to
40 fps) needs about one H100 per viewer (~$2–4/h, UNVERIFIED) and ignores our geometry. Genie 3
has no API.

### Scenario ("dream") mode: hybrid

1. The prompt becomes an action plan (the existing mission planner and `MissionProvider`).
2. A simulator decides what happens: mow masks edit grass Gaussians, a storm raises the wind
   spectrum, a flood fills the elevation model, and MuJoCo or Isaac Sim handle the robot. The
   output is a time-varying state rendered in Cesium and labelled "simulated".
3. Optionally, **Cosmos-Transfer2.5** turns renders of that state (RGB, depth, segmentation)
   into a photoreal clip, at about 4–7 min per 93 frames at 720p on an H100. The clip is a
   generated video and never 3D truth.

## Model shortlist

| Model                                                 | Use                            | 3D out? | Speed                      | Licence                                                                    | Ship?                                    |
| ----------------------------------------------------- | ------------------------------ | ------- | -------------------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| Cosmos-Predict2.5 2B/14B                              | gap-fill teacher, motion clips | pixels  | ~229 s per clip (2B, H100) | NVIDIA OML: commercial; "Built on NVIDIA Cosmos" credit; guardrails kept   | yes (resolves LIVING_WORLD's UNVERIFIED) |
| Cosmos 3 Nano/Super (2026-05)                         | newer teacher                  | pixels  | —                          | OpenMDW-1.1                                                                | yes                                      |
| NVIDIA Fixer                                          | clean up renders before distil | pixels  | 27 ms/frame (H100)         | Apache code, OML weights                                                   | yes                                      |
| Lyra 1.0                                              | generated video → 3DGS         | 3DGS    | A100/H100                  | Apache code, OML weights                                                   | yes                                      |
| Cosmos-Transfer2.5                                    | photoreal scenario clips       | pixels  | 4–7 min per 93 frames      | OML                                                                        | yes                                      |
| Wan 2.2 TI2V-5B                                       | motion teacher                 | pixels  | 80 GB class                | Apache-2.0                                                                 | yes                                      |
| gsplat (+ 3DGUT mode)                                 | training, tile patching        | 3DGS    | minutes                    | Apache-2.0                                                                 | yes                                      |
| NuRec NRE containers                                  | reconstruction, gRPC rendering | USD/PLY | —                          | NGC terms, UNVERIFIED                                                      | evaluation only                          |
| GEN3C                                                 | camera-exact generation        | pixels  | ~43 GB                     | Apache code; OML weights, but the repo points to a custom-licence form     | verify first                             |
| Matrix-Game 3.0                                       | explorable dream               | pixels  | 720p, up to 40 fps         | Apache-2.0                                                                 | not grounded                             |
| World Labs Marble / World API                         | hosted splat worlds            | splats  | $0.12–1.20 per world       | closed API; customer scans leave us; reported AMD acquisition (UNVERIFIED) | preview at most                          |
| Difix3D+, Lyra 2.0, SEVA, HY-World, NeoVerse, Genie 3 | —                              | —       | —                          | non-commercial, R&D-only, regional, or no API                              | **no**                                   |

## The NVIDIA skill and the Brev launchable

- [`physical-ai-neural-reconstruction`](https://github.com/NVIDIA/skills/tree/main/skills/physical-ai-neural-reconstruction)
  is a thin router to [`NVIDIA/nurec-skills`](https://github.com/NVIDIA/nurec-skills):
  - `ncore` converts recordings to NCore (a COLMAP converter is built in);
  - `nre` trains 3DGUT/3DGRT into USDZ, renders over a gRPC server, edits actors, and exports a
    standard 3DGS PLY;
  - `asset-harvester` extracts objects as PLY;
  - `nurec-fixer` runs DiffusionHarmonizer.

  It needs an Ampere or newer GPU with 24 GB or more (48 GB recommended) and 150 GB or more of
  disk. It is built for autonomous-vehicle and robot rigs and does no gap filling.

- The [Brev launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-3C5z7T9WtTr3dmDBU1lvBWUcfjj)
  is "Neural Reconstruction": one A100 80 GB (GCP `a2-ultragpu-1g`) with the NuRec workflows
  behind an agent UI, not notebooks. Cost isn't shown; GCP list is about $5/h (UNVERIFIED).

**Verdict:** use it as a sandbox to try Fixer, Cosmos and NRE on the Minnetonka scan. Keep the
product on our own COLMAP → gsplat → SPZ path on Modal until the NRE container terms are
confirmed. gsplat's Apache 3DGUT mode is worth adopting for fisheye and rolling-shutter drone
footage, since 3DGUT keeps standard 3DGS parameters and exports a PLY our tiler reads (test the
pinhole quality difference).

## Proposed first moves

| Step           | What                                                                                                      | Pass                                                                    |
| -------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A · data model | enduring/slow/fast/inferred layers; per-splat support; an `inferred` tileset kind and its Inspector label | contracts and catalog carry it; the viewer can hide and tint it         |
| B · static     | leave-one-tier-out on Minnetonka; Cosmos-Predict2.5 + Fixer distillation into an inferred layer           | held-out low-support pixels improve; measured splats bit-identical      |
| C · living     | Teacher A: Wan / Cosmos clips → `motion.json` tagged `fitted-generated`                                   | beats a random-parameter control; blind test prefers it over allometric |
| D · live       | live state schema; one weather station drives the wind shader; staleness shown                            | shader follows the filtered wind; a stale reading is visible            |

## Sources

- NVIDIA: [nurec-skills](https://github.com/NVIDIA/nurec-skills),
  [3dgrut](https://github.com/nv-tlabs/3dgrut), [instant-nurec](https://github.com/NVIDIA/instant-nurec),
  [Fixer](https://huggingface.co/nvidia/Fixer), [Lyra](https://huggingface.co/nvidia/Lyra),
  [Lyra-2.0](https://huggingface.co/nvidia/Lyra-2.0),
  [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B),
  [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5),
  [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano),
  [Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
  [Difix3D licence](https://github.com/nv-tlabs/Difix3D/blob/main/LICENSE.txt),
  [GEN3C](https://github.com/nv-tlabs/GEN3C)
- World models: [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/),
  [Matrix-Game 3.0](https://huggingface.co/Skywork/Matrix-Game-3.0),
  [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay),
  [World Labs export](https://docs.worldlabs.ai/marble/export/gaussian-splat/index)
- Motion and completion: [Wind on Trees](https://arxiv.org/abs/2609.17810),
  [AniGS](https://arxiv.org/html/2607.18539v1), [DynamicTree](https://arxiv.org/abs/2510.22213),
  [GSComplete](https://arxiv.org/abs/2609.08449), [MoSca](https://github.com/JiahuiLei/MoSca)
- Live: [RMGS-SLAM](https://arxiv.org/abs/2604.12942), [GLAM-SLAM](https://arxiv.org/pdf/2607.21416),
  [gsplat](https://arxiv.org/pdf/2409.06765), [MonoGS++ licence](https://arxiv.org/html/2504.02437v1)
