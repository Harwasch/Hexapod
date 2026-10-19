# Archive: parked experiment branches

Six experiment branches were parked on 2026-10-19 and kept only as annotated tags,
`archive/<branch>`. This page is the map to them: what each one asked, what it found, where its
results are written down on the tag, and what reached `main` by another route. Every number below
names the file (or commit message) it comes from. `archive/<b>:<path>` means that path on the
tag; a path written with a `main:` prefix is that path on `main`.

## How to read an archive

Tags are not fetched by default, and cloud clones often skip them. Fetch the one you need by
name:

```sh
git fetch origin tag archive/<b> --no-tags   # e.g. archive/fill-objects
git log archive/<b>                          # its history (the commit messages carry run ids and costs)
git show archive/<b>:<path>                  # one file as the branch left it
git diff main...archive/<b> --stat           # what it adds beside main
```

Nothing here needs a checkout. To run code from an archive, branch from the tag
(`git switch -c <engine>/research-<slug> archive/<b>`) and expect to merge `main` first: every
one is hundreds of commits behind.

## Index

Ahead and behind are against `main` at `f74ece8f` (2026-10-19). The numbers' sources are in
each branch's section below.

| Tag                                  | Dates            | Ahead / behind | Area                                         | Verdict in one line                                                                                                                                                                                            |
| ------------------------------------ | ---------------- | -------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `archive/bakeoff-fill-generative`    | 2026-10-05 to 06 | 8 / 348        | Fill: video models fill a scan's holes       | Wan2.1-VACE 1.3B ranked first in every held-out view (+~4 dB), but no model drew the spool's real top; about $7.05 of $12.                                                                                     |
| `archive/bakeoff-fill-v2`            | 2026-10-05 to 07 | 28 / 301       | Fill round 2: anchor, then propagate         | +5.0 dB on the spool's held-out top with or without photos; shape under the flange passes; the Splash LoRA's invented detail costs fidelity (-0.63 dB) and was published only by the owner's decision.         |
| `archive/fill-objects`               | 2026-10-05 to 07 | 30 / 301       | Fill, object round: 3D models for undersides | TRELLIS (stand-in for gated SAM 3D Objects) beat every round-2 layer on the pumpkins' held-out lower edges (16.77 against 15.82 dB); about $1.13 of $7; no verdict recorded on adoption.                       |
| `archive/bakeoff-living-view`        | 2026-10-08 to 09 | 34 / 278       | Living view: slight wind on a still render   | No overall verdict recorded. LTX-2.5 + Cinemagraph LoRA is the only arm with real wind; pinning the render at both ends (`ltx-loop`) brings sway to 2-4 px; $6.82 (round 1) and $7.41 of the $8 cap (round 2). |
| `archive/bakeoff-seg-feature-fields` | 2026-10-05       | 18 / 346       | Segmentation bake-off candidate B            | No verdict recorded. Feature-field segmentation built, run ($0.19 spool, $0.15 pumpkin) and published as the `feature-fields` objects variant; never merged.                                                   |
| `archive/search-meaning`             | 2026-10-02       | 5 / 621        | Object search by meaning (SigLIP 2 text)     | Decided 2026-10-02: the text tower on the API's CPU (cos ≥ 0.9999998, ~140 ms a query, +565 MB image); never merged. Re-port is Harrison's decision.                                                           |

The three fill branches are one line of work: every commit of `bakeoff-fill-generative` is also in
`bakeoff-fill-v2` and `fill-objects`, and those two share 19 commits (fill-generative's 8, a merge
of main, round 2's 9 and the spool experiments' set-up commit `0b45df32`). After that,
`bakeoff-fill-v2` adds the spool experiments' results (9 commits) and `fill-objects` adds the
object round (11 commits).

## Fill: what reached main by other routes

`main:.github/workflows/fill.yml` and `main:.github/workflows/publish-fill.yml` predate these
branches (2026-10-02, the Teacher B runs with NVIDIA Fixer, `main:tools/captures/teacher_fill.py`,
`main:docs/WORLD_MODEL_RUNBOOK.md` § 5-9). The fill branches extended those workflows and
`infra/modal/fill.py` with their own jobs (`gen:`, `holdout:`, `anchor:`, `leaveout:`); those
extensions did not land. What did land is the viewer side: fill variants in the method switcher
and a fill that hides the measured splats it replaces (`main:apps/web/src/lib/supersedes.ts`,
`main:docs/SCENE_OBJECTS.md` § "Variants: other methods for the same scan (bake-offs)"). The
published fill variants themselves are data in the bucket beside the spool's and pumpkin's
tiles, not files in git.

## `archive/bakeoff-fill-generative`

**Question.** Can a video model make the novel views a scan never had, with the shape coming out
of those views fused into gaussians, with no rule-based geometry and no 3D object models (the
owner's direction)? Source: `archive/bakeoff-fill-generative:docs/WORLD_MODEL_RUNBOOK.md` § 10.

**Built.** The loop `archive/bakeoff-fill-generative:tools/captures/generative_fill.py`, the
models `archive/bakeoff-fill-generative:tools/captures/video_fill_models.py` (Wan2.1-VACE 1.3B,
Wan2.2 TI2V-5B, Cosmos-Predict2 2B, LaMa as the baseline), GPU jobs in
`archive/bakeoff-fill-generative:infra/modal/fill.py`, the variant publisher
`archive/bakeoff-fill-generative:tools/captures/publish_variants.py`, and the held-out check
(`holdout:spool`: the frames that see the spool from highest held out).

**Outcome.** From `archive/bakeoff-fill-generative:docs/WORLD_MODEL_RUNBOOK.md` § 10 "Results
(2026-10-06)":

- Cost: about $7.05 of the $12 cap over runs 37383986439, 37390698492, 37393225461 and 37395213267. An L40S clip cost about $0.06 with VACE or Wan2.2, $0.16 with Cosmos.
- Spool top, held out (PSNR in the region): VACE 10.0-11.1 dB against 5.5-7.4 dB before; Wan2.2
  9.8-10.5; Cosmos 9.0-10.1; LaMa 7.3-8.9. VACE ranks first in every view and its clips are the
  most consistent. Gate pass rates: VACE 147/147, Cosmos 107/147, Wan2.2 18/147.
- None of them draws the real top (light grey planks).
- Pumpkin ground: Cosmos paints straw, the nearest to what is there; VACE and LaMa paint a
  pumpkin back from segmentation fragments.
- Camp roof: nothing to fill with this data (no cameras; through the roof's holes the cabin
  below is seen from every viewpoint).
- The bake-off's dollar ledger is said to be "in the bake-off notes"; no such file is on the tag.

**Relationship.** Contained whole in `archive/bakeoff-fill-v2` and `archive/fill-objects`. Round 2
(`bakeoff-fill-v2`) supersedes it; of its published variants only `vace-1-3b` stayed
(`archive/bakeoff-fill-v2:docs/WORLD_MODEL_RUNBOOK.md` § 11 "Results").

**Landed by other routes.** See "Fill: what reached main by other routes" above.

## `archive/bakeoff-fill-v2`

**Question.** Round 2 of the inferred fill, the plan the owner approved after round 1: fill at the
poses that need it, with real photos as context; a few anchor views first, every later view
continuing them, nothing picked by hand. Then: treat weak (seen but badly) surfaces before any
generator, swap rather than stack, shape the unseen pocket under the spool's flange, and test a
third-party "Splash" editor LoRA. Source: `archive/bakeoff-fill-v2:docs/WORLD_MODEL_RUNBOOK.md` § 11.

**Built.** `archive/bakeoff-fill-v2:tools/captures/anchor_fill.py` (the loop),
`archive/bakeoff-fill-v2:tools/captures/fill_quality.py` (where to fill),
`archive/bakeoff-fill-v2:tools/captures/fill_views.py` (which views and photos),
`archive/bakeoff-fill-v2:tools/captures/anchor_models.py` (Qwen-Image-Edit-2511, VACE-14B, the
Splash LoRA), `archive/bakeoff-fill-v2:tools/captures/fill_surface.py` (weak surfaces
reprojected from the photos), `archive/bakeoff-fill-v2:tools/captures/fill_pockets.py` (free
space and the shape under the flange), solidity terms in
`archive/bakeoff-fill-v2:tools/captures/distill_fill.py`, and tests under
`archive/bakeoff-fill-v2:tools/captures/tests/`. Arms: `refs` (anchors with photos), `norefs`
(no photos, the control) and `vace`.

**Outcome.** All from `archive/bakeoff-fill-v2:docs/WORLD_MODEL_RUNBOOK.md` § 11:

- Anchor and propagate ("Results (2026-10-06)"), about $8.33 of the $15 cap: on the spool's
  held-out top, PSNR from 10.9 dB before to 15.8 (`refs`), 15.9 (`norefs`) and 15.7 (`vace`),
  +5.0 dB with or without photos against round 1's +4.0. On the pumpkin, 12.2 to 14.1-14.3 dB,
  but LPIPS slightly worse (the fills are smoother than the straw). Fills are plausible but
  soft. Published on the spool and the pumpkin: `anchor-refs`, `anchor-norefs`, `anchor-vace`
  (run 37524204337); round 1's `wan22-5b`, `cosmos-p2-2b` and `lama-baseline` withdrawn.
- Swap, don't stack: once swapped, `anchor-refs` is at most 2.1 % see-through on the top face,
  against 6.4 % for the scan alone (run 37541017400).
- Spool experiments ("Experiments on the spool (2026-10-07, $18)", run 37551303121, about $4.90):
  the shape under the flange passes (14,825 discs; low-view see-through from 64 % to 0.1 % at
  -5°); the solidity terms change little for `refs` and help `norefs` most; the Splash anchors'
  gate kept 0 of 32 frames.
- Splash retries: S2 (run 37562977111, about $3.24) kept 2 frames, a patchwork; S3 (run
  37566525560, about $2.17) kept none; S4 `splash-footprint` (run 37569429589, about $1.28) kept
  12 of 16 but failed the leave-out against `refs` (15.323 against 15.951 dB, -0.63): the detail
  it paints is invented. Published anyway by the owner's decision (run 37637300421, about
  $0.95), labelled "Splash LoRA top (detail invented)".
- Open: the Splash LoRA's training data likely comes from Apple's SHARP renders; ask before
  shipping anything made with it ("Experiments on the spool", item 3).

**Relationship.** Contains all of `archive/bakeoff-fill-generative`. Shares 19 commits with
`archive/fill-objects`, which branched off at `0b45df32` (the spool experiments' set-up) and does
not have their results; read this tag's runbook, not fill-objects', for § 11 after "Experiments
on the spool".

**Landed by other routes.** The supersedes swap's viewer side is on main
(`main:apps/web/src/lib/supersedes.ts`); see also "Fill: what reached main by other routes".

## `archive/fill-objects`

**Question.** The object round of the inferred fill: can 3D object-completion models make whole
an object the cameras saw from one side only (the pumpkins' undersides on their straw), registered
to the scan, with the measured splats untouched? Source: the docstring of
`archive/fill-objects:tools/captures/object_fill.py`.

**Built.** `archive/fill-objects:tools/captures/object_fill.py` (objects from the segmentation,
frames, register, carve by free space and silhouettes, keep only unseen directions, colour match,
one inferred layer, grading against round 2's leave-out),
`archive/fill-objects:tools/captures/object_models.py`,
`archive/fill-objects:infra/modal/fill_objects.py`,
`archive/fill-objects:.github/workflows/fill-objects.yml` and
`archive/fill-objects:tools/captures/tests/test_object_fill.py`. Model: TRELLIS (v1) image-large
(MIT) as a stand-in, because SAM 3D Objects and the DINOv3 encoder Pixal3D and TRELLIS.2 load were
gated for the workspace's token (commit `2bf4e500` message).

**Outcome.** There is no result doc on the tag; the results are in the commit messages
(`git log archive/fill-objects`), each carrying its run id, cost and the running total:

- Commit `aeed590a` (run 37552051186): with the straw cut from the masks, TRELLIS made both
  pumpkins whole; the object layer alone scored 16.77 dB / LPIPS 0.224 / DreamSim 0.105 on the
  held-out lower edges, better than any round-2 layer (best 15.82 dB / 0.274 / 0.117), and
  improved all three on top of each round-2 layer.
- Commit `d8fb5da1` (run 37555215455, the last): silhouette IoU 0.9998 / 0.9992, the object layer
  alone 16.32 dB / 0.239 / 0.120; ~$1.13 spent of the $7 cap.
- The object variants (`objects-trellis`, `objects-sam3d`, `objects-pixal3d`) were made
  publishable (commit `6d80f5a7`); no commit records publishing one.
- No verdict recorded on whether the object round is adopted.

**Relationship.** Contains all of `archive/bakeoff-fill-generative` and the first 19 commits of
`archive/bakeoff-fill-v2` (merged at `d368d329` for round 2's free-space voxels). Its
`docs/WORLD_MODEL_RUNBOOK.md` stops at round 2's spool experiment plan, without results.

**Landed by other routes.** Nothing of the object round; see "Fill: what reached main by other
routes" for the viewer side.

## `archive/bakeoff-living-view`

**Question.** The owner's idea: when the camera stops, a video model adds slight wind to the
plants, conditioned on the exact render of the splat, in short segments restarting from that
render. Which model gives believable, slight, anchored motion, and is the model's motion applied
to our full-resolution render better than its pixels? Round 2 added generic prompts, the LTX
ladder, three world models (Waypoint 1.5, Matrix-Game 3.0, Yume 1.5) and 2560x1408 upscales.
Source: the docstring of `archive/bakeoff-living-view:infra/modal/living_bakeoff.py`.

**Built.** `archive/bakeoff-living-view:infra/modal/living_bakeoff.py` (arms `ltx`, `causal`,
`flf`, `wan`), `archive/bakeoff-living-view:infra/modal/living_upscale.py` (FlashVSR v1.1,
SeedVR2-3B), `archive/bakeoff-living-view:infra/modal/living_waypoint.py`,
`archive/bakeoff-living-view:infra/modal/living_mg3.py`,
`archive/bakeoff-living-view:infra/modal/living_yume.py`,
`archive/bakeoff-living-view:tools/captures/living_view.py` (motion-only: the model's flow applied
to our render), `archive/bakeoff-living-view:.github/workflows/living-view.yml`, and the
comparison page and delivery scripts
`archive/bakeoff-living-view:bakeoff/living-view/build_page.py` and
`archive/bakeoff-living-view:bakeoff/living-view/r2_deliver.py`.

**Outcome.** No overall verdict recorded on which model or approach ships. What is recorded:

- Cost (`archive/bakeoff-living-view:bakeoff/living-view/cost-ledger.md`, the ledger, with
  every run's GitHub run, app id, GPU minutes and dollars): round 1 $6.82 metered against a
  ~$7 target and $10 cap; round 2 $7.41 metered of the $8 cap. GPU work stopped short of the 4K
  clips. The ledger also lists every model's licence.
- Round 1 by eye (`BY_EYE` in `archive/bakeoff-living-view:bakeoff/living-view/build_page.py`):
  LTX is the only arm with real wind (p95 15-59 px) and no camera creep, but redraws frame 0;
  Causal Forcing leaves the anchor within a second; FLF2V hardly moves; Wan 2.2 creeps and
  smears. The motion-only path keeps our pixels but large motion tears them. FlashVSR stays
  close to the source, SeedVR2 over-sharpens.
- Round 2 prompts (`archive/bakeoff-living-view:bakeoff/living-view/prompts.md`, "Scores" and
  "What sets the amplitude"): winner `ltx-p1` (least camera creep, at most 0.7 px over 4 s). No
  generic wording brings plant motion under 20-60 px p95; pinning the render as both first and
  last keyframe (`ltx-loop`) brings it to 2-4 px and loops without a jump.
- World models (ledger, run R2-9): Waypoint 1.5 first frame 0.4-1.0 s, 21-47 fps sustained;
  Matrix-Game 3.0 25.8 s per 97-frame rollout; Yume 1.5 27 s per 58 frames, and its idle is not
  still (`prompts.md`, "Yume 1.5: what it did (R2-9)").
- Clean-up the ledger asks for: delete the Modal volumes `hexapod-living-view-weights` (about
  220 GB) and `hexapod-living-view` once decided.

**Relationship.** Independent of the other five.

**Landed by other routes.** Not this code. Main's Worlds explorer has its own LTX 2.5 and
Matrix-Game 3 adapters (`main:docs/WORLDS_MODELS.md`), and `main:docs/WORLDS_LAND_TESTING.md`
says the Living View experiments do not validate the Worlds LTX adapter.

## `archive/bakeoff-seg-feature-fields`

**Question.** Segmentation bake-off candidate B: can a scale-conditioned feature field trained on
the splat (GARField and SAGA, reimplemented on gsplat) give objects, their parts and the ground's
cover classes, in the same `instances.json` the viewer reads? Source: the docstring of
`archive/bakeoff-seg-feature-fields:tools/captures/feature_fields.py`.

**Built.** `archive/bakeoff-seg-feature-fields:tools/captures/feature_fields.py`,
`archive/bakeoff-seg-feature-fields:infra/modal/feature_fields.py`,
`archive/bakeoff-seg-feature-fields:.github/workflows/segment-feature-fields.yml` (a GPU `field`
job and a free CPU `finish` job on saved fields) and
`archive/bakeoff-seg-feature-fields:tools/captures/tests/test_feature_fields.py`. The method and
measurements are written up in `archive/bakeoff-seg-feature-fields:docs/SCENE_OBJECTS.md` §
"Bake-off candidate B: feature fields". The first commit is "WIP feature fields module (not yet
wired)"; later commits wire the workflow and publish, but nothing was merged.

**Outcome.** No verdict recorded. Measured (`archive/bakeoff-seg-feature-fields:docs/SCENE_OBJECTS.md`
§ "Bake-off candidate B"): run 37382311130 on an L4, spool 597 s and $0.19, pumpkin 468 s and
$0.15. The published files (finished on the CPU, run 37389686654): the spool one thing with top
and drum as parts, 13 things, 8 cover classes; the pumpkins two whole things in the hay, 5 cover
classes. Published as the `feature-fields` objects variant.

**Relationship.** Merges `bakeoff-seg-ground-first` (candidate A), which is on main. Independent
of the fill branches.

**Landed by other routes.** Candidate A, ground first, and candidate C, concept first, are on
main (`main:tools/captures/segment_ground_first.py`, `main:docs/SCENE_OBJECTS.md` § 3b). The
published `feature-fields` variant is still one of the objects variants main's selection checks
cover (`main:docs/SCENE_OBJECTS.md` § "Selecting in the scene": "Saucer" under feature fields).

## `archive/search-meaning`

**Question.** How should a viewer's object search match by meaning, not only by tags, and what
does it cost? Source: `archive/search-meaning:docs/SCENE_OBJECTS.md` § 3 step 4, "Search by
meaning (decided 2026-10-02)".

**Built.** Across the API, the web app and the contracts:
`archive/search-meaning:tools/captures/export_text_encoder.py` (exports SigLIP 2's text tower as
ONNX in the image's build stage), `archive/search-meaning:apps/api/app/services/text_encoder.py`
and `archive/search-meaning:apps/api/app/api/v1/text.py` (`GET /api/v1/text-embeddings`, on the
API's CPU with onnxruntime), the `text-encoder` stage of `archive/search-meaning:infra/api.Dockerfile`,
the viewer's blend in `archive/search-meaning:apps/web/src/lib/instances.ts` and
`archive/search-meaning:apps/web/src/state/instances.ts`, regenerated
`archive/search-meaning:packages/contracts/openapi.json`, and tests
`archive/search-meaning:apps/api/tests/test_text_embedding.py` and
`archive/search-meaning:apps/web/src/__tests__/instancesMeaning.test.ts`.

**Outcome.** Decided 2026-10-02: the text tower on the API's CPU. From
`archive/search-meaning:docs/SCENE_OBJECTS.md` § 3 step 4 (measured 2026-10-02, 4 shared CPUs):

| Option                             | Size           | Parity to the reference embedder            | Latency                                 | Result   |
| ---------------------------------- | -------------- | ------------------------------------------- | --------------------------------------- | -------- |
| Browser, onnx-community int8 tower | 283 MB + 34 MB | cos 0.92-0.98; top-10 overlap down to 3/10  | download alone fails "first query < 5s" | rejected |
| Modal function                     | -              | exact                                       | container cold start per idle period    | rejected |
| API CPU, onnxruntime, fp32 maths   | +565 MB image  | cos ≥ 0.9999998, same top-10 on every query | first 1.24 s (load), then ~140 ms       | chosen   |

From `archive/search-meaning:docs/DEPLOYMENT.md` § "Search by meaning runs on the app machine":
the app process goes from ~130 MB to ~640 MB resident once the encoder loads (measured locally),
the first search on a fresh machine pays ~1.3 s, later ones ~140 ms on one core; without the
encoder the route answers 503 and the viewer searches by tags.

**Relationship.** Independent of the other five; the oldest (621 behind).

**Landed by other routes.** Main has image-embedding search, not text: **More like this** ranks a
scan's objects by their SigLIP 2 image embeddings (`main:apps/web/src/features/sites/MoreLikeThis.tsx`,
`main:docs/SCENE_OBJECTS.md` § "More like this"), with `rankByVector` and `loadEmbeddings` in
`main:apps/web/src/lib/instances.ts`. No text-embedding route is on main.

Re-port assessment (2026-10-19): medium, about 3-5 agent hours. The API half
(`archive/search-meaning:tools/captures/export_text_encoder.py`,
`archive/search-meaning:apps/api/app/services/text_encoder.py`,
`archive/search-meaning:apps/api/app/api/v1/text.py`, the `text-encoder` stage of
`archive/search-meaning:infra/api.Dockerfile`) merges almost cleanly. Regenerate the contracts
rather than merging them. The web blend must be rewritten on main's `rankByVector` and
`loadEmbeddings` in `main:apps/web/src/lib/instances.ts`. Conflicts with main:
`archive/search-meaning:apps/api/app/main.py`, `archive/search-meaning:apps/api/app/api/v1/router.py`,
`archive/search-meaning:apps/api/uv.lock`, `archive/search-meaning:apps/web/src/lib/instances.ts`,
`archive/search-meaning:apps/web/src/state/instances.ts`,
`archive/search-meaning:apps/web/src/features/sites/InstanceSearch.tsx`,
`archive/search-meaning:apps/web/src/cesium/splatInstances.ts`,
`archive/search-meaning:docs/DEPLOYMENT.md` and the three contracts files
(`archive/search-meaning:packages/contracts/openapi.json`,
`archive/search-meaning:packages/contracts/src/generated/api.d.ts`,
`archive/search-meaning:packages/contracts/src/index.ts`). Risks: a 565 MB larger API image, and
the ~640 MB resident figure was measured locally, not on Fly (`main:fly.toml` gives the app
machine 1 GB). Whether to re-port is Harrison's decision, still open.
