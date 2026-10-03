# World-model runbook: the first GPU run

Everything below runs on the CPU today with stand-ins (Telea inpainting for fill, damped
oscillators for motion). This page covers swapping in the real models on Modal.

| Teacher         | Stand-in (CPU, tested)         | GPU model                                               | Modal class                                  | Client                                |
| --------------- | ------------------------------ | ------------------------------------------------------- | -------------------------------------------- | ------------------------------------- |
| B: fill         | `InpaintFiller` (Telea)        | NVIDIA Fixer                                            | `Fixer.fix`                                  | `world_model_client:FixerFiller`      |
| B: fill (holes) | `InpaintFiller` (Telea)        | Qwen-Image inpainting ControlNet; LaMa; SDXL inpainting | `InpaintQwen.inpaint`, `InpaintSDXL.inpaint` | `world_model_client:GenerativeFiller` |
| B: refine       | `distill_fill.torch_rasterize` | gsplat 1.5.3                                            | `Distill.run`                                | `teacher_fill.py fill --distill N`    |
| A: motion       | `OscillatorClips`              | Wan 2.2 TI2V-5B                                         | `Wan.clip`                                   | `teacher_motion.py --source wan`      |
| A: motion (alt) | (same)                         | Cosmos-Predict2.5-2B                                    | `Cosmos.clip`                                | `teacher_motion.py --source cosmos`   |
| C2: materials   | `teacher_materials.py synth`   | Wan / Cosmos                                            | `Wan.clip`                                   | `teacher_materials.py world`          |

Code: `infra/modal/world_models.py` (server), `tools/captures/world_model_client.py`
(client), `teacher_fill.py`, `teacher_motion.py`, `distill_fill.py`.

## 1. Credentials

| Need                  | Where it goes                                        | Why                                                                                                                      |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Modal token           | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the shell | deploy and call                                                                                                          |
| Hugging Face token    | Modal secret `huggingface` (`HF_TOKEN`)              | Wan, Cosmos weights (Fixer's `nvidia/Fixer` is not gated and needs none); the workspace had no such secret on 2026-10-02 |
| HF licence acceptance | huggingface.co, on the token's account               | Cosmos only: Cosmos-Predict2.5-2B, Cosmos-Reason1-7B, Cosmos-Guardrail1 (all gated; `harwasch` had none on 2026-10-01)   |
| ~~NGC API key~~       | not needed                                           | Fixer's NGC container is replaced by the same environment built from cosmos-predict2's `uv.lock` (section 5)             |
| R2 credentials        | already used by `infra/modal/app.py`                 | only to publish an inferred layer                                                                                        |

```sh
modal secret create huggingface HF_TOKEN=hf_...   # for Wan / Cosmos only
modal deploy infra/modal/world_models.py
```

Fixer itself runs from CI with nothing but the repository's Modal token: push a `wm-*`
branch touching `infra/modal/fill.py` (or `teacher_fill.py`, `world_model_client.py`) and
`.github/workflows/fill.yml` runs `modal run infra/modal/fill.py` and uploads the
reports, strips and inferred tileset as the `fill` artifact.

## 2. Order of runs (cheapest proof first)

1. **Fixer on the drop test.** Holds the truth, so it gives a score.
   ```sh
   cd tools/captures
   uv run --with modal python teacher_fill.py drop ../../data/tiles/synthetic-yard/splat/tileset.json \
     --filler world_model_client:FixerFiller --save /tmp/drop-fixer
   ```
   Pass: `psnrFill > psnrHole` in the held-out view, by more than Telea's margin
   (Telea on the yard: 14.94 vs 14.78 dB). Then repeat on the spool scan (phone0924).
2. **Fixer on the camp, from outside.** The real target.
   ```sh
   uv run --with modal python teacher_fill.py fill <camp>/tileset.json <camp>/../inferred \
     --filler world_model_client:FixerFiller --views 8 --save /tmp/camp-fixer
   uv run python teacher_fill.py link <camp>/tileset.json <camp>/../inferred/tileset.json
   ```
   Inspect the strips in `--save` (full render | seen + mask | fill), then the globe.
3. **Distill.** Same command plus `--distill 1500` (runs `Distill.run` on an L40S).
   Pass: every view's `maskedL1After < maskedL1Before` and `outsideL1After ≤ outsideL1Before`.
4. **Wan on the synthetic tree (test L1, real model).**
   ```sh
   uv run --with modal python teacher_motion.py ../../data/tiles/synthetic-tree/source /tmp/motion.json \
     --source wan --seeds 12 --cameras 3 --width 1280 --height 704 --report /tmp/report.json
   ```
   36 clips at about 2–4 min each on an A100. There is no ground truth here: the
   oscillator prior is only a sanity range. Pass: the trunk is fitted, at least 10 limbs
   are `fitted`, and their frequencies are plausible (0.2–3 Hz) and differ from the prior.
5. **Cosmos** (after licences): step 4 with `--source cosmos`. It runs 16 fps × 77 frames,
   so the Welch segment is about 4.8 s.

6. **Materials (C2) from a world-model clip.** Same secret as 4.
   ```sh
   uv run --with modal python teacher_materials.py world ../../data/tiles/synthetic-yard/splat \
     ../../data/tiles/synthetic-yard/skin --instance 10 --model Wan --seeds 4 \
     --strength 0.1 --bearing 60 --materials /tmp/materials.json --report /tmp/c2.json
   ```
   A 5 s clip resolves shrubs (2-8 Hz), not a tree's 0.2 Hz sway; the clip's wind speed is
   unknown, so `drag` is relative to the `--strength` given. Records are `fitted-generated`.

## 3. Things that will probably need a fix on first contact

- **Package pins in the images** (`diffusers==0.35.1`, `torch==2.6.0` for Wan): these are
  first guesses. The Wan model card says diffusers main was once required, so pin a commit
  if 0.35.1 lacks `WanImageToVideoPipeline` support for TI2V.
- **Fixer's model code imports** from `/work/fixer/src`, and it expects
  `/work/models/base/*`. The volume is mounted there, and `nvidia/Fixer` is snapshotted
  into it on first start (5.5 GB).
- **Cosmos `uv sync`** builds a large CUDA 12.8 environment at image build (slow).
  The guardrail must stay on (licence), and a blocked clip raises.
- **Camera motion in generated clips.** Teacher A assumes a locked-off camera: the prompt
  says so, and the trunk regression absorbs a little. If clips pan, add global-motion
  compensation before tracking (estimate an affine transform per frame on background pixels).

## 4. What "done" looks like for the prototype

- Camp from outside: the inferred layer fills the faded shell, labelled **Inferred fill**
  in the site bar (hover shows filler, views and confidence), and switchable off.
- Spool: drop-and-fill scores for Fixer beat Telea in the held-out view.
- Tree: a `fitted-generated` motion sidecar from Wan clips that the Living Survey plays.

## 5. What happened on the first GPU runs (2026-10-02, branch `wm-fixer`)

Runner: `infra/modal/fill.py` + `.github/workflows/fill.yml` (CI runs 36973559815 …
36981733755). Each job runs `teacher_fill.py drop|fill` in a CPU container; `Fixer` is a
class of the same app on an L40S (`world_model_client.LOCAL_CLASSES`), nothing deployed.

- **No NGC, no secrets.** NGC's `cosmos-predict2-container` needs a key (anonymous pull:
  401). Its environment is cosmos-predict2's own `uv.lock` (commit `661da47` = 1.0.9) on
  `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`, installed with `uv sync --frozen` into the
  image's Python 3.10 (`--locked` fails: Modal's PyPI mirror reads as a stale lock). Image
  builds in a few minutes. `nvidia/Fixer` is not gated. The workspace's only Modal secret
  is the object-storage one; `huggingface`/`ngc` do not exist.
- **The model is sound**: every checkpoint key loads (`load.json`), and Fixer's own
  examples come out cleaned (`fill.py --selftest`).
- **Our renders are out of its distribution.** The CPU renderer's point-sampled frames
  (black background, speckle) come back as a blur at the README's timestep 250, and the
  camp from outside as a uniform textured field whatever the input filter or resolution
  (`fill.py --probes camp`). At timestep 100 it keeps the layout, at 50 more so
  (`FixerFiller?timestep=50`). A gsplat-rasterized input is the likely real fix.
- **Gate.** Fixer re-renders every pixel, so a `reads_full_render` filler is now gated on
  the blurred (2 px) frame against the render it was shown, at 20 dB
  (`GATE_FULL_RENDER_PSNR_DB`); an inverted frame still scores under 10.
- **Drop test** (640x360, 4 views + held-out): yard (0.4 m) held-out psnrFill/psnrHole
  Telea 20.55/20.57, Fixer t50 20.55/20.57, t250 20.56/20.57; spool (0.15 m) Telea
  8.92/8.93, Fixer t50 8.89/8.93. Nobody moves the held-out view: the lift puts the fill at
  the depth seen through the hole (behind the dropped region), so the held-out score does
  not discriminate fillers yet. In the fill views Fixer beats Telea in most views (yard t250:
  22.6/22.8/20.1/23.8 vs 22.2/22.6/19.8/23.7 dB).
- **Camp from outside, Telea**: 500,901 inferred gaussians, 8 views, mean confidence
  0.42, ~8 min per run. Fixer at 250 and presmoothed: every view refused. Fixer t50/t100
  with the corrected gate: run 36981733755.

## 6. gsplat renders (2026-10-02, branch `wm-gsplat`)

`teacher_fill --renderer gsplat` (`splat_render.GsplatRenderer`, through
`distill_fill.gsplat_frame`) rasterizes every view on a GPU; `fill.py --renderer gsplat`
runs the job on an L4 (`run_job_gsplat`). Runs 36992095315 (gsplat), 36994749804
(+ `--max-scale-m 0.5`, drop-depth fix), 36997466266 (+ Distill 1500).

- **Parity** (`tests/test_gsplat_parity.py`, `fill.py --parity-test`, yard, 8 ring views):
  depth median 0.5-0.8 % apart, p90 2.5-11 %; gsplat covers 98-99 % of what the CPU covers
  (IoU 0.70-0.77: it also covers the CPU's speckle gaps); fill masks agree on 98.4-99.8 %
  of pixels. From inside the yard's canopy they differ by design: the CPU renderer thins any
  gaussian large on screen to `max_samples` samples, gsplat draws it whole.
- **What gsplat showed.** From outside, the camp's full render is mostly its edge
  floaters (scale 0.5-2.7 m, 0.4 % of 22.6 M gaussians), which the CPU renderer had all but
  hidden. `fill --max-scale-m 0.5` conditions without them; what is left of the unseen side
  is the surrounding forest's back faces.
- **Camp from outside, gsplat + max-scale 0.5** (8 views, gate 20 dB blurred): Telea 8/8;
  Fixer t250 0/8 (11.9-15.9 dB, a grey fog); t100 6/8 (19.4-22.0); t50 8/8 (25.7-27.7, was
  25.7-26.5 on CPU renders). t50 keeps the layout and cleans the foliage; t250 does not
  keep it. With Distill 1500 (L40S, ~4 min) Fixer t50's layer is coherent forest, masked L1
  0.05 -> 0.015 and the measured pixels improve too (outside L1 0.015 -> 0.003); Telea's
  goes 0.21 -> 0.07 masked but spoils the measured in 4 of 8 views (grey patches).
- **Drop test** lifts the hole at the depth interpolated from the scan just outside the
  region (`hole_depth="surround"`, a shell of 2x its half size), not what shows through
  the hole. Held-out psnrFill/psnrHole (gsplat): yard Telea 16.71/15.56, Fixer t250
  16.53, t100 16.53, t50 16.53; spool Telea 8.05/7.37, Fixer t250 7.52, t100 7.48, t50 7.47.
  Telea is ahead on both; on the spool Fixer's fill views are near the hole's own score.

## 7. Split objects: the pumpkin's hole (2026-10-02, branch `wm-c4`)

`fill.py --jobs split:pumpkin` runs `split_objects.py split` (C4) on the pumpkin scan with
`--ids 3 --absorb` (the red pumpkin and 40 fragments segmentation left under other ids,
27,319 leaf gaussians; 2,124 merged parents), gsplat renders (L4), 6 views at 1024x576,
`--max-scale-m 0.5`, Distill 1500 (L40S). Runs 37028005624, 37029416454, 37029881769.

- **Split**: 4 tiles rewritten (all of them held some of it), instances re-bound, the object a
  one-tile tileset; the plane under it from 101,974 gaussians around its footprint.
- **The hole**: the held-out view sees through 91,480 px where it stood, 0.4% covered;
  after the fill 99.2% (Telea), 99.2% (Fixer t50), 99.2% (t100), 99.2% (t250). Every view
  passed the gate (Fixer t50 34-39 dB blurred, t100 29-34, t250 25-29).
- **Fixer is a cleaner, not an inpainter.** Shown the scan with an empty hole, t50 kept it
  black (the first run). Shown a rough Telea fill of the hole (now `fill_hole`'s input to
  every filler), t50, t100 and t250 all clean the grass around it and leave the flat Telea
  colours inside as they were: the fill closes the hole in a plausible ground colour, with
  no texture, and a red speck of a fragment that stayed behind seeds a red blob in some
  views. A hole the size of an object wants a generative inpainter (an inpainting diffusion
  model behind the same `Filler` interface), not Fixer.
- **Distill 1500** (L40S): masked L1 0.035 -> 0.016 / 0.027 -> 0.006 / 0.040 -> 0.006 in the
  first three views; 8-22 s.
- **GPU time**: each job 44-98 s on the L4 (Fixer 36-53 s and Distill 8-22 s of it on
  L40S calls); about 10 min of L4 wall over the four runs (one a duplicate push).

## 8. Generative inpainting for holes (2026-10-02, branch `wm-inpaint`)

`world_model_client.GenerativeFiller` is a `Filler` that paints the mask with an inpainting
model (`tools/captures/inpaint_models.py`, run by `InpaintSDXL` / `InpaintQwen` /
`InpaintFlux` in `infra/modal/fill.py` and `world_models.py`). `fill.py` checks the
workspace's Hugging Face token against every model first (`inpaint-access.json`), drops jobs
whose model it cannot read, and fetches the weights into the volume once. Runs 37051163053,
37053677004, 37057636855 (pumpkin), 37059869195 (camp). Fillers: `lama`, `sdxl`, `qwen`,
`sdxl-lama` (LaMa then SDXL at strength 0.6), each with `-chain`.

| Model                                                      | Weights                                                        | Licence                                                                           | GPU, per 1024x576 view                                             |
| ---------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Qwen-Image + inpainting ControlNet                         | `Qwen/Qwen-Image`, `InstantX/Qwen-Image-ControlNet-Inpainting` | Apache-2.0 (both)                                                                 | H100, 7-9 s (58 GB; cold load from the volume 12 min, warm ~1 min) |
| LaMa (big-lama, IOPaint's TorchScript export, md5 checked) | GitHub release `Sanster/models`                                | Apache-2.0                                                                        | L40S, 0.3-2 s                                                      |
| SDXL inpainting 0.1                                        | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`             | CreativeML OpenRAIL++-M (use restrictions travel with the weights)                | L40S, 1.5-3 s                                                      |
| FLUX.1 Fill [dev] (not run)                                | `black-forest-labs/FLUX.1-Fill-dev`                            | FLUX.1 [dev] Non-Commercial: the model may not be used commercially (outputs may) | gated: `GatedRepoError 403` for the workspace's token              |

- **Secrets.** The workspace's Hugging Face secret is not named `huggingface` (it starts
  with `huggingface-secr`): `fill.yml` finds it by prefix (`HEXAPOD_HF_SECRET`), and
  `inpaint_models.find_token` reads whichever key holds an `hf_` value.
- **What made it work on the pumpkin.** (1) Shown the whole frame, both diffusion models
  painted an _object_ into the object-shaped hole (Qwen a blue bowl, then a mushroom; SDXL
  an orange disc). (2) A crop 2.5x the hole, a texture prompt ("<labels>: a top-down close-up
  photograph of the ground, a seamless natural texture ...") and the object's own labels
  only as negative (its parts carry the ground's labels: dirt, leaves, rock) removed the
  objects, but every model then copied the rough pre-fill's flat polygons beyond the scan's
  edge. (3) The void (`Conditioning.void`: nothing measured, nothing asked) is now repainted
  with the hole and discarded (`reads_void`), so the model reads only measured pixels.
  The prompt is `teacher_fill.describe_surroundings` over instances.json around the
  footprint (tags weighted by score x gaussians x footprint share): "Dirt, ground, forest
  floor and moss"; the bale is straw, which no tag says.
- **Multi-view consistency.** Same seed in every view; `chain=1` fills the views in turn,
  each shown the earlier views' fill lifted onto the plane and re-rendered, only the rest
  masked (`teacher_fill._chained_fill`). The masked L1 between the lifted layer and each
  view's fill before distill measures how much the views disagree.
- **Pumpkin split** (ids 3 --absorb, gsplat L4, 6 views, Distill 1500; held-out
  coverage of the 91,480 see-through px 0.4% before):

  | filler          | held-out covered | masked L1 before -> after distill (mean) | outside L1     | look                                     |
  | --------------- | ---------------- | ---------------------------------------- | -------------- | ---------------------------------------- |
  | telea           | 99.2%            | 0.035 -> 0.009                           | 0.020 -> 0.005 | flat Telea polygons, a red blob          |
  | fixer-t50       | 99.0%            | 0.034 -> 0.009                           | 0.020 -> 0.005 | the same, cleaned at its edge            |
  | lama            | 99.6%            | 0.043 -> 0.015                           | 0.016 -> 0.004 | dark, blurred straw texture              |
  | lama-chain      | 99.4%            | 0.017 -> 0.008                           | 0.016 -> 0.004 | the same, smoother across views          |
  | sdxl            | 99.1%            | 0.069 -> 0.016                           | 0.024 -> 0.005 | an orange pumpkin-like streak            |
  | sdxl-chain      | 99.3%            | 0.011 -> 0.005                           | 0.026 -> 0.005 | orange streak, consistent                |
  | sdxl-lama-chain | 99.5%            | 0.015 -> 0.007                           | 0.018 -> 0.004 | dark smooth patch                        |
  | qwen-chain      | 99.6%            | 0.030 -> 0.015                           | 0.019 -> 0.004 | straw-and-soil texture, no object (best) |

  Every view of every filler passes the gate (only the mask is kept, so the measured
  pixels are untouched; the models' own re-encoding of the rest scores 22-31 dB, discarded).
  Coverage does not separate fillers; the strips do. The fill is darker than the bale:
  the scan's ground around the pumpkin is in its shadow.

- **Camp from outside** (gsplat, max-scale 0.5, 8 views, Distill 1500): `qwen-chain`
  paints sharp, plausible forest in each view, but independent views disagree (masked L1
  0.19 -> 0.11 after distill, against Fixer t50's 0.05 -> 0.015) and distill cannot fit the
  measured pixels better (outside L1 0.019 -> 0.017; Fixer 0.015 -> 0.003): the layer is a
  blotchy mix. `lama-chain`: 0.16 -> 0.07, outside 0.017 -> 0.016, a smeared green.
  For the camp from outside Fixer t50 + distill stays the better layer: it cleans one
  consistent render rather than inventing eight.
- **GPU time** (caller-side wall, overlapping): L4 jobs 86 min over 23 jobs; H100 (Qwen)
  40 min, of which ~25 min cold loads of 58 GB from the volume in the first runs; L40S
  SDXL/LaMa 9 min; L40S Distill 17 min; weight prefetch (CPU) 7 min.
- **Next.** A hole wants the generative fill inside it and the scan's own lighting: Qwen
  chained is the one to keep for split objects. The prompt would gain from a caption of the
  surroundings (the tags miss "straw"); the camp wants Fixer, or a multi-view model.
