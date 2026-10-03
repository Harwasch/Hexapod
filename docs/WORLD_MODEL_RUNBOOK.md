# World-model runbook: the first GPU run

Everything below runs on the CPU today with stand-ins (Telea inpainting for fill, damped
oscillators for motion). This page covers swapping in the real models on Modal.

| Teacher         | Stand-in (CPU, tested)         | GPU model            | Modal class   | Client                              |
| --------------- | ------------------------------ | -------------------- | ------------- | ----------------------------------- |
| B: fill         | `InpaintFiller` (Telea)        | NVIDIA Fixer         | `Fixer.fix`   | `world_model_client:FixerFiller`    |
| B: refine       | `distill_fill.torch_rasterize` | gsplat 1.5.3         | `Distill.run` | `teacher_fill.py fill --distill N`  |
| A: motion       | `OscillatorClips`              | Wan 2.2 TI2V-5B      | `Wan.clip`    | `teacher_motion.py --source wan`    |
| A: motion (alt) | (same)                         | Cosmos-Predict2-2B   | `Cosmos.clip` | `teacher_motion.py --source cosmos` |
| C2: materials   | `teacher_materials.py synth`   | Wan / Cosmos         | `Wan.clip`    | `teacher_materials.py world`        |

Code: `infra/modal/world_models.py` (server), `tools/captures/world_model_client.py`
(client), `teacher_fill.py`, `teacher_motion.py`, `distill_fill.py`.

## 1. Credentials

| Need                  | Where it goes                                        | Why                                                                                                                      |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Modal token           | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the shell | deploy and call                                                                                                          |
| Hugging Face token    | Modal secret `huggingface-secret` (found by prefix)  | Wan, Cosmos weights (Fixer's `nvidia/Fixer` is not gated and needs none); token of account `harwasch` (section 8)          |
| HF licence acceptance | huggingface.co, on the token's account               | Cosmos only: nvidia/Cosmos-Predict2-2B-Video2World and nvidia/Cosmos-1.0-Guardrail (both still 403 for `harwasch` on 2026-10-03) |
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

## 8. Dream clips and C2 on generated footage (2026-10-03, branch `wm-dream`)

Runner: `infra/modal/dream.py` + `.github/workflows/dream.yml` (deploys `world_models.py`,
then runs). Run 37079912011 is the complete one. Video models are diffusers 0.40 /
transformers 5 / torch 2.8 on H100s (`world_models.video_image`); `world_models.access()`
reports what the token can read.

- **Wan 2.2 TI2V-5B runs**: 1280x704, 121 frames at 24 fps (5.04 s), 50 steps,
  **~178 s of H100 per clip** warm (176-181 s over 24 clips), plus ~25 s of model load per
  new container (117 s on the very first, downloading into the weights volume). At Modal's
  H100 rate (~$3.95/h) that is about **$0.20 per clip**.
- **Cosmos-Predict2-2B Video2World does not run yet**: the token's account (`harwasch`)
  gets 403 on `nvidia/Cosmos-Predict2-2B-Video2World` and `nvidia/Cosmos-1.0-Guardrail`
  (the guardrail the diffusers pipeline builds; its prompt model Qwen3Guard is open).
  Accept both licences on huggingface.co. A model whose start fails is retried by Modal
  without end (run 37054515134 waited 4 h), so `dream.py` now skips a model `access()`
  reports unreadable, and waits on every call with a timeout.
- **Starts**: gsplat renders over a pale sky from eye-level candidates scored by coverage
  and depth (`starts/*-candidates.png`). The camp's two from its clearing look like
  photographs; the pumpkin's (1.4 m over the patch) is a close, low view.
- **Clips** (`gentle`, `gusty`, `rain` x 3 starts): the camera stays put (the sign, trunks
  and benches do not move); `gusty` moves canopy and ferns visibly more than `gentle`;
  `rain` adds faint streaks and little else. On the pumpkin the model reinvents the
  foreground: the pumpkin turns into a leaf blowing through. A view where plants, not an
  object, fill the frame is what the motion teachers want.
- **C2** (`teacher_materials.py world`, strength 0.1 assumed, `--auto-bearing`, gsplat
  still, three Wan calls chained = 361 frames, 15.0 s): 4 of 5 camp plants `fitted`
  (`fitted-generated`), the 9.7 m tree 231 `weak` (445 of 14,387 px unoccluded from its
  best bearing). Fitted c / zeta / D against the prior (3.6-3.9 / 0.10 / 0.025):
  225 bush 0.9 m 5.0 / 0.62 / 0.008; 244 bush 1.0 m 24.0 / 0.49 / 0.010; 113 bush 4 m
  30.7 / 0.95 / 0.034; 414 conifer 3.8 m 20.7 / 0.69 / 0.148.
- **Read them as "no resonance seen"**, not as materials. The tracked spectra are a
  smooth decay from 0.5 to 8 Hz (`materials/instance-*/spectrum.png`) with no peak, two
  orders below what the prior predicts at 0.5 Hz; the fit reaches for very stiff,
  overdamped (resonance above the band, flat response), two of four past the grid's
  6x-prior edge and a third at it. Wan's breeze is small, broadband jitter of foliage, not the sway of an
  anchored modal structure.
- **Clip length**: 5 s gives Welch segments of ~1.25 s (lowest band ~1.6 Hz); three
  chained calls (15 s) reach ~0.5 Hz. A shrub's 2-8 Hz would be inside that, a tree's
  0.17-0.3 Hz is not. Chaining holds the pose at the joins but drifts colour (the 15 s
  clip desaturates) and restarts the motion's phase each 5 s; longer chains need the
  tracker to re-anchor per link. More length will not by itself make the fit physical.
