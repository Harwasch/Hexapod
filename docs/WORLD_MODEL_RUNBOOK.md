# World-model runbook: the first GPU run

Everything below runs on the CPU today with stand-ins (Telea inpainting for fill, damped
oscillators for motion). This page covers swapping in the real models on Modal.

| Teacher         | Stand-in (CPU, tested)         | GPU model            | Modal class   | Client                              |
| --------------- | ------------------------------ | -------------------- | ------------- | ----------------------------------- |
| B: fill         | `InpaintFiller` (Telea)        | NVIDIA Fixer         | `Fixer.fix`   | `world_model_client:FixerFiller`    |
| B: refine       | `distill_fill.torch_rasterize` | gsplat 1.5.3         | `Distill.run` | `teacher_fill.py fill --distill N`  |
| A: motion       | `OscillatorClips`              | Wan 2.2 TI2V-5B      | `Wan.clip`    | `teacher_motion.py --source wan`    |
| A: motion (alt) | (same)                         | Cosmos-Predict2.5-2B | `Cosmos.clip` | `teacher_motion.py --source cosmos` |
| C2: materials   | `teacher_materials.py synth`   | Wan / Cosmos         | `Wan.clip`    | `teacher_materials.py world`        |

Code: `infra/modal/world_models.py` (server), `tools/captures/world_model_client.py`
(client), `teacher_fill.py`, `teacher_motion.py`, `distill_fill.py`.

## 1. Credentials

| Need                  | Where it goes                                                                   | Why                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Modal token           | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the shell                            | deploy and call                                                                                                        |
| Hugging Face token    | Modal secret `huggingface` (`HF_TOKEN`)                                         | Wan, Cosmos weights (Fixer's `nvidia/Fixer` is not gated and needs none); the workspace had no such secret on 2026-10-02 |
| HF licence acceptance | huggingface.co, on the token's account                                          | Cosmos only: Cosmos-Predict2.5-2B, Cosmos-Reason1-7B, Cosmos-Guardrail1 (all gated; `harwasch` had none on 2026-10-01) |
| ~~NGC API key~~       | not needed                                                                      | Fixer's NGC container is replaced by the same environment built from cosmos-predict2's `uv.lock` (section 5)            |
| R2 credentials        | already used by `infra/modal/app.py`                                            | only to publish an inferred layer                                                                                      |

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
