# Living view, round 2: one generic prompt per model

The owner's rule: one prompt per model for every view, no scene words, so it scales to any
view in real time. It has to make whatever is there come alive slightly (foliage, grass, water,
flags, cloth) and keep everything else rigid and the camera frozen, without naming what is in
the particular view. This file records what each model's own sources say about prompting, the
prompt derived from that, and why. Read 2026-10-08.

## LTX-2.5 + Cinemagraph LoRA

### What the sources say

- **The LoRA's card** ([Lightricks/LTX-2.5-22b-LoRA-Cinemagraph](https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Cinemagraph),
  gated, read with the harwasch token). Every example and widget prompt has one shape, a
  comma-separated caption in this order:
  1. the trigger `CINEMAGRAPH_MOTION`, first;
  2. `tripod locked-off static camera, zero camera movement`;
  3. what stays frozen, listed (`the man, face, hair, clothing, beach, sky, and background remain
     completely frozen`);
  4. `only the <element> moves`, then how it moves (`roll and shimmer`, `drift and roll`);
  5. a containment clause (`everything outside the glasses stays still`,
     `everything else stays perfectly still`);
  6. `seamless natural loop`.

  Its tips: "Be very specific about what moves and what doesn't. 'Only the [element] moves;
  everything else frozen' works better than vague descriptions." It also says to leave out
  camera words such as "camera movement", "pan", "zoom" and "parallax", because the LoRA was
  trained on static-camera footage. Read with its own examples, that means don't *ask* for
  camera motion: every example still says `zero camera movement`.

  The recommended settings are for the full model: 30 steps, guidance 4.0, STG (`stg_v`,
  scale 1.0, block 29). LoRA strength runs 0.5–1.5: "`0.5` is conservative; `1.0–1.2` is the
  sweet spot; pushing toward `1.5` drives selective motion more aggressively."

  It was trained at 512×704 with **25 frames**. Our 97 frames at 1280×704 is four times longer
  than it was trained on, which is one reason round 1's motion grew large (p95 15–59 px).
- **LTX's own guides** ([LTX-2.5 prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide),
  [prompting guide for LTX-2](https://ltx.io/blog/prompting-guide-for-ltx-2),
  [camera movement guide](https://ltx.io/blog/camera-movement-prompt-guide),
  [negative prompts](https://ltx.io/blog/negative-prompts)):
  - write one flowing paragraph in the present tense, about 4–8 sentences, in the order shot,
    scene, action, camera;
  - "Static is a real instruction and worth writing explicitly. 'Static frame' pins the camera
    … Leaving camera language out entirely is not the same as asking for static." `static frame`
    is in the documented camera vocabulary, and synonyms do worse;
  - for image-to-video, prefer a single continuous take, with no cuts;
  - negative prompts act only through classifier-free guidance, so keep them to 5–15 tokens.
    For image-to-video the guide adds "static, frozen, no motion, Ken Burns zoom".
- **The pipeline's prompt enhancer** (`LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` in diffusers
  [`pipelines/ltx2/utils.py`](https://github.com/huggingface/diffusers/blob/7564fb016dabda0c943416190fc92398c50b1b20/src/diffusers/pipelines/ltx2/utils.py),
  run by LTX-2.5's dedicated Gemma 4 `prompt_enhancer`). It turns the image plus the user's
  request into a 150–220-word caption in the style of the training captions:
  - the first frame described exactly;
  - one shot type and one viewpoint;
  - "Camera motion (always stated; if none, explicitly say the camera remains static)";
  - a full soundscape, a chronological single paragraph, and film-grade quality words.

  It also says "Camera movement is expected and good — match the user if they specified it",
  so the request has to state the static camera, or the enhancer may invent a move.
- **Our recipe and guidance.** We run the distilled path as the LoRA's distilled ComfyUI
  workflow does: 8 + 3 sigmas, guidance 1, STG off. At guidance 1 the negative prompt is inert,
  so freezing the scene rests entirely on the positive prompt and the LoRA strength.

### The generic prompts tested (arms ltx-p1 to ltx-p5)

**ltx-p1, derived from the research.** It keeps the card's caption shape, which is what the
LoRA was trained on. It lists the rigid classes generically rather than naming this view's
objects, adds the documented `static frame`, makes the motion verbs small, and names water and
fabric only as "already in the frame", so nothing new is invited:

> CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera movement, static frame, the
> sky, the ground, every building, wall, rock, trunk and object remain completely frozen, only
> the existing leaves, grass and thin twigs sway very slightly in a light breeze, any water or
> fabric already in the frame ripples gently, nothing new appears, everything else stays
> perfectly still, seamless natural loop

**ltx-p2, the coordinator's simple generic prompt** (the owner's rule as first given):

> CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera movement, only the leaves,
> grass and thin branches sway slightly in a gentle breeze, everything else stays perfectly
> still, seamless natural loop

**ltx-p3, auto-captioned.** No person writes it. The pipeline's own prompt enhancer (Gemma 4,
the default LTX-2.5 image-to-video system prompt) gets the render and ltx-p1 as the user's
request and writes the caption; `CINEMAGRAPH_MOTION, ` is put in front if it is missing. It is
scalable, but it costs the enhancer's time (recorded) and its text differs per view (each one
is kept in `numbers.json`).

All three use the card's generic negative prompt, as in round 1; it is inert at guidance 1.
The LoRA strength is 1.0 (the card's sweet spot). If even the best prompt moves more than a
few pixels, the card's own lever is strength (0.5 is "conservative"), so ltx-p1 is also run at
0.6 as ltx-p4.

The winner, by plant motion in a slight range (a few px p95 at 1280 wide), low non-plant drift
and camera creep, and a read by eye, is used for the rest of the LTX ladder. The scores are in
`numbers.json` and below.

### Scores (run R2-5, H200, the round-1 recipe: 97 frames at 1280x704, seed 42)

Measured on the model's own frames (`r2_deliver.py`). Plant motion is DIS optical flow against
frame 0 inside the plant mask, in pixels at 1280 wide. Creep is a homography fitted on the
textured non-plant pixels. Drop is how many dB the non-plant PSNR against our render loses from
the first frame to the last. Round 1's LTX clip (scene prompts) is given for scale.

| arm | prompt | plant p95 px (tree-1, tree-2, camp-1, camp-2) | mean | camp creep px (camp-1, camp-2) | worst drop dB | mean abs frame change |
| --- | --- | --- | ---: | --- | ---: | ---: |
| ltx (round 1) | scene words | 24.4, 26.6, 14.5, 59.1 | 31.1 | 0.89, 1.75 | 2.0 | 12.8 |
| **ltx-p1** | researched | 23.4, 27.8, 20.1, 56.8 | 32.0 | **0.53, 0.72** | 1.7 | **12.7** |
| ltx-p2 | coordinator's | 25.1, 30.8, 18.3, 41.5 | **28.9** | 0.56, 2.06 | 1.4 | 13.8 |
| ltx-p3 | auto-captioned | 21.2, 29.0, 20.8, 64.0 | 33.8 | 0.78, 2.48 | **9.5** (tree-1) | 13.9 |
| ltx-p4 | p1 at LoRA 0.6 | 26.5, 33.2, 20.2, 62.2 | 35.5 | 0.74, **8.37** | 1.0 | 15.2 |

All of them take about 10 s for 4 s of video on a warm H200 (0.34-0.40 s of video per second
of compute). The prompt encodes in 0.1-0.2 s. The enhancer adds 5.7-7.7 s per view (ltx-p3).

**Winner: ltx-p1.** It has the least camera creep (at most 0.7 px over 4 s) and the least
change overall. Its plant motion ties with ltx-p2 (mean flow 7.0 vs 7.2 px, p95 32 vs 29: p2
moves camp-2's near conifer less, and the trees and camp-1's ferns more). By eye, p1 is the
most contained: on camp-1 the motion stays in the fern bush, where p2, p3 and round 1 also stir
the tree on the left.

**What the scores show.**

- **The wording does not set the amplitude.** No generic prompt gets plant motion into the
  "few px" range. Every variant swings the near foliage 20-60 px (p95) against frame 0, the same
  as round 1's scene prompts. "Very slightly" (p1) and "slightly" (p2) score alike. At guidance
  1 the text steers what moves more than how far it moves.
- **The LoRA strength works the other way from what we guessed.** At 0.6 (ltx-p4) everything
  moves more, and camp-2's camera creeps 8.4 px. A stronger LoRA holds the frame better, so
  ltx-p1 at 1.4 runs with the ladder as **ltx-p5**. The card calls 1.5 "aggressive".
- **Auto-captioning (ltx-p3) is a regression.** The enhancer writes a 150-220-word cinematic
  caption with its own mistakes: for tree-1's crown it wrote "a high-angle viewpoint looking
  down". It also appends a soundscape. Its tree-1 sky drifts 9.5 dB by the last frame, and it
  costs 6-8 s per view.

## Matrix-Game 3.0 (Skywork/Matrix-Game-3.0)

- **Text: yes.** A umT5-XXL encoder feeds cross-attention (`use_text_crossattn: true` in
  `base_distilled_model/config.json`). The documented prompts are plain one-sentence scene
  captions: "a vintage gas station with a classic car parked under a canopy, set against a
  desert landscape." ([model card](https://huggingface.co/Skywork/Matrix-Game-3.0),
  [code](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3), Apache-2.0
  weights, MIT/Apache code).
- **The camera is not text.** It comes from keyboard (W/S/A/D) and mouse (pitch/yaw) actions,
  turned into camera poses and Plücker embeddings. "Idle" is all keys up and the mouse at zero,
  so every pose stays at the first. The distilled model runs 3 steps without guidance, so its
  negative prompt (which lists "static, still image") is unused.
- **Generic prompt** (no camera words, because the actions own the camera):

  > A realistic view in a light breeze: leaves, grass and thin branches sway slightly and
  > settle, any water ripples gently, while the ground, buildings and objects stay rigid and
  > still.

## Waypoint-1.5 (Overworld/Waypoint-1.5-1B)

- **Text: no.** The checkpoint's `transformer/config.json` has `prompt_conditioning: null`, so
  the transformer builds no cross-attention and ignores any prompt embedding. The pipeline's
  text-encoder step exists but its output is unused.
  ([model card](https://huggingface.co/Overworld/Waypoint-1.5-1B)).
- **Conditioning is the starting image plus controls**: a set of pressed buttons (256 key
  codes), mouse velocity (x, y) and scroll. "Idle" is no buttons, mouse (0, 0), scroll 0, frame
  after frame. A pan is a small constant mouse x velocity.
- **No prompt.** The card's example passes "An explorable world", which this checkpoint ignores.
- **Licence:** the weights are Apache-2.0. The Python that runs them (`modular_blocks.py`,
  `transformer/model.py`, `vae/ae_model.py` in the same repository) is **GPL-3.0**. That allows
  commercial use but is copyleft, so a shipped integration would have to comply or reimplement.

## Yume 1.5 (stdstu123/Yume-5B-720P)

- **Text: yes, and the camera is text.** Every 2-second chunk is captioned as a camera clause
  followed by an event clause ([repo](https://github.com/stdstu12/YUME), Apache-2.0;
  `webapp_single_gpu.py`, the single-GPU 5B path):
  `First-person perspective.` + a movement phrase from a fixed vocabulary + a rotation phrase +
  the user's event text. The movement phrases are "The camera pushes forward (W).", "… moves to
  the left (A)." and so on. The rotation phrases are "The camera pans to the right (→).", "…
  tilts up (↑)." and so on. For no movement and no rotation, the web app leaves both phrases
  out. The README's tips: "When executing Camera remains still (·), reduce the Actual distance
  value", and the speed numbers may be dropped altogether.
- **Idle caption:** `First-person perspective. ` + the event text.
  **Pan caption:** `First-person perspective. The camera pans to the right (→). ` + the event
  text.
- **Generic event text:**

  > A gentle breeze: leaves, grass and thin branches sway slightly and settle. Nothing else
  > changes.

  This is the coordinator's generic sentence without its camera sentence, because Yume's
  camera clause already sets the camera and a second camera instruction would contradict it on
  the pan.

## Models without text, and negatives

- Waypoint takes no text; its idle is the zero control.
- For the guided models, the round-1 `NEGATIVE` is kept where a negative is used. LTX's
  distilled path, MG3's distilled path and Yume's distilled sampling run without classifier-free
  guidance, so negatives have no effect there; the positive prompt and the control inputs carry
  everything.
