# Living view bake-off: cost ledger

Target about $7 (raised from $6 for the upscalers), hard cap **$10** for the whole bake-off,
model downloads, image builds and cold starts included. Every run is estimated before it is
pushed (the worst case: each call's timeout times the list price of what its container
reserves -- GPU, CPU cores, memory -- must fit what is left) and recorded after from what
Modal metered (below).

Modal list prices, read 2026-10-08 (modal.com/pricing): H200 $0.001261/s, H100 $0.001097/s,
L40S $0.000542/s, L4 $0.000222/s; CPU $0.0000131 per physical core-second; memory
$0.00000222 per GiB-second. Volume storage $0.09/GiB-month past 1 TiB free (the weights
volume `hexapod-living-view-weights`, about 220 GB, is inside that; delete it once decided).
GitHub runner minutes are not Modal spend and are not counted.

## Models and licences

| arm | model | licence | shippable? |
| --- | --- | --- | --- |
| ltx | Lightricks/LTX-2.5-Diffusers (distilled) + Lightricks/LTX-2.5-22b-LoRA-Cinemagraph | LTX-2.x Community License: commercial use free under $10M annual revenue (accepted by the owner) | yes, under the revenue threshold |
| causal | zhuhz22/Causal-Forcing (frame-wise) on Wan-AI/Wan2.1-T2V-1.3B; code thu-ml/Causal-Forcing | Apache-2.0 (weights, code, base) | yes |
| flf | Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers | Apache-2.0 | yes |
| wan | Wan-AI/Wan2.2-TI2V-5B-Diffusers (control) | Apache-2.0 | yes |
| (B) | OpenCV DIS optical flow | Apache-2.0 (OpenCV), no learned weights | yes |
| A↑ | JunhaoZhuang/FlashVSR-v1.1 (tiny decoder); code OpenImagingLab/FlashVSR; its sparse attention re-written in PyTorch here | Apache-2.0 (weights and code) | yes |
| A↑ (second) | ByteDance-Seed/SeedVR2-3B; code from the Space ByteDance-Seed/SeedVR2-3B; flash-attn 2.7.4 (BSD-3-Clause); apex replaced by torch norms | Apache-2.0 (weights and code) | yes |

Fallbacks, only if Causal Forcing cannot start from a frame: TencentARC/RollingForcing (MIT);
krea/krea-realtime-video (the Hub card says Apache-2.0, its GitHub repository says CC BY-NC-SA
4.0 -- non-commercial; not for anything that could ship).

## Runs

Dollars are what Modal metered (`modal billing report --for today -r h --show-resources`,
saved by the workflow's `meter` step) for the run's app, so they include image builds,
network egress and cold starts, which the first version of this ledger (booked from the
runner's estimates) under-counted by $0.49 before run 10. Where one app ran several arms, its
GPU line is exact per GPU type (the two H100 arms split by container seconds) and its CPU and
memory lines are shared by reservation x container seconds. Minutes are GPU minutes.

| # | GitHub run | what | GPU | minutes | $ | running total |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 37794816442 | app ap-LgteCwzYzFSlszWG4fMJhD. Access check: the `huggingface` secret is account harwasch and reads every repo, the three LTX ones included; weights into the volume on CPU (FLF2V 90.1 GB, Causal Forcing + Wan 2.1 1.3B 23.2 GB, LTX-2.5 82.6 GB; $0.22 of it network egress); starts failed at once (a module-name clash, fixed). Estimated $0.30, booked $0.06 at first | CPU, L4 | 1.2 | 0.34 | 0.34 |
| 2 | 37796863736 | app ap-PQ0SED6xnlOFU7LXCNiPFq. Cancelled before any arm ran: the app's images build before anything runs, and FlashVSR's kernel compile (then in the same app) held up the starts; the build itself was metered. The upscaler moved to an app and job of its own | builder | 0 | 0.13 | 0.48 |
| 3 | 37799382669 | apps ap-orYY1j7AajnebYceox9d90, ap-Hkns9QVJX7DMEURsvY4vqd. Starts: camp done (two views; one was a wall of leaves, re-chosen in run 4), tree failed reading back its result (a numpy scalar, fixed); FlashVSR weights and image build on CPU in the other job | L4 | 2.1 | 0.12 | 0.60 |
| 4 | 37799908960 | app ap-ZIRh7lAvCNMNxD2FSG8eO0. Starts, both scenes (the tree's two were both close-ups; re-chosen in run 5) | L4 | 1.8 | 0.04 | 0.64 |
| 5 | 37800517764 | app ap-5dG6wMxXwwoErbeMKNi26e ($2.50 in all). Starts again (tree: a close crown and the whole crown; camp: shrubs and signboard, conifers and cabin) | L4 | 3.6 | 0.08 | 0.71 |
| 5 | 37800517764 | arm wan: 4 clips, 124 s each warm, load 98 s, container 606 s | H100 | 10.2 | 0.74 | 1.46 |
| 5 | 37800517764 | arm flf: 4 clips, 200 s each warm, load 183 s, container 993 s | H100 | 16.6 | 1.34 | 2.80 |
| 5 | 37800517764 | arm ltx: 4 clips, 9-11 s each warm, load 85 s, container 136 s (metered 141 s of H200) | H200 | 2.4 | 0.22 | 3.02 |
| 5 | 37800517764 | arm causal failed: Wan's cross-attention calls flash-attn directly (`assert FLASH_ATTN_2_AVAILABLE`), after the model load | L40S | 2.7 | 0.11 | 3.14 |
| 6 | 37802115592 | app ap-nSUsLfMMgPaZOY4guTBzT5. FlashVSR image build on Modal's builders: Block-Sparse-Attention's kernels compiled (28 min) but its link step called clang++, absent from the image; replaced by the same block-sparse attention in PyTorch (checked against dense masked attention) | builder | 0 | 0.16 | 3.30 |
| 7 | 37805436202 | app ap-ownvnI9x4IFZAjijgXb0Jr. Arm causal again, with flash-attn 2.7.4: never ran -- Modal had no L40S to give for 31 minutes ("waiting to be scheduled on a GPU_L40S worker"); cancelled. Estimated $0.20 (worst $0.63) | none | 0 | 0.00 | 3.30 |
| 8 | 37806924930 | app ap-Ygi4GEYlrke7PkaS4o1dhi. FlashVSR weights (7 GB, 31 s) and the image check on CPU: diffsynth needs `modelscope`, not in FlashVSR's requirements (caught before any GPU) | CPU | 0 | 0.01 | 3.30 |
| 9 | 37807558216 | app ap-l11LEUbKwrwvvulb1WnNfQ. FlashVSR image check: imports clean | CPU | 0 | 0.00 | 3.31 |
| 10 | 37807747533 | upscale flashvsr: the 12 clips of ltx, flf, wan in one A100 container (app ap-xeM3pdygX2xMa08RkhwD5W): load 25 s, then 151-237 s a clip (0.3 fps at 4x for the 848 x 464 clips, 0.55-0.64 fps at 2x for the 1280 x 704 ones); container 2257 s of its 2400. Estimated $0.9 (worst $2.26) | A100-80GB | 37.8 | 2.14 | 5.44 |
| 11 | 37809248102 | SeedVR2-3B weights (14.6 GB, 46 s) and its image check on CPU: flash-attn imports, the code imports with torch norms for apex's, the configs make the DiT and the VAE (app ap-wHgQNcC3lKMT6OTId3HGyg). Estimated $0.05 | CPU | 0 | 0.01 | 5.45 |
| 12 | 37809859637 | meter: Modal's billing report and run 7's app logs, from the runner (no container) | none | 0 | 0.00 | 5.45 |
| 13 | 37810483113 | arm causal: on an H100, 8 clips (4 starts, still contexts of 1 and 3 latent frames), first motion 0.41-0.82 s, 65 frames in 5.6-6.3 s, load 80 s, container 143 s (app ap-pJTVVAtQwy0DAP1I9BRqE0). Estimated $0.45 (worst $1.13) | H100 | 2.4 | 0.19 | 5.64 |
| 14 | 37813220982 | upscale flashvsr: the 8 causal clips at 2x (it reaches the render), 41-51 s a clip (1.3-1.6 fps), load 18 s, container 402 s (app ap-eMUihRp456vIuKe7Rl7X1E, $1.18 with the next row). Estimated $0.45 (worst $1.13) | A100-80GB | 6.8 | 0.38 | 6.02 |
| 14 | 37813220982 | upscale seedvr2: all 20 clips inside its 600 s wall (none skipped): load 188 s, then 14 s a 65-frame clip and 21 s a 97-frame clip at 1280 x 704 (4.7 fps), container 591 s. Estimated $1.0 (worst $1.61) | H100 | 9.9 | 0.80 | 6.82 |
| 15 | 37815636536 | meter: Modal's billing report again, for runs 10-14 (no container) | none | 0 | 0.00 | 6.82 |

## Total

**$6.82** metered by Modal for the whole bake-off, against the target of about $7 and the
hard cap of $10: the four arms $2.60 (causal's failed first try included), the upscalers $3.32
(FlashVSR $2.52 for 20 clips, SeedVR2-3B $0.80 for 20), starts, downloads, image builds and
failed runs the rest. Still to do once the comparison is decided: delete the weights volume
`hexapod-living-view-weights` (about 220 GB; inside the free TiB, so it costs nothing while
it stays) and the results volume `hexapod-living-view`.

## Round 2

Approved separately: target about $6, hard cap **$8**, on top of round 1's $6.82. Same rules:
each run's worst case (timeouts x list rate) must fit what is left before it is pushed, and it
is recorded from what Modal metered. Prompts: `bakeoff/living-view/prompts.md`.

| # | GitHub run | what | GPU | minutes | $ | running total |
| --- | --- | --- | --- | ---: | ---: | ---: |
| R2-1 | 37842888808 | the 2k and 4k stills (done: L4, 23 s and 89 s); r2 ltx prompts failed at load in 18 s: LTX-2.5's prompt enhancer config (saved by transformers 5.15) is heterogeneous and 5.14.1 refuses it. Runner's estimate | L4, H200 | 0.3 | 0.07 | 0.07 |
| R2-2 | (pending) | FlashVSR image with Block-Sparse-Attention's kernels (clang added), built and checked on CPU | builder | | | |
| R2-3 | (pending) | r2 ltx prompts again (the enhancer loaded on its own, global per-layer reads allowed; ltx-p3 skipped if it still fails). Estimate $0.7 (worst $3.0) | H200 | | | |
| R2-4 | (pending) | world models' weights on CPU: Waypoint-1.5-1B (11 GB), Matrix-Game-3.0 distilled + T5 + VAEs (41 GB), Yume-5B-720P (35 GB). Estimate $0.15 with egress | CPU | | | |
