# Living view bake-off: cost ledger

Target about $7 (raised from $6 for the upscalers), hard cap **$10** for the whole bake-off,
model downloads, image builds and cold starts included. Every run is estimated before it is
pushed (the worst case: each call's timeout times the list price of what its container
reserves -- GPU, CPU cores, memory -- must fit what is left) and recorded after from what
Modal metered (below).

Modal list prices, read 2026-10-08 (modal.com/pricing): H200 $0.001261/s, H100 $0.001097/s,
L40S $0.000542/s, L4 $0.000222/s; CPU $0.0000131 per physical core-second; memory
$0.00000222 per GiB-second. Volume storage $0.09/GiB-month past 1 TiB free (the weights
volume `hexapod-living-view-weights`, about 190 GB, is inside that; delete it once decided).
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
| 10 | 37807747533 | app ap-xeM3pdygX2xMa08RkhwD5W. FlashVSR on the 12 clips already in the volume (ltx, flf, wan), one A100 container. Estimate $0.9 (worst $2.26); still running ($0.86 metered by 16:36) | A100-80GB | | | |
| 11 | 37809248102 | SeedVR2-3B weights (14.6 GB) and its image check, on CPU. Estimate $0.05 | CPU | | | |
| 12 | 37809859637 | meter: Modal's billing report and run 7's app logs, from the runner (no container) | none | 0 | 0.00 | 3.31 |
| 13 | (pending) | arm causal on an H100 (no L40S to be had). Estimate $0.45 (worst $1.13) | H100 | | | |
