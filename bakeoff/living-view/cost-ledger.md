# Living view bake-off: cost ledger

Target about $6, hard cap **$10** for the whole bake-off, model downloads and cold starts
included. Every run is estimated before it is pushed and recorded after, from the run's
`summary.json` (`costs`: each Modal call's wall time from spawn to result, times the list
price of what the container reserved -- GPU, CPU cores, memory). Wall time from the runner
includes queueing and the cold start, so it over-counts slightly, never under.

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

Fallbacks, only if Causal Forcing cannot start from a frame: TencentARC/RollingForcing (MIT);
krea/krea-realtime-video (the Hub card says Apache-2.0, its GitHub repository says CC BY-NC-SA
4.0 -- non-commercial; not for anything that could ship).

## Runs

| # | GitHub run | what | GPU | minutes | $ | running total |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 37794816442 | access check: the `huggingface` secret is account harwasch and reads every repo, the three LTX ones included; weights into the volume on CPU (FLF2V 90.1 GB, Causal Forcing + Wan 2.1 1.3B 23.2 GB, LTX-2.5 82.6 GB); starts failed at once (a module-name clash, fixed). Estimated $0.30 | CPU, L4 | 13.5 | 0.06 | 0.06 |
| 2 | 37796863736 | cancelled before any container ran: the app's images build before anything runs, and FlashVSR's kernel compile (then in the same app) held up the starts; the upscaler moved to an app and job of its own | none | 0 | 0.00 | 0.06 |
| 3 | 37799382669 | starts: camp done (two views; one was a wall of leaves, re-chosen in run 4), tree failed reading back its result (a numpy scalar, fixed); FlashVSR weights and image build on CPU in the other job (see run 3b) | L4 | 1.5 | 0.03 | 0.09 |
| 4 | 37799908960 | starts, both scenes (the tree's two were both close-ups; re-chosen in run 5) | L4 | 1.5 | 0.03 | 0.12 |
| 5 | 37800517764 | starts again (tree: a close crown and the whole crown; camp: shrubs and signboard, conifers and cabin) | L4 | 2.3 | 0.05 | 0.17 |
| 5 | 37800517764 | arm wan: 4 clips, 124 s each warm, load 98 s, container 606 s (wall 642 s) | H100 | 10.7 | 0.78 | 0.95 |
| 5 | 37800517764 | arm flf: 4 clips, 200 s each warm, load 183 s, container 993 s (wall 1000 s) | H100 | 16.7 | 1.34 | 2.30 |
| 5 | 37800517764 | arm ltx: 4 clips, 9-11 s each warm, load 85 s, container 136 s; booked at container time + 60 s for the unmeasured boot (the runner's 1816 s was the wait behind the causal arm, collected in spawn order; fixed) | H200 | 3.3 | 0.31 | 2.61 |
| 5 | 37800517764 | arm causal failed: Wan's cross-attention calls flash-attn directly (`assert FLASH_ATTN_2_AVAILABLE`); container time not measured (it failed on its first clip, after the model load), booked at 5 min | L40S | 5.0 | 0.21 | 2.82 |
| 3b, 6 | 37799382669, 37802115592 | FlashVSR image builds on Modal's builders: Block-Sparse-Attention's kernels compiled (28 min) but its link step called clang++, absent from the image; replaced by the same block-sparse attention in PyTorch (checked against dense masked attention) | builder | | 0.00 | 2.82 |
| 7 | (pending) | arm causal again, with flash-attn 2.7.4. Estimate $0.20 (worst $0.63) | L40S | | | |
| 8 | (pending) | FlashVSR weights (CPU) and its image check (CPU), no kernels to compile | CPU | | | |
