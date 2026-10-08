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
| 3 | (pending) | starts (L4) and, side by side, FlashVSR's weights and image build (CPU). Estimate $0.10 (worst $1.8) | L4, CPU | | | |
