# Living structure: Minnetonka feasibility probe

Research only, based on `9eb3bac0`, with no merge review requested. All motion is
**Simulated**. The parallel `codex/research-living-breeze` work is untouched.

**NO-GO for building clean, general branch/leaf articulation from the tested CPU
cuts.** The scan contains recognizable outer leaf silhouettes, but the tested cuts
mostly recover fragments or foliage patches, without reliable woody attachment.
This is a practical build-out decision, not proof that every segmentation method
must fail. The optional SAM 2 arm is prepared and dry-run only: read-only Git
metadata prevents staging its code into the cloud helper's tracked-file snapshot.
That missing experiment limits the conclusion about the scan itself.

## Inputs and reproducibility

The archive was checked first. Its living-view experiments generated motion;
the feature-field segmentation experiment concerned spool/pumpkin objects.
Neither tested this tree's leaf/twig identity. This probe adds covariance/colour
adjacency, seeded skeleton refinement, native-detail visibility and semantic audits.

The immutable round-3 local mirror is
`../codex-research-living-spectra/.lab-data/minnetonka-tree`. Only terminal tiles
are analyzed: **2,221,311 splats**, including **2,187,227 crown splats** at z ≥ 2 m.
Replacement LOD parents are excluded. The composed tile offsets are retained.
SHA-256 of ordered leaf URI strings and their GLB bytes:
`a5416c3768ab7b88b6f6df2ff2d108633851055a0208a499d44a5ef3832352c0`.
The scan is Matthew Guertin's Minnetonka dataset, CC BY 4.0;
[upstream provenance](https://github.com/Matt1Up/tree-photogrammetry-dataset).

Artifacts are in
`.claude/ops/notes/living-structure/`;
large arrays and playback bindings are under
`~/.cache/hexapod-living-structure/`.
See the notes' `results.md`, `metrics.json`, `renders/`, audit JSON/contact sheets,
recording scripts, gate logs and `cloud-dry-run.txt`.

From `tools/captures`, run each heavy command **sequentially**, after checking
`pgrep -fa 'pytest|vitest|playwright|chromium|living_'` and waiting for other Hexapod
jobs. Set `TMPDIR` and `UV_CACHE_DIR` under the task cache; use at most four workers.

```bash
export TMPDIR=~/.cache/hexapod-living-structure/tmp
export UV_CACHE_DIR=~/.cache/hexapod-living-structure/uv
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
heavy uv run python living_structure.py \
  ../../../codex-research-living-spectra/.lab-data/minnetonka-tree \
  ~/.cache/hexapod-living-structure/probe
heavy uv run python living_structure_render.py \
  ~/.cache/hexapod-living-structure/probe \
  .claude/ops/notes/living-structure/renders
```

The scan cache is for the recorded immutable input only; remove `probe/scan.npz`
before analyzing another source. The CPU renderer needs `g++` locally; cloud uses
Numba instead. No production assets, registration or runtime hooks are changed.

## Arm A: geometry and appearance

One randomly selected representative per 15 mm voxel gives 784,905 graph vertices.
Their median nearest-neighbour spacing is 9.90 mm. A bounded 12-neighbour graph
uses Euclidean separation, RGB distance, unoriented covariance normals, opacity
and covariance size. Splats below alpha 0.15 or above 12 cm maximum sigma cannot
bridge components. Normal agreement is required only for flat Gaussians.
Voxel membership lifts the graph labels back to every native-detail crown splat.

Three parameter combinations are swept, rather than selecting a tolerance to
make one result look good. RGB distances are in [0,1] RGB units; the last value is
minimum absolute normal dot product. Seeds 17 and 29 choose different voxel representatives.

| Radius / RGB / normal | Components, seed 17 | Largest component | Median splats | 3–30 cm pieces ≥8 splats | Crown in those pieces | Contact-supported crown | Seed ARI |
| --------------------- | ------------------: | ----------------: | ------------: | -----------------------: | --------------------: | ----------------------: | -------: |
| 25 mm / .12 / .8      |             526,434 |            0.019% |             2 |                   68,826 |                50.40% |                   1.31% |     .478 |
| 50 mm / .24 / .8      |             321,067 |            12.70% |             2 |                   34,774 |                31.57% |                   0.55% |     .216 |
| 100 mm / .36 / .5     |             266,828 |            57.33% |             2 |                   22,963 |                15.17% |                   0.17% |     .343 |

Size is an axis-aligned bounding-box diagonal including conservative 2σ support,
not botanical leaf length. The 3–30 cm prior spans small leaf clusters and short
twigs; the species and true leaf dimensions have not been measured. For the tight
cut, size p10/p50/p90 is 4.06/7.76/15.31 cm; splat-count p10/p50/p90 is 1/2/9.
Allowing one rather than eight splats raises plausible-size coverage to 96.80%
(502,567 pieces); four splats gives 71.96%. Thus a small component count alone is
not the rejection criterion. The visibility, audits and stability matter.

An additional control removes the representative grid and uses **every crown splat**.
The tight cut has **1,236,369 components**, median one splat, 50.27% of splats in
singleton components; 28,575 plausible-size pieces cover **30.07%** of the crown,
with **1.08%** contact-supported. Input-order seeds give **ARI 0.99983**. The medium
cut has 878,983 components and 18.19% plausible-size coverage; the loose cut has
776,842 components, 5.49% plausible-size coverage and one component holding 35.17%
of the crown. These controls do not improve measured attachment.

**Component counts include vertices deliberately isolated by the opacity/size
filters. They are not all recovered pieces, and the singleton share is not by
itself a measure of missing geometry.** The notes' separate opacity/contribution
control reports this selection effect and lowers alpha to .02. Repeatability can
coexist with bad biological cuts; the native control separates representative
sampling instability from that more fundamental issue.

All pieces can be **mechanically** assigned to a nearest skeleton segment (100%).
That is not recovered biological attachment. “Contact-supported” means a
plausible-size component has at least one splat centre within 4 cm of the original
skeleton. It is only a geometric proxy, not wood truth; missing skeleton branches
can undercount actual attachment. Segment search overestimates exact minimum
distance by at most 1 cm. Thresholds are experimental priors, not calibrated errors.

### Refining the 200-node rig

Space colonisation attracts the existing skeleton toward representative crown
centres: step 6 cm, attraction radius 35 cm, kill radius 8 cm, maximum 1,000 nodes.
This respects the unchanged player's 1,024-node limit. It is generated connectivity
through foliage, not an extracted woody skeleton.

Both seeds add 800 nodes. Crown centres within 10 cm of a segment rise from **7.81%**
to **12.84% / 13.21%**. Only **15.88% / 17.38%** of added nodes are within 3 cm of
the conservative wood candidates, which themselves fail the semantic audit below.
Nearest-segment groups have median support diagonal **69.0 / 68.2 cm** and p90
**1.24 / 1.22 m**; **99.71% / 99.70%** of crown splats remain in groups above 30 cm.
For comparison, the original 200-node rig's 187 nonempty crown groups have median
diagonal **1.53 m**, p90 **2.01 m**, and 99.993% of splats in groups above 30 cm.
79.75% of seed-17 added nodes have a seed-29 node within 5 cm. The 200-node rig can
be refined numerically, but five times the node count does not reveal five times
the verified branch structure. No optimum or convergence beyond this budget is claimed.

### What is visually separable

Full-detail CPU renders project covariance through the perspective Jacobian and
alpha-composite 3σ support front-to-back. They use DC colour and 0.3 pixel²
antialias covariance, without view-dependent SH. Dominant visible contributions
retain splat identities. The cameras use round 3's heading 35°, pitch −12°,
whole-tree range 15.4 m and crown target [0,0,4.6], range 7.8 m; an opposite
azimuth adds a check on occlusion. Resolution is 1024×640.

| View          | Components dominating ≥16 pixels | ≥64 pixels |
| ------------- | -------------------------------: | ---------: |
| Whole tree    |                               61 |          1 |
| Crown         |                            1,167 |         27 |
| Crown reverse |                            1,174 |         37 |

These are visibility proxies, not counts of real leaves. Non-crown geometry has
one shared colour ID. A seeded sample of 24 plausible-size crown components from
the 1,024 with ≥16 dominant pixels yields **18 fragments/mixed patches, four
ambiguous, two possible complete leaf silhouettes, zero confirmed attached twigs**.
These are Codex visual judgments, with source pixels, IDs and overlays saved for
human checking. Even the two silhouette candidates are not verified 3D leaves.
This is a small audit, not a population accuracy claim.

The native-vertex tight cut exposes **2,035 / 149** crown components at ≥16 / ≥64
dominant pixels, excluding the shared non-crown colour. A second 24-component
audit, sampled from 1,767 visible plausible-size candidates, yields **15 fragments
or mixed patches, four ambiguous and five possible complete leaf silhouettes**;
again zero confirmed attached twigs. Native contact sheets and random-colour
renders are retained separately. Thus the representative grid is not the only
source of poor biological grouping, although some native components look better.

### Opacity selection control

At alpha .15, **65.37%** of native crown vertices can participate in edges,
carrying **86.14%** of the crown render's summed alpha contributions. Thus 34.63%
of raw vertices are isolated by selection, not necessarily missing structure.
At alpha .02, **99.74%** of crown vertices and **99.985%** of rendered contribution
are retained. The 25 mm / .12 / .8 native cut then has **633,010 components**,
**21.38%** of splats in singletons, 27,536 plausible-size pieces covering **44.30%**
of the crown, and **1.52%** contact-supported crown coverage. Seed 29 gives 44.30%
coverage too; input-order ARI is **.99835**. The largest component is 0.171% of
the crown; size p10/p50/p90 is 2.55/5.29/12.69 cm, with one splat at the median.

This less restrictive cut still fails a further 24-component visual audit:
**15 fragments/mixed patches, five ambiguous, four possible leaf silhouettes,
zero verified attached twigs**. The sample is drawn from 1,884 visible plausible-size
candidates with seed 91. See `quality-control.json` and `low-alpha-piece-audit*`
in the notes. The conclusion rests on these audits and attachment limitations,
not on treating every low-opacity rejected Gaussian as defective geometry.

## Arm B: optional 2D-to-3D masks

Arm A cannot distinguish missing structure from a weak grouping method: some
outer leaves are recognizable. Therefore a mask probe is informative. Prepared:
SAM 2.1 tiny, open weights, pinned source revision, six crown azimuths, two jittered
prompt-grid seeds. It fetches the public scan itself, renders full-detail images,
lifts masks by **dominant-alpha contribution voting**, merges overlapping masks
across views and abstains on conflicts. This is a deliberately cheaper approximation
to [FlashSplat](https://arxiv.org/abs/2409.08270), not its optimal full-contribution
method or a reproduction of Gaussian Grouping/SAGA.

**Not executed. No coverage, piece, visibility or stability results exist for B.**
The sandbox cannot create the Git index lock, so its new entrypoint cannot enter
the helper's tracked-file-only snapshot. Launching that incomplete snapshot would
not perform the experiment. The successful dry-run is retained; Claude's exact
staging/launch instructions are in the notes. Do not interpret B as a negative result.

One T4, two CPU cores, 8 GiB, timeout 20 minutes. At the checked
[Modal prices](https://modal.com/pricing), the requested resources cost at most
`1200 × (.000164 + 2 × .0000131 + 8 × .00000222) = $0.2496`, plus small storage/egress.
No retry is authorized automatically. **Actual GPU spend: $0; no output volume was created.**
After a future run, verify/download results and delete its printed output volume.

## Arm C: wood versus foliage

The conservative wood candidate combines excess green < .06 with proximity
within 10 cm of the skeleton or covariance elongation >2. The predicted wood share
is **1.57% overall / 1.41% crown**. The classifier is deterministic, so a seed rerun
does not change it; sampling uncertainty and occlusion dominate its evaluation.

The hand-checkable audit samples 12 visible splats per predicted class per view,
three views, seed 81. A crosshair marks the dominant-contribution splat inside an
80×80-pixel context enlarged 2×. Sampling is deliberately balanced, not a natural
class distribution. The same physical surface may appear in multiple views.

| Prediction | Judged wood | Judged foliage | Ambiguous |
| ---------- | ----------: | -------------: | --------: |
| Wood       |          15 |             15 |         6 |
| Foliage    |           1 |             34 |         1 |

Wood precision among judged samples is **50% (15/30)**; its Wilson 95% interval is
approximately 33–67%. Foliage precision is 97% (34/35), which does not establish
wood recall in the interior. Seven of 72 points are ambiguous. This is a visual
confusion estimate, not independent manually labelled 3D ground truth. Dark leaves
and baked shade are poor wood evidence. Do not promote this classifier to a woody rig.
Arm C classifies the existing groups; it creates no new pieces or twig connectivity,
so piece sizes and attachment remain Arm A's. Its colour overlay is saved in the renders.

## Playback evidence and limits

`living_structure_playback.py` exports the 68,826 plausible-size graph pieces to
the existing round-3 articulation format. Original 200-node motion coefficients
are copied unchanged; each piece has one nearest branch and one independent
rotation. The flutter frequency/amplitude distribution matches round 3, but the
packet memberships, phases and pivots differ. This is a diagnostic, not calibrated
wind or an attachment-only ablation. Unselected fragments retain branch motion.

The browser uses original immutable GLBs, depth-two LOD, 687,226 selected splats,
512×320 and round-3 cameras. LOD splats inherit the nearest native component only
within 5 cm, a display approximation excluded from all structural metrics. The
gallery places the new simulated clip beside the existing round-3 frames and
DynamicTree's published 75-second excerpt. Different plant/camera: qualitative
only. Static/zero-strength controls and frame uniqueness are checked in the notes.

[DynamicTree](https://arxiv.org/html/2510.22213v2) transfers mesh motion to
surface-bound Gaussians. Our recovered fragments lack that verified surface and
connectivity. Its published clip is a visual reference, not a matched benchmark.

## Capture guidance derived from the failure

The current crown has median maximum Gaussian sigma **7.64 mm** (p90 **15.18 mm**).
Those are representation footprints, not sensor GSD: the median 2σ diameter is
30.5 mm. The median crown-to-current-skeleton distance is **31.8 cm**. A denser
spatial partition cannot reliably turn these footprints into 1–3 mm twig geometry.
The dataset already has 812 source images; more distant photographs alone are
not the proposed fix. Source photos and retraining were not evaluated here.

Start with **one isolated 30–50 cm leafy branch or a sparse sapling**, proving that
stems and both leaf faces survive reconstruction before a whole mature tree.
The following are proposed engineering targets, not empirically validated capture
settings for this tree:

- Target **0.15–0.35 mm/pixel** on twigs: a 2 mm twig then spans about 6–13 pixels.
  For a 4,000-pixel-wide image, GSD ≈ `distance × 36 / (equivalentFocalLength × 4000)`.
  A phone's 26 mm-equivalent main lens at **0.5–1 m from the vegetation** gives
  **0.17–0.35 mm/pixel**; a real 50 mm-equivalent lens at **1–2 m** gives
  **0.18–0.36 mm/pixel**. Use the full-resolution still mode, fixed lens/focus and
  exposure; avoid digital zoom, ultrawide, portrait blur and temporally mixed video frames.
- Capture **three or four elevation rings**, **36–48 positions per ring**, plus
  **40–60 underside/grazing views**: approximately **150–250 sharp stills per proof branch**.
  Keep **80–90% local overlap**, adjusting steps to framing. Include upward and
  downward 20–60° views and both leaf faces, with overlapping bridge views between
  rings. Require each target twig/leaf junction to appear sharply in at least three
  genuinely separated views. Hidden junctions need new camera access, not more views
  from the same orbit.
- Use diffuse overcast light and a visibly still plant. A suggested shutter starting
  point is **1/500–1/1000 s**; inspect images for motion blur rather than accepting
  that number alone. Capture the small local sequence quickly, repeat blurred or
  wind-disturbed segments, and include a scale bar.
- Prefer **phone/ground camera plus pole** for under-canopy and local detail.
  A drone with the dataset's 28 mm-equivalent/5,464-pixel imaging geometry at a
  5 m subject distance gives about **1.18 mm/pixel**, fewer than two pixels across
  a 2 mm twig. Use drone obliques for upper/exterior coverage and registration,
  with optical telephoto where available; they do not replace interior access.
  For the roughly 6.55 m mature tree, budget a proposed **800–1,500 useful images**
  combining overview rings and close sectors only after the small proof passes.
- Accept the proof only when multiple views show a continuous stem to the selected
  leaf cluster, independent cuts retain the same piece, and moving it exposes neither
  torn neighbouring leaves nor a large baked background patch. Record those checks
  separately from image sharpness. Keep the existing scan as the static survey asset.

[Agisoft's capture guidance](https://www.agisoft.com/pdf/tips_and_tricks/Image%20Capture%20Tips%20-%20Equipment%20and%20Shooting%20Scenarios.pdf)
supports overlapping object views, suitable optics and measured scale. The tighter
vegetation targets above are this probe's inference from the measured footprints,
occlusion and failed attachment, not vendor guarantees.

## Validation and handoff

Run Ruff, Ruff format, mypy and the complete captures pytest suite after the final
Python edit, sequentially through `heavy`. Exact tails and any baseline blocker
are recorded in the notes' `gate-status.txt`. No tracked web code changed, so web
gates/e2e are not triggered by this probe; the browser recording is a separate
research verification. Git metadata is read-only in this sandbox: the notes contain
an applyable patch and commands to commit/push and optionally execute B from a
writable session. No commit or push is claimed when it could not be made.
