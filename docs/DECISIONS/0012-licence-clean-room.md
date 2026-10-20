# ADR 0012 — Licence clean-up: what came from non-commercial code is removed or rewritten clean-room

**Status:** accepted · **Date:** 2026-10-19

## Context

The splat pipeline had picked up code and comments derived from projects whose licences forbid
commercial use. Named here in prose only:

- the level-of-detail **merge** in the tiler (`tools/captures/splat_tiles.py`), whose moment
  matching followed the reference code of Hierarchical 3D Gaussians (arXiv 2406.12080);
- the **block-training geometry** (`tools/pipeline/block_maths.py`, `block_views.py`): the
  partition, the camera-contribution test and the cross-fade followed the reference code of
  CityGaussian (arXiv 2404.01133) and of the same hierarchy paper;
- a **parent-optimisation stage**, `optimise_lod` (impl `h3dgs_parents`), and the numpy half it
  ran (`lod_maths.py`), following the hierarchy paper's reference code. It never ran on a GPU;
- a test that reproduced Inria's gaussian-splatting spherical-harmonics evaluation, and the
  basis constants in `tools/pipeline/harmonics.py` that matched it;
- an inpainting option using a gated FLUX model whose weights forbid commercial use;
- leftovers of a withdrawn Cesium ion sample mesh (a helicopter survey of San Francisco): test
  ids, a comment and two documentation rows.

The maths is published and free to use; the reference code is not. A product cannot carry the
code, and comments that say where code came from make that question permanent.

## Decision

Decision D2 of the splat pipeline build:

1. **Delete what nobody uses.** The FLUX inpainting class everywhere (client, both Modal apps,
   model table, runbook). The `optimise_lod` stage, its impl, `lod_optimise.py`, `lod_parents.py`
   and their stand-ins; `placement.json` moves to `tools/pipeline/placement.py`, because `place`
   still records the similarity it applied. Recipe `photo-reconstruct` goes to version 11
   (`SHIPPED_RECIPE_VERSIONS` follows; a test holds the two equal). The withdrawn mesh leaves the
   tests and docs; the seed keeps its slug in `WITHDRAWN_SITES` so old databases still delete
   the row.
2. **Rewrite the rest clean-room.** `tools/captures/splat_merge.py` (new; `splat_tiles` keeps
   thin adapters), `tools/pipeline/block_maths.py`, `block_views.py`, `lod_maths.py` (only the
   viewer's cut that `experiments/lod_compare.py` draws) and `harmonics.basis` with its
   constants. Each has our own rules where the method left a choice, stated in the Appendix: the
   opacity x Thomsen-area weight and the bounded widening of the merge, the SSIM boundary rule,
   the blend width and its cap.
3. **Regenerate every fixture the merge feeds**, in the same change (commands below).
4. **Keep it from coming back**: `scripts/licence_scan.py`, a standard-library scan with an
   allowlist (`scripts/licence_allowlist.json`), run by tests in `scripts/tests` and in
   `tools/pipeline/tests`.
5. **Record provenance**: `tools/captures/NOTICE` and this ADR.

`ParentOverrides`, `hierarchy()` and `--parents` stay in `splat_tiles.py`: they are our own file
format, keyed by tile, cell and PLY checksum, and the seam a future licence-clean optimiser
would write to. `package` no longer reads optimised parents, because nothing produces them.

## The clean-room procedure

- One agent, who had read the old code, wrote a specification of each flagged module from the
  mathematics, the papers and the module's interface alone: no code, no names of the old
  internals. It is the Appendix of this ADR.
- A different agent implemented from that specification without reading the flagged originals.
  Whole files were deleted with `git rm` unopened; named definitions inside kept files were cut
  by a script that parses the file and prints only names and line ranges; diffs of those files
  were never displayed. Each rewrite commit records that it was written clean-room.
- Comments in the new code cite only the mathematics (the references in the Appendix), the
  permissively licensed projects (PlayCanvas splat-transform and Spark, MIT; gsplat and
  CesiumJS, Apache-2.0), and the non-commercial papers by arXiv id.
- Each rewritten module's behavioural tests were kept; tests that compared against copies of
  the old code were replaced by tests against references written in the tests from the
  mathematics.

### What actually happened (correction, 2026-10-20)

The procedure above was not kept perfectly, and this records exactly how:

- **The first implementer's incidental exposures.** Two grep outputs showed it a few lines of
  the old code: old docstring and comment lines, and a few lines of the old merge. The merge's
  opacity rule, opacity = weight / surface area, is in that old code too; it is also what spec
  5A specifies, and the implementer took it from the spec.
- **A review found matches.** An independent review compared the result with the old code and
  found that five helper bodies in `tools/captures/splat_merge.py` (`Moments.concat`,
  `Moments.take`, `merge_cells`, `coarsen`, `_rows`) and the `tensor()` helper nested in
  `tools/pipeline/block_views.py`'s `main()` matched the old code character for character, so
  the clean-room claim could not stand for those six bodies.
- **They were rewritten by a second implementer.** It removed each body with a script that
  printed only line numbers, and wrote new ones having seen only the signatures and docstrings,
  the tests, metadata of the call sites (where each is called and with what kinds of argument,
  not the calling code), the name of `main()`'s device variable and spec 5A. It never displayed
  the old bodies, the originals or a diff of either file. `merge_cells` now weighs members by
  their share of the cell and copies a one-member cell bit for bit. The rewrite changed the
  last bits of the arithmetic, which flips the arbitrary sign of a few eigenvectors, so four
  parent tiles of the yard and far fixtures carry an equivalent but differently quantised
  rotation; no leaf changed.

Nothing more is claimed: the rest of `splat_merge.py` and `block_views.py` is the first
implementer's work under the procedure above, with the exposures listed.

### Second blind rewrite (2026-10-20)

- **A second review found two more matches.** A reviewer from another model found that `ring`
  and `_cell_weights` in `tools/pipeline/block_maths.py` were near-identical to the original
  non-commercial implementation, down to its guards and the form of its ramp, so the
  clean-room claim could not stand for those two bodies either.
- **They were re-derived blind.** A further implementer cut both bodies with an `ast` script
  that printed only line numbers, and wrote new ones having seen only the module with the
  bodies of `ring`, `_cell_weights`, `weights` and `blended_opacity` removed (signatures and
  docstrings kept), the tests, the call sites in `blocks.py`, and the prose and tables of the
  papers (Hierarchical 3DGS arXiv 2406.12080, CityGaussian arXiv 2404.01133, VastGaussian arXiv
  2402.17427), never their code. It never displayed the old bodies, any earlier version of
  them on any branch, or a diff of the file. One incidental exposure: a grep for call sites
  printed the two lines of `weights` that call `_cell_weights`. The derivation of each formula
  is in the module docstring. One behaviour was added: the ramp's half-width is also capped
  at half the narrower cell, so no weight is negative for any margin (inactive at the default
  margin of 0.1).
- **Hyperparameters kept as values, with citations.** `DEFAULT_EPSILON = 0.05` is
  CityGaussian's Table S3 value; the frozen ring is Hierarchical 3DGS's idea, but
  `DEFAULT_RING_OUTER = 1.5`, `DEFAULT_BLEND = 0.0125` and `MIN_IMAGES = 50` are this
  project's values (none of the papers blends at the seams or sets a minimum of images per
  block), and the comments say so.

## Regenerating the fixtures

Parents go through `eigh` and the Thomsen area through `pow`, so their bytes depend on the
BLAS/LAPACK kernel; each fixture is written under the kernel its gate pins. From
`tools/captures`, in order:

```sh
# 1. LOD fixture and its rig
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python -c "import pathlib, splat_tiles; splat_tiles.convert(pathlib.Path('../../data/tiles/synthetic-tree/source/splat.ply'), pathlib.Path('../../data/tiles/synthetic-tree-lod'), 28.0389, -82.6966, 0.0, 0.02, 1500)"
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python rig_tiles.py ../../data/tiles/synthetic-tree-lod ../../data/tiles/synthetic-tree/source/rig.json
# 2. SH fixture
OPENBLAS_CORETYPE=Sandybridge PYTHONPATH=. uv run --no-sync python tests/test_splat_tiles_sh.py
# 3. Yard tiles and scene step (writes source/, which is git-ignored)
OPENBLAS_CORETYPE=Haswell uv run --no-sync python synthetic_yard.py ../../data/tiles/synthetic-yard
# 4. The far site
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python synthetic_far.py ../../data/tiles/synthetic-far
# 5. Yard instances re-keyed (leaves and the instance list kept byte for byte)
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python rebind_instances.py ../../data/tiles/synthetic-yard/splat --instances ../../data/tiles/synthetic-yard/instances/instances.json --rekey
# 6. Skins
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python skin_scene.py ../../data/tiles/synthetic-yard/splat ../../data/tiles/synthetic-yard/instances/instances.json --out ../../data/tiles/synthetic-yard/skin --only 1,9,10,12
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python skin_variants.py build ../../data/tiles/synthetic-yard/splat --only 1,9,10,12 --variants freeform-stiff --flat --instances ../../data/tiles/synthetic-yard/instances/instances.json --materials ../../data/tiles/synthetic-yard/skin/materials.json --out ../../data/tiles/synthetic-yard/skin-wide
# 7. Variants
OPENBLAS_CORETYPE=Sandybridge uv run --no-sync python yard_variants.py ../../data/tiles/synthetic-yard
# 8. Wind parity, only if packages/world skinWind.test.ts fails after 6:
cd ../.. && UPDATE_SKIN_WIND_PARITY=1 TZ=UTC corepack pnpm --filter @twin/world exec vitest run src/skinWind.test.ts
```

The tests are the judge (`test_committed_lod_fixture_matches_the_packer`,
`test_committed_sh_fixture_matches_the_packer`, `test_rig_tiles.py`, `test_scene_plants.py`, the
committed-skin, wide-skin, variants, wind and far-fixture tests). No leaf GLB, `collision.bin`
or `viewcones.bin` changes, and `data/tiles/synthetic-tree` is untouched: it is one tile, so the
merge never runs on it.

## Consequences

- **Published scans keep their old parents.** Their leaves are unaffected; their parent tiles
  were produced by the old merge. Re-packing them (re-run `package` from the stored
  `canonical.ply`, then `rebind_instances --rekey` and re-skin) is a follow-up, dispatch-only
  workflow, not part of this change.
- **A re-pack changes parent checksums.** `carry.py` then drops the instances, skin, rig and
  streamed-LOD package and flags them "need re-segmenting", which is honest and visible.
  `rebind_instances --rekey` restores the instances (every leaf is kept, so its precondition
  holds); skins need `skin_scene` again.
- **Parent quality** is measured on the CPU only (`tools/captures/lod_parity.py`, synthetic
  fixtures) and is not gated: no threshold has a recorded measurement. The visual comparison on
  real captures (`experiments/lod_compare.py`) needs a GPU and has not been run.
- **The camera test's SSIM border rule changed**, unmeasured on a GPU run: assignments at frame
  borders may differ slightly from the measured spool run's. `block_blend` now means a
  half-width fraction of the centre spacing; its default gives the band the spool run used.
  Neither is exposed by the phone API.
- **A later change that edits an allowlisted line** (in `gaussians.py`, `splat_io.py`,
  `harmonics.py`, `experiments/benchmark.py`, the seed's `WITHDRAWN_SITES`, the Worlds catalogue
  or the research registry) must update `scripts/licence_allowlist.json` in the same commit: a
  stale entry fails the scan.
- **Dependency licences** are checked by `scripts/dependency_licences.py` (standard library):
  the installed metadata (`License-Expression`, `License`, `License ::` classifiers) of every
  package in the `tools/pipeline`, `tools/captures` and `apps/api` venvs, and
  `pnpm licenses list --json --prod --recursive` for the workspace's production packages. A
  non-commercial or undeclared licence fails unless `scripts/dependency_licence_allowlist.json`
  lists it with a reason (empty today: 2026-10-20, 176 Python and 102 npm production packages,
  none non-commercial; the API image's `modal` extra, 124 packages, also clean). CI runs it
  through each project's own tests and, for pnpm, a web vitest test. Not covered: the Modal GPU
  images and the Worlds worker images, which install third-party repositories and
  `requirements*.lock` files outside these venvs.

## Appendix: the clean-room specifications

As handed to the implementer, unchanged. It contains no code.

Only maths, behaviour, invariants, interfaces and test properties. No code.

### 5A. CLEAN-ROOM SPEC — `tools/captures/splat_merge.py` (parent construction for the octree)

**Purpose.** A parent tile of the REPLACE octree holds one gaussian per occupied grid cell at the tile's level. That gaussian stands in for every original gaussian (leaf) in the cell. This module turns leaves into additive _moments_, merges them per cell, and turns merged moments back into gaussians. Pure numpy. It must **not** import `splat_tiles` (the dependency runs the other way).

**Representation.** A gaussian i has:

- centre μ_i ∈ ℝ³;
- unit quaternion q_i = (w, x, y, z);
- per-axis standard deviations σ_i = (σ_i1, σ_i2, σ_i3) > 0;
- opacity α_i ∈ (0, 1), stored as a logit ℓ_i, with α = 1/(1 + e^(−ℓ));
- DC colour coefficients c_i ∈ ℝ³;
- optional higher SH coefficients h_i ∈ ℝ^(3K).

The covariance is Σ_i = R(q_i) · diag(σ_i²) · R(q_i)ᵀ, the standard 3DGS parameterisation (Kerbl et al. 2023, arXiv 2308.04079, eq. 6). The columns of R are the gaussian's axes.

**Rotation from a quaternion.** Normalise q, flooring its norm at 1e-12. R is the standard rotation matrix of a unit quaternion (any textbook, e.g. Shoemake 1985). Its first row is (1 − 2(y² + z²), 2(xy − wz), 2(xz + wy)); the other rows follow by the usual cyclic pattern. det R = +1.

**Quaternion from a rotation.** For a proper rotation matrix, return a unit quaternion (w, x, y, z) with w ≥ 0, so that R(q) reproduces the matrix. Method: Shepperd (J. Guidance and Control 1(3), 1978). Pick the largest of {trace, R₁₁, R₂₂, R₃₃} and derive the other components from it, so no square root is taken of a small number. Vectorised over n matrices. Normalise the result, then flip its sign if w < 0.

**Leaf weight (our choice).** w_i = max(α_i · S(σ̂_i), 10⁻³⁰), where σ̂_i = exp(clip(log σ_i, lo, hi)) and (lo, hi) is SPZ's representable log-scale range, passed in by the caller. S is the surface area of the 1σ ellipsoid by Knud Thomsen's approximation:

S(a, b, c) = 4π · [((ab)^p + (ac)^p + (bc)^p) / 3]^(1/p), with p = 1.6075.

Its relative error is at most 1.061%. It is exact for spheres (S = 4πr²) and homogeneous of degree 2: S(ka, kb, kc) = k²·S(a, b, c).

Justification:

- By Cauchy's surface-area formula, the mean area of a convex body's orthogonal projection over all view directions is S/4. So α·S is proportional to the "ink" a gaussian deposits on screen, averaged over viewpoints. That is the quantity a parent must preserve.
- PlayCanvas splat-transform 3.8.0 (MIT) uses the same weight. Verified in its published `dist/index.mjs`, `splatMass` / `mergeGroup`.
- Clipping the scales first means a parent merges exactly the sizes its children are drawn at.

**Moments.** Per row:

| Field    | Shape           | Content                                                                                                                                |
| -------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `key`    | int64 (n,)      | cell id, ascending                                                                                                                     |
| `weight` | float64 (n,)    | unnormalised W                                                                                                                         |
| `mean`   | float64 (n,3)   | centre                                                                                                                                 |
| `cov`    | float64 (n,6)   | the 6 unique entries in order (xx, xy, xz, yy, yz, zz)                                                                                 |
| `colour` | float64 (n,3)   | DC coefficients                                                                                                                        |
| `sh`     | float64 (n, 3K) | higher bands, coefficient-major with the colour channel innermost (reshaping to (n, K, 3) gives SPZ order); (n, 0) when there are none |

Methods: `__len__` (the number of rows); static `concat(parts)` (field-wise concatenation); `take(index)` (field-wise fancy index). Module constant `MOMENT_FIELDS = ("key", "weight", "mean", "cov", "colour", "sh")`.

**`leaf_moments(keys, xyz, opacity_logit, log_scales, quat_wxyz, f_dc, sh_flat, *, log_scale_range) -> Moments`.** One row per input, in input order:

- weight = w_i;
- mean = μ_i (float64);
- cov = Σ_i from the clipped scales. Compute each entry as the elementwise sum Σ_ab = Σ_k R_ak·R_bk·σ̂_k², with no matrix-multiply call: elementwise arithmetic is identical on every BLAS kernel;
- colour = c_i;
- sh = h_i.

Every input is cast to float64 first. `keys` is cast to int64.

**`merge_cells(m) -> Moments`.** Precondition: `m.key` is non-decreasing (equal keys are contiguous). Output: one row per distinct key, keys ascending. For a cell C with W = Σ_{i∈C} w_i:

- μ = Σ w_i μ_i / W;
- Σ = Σ w_i (Σ_i + δ_i δ_iᵀ) / W, with δ_i = μ_i − μ. This is the law of total covariance. Compute δ against the cell's own mean, in two passes, never as E[xxᵀ] − μμᵀ, which cancels catastrophically for small cells far from the origin;
- colour = Σ w_i c_i / W, and sh likewise. SH is linear in its coefficients, so the merged colour from any direction is the weighted mean of the members' colours from that direction;
- weight = W, unnormalised, so the result merges again.

An empty input returns itself. Use run-wise reductions; no Python loop over cells.

**`coarsen(m, levels) -> Moments`.** For levels ≤ 0, return `m` unchanged. Otherwise shift each key right by 3·levels bits (one octant per level), which keeps keys sorted, then `merge_cells`.

**Associativity invariant.** Merging leaves into fine cells and coarsening by L levels equals merging the leaves directly at the coarse level. Weights agree to 1e-12 relative; mean, cov, colour and sh to rtol 1e-9, atol 1e-12. This holds exactly in algebra, by the pairwise combination identity for weighted means and covariances (Chan, Golub and LeVeque, _Am. Statistician_ 37(3), 1983). `build_tiles` relies on it to build each parent from its children's cells.

**`to_splats(m, *, log_scale_range) -> MergedSplats`.** For each row:

1. **Eigendecomposition.** Assemble the symmetric 3×3 Σ and take its eigen-decomposition with `numpy.linalg.eigh` (eigenvalues ascending, eigenvectors as columns V). If det V < 0, negate column 0, so V is a proper rotation.
2. **Floor (our choice; amended 2026-10-20 after review, see "Why the merge differs").** σ_k = sqrt(max(λ_k, e^(2·lo), 2^-20·λ_max)), aligned with the columns of V. The absolute part is the smallest scale SPZ can store, and keeps S(σ) > 0. The relative part (`MERGED_EIGEN_FLOOR`) keeps a rank-1 or rank-2 cell a gaussian that a float32 consumer can rebuild: as first written the floor was absolute only, which left such cells with axes 10^14 apart.
3. **Mass-conserving opacity with bounded widening (our choice).** F = W / S(σ) is the opacity that would deposit exactly the members' total ink. If F ≤ α_max, keep σ and set α = F. Otherwise, widen every axis by k = clip(sqrt(F/α_max), 1, k_max). Then S grows by k² and α = F/k², so α·S = W is conserved while k < k_max. Finally α = clip(F/k², α_min, α_max). Constants:
   - `MERGED_OPACITY_MAX` α_max = 511/512. The largest α whose logit is finite and small (≈ 6.24) and whose SPZ alpha byte round(255·α) is still 255.
   - `MERGED_WIDEN_MAX` k_max = 2.0. A parent widened past twice its moment-matched size spreads past its cell's neighbourhood. F > 4·α_max means about four or more stacked layers, whose extra mass is not recoverable as coverage.
   - `MERGED_OPACITY_MIN` α_min = 1e-6, a numeric guard only.

   The widening is the idea Spark (MIT) applies with `lodInflate`. splat-transform caps at 1 without widening. We widen, to avoid see-through parents.

4. **Outputs.**

   | Output        | Value                                                |
   | ------------- | ---------------------------------------------------- |
   | xyz           | μ as float32                                         |
   | sh0           | colour as float32                                    |
   | opacity_logit | log(α/(1−α)) as float32                              |
   | log_scales    | log(k·σ) as float32                                  |
   | quat_xyzw     | quaternion of V, reordered to (x, y, z, w), float32  |
   | sh_rest       | sh reshaped to (n, K, 3) float32, or None when K = 0 |

`MergedSplats` is a dataclass with exactly these six fields.

**Adapters in `splat_tiles.py`** (new code; the contract callers rely on):

- `gaussian_moments(records, keys)` = `splat_merge.leaf_moments(keys.astype(int64), records["xyz"], records["opacity"], records["scale"], records["rot"], records["f_dc"], sh_flat, log_scale_range=SPZ_LOG_SCALE_RANGE)`. `records["rot"]` is w-first. `sh_flat = unquantize_sh(records["sh"]).reshape(n, -1)` when the dtype has `"sh"`, else an (n, 0) array.
- `to_gaussians(moments) -> Gaussians` wraps `to_splats` with the same `log_scale_range`. Same field names.
- `build_tiles` and `write_tiles` are otherwise unchanged: leaves are still written in PLY order, `emit` still receives `merged.key`, and errors, boxes and widening feed `geometric_error` as before.

**Test properties** (`tests/test_splat_merge.py`; every reference written in the test from this section):

1. `test_a_single_gaussian_merges_to_itself`: one leaf with α ≤ α_max. The merged mean is identical. The reconstructed covariance R(q)·diag(e^(2·log s))·R(q)ᵀ matches Σ_i within 1e-5 of ‖Σ‖_F (float32 outputs). The logit matches within 1e-5.
2. `test_coincident_gaussians_widen_to_keep_their_mass`: two identical leaves with α = 0.8. α_out ≈ α_max and α_out·S(σ_out) ≈ 2·w_1 within 1e-5 relative.
3. `test_mass_beyond_the_widening_cap_is_not_smeared`: eight identical leaves with α = 0.9. k = 2 exactly and α_out = α_max.
4. `test_the_moments_are_the_law_of_total_covariance`: 50 random members in one cell, compared with an explicit loop over members. rtol 1e-12.
5. `test_merging_merged_cells_is_merging_their_gaussians`: the associativity invariant on 5,000 random gaussians with keys at two levels.
6. `test_the_weight_is_opacity_times_thomsen_area`:
   - a sphere: w = α·4πr² to 1e-12;
   - a 3:2:1 ellipsoid: within 1.1% of the exact area, computed in the test by a fine (θ, φ) quadrature of the parametric surface;
   - scales outside the range: clipped before the area is taken.
7. `test_quaternions_round_trip`: 2,000 random unit quaternions (w ≥ 0), including angles within 1e-6 of π about each axis. Quaternion → R → quaternion matches within 1e-12 up to sign, and det R = 1.
8. `test_the_axes_reproduce_the_covariance`: 500 random SPD matrices. Reconstruction from the float32 outputs is within 1e-5·‖Σ‖_F + 1e-12.
9. (Amended.) `test_a_flat_cell_is_floored_relative_to_its_longest_axis` and `test_a_cell_on_a_line_is_floored_in_both_thin_axes`: rank-2 and rank-1 cells give their thin axes ½·log(2^-20·λ_max), and their float32 rebuild stays positive definite; `test_a_small_flat_cell_is_floored_at_the_smallest_spz_scale`: a 1 cm flat cell gives lo.
10. `test_colour_and_sh_are_weighted_means`.
11. `test_opacity_stays_finite_and_in_range`: random F spanning 1e-9 to 1e3 gives α_out in [α_min, α_max] and finite logits.

Also, in `test_splat_tiles_lod.py`, new `test_a_parent_tile_is_its_subtree_merged`: decode `splat.glb` (the root) of the `packed` fixture and recompute its cells in the test (Morton keys at `plan_tree`'s root level). Each decoded parent matches the per-cell maths above, after SPZ quantisation: position within 1.5/4096 m; colour, scale and alpha bytes within ±1. These are SPZ's quanta, so first-principles tolerances, not tuned ones.

### 5B. CLEAN-ROOM SPEC — `tools/pipeline/block_maths.py` (block-training geometry)

**Context.** Pure numpy, importable by CPython 3.10 (the trainer's interpreter). Imports only numpy and the standard library.

**Constants (names callers use):**

| Name                 | Value        | Meaning                                                                                                                                                                                                          |
| -------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MARGIN`     | 0.1          | Trainable region grows by 10% of each cell's width per side. VastGaussian arXiv 2402.17427 uses a 20% expansion. Kept so the measured 2-block spool run (26.42 dB against 26.39 whole) still describes this code |
| `DEFAULT_RING_OUTER` | 1.5          | Ring reaches 1.5 block sizes from the centre, one full block width beyond each edge, so a camera anywhere in the block sees at least a block width of frozen surroundings in every ground direction              |
| `DEFAULT_BLEND`      | 0.0125       | Half-width of the cross-fade, as a fraction of the spacing between adjacent cell centres (the band the spool run used)                                                                                           |
| `DEFAULT_EPSILON`    | 0.05         | Camera-test threshold, the value shipped and measured with until now; `block_epsilon` is the knob                                                                                                                |
| `MIN_IMAGES`         | 50           | Fewest training cameras a block may have                                                                                                                                                                         |
| `OUTER_QUANTILES`    | (0.01, 0.99) | Robust outer bounds                                                                                                                                                                                              |
| `SSIM_WINDOW`        | 11           | Wang et al. 2004                                                                                                                                                                                                 |
| `SSIM_SIGMA`         | 1.5          | Wang et al. 2004                                                                                                                                                                                                 |
| `SSIM_K`             | (0.01, 0.03) | With dynamic range L = 1                                                                                                                                                                                         |

**`horizontal_axes(up, xyz) -> (a1, a2)`.** Let û = up/|up|.

1. Seed vector e = (1,0,0) if |û_x| < 0.9, else (0,1,0). Set e1 = normalise(e − (e·û)û) and e2 = û × e1.
2. Keep only finite points. If fewer than 2 remain, return (e1, e2).
3. Otherwise project the points onto (e1, e2), subtract the 2D mean, and take the eigenvector of the 2×2 scatter matrix with the largest eigenvalue.
4. Fix its sign so its first component is positive (or, if that is 0, its second).
5. a1 = normalise(m₁e1 + m₂e2), a2 = û × a1.

**`grid_shape(count, extent_u, extent_v, *, exact) -> (columns, rows)`.** N = max(1, ⌊count⌋); Lu and Lv are floored at 1e-12. For m = 1…N, take k = ⌊N/m⌋ if `exact` else ⌈N/m⌉. Skip k < 1, and, when `exact`, any m·k ≠ N. Score each candidate with the tuple

(round(|ln((Lu/m)/(Lv/k))| + ln(m·k/N), 12), m·k, −m)

and take the lexicographic minimum. Default: (N, 1). Ties go to more columns.

Required results:

| Call                  | Result |
| --------------------- | ------ |
| (2, 1, 1, exact)      | (2, 1) |
| (4, 1, 1, exact)      | (2, 2) |
| (3, 1, 1, not exact)  | (2, 2) |
| (5, 10, 1, not exact) | (5, 1) |

**`Partition`** is a frozen dataclass, _not_ slotted (tests use `__dict__`). Fields, in order:

| Field     | Type                    |
| --------- | ----------------------- |
| `origin`  | (float, float, float)   |
| `axis_u`  | (float, float, float)   |
| `axis_v`  | (float, float, float)   |
| `u_edges` | tuple[float, …]         |
| `v_edges` | tuple[float, …]         |
| `blocks`  | tuple[tuple[int, …], …] |

Properties: `columns = len(u_edges) − 1`, `rows = len(v_edges) − 1`, `count = len(blocks)`. Cell index = iu·rows + iv. Methods:

- `project(xyz)`: (n,2) of ((p − origin)·axis_u, (p − origin)·axis_v), in float64.
- `cell_of(uv)`: a non-finite coordinate is treated as 0.0. iu is the number of inner u-edges (`u_edges[1:-1]`) that are ≤ u (so a point exactly on an edge goes to the cell after it); iv likewise. The outer edges are open: every point lands in exactly one cell.
- `block_of_cell()`: int64 table of length columns·rows, −1 by default, each block's cells set to its index.
- `block_of(uv) = block_of_cell()[cell_of(uv)]`.
- `cell_rect(cell)`: (u0, u1, v0, v1) from the edges.
- `rect(block)`: the bounding rectangle of its cells.
- `centre(block)`: the midpoint of `rect`.
- `in_expanded(uv, block, margin)`: true when the point lies in at least one of the block's cells grown by margin × (that cell's width) on each side, closed intervals. A side on the grid's outer edge is unbounded.
- `to_dict()` returns exactly the keys `origin`, `axisU`, `axisV`, `uEdges`, `vEdges`, `blocks` (lists; blocks as lists of int). Persisted in `checkpoint/blocks/plan.json` and in `partition.json` for `block_views`.
- `from_dict(d)` inverts it with float and int casts. `from_dict(json round-trip of to_dict(p)) == p`.

**`partition(xyz, up, count, *, exact) -> Partition`.**

1. Keep finite rows. If none remain, raise `ValueError("no finite positions to partition")`.
2. Axes = `horizontal_axes(up, points)`, each component rounded to 9 decimals. This makes the result independent of input order.
3. origin = the component-wise median.
4. u = (p − origin)·a1, v = (p − origin)·a2.
5. The extents are the differences of the `OUTER_QUANTILES` (numpy's default linear quantile) of u and of v. `grid_shape` gives (columns, rows).
6. Edges per axis, for c cells:
   - inner edges at the j/c quantiles, j = 1…c−1;
   - the low outer edge is min(Q(0.01), all inner edges), the high one max(Q(0.99), all inner edges);
   - make the sequence non-decreasing;
   - if the last edge equals the first, set last = first + 1e-9.
7. blocks = ((0,), (1,), …, (columns·rows − 1,)).

Invariants: equal counts per column and per row, within 1; deterministic for permuted input (`to_dict` equal); every point in exactly one block.

**`cameras_inside(part, centres)`** returns a (cameras, blocks) one-hot of `block_of(project(centres))`.

**`ssim(a, b) -> float` (our boundary rule).** Inputs: images (H, W, C) or (H, W), in [0, 1], computed in float64. Local statistics use the normalised 11-tap Gaussian (σ = 1.5), separable, **renormalised to the in-frame pixels**: every local average is (G ∗ (x·M)) / (G ∗ M), where M is 1 inside the frame and G ∗ uses zero outside it. So border windows use only real pixels, with the weights rescaled to sum to 1. Then SSIM per pixel and channel is Wang, Bovik, Sheikh and Simoncelli (IEEE TIP 13(4), 2004):

((2μ_aμ_b + C1)(2σ_ab + C2)) / ((μ_a² + μ_b² + C1)(σ_a² + σ_b² + C2)), with C1 = (0.01)² and C2 = (0.03)².

The result is the mean over all pixels and channels.

Why this boundary rule: a block seen only at the frame edge must count. Valid-only filtering ignores a 5-pixel border, and edge replication over-weights it. I checked this against the existing camera-test data (random 24×32 image, seed 2):

| Columns zeroed     | Normalised window | Edge replication |
| ------------------ | ----------------- | ---------------- |
| 16 (half the view) | 0.512             | —                |
| 1 (a sliver)       | 0.0386            | 0.0502           |

So the normalised window keeps the ordering half-view > ε > sliver, where edge replication would push the sliver over ε.

**`contribution(render, cameras, members, *, skip=None, ssim_fn=ssim)`** returns a (cameras, blocks) float64 array, initially 0.

- For each camera c and block b, record 0 _without rendering_ when `skip[c, b]` is true or `members[b]` has no true entry.
- Otherwise compute 1 − `ssim_fn(render(c, members[b]), full_c)`. `full_c = render(c, None)` is rendered at most once per camera, and only if some block of that camera needs it.
- `render` must be handed **the identical array object** `members[b]` (`block_views` relies on that identity).

**`assign(inside, loss, epsilon)`** = inside OR (loss > epsilon). If loss is None, it is a copy of `inside`.

**`merge_small(part, assigned, train, gaussians, minimum=MIN_IMAGES) -> (Partition, matrix, merges)`.**

While more than one block remains:

1. For each block b, cams_b = the number of cameras with assigned[:, b] ∧ train.
2. s = argmin over b of (cams_b, b). If cams_s ≥ minimum, stop.
3. The candidates are the blocks edge-adjacent to s: some cell of one is at Manhattan distance 1 (|Δiu| + |Δiv| = 1) from some cell of the other. If there are none, every other block is a candidate.
4. o = argmin over candidates of (gaussians_o, o): the block least likely to overflow the GPU.
5. Record (o, s, cams_s), using indices in the _current_ list.
6. o's cells become the sorted union of both; its assigned column becomes the OR of both; its gaussian count the sum. Delete s.

Finally, order the blocks by their smallest cell, and reorder the matrix columns the same way. Return a Partition with the same grid and the new blocks, that matrix, and the list of merges.

Note: the union of cameras is never more than a re-test would give, since removing two blocks changes a render at least as much as removing one.

**`ring(part, uv, block, *, margin, outer) -> bool (n,)`.** Take rect(block) = (u0, u1, v0, v1), centre (c_u, c_v), and widths w_u = max(u1 − u0, 1e-12), w_v likewise. Let

d = max(|u − c_u|/w_u, |v − c_v|/w_v).

The ring is d ≤ outer AND NOT `in_expanded(uv, block, margin)`. This is a square annulus in block-normalised L∞ distance, so nothing is both trainable and frozen.

**`weights(part, uv, block, *, blend=DEFAULT_BLEND, margin=DEFAULT_MARGIN) -> float (n,)` (our cross-fade).** Per axis, with edges e₀…e_c, cell widths w_j and centres m_j:

1. For each interior edge k = 1…c−1: D_k = m_k − m_{k−1} and h_k = min(blend·D_k, margin·min(w_{k−1}, w_k)).
2. Step function: S₀ ≡ 1, S_c ≡ 0. For an interior edge, S_k(x) = 1 if x ≥ e_k when h_k = 0; otherwise S_k(x) = clip(½ + (x − e_k)/(2h_k), 0, 1). A non-finite coordinate is treated as 0.
3. Cell weight ω_j(x) = S_j(x) − S_{j+1}(x).
4. The block's weight is Σ over its cells (iu, iv) of ω^u_iu(u)·ω^v_iv(v), clipped to [0, 1].

Invariants:

- **Partition of unity.** Summed over all blocks it is exactly 1 (the steps telescope). This also holds after `merge_small`.
- **Never negative.** Weights are ≥ 0 everywhere: the cap makes h_k + h_{k+1} ≤ 2·margin·w_k ≤ w_k for margin ≤ ½, so the ramps never overlap.
- **The band stays inside both trained margins** (the cap is our addition).
- With blend = 0 the cross-fade is a hard crop.

**`blended_opacity(logits, weight) -> float32`.** α = 1/(1 + e^(−ℓ)), α′ = clip(α·weight, 1e-7, 1 − 1e-7), out = log(α′/(1 − α′)). Where weight ≥ 1, return the input logit bit for bit. Overflow in exp is silenced.

**`val_order(train, val) -> (list[(name, is_val)], every)`.** gsplat 1.5.3's parser sorts image names and holds out every index with index mod every = 0. Blocks rename their frames `<position>_<name>` in the returned order.

- If `val` is empty: return [(name, index == 0)] over `train` in order, and every = max(2, len(train) + 1). The first training frame is held out.
- Otherwise, for every = 2, 3, …, len(train) + 2, build the order by repeatedly appending:
  - at each position p with p mod every = 0, the next val name (flag true). If none is left, this `every` fails;
  - at every other position, the next train name (flag false).

  Stop when every train name is placed. Return the first `every` that does not fail. Leftover val names are not used.

- This is unreachable for every = len(train) + 1. If reached anyway, raise `AssertionError`.

**Interfaces callers rely on:** `blocks.py` (`partition`, `cameras_inside`, `assign`, `merge_small`, `ring`, `weights`, `blended_opacity`, `val_order`, `Partition.*`, the constants above), `block_views.py` (`Partition.from_dict`, `block_of`, `project`, `cameras_inside`, `contribution`) and `tests/block_views_stand_in.py`.

**Test properties** (`test_blocks.py`):

- **Kept unchanged**:
  - `test_blocks_hold_equal_counts_per_column_and_row_at_the_quantiles`;
  - `test_the_partition_is_deterministic_and_ignores_input_order`;
  - `test_the_ground_plane_follows_the_cameras_up`;
  - `test_a_forced_count_is_exact_and_a_budget_count_is_a_minimum`;
  - `test_every_point_belongs_to_exactly_one_block_the_outer_edges_open`;
  - `test_a_camera_standing_in_a_block_is_assigned_to_it`;
  - `test_ssim_is_one_for_equal_images_and_falls_with_what_changes`;
  - `test_the_camera_test_is_one_minus_ssim_of_the_render_without_the_block` (docstring reworded);
  - `test_a_block_under_the_minimum_joins_its_emptiest_neighbour`;
  - `test_a_block_count_that_cannot_reach_the_minimum_comes_down_to_one`;
  - `test_the_blend_is_a_partition_of_unity_so_the_band_never_doubles_density`;
  - `test_fading_scales_alpha_and_leaves_full_weight_bit_for_bit`;
  - both `val_order` tests;
  - every end-to-end stand-in test.
- **New**:
  - `test_the_frozen_ring_is_a_square_annulus_outside_the_trainable_region`. Grid u edges (−10, 0, 2, 4, 14), v edges (−10, 0, 1, 11), block = the cell u ∈ [0,2], v ∈ [0,1]. Points (1, .5), (2.1, .5), (2.5, .5), (3.9, .5), (4.1, .5), (1, 1.9), (1, 2.1) give ring [F, F, T, T, F, T, F] at margin 0.1, outer 1.5.
  - `test_the_blend_band_is_a_fraction_of_the_centre_spacing`. With `two_blocks()` (D = 4, h = 0.05), block 0 has weight 1 at u ≤ −0.05, 0 at u ≥ 0.05, 0.5 at 0 and 0.75 at −0.025.
  - `test_a_zero_blend_is_a_hard_crop`: u = (−0.01, 0, 0.01) gives [1, 0, 0].
  - `test_the_blend_never_reaches_past_the_margin`. Edges (0, 0.1, 10): the band is [0.09, 0.11]; weights ≥ 0 and their sum is 1 on 10,000 points.
  - `test_weights_are_never_negative_on_uneven_grids`: 50 random edge sets with width ratios up to 1e3.
  - `test_ssim_counts_the_frame_border`. Zeroing one edge column of a random image gives 0 < 1 − SSIM < 0.05; zeroing half gives > 0.5.

### 5C. CLEAN-ROOM SPEC — `tools/pipeline/block_views.py` (the camera test's renders, GPU)

**Command-line interface:**

| Argument           | Required / default                                                                                                   |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `--data_dir PATH`  | required                                                                                                             |
| `--prior PATH`     | required; npz with `xyz`, `quats` (w-first), `log_scales`, `logit_opacity`, `f_dc`                                   |
| `--partition PATH` | required; `Partition.to_dict` JSON                                                                                   |
| `--out PATH`       | required                                                                                                             |
| `--trainer PATH`   | required; gsplat's `simple_trainer.py`, whose directory goes first on `sys.path` so `datasets.colmap.Parser` imports |
| `--max-side`       | 800                                                                                                                  |
| `--budget-s`       | 1800.0                                                                                                               |
| `--device`         | hidden, default `cuda`                                                                                               |

The module docstring's first line is the argparse description.

**Behaviour:**

1. Start a daemon watchdog. After `budget_s` seconds it writes `block_views: out of time after <s> s` to stderr and exits the process with code 3. Then the stage falls back to the CPU visibility test.
2. Import torch, `datasets.colmap.Parser` and `gsplat.rasterization` inside `main`. If the device is cuda and torch has no CUDA, write the reason to stderr and return 2.
3. Load the partition and the prior. member[b] = (block_of(project(xyz)) == b), as boolean arrays kept in a list.
4. Read the dataset with Parser(data_dir, factor = 1, normalize = False, test_every = 8). names = its `image_names`, in order; centres = the translation column of `camtoworlds`; `inside = cameras_inside(part, centres)`.
5. Per frame: scale = min(1, max_side/max(W, H)); scale the first two rows of its K; size = (max(1, round(W·scale)), max(1, round(H·scale))); viewmat = inverse(camtoworld).
6. Render(camera, drop): one call to gsplat's `rasterization` with:
   - means = xyz;
   - quats;
   - scales = e^(log_scales);
   - opacities = sigmoid(logit_opacity), with the dropped gaussians' opacity set to 0;
   - colours = `f_dc` as degree-0 SH, shape (N, 1, 3), `sh_degree = 0`;
   - `packed = False`; that frame's viewmat, K and size.

   Take the RGB clamped to [0, 1] on the CPU as (H, W, 3). Map `drop` to its precomputed device mask by object identity with `members`; anything else becomes a fresh mask. Cache the full render per camera.

7. For each frame:
   - render the full view first;
   - a gaussian is _visible_ when any of its projected radii (`meta["radii"]`, reduced over trailing axes) are > 0;
   - skip[c, b] = inside[c, b] OR no visible gaussian of block b;
   - row = `contribution(render-for-this-frame, 1, members, skip=skip[c:c+1])[0]`;
   - drop the cache entry;
   - print `block_views: k of n frames` every 25 frames and at the last.
8. Write `--out` (parents created): `{"names": names, "loss": loss rounded to 6 decimals, as nested lists, "seconds": wall seconds rounded to 0.1}`. Return 0.
9. `if __name__ == "__main__": raise SystemExit(main())`.

Docstring: "Not executed on a GPU in this repository yet." State the rule as in §5B (`contribution`, `ssim`), citing CityGaussian arXiv 2404.01133 for the idea of testing a camera by removing a block from the coarse model's render.

### 5D. CLEAN-ROOM SPEC — `harmonics.basis` and its constants

**Definition.** The real spherical harmonics of bands l = 1…3, ordered m = −l…l. Take Y_l^m to be the complex orthonormal SH **including** the Condon–Shortley phase (−1)^m, with θ the polar angle from +z and φ the azimuth from +x towards +y. Then:

- Y_{l,m<0} = √2·Im Y_l^{|m|};
- Y_{l,0} = Y_l^0;
- Y_{l,m>0} = √2·Re Y_l^m.

The normalisation is N_l^m = sqrt((2l+1)/(4π) · (l−m)!/(l+m)!).

This is the basis 3DGS trainers fit and CesiumJS's `PrimitiveGaussianSplatVS.glsl` evaluates (its constants agree to 8 digits). The convention is the one gsplat's `_eval_sh_bases_fast` (Apache-2.0) uses; standard tables: Sloan, "Stupid Spherical Harmonics Tricks", 2008.

**Constants**, each written as `math.sqrt(n / (d * math.pi))`. Verified bit-identical (`==`) to today's literals:

| Name    | Value                                                                             |
| ------- | --------------------------------------------------------------------------------- |
| `SH_C0` | √(1/4π)                                                                           |
| `SH_C1` | √(3/4π)                                                                           |
| `SH_C2` | (√(15/4π), −√(15/4π), √(5/16π), −√(15/4π), √(15/16π))                             |
| `SH_C3` | (−√(35/32π), √(105/4π), −√(21/32π), √(7/16π), −√(21/32π), √(105/16π), −√(35/32π)) |

**`basis(degree, dirs) -> float64 (n, SH_DIMS[degree])`**, for unit directions (x, y, z). Columns in this order, evaluated in these Cartesian forms, not simplified with x² + y² + z² = 1. The spec asked for values bit-identical to the previous basis; they are not quite: forming xy, yz and xz once and reusing them orders the multiplies differently, and a review measured differences of up to 2.2e-16 (1 ulp of 1.0) in columns 3, 4, 6 and 9. So the claim is equality to 1 ulp, not bit identity; the test holds every column to 1e-12 of a reference built from the Legendre recurrence, and claims nothing about the previous code.

- band 1: −C1·y, C1·z, −C1·x;
- band 2: C2₀·xy, C2₁·yz, C2₂·(2z² − x² − y²), C2₃·xz, C2₄·(x² − y²);
- band 3: C3₀·y(3x² − y²), C3₁·xyz, C3₂·y(4z² − x² − y²), C3₃·z(2z² − 3x² − 3y²), C3₄·x(4z² − x² − y²), C3₅·z(x² − y²), C3₆·x(x² − 3y²).

Degree 0 returns an (n, 0) array. `rotate`, `band_rotation`, `evaluate`, `sources` and `truncate` are unchanged.

### 5E. CLEAN-ROOM SPEC — `tools/pipeline/lod_maths.py` (the viewer's cut only)

Purpose: CesiumJS's REPLACE traversal on screen-space error, used by `experiments/lod_compare.py`. CPython 3.10 and numpy only. Exports `F64`, `I64`, `Tree`, `tree_from`, `box_distance`, `screen_space_error`, `cut`, `orbit_eye`, `look_at`.

- **`Tree`** is a frozen dataclass:

  | Field      | Type              |
  | ---------- | ----------------- |
  | `uris`     | tuple[str]        |
  | `low`      | (n, 3) f64        |
  | `high`     | (n, 3) f64        |
  | `error`    | (n,) f64          |
  | `children` | tuple[tuple[int]] |
  | `rows`     | (n, 2) int64      |

  Properties: `parents` (indices with children) and `leaves` (without).

- **`tree_from(tiles, counts)`** takes tiles in `Tile.walk()` order (root first), each with `.children`, `.uri`, `.low`, `.high` and `.geometric_error`. `rows` gives each tile its half-open run in one table: all leaves first, in walk order, then all parents, in walk order, each run `counts[t]` long. Accept low and high as any array-likes; reshape to (n, 3).
- **`box_distance(eye, low, high)`** = ‖max(max(low − eye, eye − high), 0)‖. It is 0 inside the box.
- **`screen_space_error(tree, t, eye, focal)`** = error_t·focal/d, or +∞ when d ≤ 0.
- **`cut(tree, eye, focal, tau)`**: depth-first from tile 0, with a stack. A tile with children whose error is greater than tau is replaced by its children, pushed so they are visited in order; any other tile is chosen. Return the chosen tiles in visit order.
- **`orbit_eye(target, direction, low, high, distance)`**: d̂ = the normalised direction. Bisect t over [0, max(1, 4·distance + 4·‖high − low‖)] for 80 iterations on whether box_distance(target + t·d̂) < distance. Return target + t_hi·d̂.
- **`look_at(eye, target, up=(0, 0, 1))`**: a 4×4 world-to-camera matrix with OpenCV axes (x right, y down, z forward):
  - forward f = normalise(target − eye);
  - right r = normalise(f × up); if |f × up| < 1e-9, use up = (0, 1, 0) instead;
  - down d = f × r;
  - rotation rows (r, d, f); translation −rotation·eye.

### 5F. CLEAN-ROOM SPEC — `tools/pipeline/placement.py`

- **`PLACEMENT`** = `ArtifactDecl("placement.json", content_type="application/json", summary="the similarity place applied, x' = scale * rotation @ x + translation, from the trained splat's (COLMAP) frame to canonical.ply's")`.
- **`compose(rotation, translation, scale, then) -> dict[str, object]`.** The first step is x′ = s₁R₁x + t₁ (t₁ = 0 when `translation` is None). If `then` is not None (it has `.rotation`, `.translation` and an optional `.scale`, default 1), the second step is y = s₂R₂x′ + t₂. The composition is (s₁s₂, R₂R₁, s₂R₂t₁ + t₂). Return:
  - `{"scale": float, "rotation": 3×3 nested lists of float, "translation": [3 floats], "from": "the trained splat's frame (COLMAP)", "to": "canonical.ply (east/north/up about the placed origin)"}`.

  numpy is imported inside the function, as before (cheap module import).

---
