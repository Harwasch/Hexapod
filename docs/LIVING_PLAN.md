# Living plan: checklist

Architecture: [SCENE_OBJECTS.md](SCENE_OBJECTS.md). Branch: `living-models`.
`[x]` done · `[~]` in progress · `[ ]` not started.

## A. Segmentation (camp first)

- [x] A1 Lift masks to splats: views, voting, instances, hierarchy, `instances.json` (synthetic yard: mean IoU 0.87 vs ground truth)
- [x] A2 Model adapters: SAM 2 masks, SigLIP embeddings, open-vocabulary tags, property scores (CPU and Modal GPU via `segment.yml`)
- [x] A3 Run on the camp (GPU, 252 views, 20 min): 816 top-level objects, 92% assigned; fort wall, cabin, trails, bushes, flagpole found; touching conifer crowns still merge
- [x] A4 Viewer: load `instances.json`, hide / highlight by instance, text search (checked in a real browser on the spool; e2e on the yard)
- [x] A5 Camp in the viewer: instances published beside the camp, spool and pumpkin tiles (`publish-instances.yml`); live-checked; hide / highlight now work under PlayCanvas, Spark and Cesium, plus "hide all N matches"
- [~] A6 Segment the pumpkin (phone capture 09-28) and spool (09-26) scans — both segmented; pumpkins tagged "pumpkin", spool split well but the vocabulary has no "spool" (needs search by meaning)

## B. Skin

- [x] B1 Skin method chosen: Kaolin Simplicits/FreeForm (Apache-2.0); PhysSkin rejected (no licence, unstable on the tree)
- [x] B2 Skin format in tiles: `skin.json` + `skin.bin` (SCENE_OBJECTS.md §4), dense int8, 16 B per skinned splat (top-k tears: 22% rms error at k = 4); Kaolin RKPM vendored, NumPy only, 3–8 s per tree
- [x] B3 GPU skinning path for any instance: a motion-chain part beside the rig, covariances through `I + Σ w_j A_j` (engine patch: optional `splatVertexJacobian`); drivers call `skinningOf(asset).setInstanceHandles(id, Z)` (e2e on the yard)

## C. Drive

- [x] C1 Wind forces on handles (living): anchored modal model per skin (`ω_j = c·√λ_j / scale`, the lowest tenth anchors), exact 60 Hz grid, bounded at a quarter of a support radius; priors from properties, `materials.json` overrides (SCENE_OBJECTS.md §4); 0.34 ms a frame for 30 objects (e2e on the yard)
- [x] C2 Video teacher fits materials per instance (stiffness, damping, drag): writes `materials.json` (SCENE_OBJECTS.md §4); `teacher_materials.py` matches the C1 model's predicted screen-motion spectrum (Python port `skin_wind.py`, parity-tested against `skinWind.ts`) to tracked points; synthetic yard from a 4×-wrong prior: c within 4%, ζ within 16%, D within 14%; a world-model clip on Modal not yet run (needs the `huggingface` secret)
- [ ] C3 Telemetry drives a rigid instance (live)
- [ ] C4 Movable instances to their own tilesets + fill the hole

## E. Merge

- [x] E1 Pulled in `claude/funny-carson-937ydv` (rendering + UX), merged cleanly at `0e0dd71`
- [x] E2 After E1: this session owns releases and infrastructure
  - [x] Retarget the 6 workflows that ran on pushes to `claude/funny-carson-937ydv` to `living-models` (modal, modal-benchmark, living-plants, minnetonka-tree, collision-backfill, streamed-lod-backfill)
  - [x] Bring `main` up to date: Harwasch/Hexapod#2 merged at `4429290`, deployed to production 2026-10-02 (web: twin-web-f57.pages.dev, API: twin-api.fly.dev)

## D. Already done (this branch)

- [x] View cones: fade what a capture never saw
- [x] Fill pipeline (Teacher B) + inferred layer in the viewer
- [x] Modal functions for Fixer, Wan, Cosmos, gsplat distill (not yet run on GPU)
- [x] Motion teacher (Teacher A) on a stand-in
