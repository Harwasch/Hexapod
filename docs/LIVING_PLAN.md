# Living plan: checklist

Architecture: [SCENE_OBJECTS.md](SCENE_OBJECTS.md). Branch: `living-models`.
`[x]` done · `[~]` in progress · `[ ]` not started.

## A. Segmentation (camp first)

- [x] A1 Lift masks to splats: views, voting, instances, hierarchy, `instances.json` (synthetic yard: mean IoU 0.87 vs ground truth)
- [x] A2 Model adapters: SAM 2 masks, SigLIP embeddings, open-vocabulary tags, property scores (CPU and Modal GPU via `segment.yml`)
- [x] A3 Run on the camp (GPU, 252 views, 20 min): 816 top-level objects, 92% assigned; fort wall, cabin, trails, bushes, flagpole found; touching conifer crowns still merge
- [x] A4 Viewer: load `instances.json`, hide / highlight by instance, text search (checked in a real browser on the spool; e2e on the yard)
- [ ] A5 Camp in the viewer: search "tent", "tree", "table"; hide vegetation
- [~] A6 Segment the pumpkin (phone capture 09-28) and spool (09-26) scans — both segmented; pumpkins tagged "pumpkin", spool split well but the vocabulary has no "spool" (needs search by meaning)

## B. Skin

- [x] B1 Skin method chosen: Kaolin Simplicits/FreeForm (Apache-2.0); PhysSkin rejected (no licence, unstable on the tree)
- [ ] B2 Skin format in tiles (weights per splat, handles per instance)
- [ ] B3 GPU skinning path for any instance (generalise the tree rig hook)

## C. Drive

- [ ] C1 Wind forces on handles (living)
- [ ] C2 Video teacher fits materials per instance (stiffness, damping, drag)
- [ ] C3 Telemetry drives a rigid instance (live)
- [ ] C4 Movable instances to their own tilesets + fill the hole

## E. Merge

- [x] E1 Pulled in `claude/funny-carson-937ydv` (rendering + UX), merged cleanly at `0e0dd71`
- [~] E2 After E1: this session owns releases and infrastructure
  - [x] Retarget the 6 workflows that ran on pushes to `claude/funny-carson-937ydv` to `living-models` (modal, modal-benchmark, living-plants, minnetonka-tree, collision-backfill, streamed-lod-backfill)
  - [ ] Bring `main` up to date (it is 108 commits behind `living-models`) by pull request

## D. Already done (this branch)

- [x] View cones: fade what a capture never saw
- [x] Fill pipeline (Teacher B) + inferred layer in the viewer
- [x] Modal functions for Fixer, Wan, Cosmos, gsplat distill (not yet run on GPU)
- [x] Motion teacher (Teacher A) on a stand-in
