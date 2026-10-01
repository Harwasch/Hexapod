# Living plan: checklist

Architecture: [SCENE_OBJECTS.md](SCENE_OBJECTS.md). Branch: `living-models`.
`[x]` done · `[~]` in progress · `[ ]` not started.

## A. Segmentation (camp first)

- [x] A1 Lift masks to splats: views, voting, instances, hierarchy, `instances.json` (synthetic yard: mean IoU 0.87 vs ground truth)
- [~] A2 Model adapters: SAM 2 masks, SigLIP embeddings, open-vocabulary tags, property scores (CPU now, Modal later)
- [ ] A3 Run on the camp, render a colour-by-instance image
- [~] A4 Viewer: load `instances.json`, hide / highlight by instance, text search (built + unit-tested; needs a browser check with a real file)
- [ ] A5 Camp in the viewer: search "tent", "tree", "table"; hide vegetation
- [ ] A6 Segment the pumpkin (phone capture 09-28) and spool (09-26) scans

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

- [ ] E1 Pull in `claude/funny-carson-937ydv` (rendering + UX) after its final push, once A–B1 are merged
- [ ] E2 After E1: this session owns releases and infrastructure

## D. Already done (this branch)

- [x] View cones: fade what a capture never saw
- [x] Fill pipeline (Teacher B) + inferred layer in the viewer
- [x] Modal functions for Fixer, Wan, Cosmos, gsplat distill (not yet run on GPU)
- [x] Motion teacher (Teacher A) on a stand-in
