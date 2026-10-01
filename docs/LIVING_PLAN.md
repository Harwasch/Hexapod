# Living plan: checklist

Architecture: [SCENE_OBJECTS.md](SCENE_OBJECTS.md). Branch: `living-models`.
`[x]` done · `[~]` in progress · `[ ]` not started.

## A. Segmentation (camp first)

- [~] A1 Lift masks to splats: views, voting, instances, hierarchy, `instances.json` (tested with oracle masks on the synthetic yard)
- [~] A2 Model adapters: SAM 2 masks, SigLIP embeddings, open-vocabulary tags, property scores (CPU now, Modal later)
- [ ] A3 Run on the camp, render a colour-by-instance image
- [~] A4 Viewer: load `instances.json`, hide / highlight by instance, text search
- [ ] A5 Camp in the viewer: search "tent", "tree", "table"; hide vegetation

## B. Skin

- [~] B1 PhysSkin / Simplicits: licences, dependencies, CPU or GPU run on the synthetic tree
- [ ] B2 Skin format in tiles (weights per splat, handles per instance)
- [ ] B3 GPU skinning path for any instance (generalise the tree rig hook)

## C. Drive

- [ ] C1 Wind forces on handles (living)
- [ ] C2 Video teacher fits materials per instance (stiffness, damping, drag)
- [ ] C3 Telemetry drives a rigid instance (live)
- [ ] C4 Movable instances to their own tilesets + fill the hole

## D. Already done (this branch)

- [x] View cones: fade what a capture never saw
- [x] Fill pipeline (Teacher B) + inferred layer in the viewer
- [x] Modal functions for Fixer, Wan, Cosmos, gsplat distill (not yet run on GPU)
- [x] Motion teacher (Teacher A) on a stand-in
