#!/opt/hw-py/bin/python
"""Export each concept's skeleton as binary STL, one file per colour group,
into build/models/<concept>/ for the review page's 3-D viewer.

    /opt/hw-py/bin/python scripts/export_models.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "concepts"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from build123d import export_stl  # noqa: E402
from hexapod_model import STANCE_A, STANCE_B  # noqa: E402
from hexapod_skeleton import build_groups, ground_level  # noqa: E402

CONCEPTS = {"concept_a_tibia2x": STANCE_A, "concept_b_tibia2p5x": STANCE_B}
OUT = os.path.join(ROOT, "build", "models")

manifest = {}
for name, stance in CONCEPTS.items():
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    groups = build_groups(stance)
    files = {}
    for g, comp in groups.items():
        p = os.path.join(d, f"{g}.stl")
        export_stl(comp, p, tolerance=1.0, angular_tolerance=0.4, ascii_format=False)
        files[g] = os.path.relpath(p, ROOT)
    manifest[name] = {"title": stance.name, "ground_z": ground_level(stance), "files": files,
                      "hip_height": stance.hip_height, "knee_height": stance.knee_height}
    print(name, {g: f"{os.path.getsize(os.path.join(ROOT, f))//1024} kB" for g, f in files.items()})

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", os.path.join(OUT, "manifest.json"))
