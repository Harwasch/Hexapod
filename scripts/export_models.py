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
from leg3d import TOPOLOGIES, CHOSEN  # noqa: E402
from hexapod_skeleton import build_groups, ground_level  # noqa: E402

BY_KEY = {t.key: t for t in TOPOLOGIES}
CONCEPTS = {"concept_a_sprawl": ("A — Sprawl, yaw–pitch–pitch", CHOSEN),
            "concept_b_mammal": ("B — Mammal, roll–pitch–pitch", BY_KEY["mammal"])}
OUT = os.path.join(ROOT, "build", "models")

manifest = {}
for name, (title, topo) in CONCEPTS.items():
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    groups = build_groups(topo)
    files = {}
    for g, comp in groups.items():
        p = os.path.join(d, f"{g}.stl")
        export_stl(comp, p, tolerance=1.0, angular_tolerance=0.4, ascii_format=False)
        files[g] = os.path.relpath(p, ROOT)
    manifest[name] = {"title": title, "ground_z": ground_level(topo), "files": files}
    print(name, {g: f"{os.path.getsize(os.path.join(ROOT, f))//1024} kB" for g, f in files.items()})

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", os.path.join(OUT, "manifest.json"))
