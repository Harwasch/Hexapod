#!/usr/bin/env bash
# Regenerate everything the vision + sizing reviews are built from:
#   docs/design/01-sizing.md + docs/design/sizing/*.png       (analysis/sizing.py)
#   docs/design/02-leg-topology.md + docs/design/topology/*   (analysis/topology.py)
#   docs/design/vision.md + docs/design/vision/**             (vision-board)
# then fix two things the renderer gets wrong at robot scale (see friction log).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/hw-py/bin/python

$PY analysis/sizing.py
$PY analysis/topology.py

vision-board concepts/concept_a_sprawl.py concepts/concept_b_mammal_mode.py \
  --project "Hexapod" \
  --description "A hexapod the size of a large dog for outdoor missions: first navigating complex terrain, then picking up trash. Off-the-shelf where sensible, custom where it meaningfully improves cost or performance. Carrying an adult is a stretch goal. All eighteen motors housed statically in the body, power transmitted to the joints; the motors themselves axial-flux PCB stators with in-plane cycloidal reductions, each DOF geared to its own rating. Legs sprawl (confirmed): a 150 mm coxa carries each leg out from a hip under the body, the 250 mm femur sticks out and up, and a 500 mm tibia comes straight down. The same leg yaws 90 degrees into a mammal stance when the robot needs to be tall or narrow; both stances are rendered. Later: modular payloads and tools, two hot-swap batteries, wireless charging, a solar top, networking, SLAM, and a wheeled ground transport vehicle." \
  --question "Concept B is the same robot reconfigured, not a different one: hips 0.64 m up, 0.24 m stance. Is that the mammal stance you had in mind, or should the mammal stance keep some sprawl (yaw 60 degrees, say)?" \
  --question "The tibia is now 500 mm (2.0x): hips 0.32 m up in the sprawl stance. Is that enough clearance for the terrain demo, given the tall mammal stance is available?" \
  --question "Motor and reduction: does 04-motor-reduction-options.md change the plan, or is PCB stator plus in-plane cycloid still the path, with the wound flat-coil stator as the torque upgrade?"

$PY - <<'PYEOF'
import glob, re
# 1. Isometric SVGs: stroke width is fixed for a pocket-scale part; scale it to the viewBox.
for p in glob.glob("docs/design/vision/*/iso.svg"):
    s = open(p).read()
    vb = re.search(r'viewBox="([^"]+)"', s).group(1).split()
    w = float(vb[2])
    s = re.sub(r'<svg width="[^"]+" height="[^"]+"', '<svg width="100%"', s, count=1)
    def fix_group(m):
        tag = m.group(0)
        sw = w * (0.0008 if 'id="hidden"' in tag else 0.0016)
        return re.sub(r'stroke-width="[^"]+"', 'stroke-width="%.3f"' % sw, tag)
    s = re.sub(r'<g [^>]*>', fix_group, s)
    s = re.sub(r'stroke-dasharray="[^"]+"', 'stroke-dasharray="%.2f %.2f"' % (w * 0.004, w * 0.004), s)
    open(p, "w").write(s)
# 2. vision.md: the review page escapes raw HTML, so drop the <details> wrapper and the <sub> footer.
p = "docs/design/vision.md"
s = open(p).read()
s = s.replace("<details><summary>Dimensioned isometric line drawing</summary>\n", "**Isometric line drawing, hidden edges dashed**\n")
s = s.replace("</details>\n", "")
s = re.sub(r"\n---\n\n<sub>(.*?)</sub>\s*$", r"\n*\1*\n", s, flags=re.S)
open(p, "w").write(s)
print("post-processed iso.svg strokes and vision.md")
PYEOF
