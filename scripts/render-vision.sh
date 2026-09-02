#!/usr/bin/env bash
# Regenerate everything the vision + sizing reviews are built from:
#   docs/design/01-sizing.md + docs/design/sizing/*.png   (analysis/sizing.py)
#   docs/design/vision.md + docs/design/vision/**          (vision-board)
# then fix two things the renderer gets wrong at robot scale (see friction log).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/hw-py/bin/python

$PY analysis/sizing.py

vision-board concepts/concept_a_tibia2x.py concepts/concept_b_tibia2p5x.py \
  --project "Hexapod" \
  --description "A hexapod the size of a large dog for outdoor missions: first navigating complex terrain, then picking up trash. Off-the-shelf where sensible, custom where it meaningfully improves cost or performance. Able to carry an adult male if needed. All eighteen motors housed statically in the body, power transmitted to the joints; the motors themselves axial-flux PCB stators with in-plane cycloidal reductions. Legs sprawl: a coxa carries each leg out from a hip under the body, the femur sticks out and up, and a tibia at least twice the femur comes straight down, so both links are up-and-down in the neutral stance and the joint torques stay low. Later: modular payloads and tools, two hot-swap batteries, wireless charging, a solar top, networking, SLAM, and a wheeled ground transport vehicle." \
  --question "Which leg proportion, A (tibia 2.0x femur, squatter) or B (tibia 2.5x femur, taller, 20 % less femur torque), and what made you pick it?" \
  --question "Does the rider case stay as a design driver, or is it a stretch goal we size for later?" \
  --question "Hips under the body on a 240 mm spacing with a 150 mm coxa: is that the right split between body width and coxa length?"

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
# 2. vision.md: the review page escapes raw HTML, so drop the <details> wrapper.
p = "docs/design/vision.md"
s = open(p).read()
s = s.replace("<details><summary>Dimensioned isometric line drawing</summary>\n", "**Isometric line drawing, hidden edges dashed**\n")
s = s.replace("</details>\n", "")
open(p, "w").write(s)
print("post-processed iso.svg strokes and vision.md")
PYEOF
