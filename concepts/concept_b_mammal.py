"""Concept B — for comparison: mammal, roll–pitch–pitch, feet under the body."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build123d import Compound  # noqa: E402
from leg3d import TOPOLOGIES  # noqa: E402
from hexapod_skeleton import build, describe  # noqa: E402

TOPO = [t for t in TOPOLOGIES if t.key == "mammal"][0]
_robot, _figure = build(TOPO)
c, f, t = TOPO.link_lengths()

TITLE = "B — Mammal, roll–pitch–pitch"
NOTES = (
    f"The quadruped-style alternative, same body and hip stacks: a roll joint at the hip, a {c:.0f} mm carrier, "
    f"femur {f:.0f} / tibia {t:.0f} mm hanging under the body edge, knees forward on the front and mid legs, back on the rear. "
    f"Hips {TOPO.hip_height/1000:.2f} m up, feet {TOPO.foot_radius/1000:.2f} m out from the hip axes, 0.44 m stance width. "
    f"**Robot alone: {describe(_robot)}** — the envelope, volume and mass figures below include the 1.83 m (6 ft) "
    "reference figure, and the mass treats the skeleton as solid; the budget is 49 kg."
)
MATERIAL = "graphite"
RATIONALE = (
    "Lowest joint torques when the feet stay under the hips, the proven quadruped actuator layout (pitch and knee "
    "motors coaxial on the hip carrier, no transmission through a coxa), and the whole step-up box is reachable. "
    "Pays with a 0.44 m stance, half the roll margin, and a roll joint that sees the weight the moment a foot "
    "moves sideways — see docs/design/02-leg-topology.md."
)

PART = Compound(children=[*_robot.solids(), *_figure.solids()])
