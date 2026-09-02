"""Concept A — the chosen leg: sprawl, yaw–pitch–pitch, coxa 150 / femur 250 / tibia 625."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build123d import Compound  # noqa: E402
from hexapod_model import STANCE  # noqa: E402
from leg3d import CHOSEN as TOPO  # noqa: E402
from hexapod_skeleton import build, describe  # noqa: E402

_robot, _figure = build(TOPO)
lg = STANCE.leg

TITLE = "A — Sprawl, yaw–pitch–pitch"
NOTES = (
    f"The leg agreed in review: coxa {lg.coxa:.0f} / femur {lg.femur:.0f} / tibia {lg.tibia:.0f} mm "
    f"(tibia {lg.tibia_ratio:.1f}× femur). Femur up at {STANCE.femur_deg:.0f}°, tibia vertical; hips {STANCE.hip_height/1000:.2f} m up, "
    f"knees at {STANCE.knee_height/1000:.2f} m, feet {STANCE.foot_radius()/1000:.2f} m out from the yaw axes, femur arm {STANCE.foot_reach:.0f} mm. "
    "Vertical yaw axis at the hip, so the weight never loads the first motor; the coxa carries it as bending. "
    f"**Robot alone: {describe(_robot)}** — the envelope, volume and mass figures below include the 1.83 m (6 ft) "
    "reference figure, and the mass treats the skeleton as solid; the budget is 49 kg."
)
MATERIAL = "graphite"
RATIONALE = (
    "Stability and clearance: 0.83 m stance width, 0.42 m under the body, feet placed anywhere on a ring around "
    "each hip. Pays for the long tibia with a 375 mm minimum leg extension, so a high step is taken with the foot "
    "swung outward, and with a knee that must match the femur's rating at the workspace corners."
)

PART = Compound(children=[*_robot.solids(), *_figure.solids()])
