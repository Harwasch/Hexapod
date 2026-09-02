"""Concept B — sprawled leg, tibia 2.5× femur."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build123d import Compound  # noqa: E402
from hexapod_model import STANCE_B as ST  # noqa: E402
from hexapod_skeleton import build, describe  # noqa: E402

_robot, _figure = build(ST)
lg = ST.leg

TITLE = "B — Tibia 2.5× femur"
NOTES = (
    f"Coxa {lg.coxa:.0f} / femur {lg.femur:.0f} / tibia {lg.tibia:.0f} mm. Femur up at {ST.femur_deg:.0f}°, tibia "
    f"vertical; hips {ST.hip_height/1000:.2f} m up, knees at {ST.knee_height/1000:.2f} m, feet {ST.foot_radius()/1000:.2f} m out "
    f"from the yaw axes. Femur arm {ST.foot_reach:.0f} mm. Same body, same eighteen actuators. "
    f"**Robot alone: {describe(_robot)}** — the envelope, volume and mass figures below include the 1.83 m (6 ft) "
    "reference figure, and the mass treats the skeleton as solid; the budget is 49 kg."
)
MATERIAL = "graphite"
RATIONALE = (
    "The steeper femur and longer tibia pull the foot in under the knee: a 126 mm femur arm, ~100 N·m under a "
    "rider, 20 % less than A in every load case, and 50 mm more ground clearance. Costs a longer tibia to swing, "
    "a knee at deck level, and a slightly narrower footprint."
)

PART = Compound(children=[*_robot.solids(), *_figure.solids()])
