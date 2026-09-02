"""Concept A — sprawled leg, tibia 2.0× femur."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build123d import Compound  # noqa: E402
from hexapod_model import STANCE_A as ST  # noqa: E402
from hexapod_skeleton import build, describe  # noqa: E402

_robot, _figure = build(ST)
lg = ST.leg

TITLE = "A — Tibia 2.0× femur"
NOTES = (
    f"Coxa {lg.coxa:.0f} / femur {lg.femur:.0f} / tibia {lg.tibia:.0f} mm. Femur up at {ST.femur_deg:.0f}°, tibia "
    f"vertical; hips {ST.hip_height/1000:.2f} m up, knees at {ST.knee_height/1000:.2f} m, feet {ST.foot_radius()/1000:.2f} m out "
    f"from the yaw axes. Femur arm {ST.foot_reach:.0f} mm. Skeleton view: top deck and floor plate, side rails, "
    "six hip stacks of three Ø170 pancake actuators under the deck, two hot-swap packs and the compute box "
    f"between them. **Robot alone: {describe(_robot)}** — the envelope, volume and mass figures below include the "
    "1.83 m (6 ft) reference figure, and the mass treats the skeleton as solid; the budget is 49 kg."
)
MATERIAL = "graphite"
RATIONALE = (
    "The shorter tibia keeps the swing mass and the knee height down (knee just below deck level) and the "
    "footprint wider, and pays for it with a 177 mm femur arm: ~126 N·m at the femur under a rider."
)

PART = Compound(children=[*_robot.solids(), *_figure.solids()])
