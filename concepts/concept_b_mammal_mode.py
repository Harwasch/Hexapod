"""Concept B — the same robot as A, reconfigured into its mammal stance."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build123d import Compound  # noqa: E402
from hexapod_model import BODY  # noqa: E402
from leg3d import MAMMAL_MODE as TOPO  # noqa: E402
from hexapod_skeleton import build, describe  # noqa: E402

_robot, _figure = build(TOPO)
c, f, t = TOPO.link_lengths()

TITLE = "B — Same robot, mammal stance"
NOTES = (
    f"Not a different design: concept A with every leg yawed 90° so its plane runs fore-aft, femur down and forward "
    f"at 45°, tibia back to put the foot under the hip. Hips {TOPO.hip_height/1000:.2f} m up, {BODY.width/1000:.2f} m stance, knees forward on "
    "the front and mid legs, back on the rear. The tall, narrow mode: doorways, deep obstacle fields, looking over things. "
    f"**Robot alone: {describe(_robot)}** — the envelope, volume and mass figures below include the 1.83 m (6 ft) "
    "reference figure, and the mass treats the skeleton as solid; the budget is 49 kg."
)
MATERIAL = "graphite"
RATIONALE = (
    "Reconfiguration instead of a second topology: the yaw joint's ±95° range and the pitch joints' ratings are the only "
    "things it costs. In this stance the pitch joints do the pushing, so the femur and knee continuous ratings in "
    "01-sizing.md are set here, about 15–20 % above what the sprawl stance alone needs."
)

PART = Compound(children=[*_robot.solids(), *_figure.solids()])
