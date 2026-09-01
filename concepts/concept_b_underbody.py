"""Concept B — under-body: sagittal leg planes, mammal stance."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from hexapod_model import STANCE_SAGITTAL  # noqa: E402
from hexapod_skeleton import build  # noqa: E402

TITLE = "B — Under-body"
NOTES = (
    "Leg planes fore-aft, feet under the hips, front and mid knees forward, rear knees back. "
    "Hips 0.45 m up, 0.48 m stance width, 0.96 m stance length. The pitch joints both carry "
    "and push; yaw only steers and crabs. Same body, same eighteen actuators, same leg links. "
    "Mass figure below is the solid render, not the 49 kg budget."
)
MATERIAL = "graphite"
RATIONALE = (
    "Narrow and tall: fits through a doorway, more ground clearance, and the walking motion "
    "is a pitch sweep that the femur transmission can be optimised for. Pays for it in roll "
    "margin (13° unloaded, 8° with a rider on a tripod), so a rider means a wave gait and "
    "splayed feet."
)

PART = build(STANCE_SAGITTAL)
