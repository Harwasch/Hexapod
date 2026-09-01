"""Concept A — sprawl: radial leg planes, insect stance."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from hexapod_model import STANCE_SPRAWL  # noqa: E402
from hexapod_skeleton import build  # noqa: E402

TITLE = "A — Sprawl"
NOTES = (
    "Leg planes radiate from the body; front and rear legs yawed ±30°. Hips 0.40 m up, "
    "feet 0.28 m out from the yaw axes, 0.88 m stance width. The yaw joints push, the "
    "pitch joints carry. Skeleton view: top deck and floor plate, side rails, six hip "
    "stacks of three Ø170 pancake actuators, two hot-swap packs and the compute box "
    "between them. "
    "The renderer's mass figure treats the skeleton as solid; the budget is 49 kg, see 01-sizing.md."
)
MATERIAL = "graphite"
RATIONALE = (
    "Stability first. 26° roll margin unloaded and 16° with a rider on a tripod, so the "
    "rider case does not force a wave-gait-only machine. Costs a wide footprint and puts "
    "the propulsion duty on the yaw joint."
)

PART = build(STANCE_SPRAWL)
