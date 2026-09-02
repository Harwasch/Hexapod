"""Single source of truth for the hexapod's top-level numbers.

Everything downstream — the sizing document, the torque maps, the two vision
concepts — imports this module, so a change here moves every render and every
table together.  Units: mm, kg, N, N·m, s unless a name says otherwise.

Coordinate conventions
----------------------
Body frame: +x forward, +y left, +z up, origin at the centre of the body slab
at the height of the femur pitch axes (the underside of the slab).

Leg plane: a 2-D frame attached to each leg, origin at the femur pitch axis,
`r` horizontal outward along the leg plane, `z` up.  The coxa link runs from
the yaw axis to the femur axis, horizontally, along +r.  The femur pitches up
and out from there, and the tibia comes down to the foot.

Why the leg is shaped this way (review decision, 2026-09-02): the joint
torque from a vertical foot load is the load times the *horizontal* distance
from the joint to the foot.  A steep femur keeps the knee close in over the
femur axis, a vertical tibia adds nothing, and the sprawl comes from the coxa,
which carries the load as a bending moment into the yaw bearing rather than as
motor torque.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # m/s²


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    coxa: float      # yaw axis -> femur pitch axis, horizontal
    femur: float     # femur pitch axis -> knee pitch axis
    tibia: float     # knee pitch axis -> foot contact point

    @property
    def reach(self) -> float:
        return self.femur + self.tibia

    @property
    def tibia_ratio(self) -> float:
        return self.tibia / self.femur


@dataclass(frozen=True)
class Body:
    length: float = 900.0                        # overall slab length
    width: float = 240.0                         # between left/right yaw axes (hips under the body)
    height: float = 200.0                        # slab height (actuator stacks live inside)
    hip_x: tuple = (330.0, 0.0, -330.0)          # front, mid, rear hip yaw axes
    side_rail: float = 16.0                      # structural rail outboard of the hip stacks

    def slab_width(self, act) -> float:
        """Overall slab width: yaw-axis spacing + one actuator diameter + two side rails."""
        return self.width + act.od + 2 * self.side_rail


@dataclass(frozen=True)
class Actuator:
    """Envelope of one joint actuator (motor + in-plane cycloid), a pancake."""
    od: float = 170.0        # outer diameter
    thickness: float = 42.0  # axial length incl. bearings
    mass: float = 1.1        # kg, target
    stack_gap: float = 8.0   # air/structure between stacked pancakes


@dataclass(frozen=True)
class Stance:
    """A nominal standing posture, defined by the femur angle and the tibia lean.

    The hip height and the foot reach follow from those and the leg lengths,
    so a 'tibia vertical' stance stays vertical when a length changes.

    leg_plane:
    radial   — leg planes fan out from the body (insect).  Propulsion is a
               force *perpendicular* to the leg plane, so the yaw joint does
               the pushing and the pitch joints carry the weight.
    sagittal — leg planes are fore-aft (mammal).  Propulsion is *in* the leg
               plane, so the pitch joints push and carry; yaw only steers.
    """
    name: str
    leg: Leg
    femur_deg: float           # femur angle above horizontal in the neutral stance
    tibia_lean_deg: float      # tibia angle from vertical, +ve = foot outboard of the knee
    leg_plane: str             # 'radial' | 'sagittal'
    yaw_deg: tuple             # per (front, mid, rear): leg-plane yaw from +y (outboard), +ve toward +x

    @property
    def knee(self) -> tuple[float, float]:
        a = math.radians(self.femur_deg)
        return self.leg.femur * math.cos(a), self.leg.femur * math.sin(a)

    @property
    def foot_reach(self) -> float:
        """Femur axis -> foot, horizontal, in the leg plane."""
        kx, _ = self.knee
        return kx + self.leg.tibia * math.sin(math.radians(self.tibia_lean_deg))

    @property
    def hip_height(self) -> float:
        """Femur pitch axis above ground."""
        _, kz = self.knee
        return self.leg.tibia * math.cos(math.radians(self.tibia_lean_deg)) - kz

    @property
    def knee_height(self) -> float:
        """Knee above ground."""
        return self.hip_height + self.knee[1]

    def foot_radius(self) -> float:
        """Horizontal distance from the yaw axis to the foot."""
        return self.leg.coxa + self.foot_reach


# Review decision 2026-09-02: tibia 2.5× femur, with the longer (250 mm) femur.
LEG = Leg(coxa=150.0, femur=250.0, tibia=625.0)
STANCE = Stance("Sprawl, tibia 2.5× femur", LEG, femur_deg=55.0, tibia_lean_deg=0.0,
                leg_plane="radial", yaw_deg=(30.0, 0.0, -30.0))
STANCES = (STANCE,)

# ----------------------------------------------------------------------------
# Mass budget (kg)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MassBudget:
    actuators: float = 18 * 1.1
    body_structure: float = 6.0
    legs: float = 6 * 1.2            # coxa/femur/tibia links + foot + transmission share
    batteries: float = 2 * 4.0       # two hot-swap packs, ~680 Wh each (see Energy)
    electronics: float = 3.0         # compute, drivers, sensors, harness
    payload_interface: float = 2.0   # top deck rails, tool mount, solar skin
    margin: float = 3.0
    mission_payload: float = 8.0     # trash + gripper/tool (not part of the robot)
    rider: float = 100.0             # adult male, emergency / demo carry (not part of the robot)

    ROBOT_FIELDS = ("actuators", "body_structure", "legs", "batteries",
                    "electronics", "payload_interface", "margin")

    @property
    def robot(self) -> float:
        return round(sum(getattr(self, f) for f in self.ROBOT_FIELDS), 1)


MASS = MassBudget()


# ----------------------------------------------------------------------------
# Load cases — vertical force per supporting leg, plus a propulsion force
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadCase:
    name: str
    total_mass: float      # kg on the feet
    legs_down: int         # legs sharing the vertical load at the worst instant
    dyn_factor: float      # impact / acceleration multiplier on vertical load
    slope_deg: float       # slope the propulsion force must hold
    accel: float           # m/s² fore-aft acceleration the propulsion must provide
    rating: str            # 'continuous' | 'peak' | 'stretch' (reported, not a requirement)

    @property
    def foot_force_z(self) -> float:
        return self.total_mass * G * self.dyn_factor / self.legs_down

    @property
    def foot_force_prop(self) -> float:
        need = self.total_mass * (G * math.sin(math.radians(self.slope_deg)) + self.accel)
        return need / self.legs_down


LOAD_CASES = (
    LoadCase("Walk, tripod gait", MASS.robot + MASS.mission_payload, legs_down=3,
             dyn_factor=1.5, slope_deg=30.0, accel=1.0, rating="continuous"),
    LoadCase("Stumble / step-down, two legs", MASS.robot + MASS.mission_payload, legs_down=2,
             dyn_factor=3.0, slope_deg=0.0, accel=2.0, rating="peak"),
    # Review decision 2026-09-02: the rider is a stretch goal.  Reported so the
    # margin is visible; it does not set a rating.
    LoadCase("Rider, wave gait (stretch)", MASS.robot + MASS.rider, legs_down=3,
             dyn_factor=1.2, slope_deg=15.0, accel=0.3, rating="stretch"),
)


# ----------------------------------------------------------------------------
# Kinematics in the leg plane
# ----------------------------------------------------------------------------
def ik(leg: Leg, r: float, z: float) -> tuple[float, float]:
    """Femur angle from horizontal (rad, +ve up) and knee interior angle (rad),
    knee-up configuration, for a foot at (r, z) relative to the femur axis."""
    d = math.hypot(r, z)
    if d >= leg.reach or d <= abs(leg.femur - leg.tibia):
        raise ValueError(f"foot ({r:.0f},{z:.0f}) out of reach")
    knee = math.acos((leg.femur**2 + leg.tibia**2 - d**2) / (2 * leg.femur * leg.tibia))
    a1 = math.atan2(z, r) + math.acos((leg.femur**2 + d**2 - leg.tibia**2) / (2 * leg.femur * d))
    return a1, knee


def knee_pos(leg: Leg, a1: float) -> tuple[float, float]:
    return leg.femur * math.cos(a1), leg.femur * math.sin(a1)


def joint_torques(leg: Leg, r: float, z: float, Fz: float, Fr: float = 0.0, Fy: float = 0.0):
    """Quasi-static joint torques (N·m, magnitudes) for a ground reaction force
    on the foot: Fz up, Fr outward in the leg plane, Fy normal to the plane.
    Distances in mm, forces in N."""
    a1, _ = ik(leg, r, z)
    kx, kz = knee_pos(leg, a1)
    tau_femur = abs(r * Fz - z * Fr) / 1000.0
    tau_knee = abs((r - kx) * Fz - (z - kz) * Fr) / 1000.0
    tau_yaw = abs((leg.coxa + r) * Fy) / 1000.0
    return {"yaw": tau_yaw, "femur": tau_femur, "knee": tau_knee}


def stance_torques(stance: Stance, case: LoadCase, foot_reach: float | None = None):
    """Joint torques at a stance for a load case, routing the propulsion force
    into the joints the stance geometry says it goes to.  `foot_reach`
    overrides the neutral reach (same hip height) for workspace checks."""
    Fz = case.foot_force_z
    Fp = case.foot_force_prop
    if stance.leg_plane == "radial":
        # Side legs: propulsion is normal to the leg plane (yaw does it).  The
        # front/rear legs are yawed 30° so half their push is in-plane.
        Fy, Fr = Fp * math.cos(math.radians(30)), Fp * math.sin(math.radians(30))
    else:
        # Sagittal: propulsion is in the leg plane; yaw only sees lateral
        # forces, taken as 30 % of propulsion for turning / crabbing.
        Fy, Fr = 0.3 * Fp, Fp
    r = stance.foot_reach if foot_reach is None else foot_reach
    return joint_torques(stance.leg, r, -stance.hip_height, Fz=Fz, Fr=Fr, Fy=Fy)


# ----------------------------------------------------------------------------
# Gait and speed
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Gait:
    speed: float           # m/s body speed
    stride: float          # m, foot travel during stance
    duty: float            # fraction of the cycle a leg is in stance
    lift: float            # m, swing apex height
    label: str

    @property
    def t_stance(self) -> float:
        return self.stride / self.speed

    @property
    def t_cycle(self) -> float:
        return self.t_stance / self.duty

    @property
    def t_swing(self) -> float:
        return self.t_cycle - self.t_stance

    @property
    def cycle_hz(self) -> float:
        return 1.0 / self.t_cycle

    def swing_path(self) -> float:
        # half-ellipse-ish: forward stride plus up and down
        return self.stride + 2 * self.lift

    def swing_speed_peak(self) -> float:
        # sinusoidal profile: peak ≈ π/2 × mean
        return (self.swing_path() / self.t_swing) * math.pi / 2


GAITS = (
    Gait(0.3, 0.30, 5 / 6, 0.10, "Rider, wave gait"),
    Gait(1.0, 0.40, 0.5, 0.12, "Walk, tripod gait"),
    Gait(2.0, 0.50, 0.5, 0.12, "Fast walk, tripod gait (aspirational)"),
)


def joint_speeds(stance: Stance, gait: Gait):
    """Rough peak joint angular rates (rad/s) in swing and in stance.
    Effective lever arm = horizontal distance yaw-axis→foot (yaw) or the
    femur length (pitch joints)."""
    leg = stance.leg
    arm_yaw = stance.foot_radius() / 1000.0
    arm_pitch = leg.femur / 1000.0
    swing = gait.swing_speed_peak()
    stance_v = gait.speed
    if stance.leg_plane == "radial":
        return {"yaw_swing": swing / arm_yaw, "yaw_stance": stance_v / arm_yaw,
                "pitch_swing": (2 * gait.lift / gait.t_swing) * math.pi / 2 / arm_pitch,
                "pitch_stance": 0.3 * stance_v / arm_pitch}
    return {"yaw_swing": 0.3 * swing / arm_yaw, "yaw_stance": 0.3 * stance_v / arm_yaw,
            "pitch_swing": swing / arm_pitch, "pitch_stance": stance_v / arm_pitch}


# ----------------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------------
def com_height(body: Body, stance: Stance, with_rider: bool) -> float:
    """Combined centre-of-mass height above ground (m)."""
    h_robot = (stance.hip_height + body.height * 0.25) / 1000.0
    if not with_rider:
        return h_robot
    h_rider = (stance.hip_height + body.height) / 1000.0 + 0.45   # seated on the top deck
    return (MASS.robot * h_robot + MASS.rider * h_rider) / (MASS.robot + MASS.rider)


def tip_angles(body: Body, stance: Stance, with_rider: bool):
    """Static tip-over angles (deg) for roll and pitch, using the tripod
    support polygon (the narrowest one the gait produces)."""
    fr = stance.foot_radius()
    if stance.leg_plane == "radial":
        half_w = (body.width / 2 + fr * math.cos(math.radians(stance.yaw_deg[1]))) / 1000.0
        # tripod = one mid leg on one side, front+rear on the other; roll about
        # the line between the front/rear feet, ~ half the stance width
        half_w_tripod = half_w * 0.5
        half_l = (body.hip_x[0] + fr * math.sin(math.radians(stance.yaw_deg[0]))) / 1000.0
    else:
        half_w = (body.width / 2 + stance.leg.coxa) / 1000.0
        half_w_tripod = half_w * 0.5
        half_l = (body.hip_x[0] + stance.foot_reach) / 1000.0
    h = com_height(body, stance, with_rider)
    return {"roll_deg": math.degrees(math.atan2(half_w_tripod, h)),
            "pitch_deg": math.degrees(math.atan2(half_l, h)),
            "stance_width_m": 2 * half_w, "stance_length_m": 2 * half_l, "com_m": h}


# ----------------------------------------------------------------------------
# Motor + reduction model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PcbMotor:
    """Axial-flux motor with a PCB stator between two Halbach rotors.
    Torque = shear stress × swept area × radius, integrated over the annulus."""
    r_out: float = 85.0        # mm, active annulus outer radius
    r_in: float = 55.0         # mm, inner radius (cycloid lives inside this)
    stators: int = 1           # PCB stators in the stack (rotors = stators + 1)
    sigma_cont: float = 1.5e3  # Pa, continuous shear stress (thermal limit)
    sigma_peak: float = 4.5e3  # Pa, ~2 s bursts
    bus_v: float = 48.0

    def _geom(self) -> float:
        ro, ri = self.r_out / 1000.0, self.r_in / 1000.0
        return (2 * math.pi / 3) * (ro**3 - ri**3)

    @property
    def torque_cont(self) -> float:
        return self.stators * self.sigma_cont * self._geom()

    @property
    def torque_peak(self) -> float:
        return self.stators * self.sigma_peak * self._geom()


@dataclass(frozen=True)
class Reduction:
    ratio: float
    efficiency: float = 0.88   # single-stage cycloid, greased, warm


def joint_from_motor(m: PcbMotor, red: Reduction):
    return {"cont": m.torque_cont * red.ratio * red.efficiency,
            "peak": m.torque_peak * red.ratio * red.efficiency}


def motor_speed_for(joint_rad_s: float, red: Reduction) -> float:
    """Motor rpm needed for a joint rate."""
    return joint_rad_s * red.ratio * 60 / (2 * math.pi)


# ----------------------------------------------------------------------------
# Energy
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Energy:
    cost_of_transport: float = 1.0   # dimensionless, electrical, geared hexapod at walking pace
    hotel_w: float = 120.0           # compute, sensors, radios, drivers idling
    endurance_h: float = 2.0
    pack_wh_per_kg: float = 170.0
    packs: int = 2

    def walking_power(self, mass_kg: float, speed: float) -> float:
        return self.cost_of_transport * mass_kg * G * speed + self.hotel_w

    def pack_wh(self, avg_w: float) -> float:
        return avg_w * self.endurance_h / self.packs

    def pack_kg(self, avg_w: float) -> float:
        return self.pack_wh(avg_w) / self.pack_wh_per_kg


ENERGY = Energy()
BODY = Body()
ACT = Actuator()

# A 6 ft (1829 mm) reference figure for the renders.
HUMAN_HEIGHT = 1829.0
