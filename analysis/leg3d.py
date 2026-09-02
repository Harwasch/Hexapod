"""General 3-D serial-leg model: any 3-joint topology as a chain of (axis, link).

Used for the leg-topology trade study and for the per-DOF actuator ratings,
where the planar model in hexapod_model.py is not enough — the question is
what each joint sees when the foot is anywhere in its working volume, with a
force that has fore-aft and lateral components as well as vertical.

Frame: hip yaw/roll axis at the origin, +x forward, +y outboard (left leg),
+z up.  Units mm and N.  Torque_j = axis_j · ((p_foot − p_j) × F): the joint
only sees the component of the foot moment about its own axis, which is the
whole point of the topology choice — a vertical axis never sees weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def rot(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


@dataclass(frozen=True)
class Joint:
    axis: tuple      # rotation axis, in the frame before this joint turns
    link: tuple      # link from this joint to the next, in this joint's rotated frame (neutral q = 0)


@dataclass(frozen=True)
class Topology:
    key: str
    name: str
    joints: tuple            # three Joint
    joint_names: tuple       # e.g. ('yaw', 'femur', 'knee')
    stride_joint: str        # which joint sweeps the stride
    notes: str = ""

    def fk(self, q):
        """Joint positions p[0..2], foot p[3], and world joint axes a[0..2]."""
        T = np.eye(3)
        p = np.zeros(3)
        pts, axes = [], []
        for j, qi in zip(self.joints, q):
            axes.append(T @ np.asarray(j.axis, float))
            pts.append(p.copy())
            T = T @ rot(j.axis, qi)
            p = p + T @ np.asarray(j.link, float)
        pts.append(p.copy())
        return pts, axes

    def foot(self, q):
        return self.fk(q)[0][3]

    @property
    def neutral_foot(self):
        return self.foot((0.0, 0.0, 0.0))

    @property
    def hip_height(self) -> float:
        return -float(self.neutral_foot[2])

    @property
    def foot_radius(self) -> float:
        f = self.neutral_foot
        return float(math.hypot(f[0], f[1]))

    def ik(self, target, q0=(0.0, 0.0, 0.0), tol=1.0):
        t = np.asarray(target, float)
        r = least_squares(lambda q: self.foot(q) - t, np.asarray(q0, float), bounds=(-2.6, 2.6), xtol=1e-9, ftol=1e-9)
        if np.linalg.norm(r.fun) > tol:
            return None
        return r.x

    def torques(self, q, F):
        pts, axes = self.fk(q)
        F = np.asarray(F, float)
        return tuple(float(abs(np.dot(a, np.cross(pts[3] - p, F)))) / 1000.0 for p, a in zip(pts[:3], axes))

    def link_lengths(self):
        return tuple(float(np.linalg.norm(j.link)) for j in self.joints)


# ----------------------------------------------------------------------------
# The topologies on the table (left leg, mid-body, neutral stance)
# ----------------------------------------------------------------------------
def sprawl_ypp(coxa=150.0, femur=250.0, tibia=625.0, femur_deg=55.0, key="sprawl", name=None):
    a = math.radians(femur_deg)
    return Topology(
        key, name or f"Sprawl, yaw–pitch–pitch, femur {femur_deg:.0f}°",
        joints=(Joint((0, 0, 1), (0, coxa, 0)),
                Joint((1, 0, 0), (0, femur * math.cos(a), femur * math.sin(a))),
                Joint((1, 0, 0), (0, 0, -tibia))),
        joint_names=("yaw", "femur", "knee"), stride_joint="yaw",
        notes="Vertical first axis carries no weight; the coxa reacts the foot moment as bending. Stride is a yaw sweep.")


def mammal_rpp(carrier=100.0, femur=300.0, tibia=300.0, femur_deg=-45.0, key="mammal", name=None):
    a = math.radians(femur_deg)
    kx, kz = femur * math.cos(a), femur * math.sin(a)
    # tibia set so the foot lands under the femur axis
    dz = -math.sqrt(max(tibia**2 - kx**2, 1.0))
    return Topology(
        key, name or "Mammal, roll–pitch–pitch, feet under the body",
        joints=(Joint((1, 0, 0), (0, carrier, 0)),
                Joint((0, 1, 0), (kx, 0, kz)),
                Joint((0, 1, 0), (-kx, 0, dz))),
        joint_names=("roll", "femur", "knee"), stride_joint="femur",
        notes="Quadruped hip: roll at the body, pitch/knee motors coaxial on the carrier. Feet under the hips, stride is a pitch sweep.")


def lizard_ryp(coxa=150.0, femur=250.0, tibia=625.0, femur_deg=55.0, key="lizard", name=None):
    a = math.radians(femur_deg)
    return Topology(
        key, name or "Sprawl with a roll hip, roll–yaw–pitch",
        joints=(Joint((1, 0, 0), (0, coxa, 0)),
                Joint((0, 0, 1), (0, femur * math.cos(a), femur * math.sin(a))),
                Joint((1, 0, 0), (0, 0, -tibia))),
        joint_names=("roll", "yaw", "knee"), stride_joint="yaw",
        notes="The 'first motor axis rotated 90°' case: a horizontal fore-aft hip axis lifts the whole sprawled leg, so it sees the weight times the full lateral offset.")


TOPOLOGIES = (
    sprawl_ypp(),
    sprawl_ypp(femur_deg=75.0, key="sprawl_narrow", name="Sprawl, yaw–pitch–pitch, femur 75° (feet in)"),
    mammal_rpp(),
    lizard_ryp(),
)
CHOSEN = TOPOLOGIES[0]


# ----------------------------------------------------------------------------
# Workspace evaluation
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Workspace:
    """Foot displacements from the neutral foot the leg must reach under load."""
    dx: tuple = (-200.0, 200.0)     # stride, fore-aft
    dy: tuple = (-100.0, 150.0)     # lateral, in / out
    dz: tuple = (0.0, 300.0)        # step-up onto an obstacle
    n: tuple = (9, 6, 7)

    def grid(self):
        for x in np.linspace(*self.dx, self.n[0]):
            for y in np.linspace(*self.dy, self.n[1]):
                for z in np.linspace(*self.dz, self.n[2]):
                    yield np.array([x, y, z])


def force_set(Fz: float, Fp: float, lateral_frac: float = 0.3):
    """Ground reaction on the foot: vertical, ± propulsion along x, ± lateral."""
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            out.append(np.array([sx * Fp, sy * lateral_frac * Fp, Fz]))
    return out


def evaluate(topo: Topology, Fz: float, Fp: float, ws: Workspace = Workspace()):
    """Per-joint torque maxima over the workspace (N·m), the neutral torques,
    and the fraction of the box the leg can reach."""
    forces = force_set(Fz, Fp)
    nf = topo.neutral_foot
    tmax = np.zeros(3)
    reached = total = 0
    q = np.zeros(3)
    for d in ws.grid():
        total += 1
        sol = topo.ik(nf + d, q0=q)
        if sol is None:
            sol = topo.ik(nf + d)
        if sol is None:
            continue
        reached += 1
        q = sol
        for F in forces:
            tmax = np.maximum(tmax, topo.torques(sol, F))
    neutral = np.max([topo.torques((0, 0, 0), F) for F in forces], axis=0)
    return {"max": tuple(float(v) for v in tmax), "neutral": tuple(float(v) for v in neutral),
            "reach_fraction": reached / total}


# Two working volumes: what the leg does routinely under walking load, and the
# extreme box it must reach (a 300 mm step-up, a foot reached well out).
ROUTINE = Workspace(dx=(-200.0, 200.0), dy=(-100.0, 100.0), dz=(0.0, 150.0), n=(9, 5, 4))
EXTREME = Workspace(dx=(-200.0, 200.0), dy=(-100.0, 150.0), dz=(0.0, 300.0), n=(9, 6, 7))
# A stumble happens while walking, with the feet near their nominal ring.
STUMBLE = Workspace(dx=(-200.0, 200.0), dy=(-50.0, 50.0), dz=(0.0, 50.0), n=(9, 3, 2))
