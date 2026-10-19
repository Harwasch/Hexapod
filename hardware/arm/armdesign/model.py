"""The arm and its host as maths: geometry, inverse kinematics, joint torques, stability.

Everything here is vectorised over candidates (an arm pose that might serve a task point),
because the optimizer asks the same questions of a few hundred thousand poses per design.

Frames. The ground frame has its origin on the ground under the hexapod's nominal body
centre, x forward, z up. The body frame is the hexapod body centre with the same axes when
the body is level. The arm's yaw axis is the body z axis through the mount point; the
shoulder axis is horizontal in the turret and meets the yaw axis at height `pedestal`
above the deck. Pitch angles are *absolute* in the turret frame (0 = horizontal, positive
lifts), which is what a parallelogram elbow actually controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
G = 9.81

# Joint columns used by every torque array.
YAW, SHOULDER, ELBOW, WRIST_PITCH, FOREARM_ROLL, TOOL_ROLL = range(6)
JOINTS = ("yaw", "shoulder", "elbow", "wrist_pitch", "forearm_roll", "tool_roll")

Array = np.ndarray


def load_yaml(name: str) -> Any:
    return yaml.safe_load((DATA / name).read_text())


def rot_z(a: float) -> Array:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> Array:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def unit(v: Array) -> Array:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def dot(a: Array, b: Array) -> Array:
    return np.einsum("...i,...i->...", a, b)


# --- the host ----------------------------------------------------------------------


@dataclass(frozen=True)
class BodyPose:
    """How the legs have moved the body from its nominal stance (feet stay planted)."""

    dz: float = 0.0
    pitch: float = 0.0  # radians, positive is nose down
    yaw: float = 0.0  # radians
    dx: float = 0.0  # forward shift of the body over planted feet


@dataclass(frozen=True)
class Hexapod:
    mass: float
    length: float
    width: float
    thickness: float
    center_height: float
    cg: Array
    hips: Array
    feet: Array
    coxa: float
    femur: float
    tibia: float
    leg_radius: float
    dz_range: tuple[float, float]
    pitch_range: tuple[float, float]
    yaw_range: tuple[float, float]
    lean: tuple[float, ...] = ()

    @classmethod
    def load(cls, mass: float | None = None) -> Hexapod:
        d = load_yaml("hexapod.yaml")
        b, leg, m = d["body"], d["leg"], d["body_motion"]
        return cls(
            mass=float(mass if mass is not None else d["mass_kg"]),
            length=b["length"],
            width=b["width"],
            thickness=b["thickness"],
            center_height=b["center_height"],
            cg=np.array(b["cg"], float),
            hips=np.array(d["hips"], float),
            feet=np.array(d["feet"], float),
            coxa=leg["coxa"],
            femur=leg["femur"],
            tibia=leg["tibia"],
            leg_radius=leg["radius"],
            dz_range=tuple(m["dz"]),
            pitch_range=tuple(math.radians(x) for x in m["pitch_deg"]),
            yaw_range=tuple(math.radians(x) for x in m["yaw_deg"]),
            lean=tuple(m.get("lean_dx", [])),
        )

    @property
    def deck_z(self) -> float:
        return self.thickness / 2

    def body_rt(self, pose: BodyPose) -> tuple[Array, Array]:
        """Rotation and translation taking body-frame points to the ground frame."""
        r = rot_z(pose.yaw) @ rot_y(pose.pitch)
        return r, np.array([pose.dx, 0.0, self.center_height + pose.dz])

    def legs(self, pose: BodyPose) -> Array:
        """(6, 4, 3): hip, coxa end, knee, foot of each leg in the ground frame."""
        r, t = self.body_rt(pose)
        out = np.zeros((6, 4, 3))
        for i in range(6):
            hip = r @ self.hips[i] + t
            foot = self.feet[i]
            flat = foot - hip
            flat[2] = 0.0
            coxa_end = hip + self.coxa * unit(flat)
            d_vec = foot - coxa_end
            d = float(np.linalg.norm(d_vec))
            u = d_vec / d
            w = unit(np.array([0.0, 0.0, 1.0]) - u[2] * u)
            d = min(d, self.femur + self.tibia - 1e-6)
            a = (self.femur**2 - self.tibia**2 + d**2) / (2 * d)
            h = math.sqrt(max(self.femur**2 - a**2, 0.0))
            knee = coxa_end + a * u + h * w
            out[i] = (hip, coxa_end, knee, foot)
        return out

    def pose_ok(self, pose: BodyPose, reach: float = 0.95) -> bool:
        """Whether every foot can stay planted with the body at this pose: each leg within
        its reach and its coxa within +-1 rad of its nominal heading."""
        r, t = self.body_rt(pose)
        for hip_b, foot in zip(self.hips, self.feet, strict=True):
            hip = r @ hip_b + t
            nominal = math.atan2(foot[1] - hip_b[1], foot[0] - hip_b[0])
            fb = r.T @ (foot - t) - hip_b
            heading = math.atan2(fb[1], fb[0])
            if abs((heading - nominal + math.pi) % (2 * math.pi) - math.pi) > 1.0:
                return False
            flat = foot - hip
            flat[2] = 0.0
            d = float(np.linalg.norm(foot - (hip + self.coxa * unit(flat))))
            if not (abs(self.femur - self.tibia) + 0.04 <= d <= reach * (self.femur + self.tibia)):
                return False
        return True

    def support_edges(self) -> tuple[Array, Array]:
        """Edge start points and unit directions of the support polygon (counter-clockwise)."""
        p = self.feet.copy()
        p[:, 2] = 0.0
        q = np.roll(p, -1, axis=0)
        return p, unit(q - p)


def body_pose_grid(hexapod: Hexapod, fine: bool = True, lean: bool = True) -> list[BodyPose]:
    """The body poses the planner may use, each one checked for the legs' reach. The
    nominal pose comes first. `lean` adds the body slid forward over planted feet, which
    the simulations showed shortens the lever for a heavy pull or push."""
    lo_z, hi_z = hexapod.dz_range
    lo_p, hi_p = hexapod.pitch_range
    lo_y, hi_y = hexapod.yaw_range
    if fine:
        dzs = [0.0, lo_z, hi_z / 2, hi_z]
        pitches = [0.0, lo_p, hi_p / 2, hi_p]
        yaws = [0.0, lo_y, hi_y]
    else:
        dzs = [0.0, lo_z, hi_z]
        pitches = [0.0, hi_p]
        yaws = [0.0, lo_y, hi_y]
    dxs = [0.0, *hexapod.lean] if lean else [0.0]
    poses = [BodyPose(z, p, y, x) for x in dxs for z in dzs for p in pitches for y in yaws]
    return [b for b in poses if hexapod.pose_ok(b)]


# --- the arm -----------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    shoulder: tuple[float, float] = (math.radians(-35), math.radians(150))
    # Fold angle: upper-arm angle minus forearm angle; 0 is a straight arm.
    fold_parallelogram: tuple[float, float] = (math.radians(18), math.radians(160))
    fold_direct: tuple[float, float] = (math.radians(5), math.radians(165))
    wrist_pitch: float = math.radians(125)  # |tool angle - forearm angle|, 5-DOF
    wrist_bend: float = math.radians(120)  # angle between forearm and tool axis, 6-DOF


@dataclass(frozen=True)
class MassModel:
    """Masses that are not actuators. Values are for the printed-housing + aluminium-tube
    construction the CAD builds; the CAD reports its own masses for comparison."""

    tube_kg_per_m: float = 0.36  # 25 x 1.5 mm 6061 tube, 0.30 kg/m, plus harness
    # Field kit (docs/research/rugged.md): aluminium heat plate, sun shield, drip skirt and
    # gaskets on the turret; sealed pods and a detent clutch at the wrist.
    turret_kg: float = 0.65
    elbow_housing_kg: float = 0.09
    rod_kg: float = 0.07  # parallelogram rod + crank
    wrist_housing_kg: float = 0.18
    flange_kg: float = 0.11  # tool changer, wrist camera and ToF sensor
    spring_kg: float = 0.08
    roll_offset: float = 0.045  # wrist pitch axis -> tool-roll actuator centre
    forearm_roll_offset: float = 0.07  # forearm-roll actuator centre behind the wrist


@dataclass(frozen=True)
class ArmDesign:
    arch: str  # "parallelogram" or "direct"
    dof: int  # 5 or 6
    l1: float  # shoulder axis -> elbow axis
    l2: float  # elbow axis -> wrist pitch axis
    mount_x: float  # yaw axis position along the body
    pedestal: float  # deck -> shoulder axis
    spring: bool = False  # zero-length gravity springs (shoulder, and elbow when parallelogram)
    wrist_len: float = 0.09  # wrist pitch axis -> tool flange

    @property
    def label(self) -> str:
        sp = "+spring" if self.spring else ""
        return (
            f"{self.dof}DOF {self.arch[:4]}{sp} L1={self.l1:.2f} L2={self.l2:.2f} "
            f"x={self.mount_x:.2f} ped={self.pedestal:.2f}"
        )

    def fold_limits(self, lim: Limits) -> tuple[float, float]:
        return lim.fold_parallelogram if self.arch == "parallelogram" else lim.fold_direct


@dataclass
class Poses:
    """A batch of candidate arm poses, all arrays with leading dimension N."""

    S: Array  # shoulder (ground frame)
    E: Array  # elbow
    W: Array  # wrist pitch centre
    T: Array  # tool centre point
    a: Array  # tool axis
    u2: Array  # upper-arm direction
    u3: Array  # forearm direction
    ehat: Array  # pitch "lift" axis in the ground frame
    zt: Array  # yaw axis in the ground frame
    th2: Array  # shoulder angle in the turret frame
    th3: Array  # forearm angle in the turret frame
    phi: Array  # tool angle in the turret frame (5-DOF meaning; 6-DOF: in-plane part)
    yaw: Array
    body_index: Array  # which body pose
    ok: Array  # kinematically valid (reach and joint limits)
    extra: dict[str, Array] = field(default_factory=dict)

    def take(self, idx: Array) -> Poses:
        kw = {k: getattr(self, k)[idx] for k in self.__dataclass_fields__ if k != "extra"}
        return Poses(**kw, extra={k: v[idx] for k, v in self.extra.items()})


def planar_ik(r: Array, z: Array, l1: float, l2: float) -> tuple[Array, Array, Array]:
    """Elbow-up two-link IK in the arm plane. Returns th2, th3 and a reachability mask."""
    d = np.hypot(r, z)
    ok = (d <= l1 + l2 - 1e-4) & (d >= abs(l1 - l2) + 1e-4)
    dc = np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    gamma = np.arccos(np.clip((l1**2 + dc**2 - l2**2) / (2 * l1 * dc), -1.0, 1.0))
    th2 = np.arctan2(z, r) + gamma
    ex, ez = l1 * np.cos(th2), l1 * np.sin(th2)
    th3 = np.arctan2(z - ez, r - ex)
    return th2, th3, ok


def wrap(a: Array) -> Array:
    return (a + np.pi) % (2 * np.pi) - np.pi


def solve_poses(
    design: ArmDesign,
    hexapod: Hexapod,
    body_poses: list[BodyPose],
    target: Array,
    approach: Array,
    tcp_len: Array,
    body_index: Array,
    lim: Limits | None = None,
) -> Poses:
    """IK for N candidates. target/approach in the ground frame (N, 3); tcp_len is the
    wrist pitch axis to TCP distance; body_index picks each candidate's body pose."""
    lim = lim or Limits()
    rs = np.stack([hexapod.body_rt(b)[0] for b in body_poses])[body_index]
    ts = np.stack([hexapod.body_rt(b)[1] for b in body_poses])[body_index]
    s_body = np.array([design.mount_x, 0.0, hexapod.deck_z + design.pedestal])
    # Into the body frame, relative to the shoulder.
    p_b = np.einsum("nji,nj->ni", rs, target - ts) - s_body
    a_b = np.einsum("nji,nj->ni", rs, approach)
    w_b = p_b - tcp_len[:, None] * a_b
    # A 5-DOF tool axis has to lie in the arm plane, which contains the yaw axis and the
    # target (the caller generated in-plane approach directions), so yaw follows the target;
    # a 6-DOF arm yaws to its wrist centre.
    yaw = np.arctan2(p_b[:, 1], p_b[:, 0]) if design.dof == 5 else np.arctan2(w_b[:, 1], w_b[:, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    radial = np.stack([c, s, np.zeros_like(c)], axis=1)
    r_w = dot(w_b, radial)
    z_w = w_b[:, 2]
    th2, th3, ok = planar_ik(r_w, z_w, design.l1, design.l2)
    fold = wrap(th2 - th3)
    flo, fhi = design.fold_limits(lim)
    ok &= (th2 >= lim.shoulder[0]) & (th2 <= lim.shoulder[1]) & (fold >= flo) & (fold <= fhi)
    zb = np.array([0.0, 0.0, 1.0])
    u2_b = np.cos(th2)[:, None] * radial + np.sin(th2)[:, None] * zb
    u3_b = np.cos(th3)[:, None] * radial + np.sin(th3)[:, None] * zb
    if design.dof == 5:
        off_plane = np.abs(dot(a_b, np.stack([-s, c, np.zeros_like(c)], axis=1)))
        ok &= off_plane < 0.02
        phi = np.arctan2(a_b[:, 2], dot(a_b, radial))
        ok &= np.abs(wrap(phi - th3)) <= lim.wrist_pitch
    else:
        phi = np.arctan2(a_b[:, 2], dot(a_b, radial))
        bend = np.arccos(np.clip(dot(u3_b, a_b), -1.0, 1.0))
        ok &= bend <= lim.wrist_bend
    # Back to the ground frame.
    S = ts + np.einsum("nij,j->ni", rs, s_body)
    u2 = np.einsum("nij,nj->ni", rs, u2_b)
    u3 = np.einsum("nij,nj->ni", rs, u3_b)
    E = S + design.l1 * u2
    W = E + design.l2 * u3
    a = approach
    T = W + tcp_len[:, None] * a
    # The lift axis: rotating u2 about it by a positive angle raises the arm.
    ehat = np.einsum("nij,nj->ni", rs, np.stack([s, -c, np.zeros_like(c)], axis=1))
    zt = rs[:, :, 2]
    return Poses(S, E, W, T, a, u2, u3, ehat, zt, th2, th3, phi, yaw, body_index, ok)


# --- loads and torques -------------------------------------------------------------


@dataclass(frozen=True)
class Load:
    """A point load on one segment. `s` locates it: along u2 from S ('upper'), along u3
    from E ('fore') or along the tool axis from W ('tool'). `force` overrides gravity."""

    segment: str
    s: float | Array
    mass: float | Array = 0.0
    force: Array | None = None  # (N, 3), applied by the environment


def load_point(p: Poses, load: Load) -> Array:
    s = np.asarray(load.s, float)
    s = s[:, None] if s.ndim else s
    if load.segment == "upper":
        return p.S + s * p.u2
    if load.segment == "fore":
        return p.E + s * p.u3
    return p.W + s * p.a


def wrist_split(p: Poses, m_w: Array, dof: int) -> Array:
    """Motor torques (wrist_pitch, forearm_roll, tool_roll) that hold a moment m_w about W."""
    out = np.zeros((len(m_w), 3))
    if dof == 5:
        out[:, 0] = -dot(p.ehat, m_w)
        out[:, 2] = -dot(p.a, m_w)
        return out
    pitch_axis = np.cross(p.u3, p.a)
    n = np.linalg.norm(pitch_axis, axis=1, keepdims=True)
    pitch_axis = np.where(n > 1e-3, pitch_axis / np.maximum(n, 1e-12), p.ehat)
    basis = np.stack([pitch_axis, p.u3, p.a], axis=2)  # columns
    # Near the wrist singularity the split blows up, which is the honest answer.
    det = np.abs(np.linalg.det(basis))
    basis = np.where(det[:, None, None] > 0.05, basis, np.eye(3)[None] * 0.05)
    return np.linalg.solve(basis, -m_w[:, :, None])[:, :, 0]


def load_torques(design: ArmDesign, p: Poses, load: Load, gravity_dir: Array | None = None) -> Array:
    """(N, 6) motor torques needed to hold one load. Linear in the load's mass/force."""
    n = len(p.S)
    pt = load_point(p, load)
    if load.force is not None:
        f = np.broadcast_to(load.force, (n, 3))
    else:
        g = np.array([0.0, 0.0, -G]) if gravity_dir is None else gravity_dir
        f = np.asarray(load.mass, float).reshape(-1, 1) * g
        f = np.broadcast_to(f, (n, 3))
    tau = np.zeros((n, 6))

    def holding(pivot: Array, axis: Array, at: Array) -> Array:
        return -dot(axis, np.cross(at - pivot, f))

    tau[:, YAW] = holding(p.S, p.zt, pt)
    seg = load.segment
    if design.arch == "parallelogram" and seg != "upper":
        # The forearm's own moment goes down the rod to the elbow motor on the turret; the
        # shoulder only carries the load as if it acted at the elbow pivot.
        tau[:, SHOULDER] = holding(p.S, p.ehat, p.E)
    else:
        tau[:, SHOULDER] = holding(p.S, p.ehat, pt)
    if seg != "upper":
        tau[:, ELBOW] = holding(p.E, p.ehat, pt)
    if seg == "tool":
        m_w = np.cross(pt - p.W, f)
        w = wrist_split(p, m_w, design.dof)
        tau[:, WRIST_PITCH] = w[:, 0]
        tau[:, FOREARM_ROLL] = w[:, 1]
        tau[:, TOOL_ROLL] = w[:, 2]
    return tau


def inertial_torques(design: ArmDesign, p: Poses, load: Load, accel: float) -> Array:
    """(N, 6) upper bound on the torque to accelerate a point mass at `accel` m/s^2 in the
    worst direction: m * a * (distance from each joint axis)."""
    pt = load_point(p, load)
    m = np.asarray(load.mass, float)

    def dist(pivot: Array, axis: Array, at: Array) -> Array:
        d = at - pivot
        return np.linalg.norm(d - dot(d, axis)[:, None] * axis, axis=1)

    tau = np.zeros((len(p.S), 6))
    tau[:, YAW] = dist(p.S, p.zt, pt)
    seg = load.segment
    if design.arch == "parallelogram" and seg != "upper":
        tau[:, SHOULDER] = dist(p.S, p.ehat, p.E)
    else:
        tau[:, SHOULDER] = dist(p.S, p.ehat, pt)
    if seg != "upper":
        tau[:, ELBOW] = dist(p.E, p.ehat, pt)
    if seg == "tool":
        r = np.linalg.norm(pt - p.W, axis=1)
        tau[:, WRIST_PITCH] = r
        if design.dof == 6:
            tau[:, FOREARM_ROLL] = r
    return tau * (m.reshape(-1, 1) * accel if m.ndim else m * accel)


def spring_torques(design: ArmDesign, p: Poses, load: Load) -> Array:
    """(N, 6) torque the zero-length gravity springs give back for a balanced load.

    A zero-length spring between the turret and a link gives exactly K cos(theta) about
    the pivot, theta being the link's angle in the turret frame. With a parallelogram the
    shoulder and the elbow motor each see one absolute angle, so both can be balanced
    exactly for the arm's own mass; with a direct elbow only the shoulder term is.
    """
    tau = np.zeros((len(p.S), 6))
    if not design.spring:
        return tau
    m = np.asarray(load.mass, float)
    s = np.asarray(load.s, float)
    if load.segment == "upper":
        k_sh, k_el = m * G * s, 0.0
    elif load.segment == "fore":
        k_sh, k_el = m * G * design.l1, m * G * s
    else:
        k_sh, k_el = m * G * design.l1, m * G * design.l2
    tau[:, SHOULDER] = k_sh * np.cos(p.th2)
    if design.arch == "parallelogram":
        tau[:, ELBOW] = k_el * np.cos(p.th3)
    return tau


# --- stability ---------------------------------------------------------------------


def stability_margin(
    hexapod: Hexapod,
    body_poses: list[BodyPose],
    body_index: Array,
    points: list[tuple[Array, Array]],
) -> tuple[Array, Array]:
    """Tip-over margin in metres and the fraction of weight left on the feet.

    points are (position (N, 3), force (N, 3)) pairs, gravity included. The hexapod's own
    weight is added at its centre of mass. The margin is the smallest restoring moment
    about any support-polygon edge divided by the total weight: how far the centre of
    pressure sits inside the polygon.
    """
    rs = np.stack([hexapod.body_rt(b)[0] for b in body_poses])[body_index]
    ts = np.stack([hexapod.body_rt(b)[1] for b in body_poses])[body_index]
    cg = ts + np.einsum("nij,j->ni", rs, hexapod.cg)
    allp = [(cg, np.broadcast_to([0.0, 0.0, -hexapod.mass * G], cg.shape))]
    allp += points
    starts, dirs = hexapod.support_edges()
    weight = hexapod.mass * G
    worst = np.full(len(cg), np.inf)
    fz = np.zeros(len(cg))
    for _, f in allp:
        fz += f[:, 2]
    for a, e in zip(starts, dirs, strict=True):
        m = np.zeros(len(cg))
        for pos, f in allp:
            m += -dot(e, np.cross(pos - a, f))
        worst = np.minimum(worst, m)
    total_down = -fz
    return worst / np.maximum(total_down, 1e-6), total_down / weight


# --- collisions --------------------------------------------------------------------


def _seg_point_dist(p: Array, a: Array, b: Array) -> Array:
    ab = b - a
    t = np.clip(dot(p - a, ab) / np.maximum(dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def collision_free(
    hexapod: Hexapod,
    body_poses: list[BodyPose],
    p: Poses,
    tcp_len: Array,
    ground_z: Array,
    bank_x: Array,
    arm_radius: float = 0.03,
) -> Array:
    """True where the arm clears the body, the legs and the ground."""
    samples = []
    for k in np.linspace(0.15, 1.0, 5):
        samples.append(p.S + k * (p.E - p.S))
        samples.append(p.E + k * (p.W - p.E))
    for k in (0.3, 0.6):
        samples.append(p.W + (k * tcp_len)[:, None] * p.a)
    pts = np.stack(samples, axis=1)  # (N, P, 3)
    ok = np.ones(len(p.S), bool)
    # Ground: the bank drops away beyond bank_x for tasks that need it.
    gz = np.where(pts[:, :, 0] > bank_x[:, None], ground_z[:, None], 0.0)
    ok &= np.all(pts[:, :, 2] > gz + arm_radius, axis=1)
    # Body box and legs, per body pose.
    half = np.array([hexapod.length / 2, hexapod.width / 2, hexapod.thickness / 2]) + arm_radius
    for bi, b in enumerate(body_poses):
        sel = p.body_index == bi
        if not np.any(sel):
            continue
        r, t = hexapod.body_rt(b)
        q = np.einsum("ji,npj->npi", r, pts[sel] - t)
        inside = np.all(np.abs(q) < half, axis=2)
        hit = np.any(inside, axis=1)
        legs = hexapod.legs(b)
        clear = hexapod.leg_radius + arm_radius
        ps = pts[sel]
        for leg in legs[[0, 1, 4, 5]]:
            lo = leg.min(axis=0) - clear
            hi = leg.max(axis=0) + clear
            near = np.any(np.all((ps > lo) & (ps < hi), axis=2), axis=1)
            if not np.any(near):
                continue
            idx = np.nonzero(near)[0]
            for j in range(3):
                d = _seg_point_dist(ps[idx], leg[j], leg[j + 1])
                hit[idx] |= np.any(d < clear, axis=1)
        ok[sel] &= ~hit
    return ok


@dataclass(frozen=True)
class Actuator:
    id: str
    name: str
    kind: str
    rated: float
    peak: float
    mass: float
    speed: float  # rad/s at the output, at the arm's bus voltage
    price: float
    dia: float
    length: float
    backdrivable: str
    interface: str
    backlash_deg: float = 0.5
    force_sense: str = "no"
    output_encoder: str = "yes"
    ip: str = "none"
    notes: str = ""
    ratio: float = 1.0  # total reduction, motor to output
    stage: str = ""  # added reduction stage, e.g. "capstan 4:1"; empty when none
    needs_driver: bool = False  # bare motor + gearbox, driven by the custom board
    source: str = "retail"  # where the price comes from
    retail: float = 0.0  # US retail price of the motor part, for comparison
    inertia: float = 0.0  # rotor inertia reflected to the output, kg m^2

    @cached_property
    def usd_per_nm(self) -> float:
        return self.price / max(self.rated, 1e-6)
