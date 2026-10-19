"""Task points -> candidate poses -> what each candidate asks of every joint.

`kinematic_table` answers "can this geometry reach the point, at an acceptable approach,
without hitting the robot or the ground and without tipping it over", for every task point
and every body pose the legs may use. `torque_table` then keeps the few least-effort
candidates per point and splits their joint torques into a part that does not depend on
which actuators are chosen and a part per unit of actuator mass, so the optimizer can try
thousands of actuator combinations without redoing any geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from armdesign.model import (
    ELBOW,
    FOREARM_ROLL,
    SHOULDER,
    TOOL_ROLL,
    WRIST_PITCH,
    YAW,
    ArmDesign,
    BodyPose,
    G,
    Hexapod,
    Load,
    MassModel,
    Poses,
    collision_free,
    dot,
    inertial_torques,
    load_point,
    load_torques,
    load_yaml,
    solve_poses,
    spring_torques,
    stability_margin,
    unit,
    wrist_split,
)

Array = np.ndarray

HOLD_MARGIN_M = 0.05  # tip-over margin while holding still
PEAK_MARGIN_M = 0.02  # during a short pull or push
MIN_WEIGHT_ON_FEET = 0.25  # a push may not take more than 75% of the weight off the feet
DESIGN_ACCEL = 4.0  # m/s^2 at every mass, worst direction, for the dynamic torque bound
RATED_DERATE = 0.9
PEAK_DERATE = 0.8


@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    mass: float
    length: float
    cg: float


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    category: str
    weight: float
    frame: str
    step: float
    region: dict[str, float | list[float]]
    approach: Array
    half_angle: float
    tool: str
    standoff: float
    payload: float
    force_world: Array | None
    force_axis: float | None
    duty: str
    tool_torque: float
    ground_z: float
    bank_x: float


def load_tools() -> dict[str, Tool]:
    return {k: Tool(k, v["name"], v["mass_kg"], v["length"], v["cg"]) for k, v in load_yaml("tools.yaml").items()}


def load_tasks() -> list[Task]:
    d = load_yaml("tasks.yaml")
    dflt = d["defaults"]
    out = []
    for t in d["tasks"]:
        get = lambda k, t=t: t.get(k, dflt.get(k))  # noqa: E731
        force = t.get("force", {})
        out.append(
            Task(
                id=t["id"],
                name=t["name"],
                category=t["category"],
                weight=float(get("weight")),
                frame=t.get("frame", "ground"),
                step=float(get("step")),
                region=t["region"],
                approach=unit(np.array(t["approach"]["dir"], float)),
                half_angle=math.radians(t["approach"]["half_angle_deg"]),
                tool=t["tool"],
                standoff=float(get("standoff")),
                payload=float(get("payload_kg")),
                force_world=np.array(force["world"], float) if "world" in force else None,
                force_axis=float(force["axis"]) if "axis" in force else None,
                duty=t["duty"],
                tool_torque=float(get("tool_torque_nm")),
                ground_z=float(get("ground_z")),
                bank_x=float(t.get("bank_x", 0.55 if get("ground_z") < 0 else 99.0)),
            )
        )
    return out


@dataclass
class Instances:
    task: Array  # (I,) index into tasks
    target: Array  # (I, 3) ground frame, or body frame for body-frame tasks


def task_points(task: Task, hexapod: Hexapod, coarse: bool = False) -> Array:
    step = task.step * (2.0 if coarse else 1.0)
    axes = []
    for k in "xyz":
        v = task.region[k]
        if isinstance(v, list):
            lo, hi = v
            n = max(round((hi - lo) / step) + 1, 1)
            axes.append(np.linspace(lo, hi, n))
        else:
            axes.append(np.array([float(v)]))
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if task.frame == "ground":
        # Nothing can be done exactly where a foot stands.
        d = np.linalg.norm(pts[:, None, :2] - hexapod.feet[None, :, :2], axis=2)
        pts = pts[np.all(d > 0.10, axis=1)]
    return pts


def make_instances(tasks: list[Task], hexapod: Hexapod, coarse: bool = False) -> Instances:
    idx, pts = [], []
    for i, t in enumerate(tasks):
        p = task_points(t, hexapod, coarse)
        idx.append(np.full(len(p), i))
        pts.append(p)
    return Instances(np.concatenate(idx), np.concatenate(pts))


def _cone_dirs(n: Array, half: float) -> Array:
    """Directions inside a cone: the axis and two rings."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(n, helper))
    e2 = np.cross(n, e1)
    dirs = [n]
    for frac, count in ((0.5, 6), (1.0, 8)):
        th = half * frac
        for k in range(count):
            az = 2 * math.pi * k / count
            dirs.append(math.cos(th) * n + math.sin(th) * (math.cos(az) * e1 + math.sin(az) * e2))
    return np.array(dirs)


@dataclass
class Candidates:
    instance: Array  # (N,) index into instances
    target: Array  # (N, 3) ground frame
    approach: Array  # (N, 3)
    tcp_len: Array  # (N,) wrist pitch axis -> TCP
    body_index: Array  # (N,)


def make_candidates(
    design: ArmDesign,
    tasks: list[Task],
    tools: dict[str, Tool],
    inst: Instances,
    hexapod: Hexapod,
    body_poses: list[BodyPose],
    max_dirs: int = 8,
) -> Candidates:
    """Every (body pose, approach direction) worth trying for every task point.

    A 5-DOF arm can only point its tool within the vertical plane through its yaw axis
    and the target, so its directions are sampled in that plane and kept if they fall in
    the task's cone; a 6-DOF arm samples the cone itself.
    """
    n_i = len(inst.task)
    task_ix = inst.task
    tcp = np.array([design.wrist_len + tools[t.tool].length + t.standoff for t in tasks])[task_ix]
    cos_half = np.cos(np.array([t.half_angle for t in tasks]))[task_ix]
    approach = np.stack([t.approach for t in tasks])[task_ix]
    in_body = np.array([t.frame == "body" for t in tasks])[task_ix]
    betas = np.radians(np.arange(-180, 180, 7.5))
    cone = [_cone_dirs(t.approach, t.half_angle) for t in tasks]
    rows_i, rows_t, rows_a, rows_b = [], [], [], []
    for bi, b in enumerate(body_poses):
        r, t = hexapod.body_rt(b)
        target = np.where(in_body[:, None], inst.target @ r.T + t, inst.target)
        n = np.where(in_body[:, None], approach @ r.T, approach)
        if design.dof == 5:
            pb = (target - t) @ r
            az = np.arctan2(pb[:, 1], pb[:, 0] - design.mount_x)
            radial = np.stack([np.cos(az), np.sin(az), np.zeros(n_i)], axis=1)
            a_b = np.cos(betas)[None, :, None] * radial[:, None, :]
            a_b[:, :, 2] += np.sin(betas)[None, :]
            dirs = a_b @ r.T  # (I, B, 3)
            mask = np.einsum("ibk,ik->ib", dirs, n) >= cos_half[:, None] - 1e-9
            # Thin to at most max_dirs evenly spread directions per point.
            rank = np.cumsum(mask, axis=1) - 1
            count = mask.sum(axis=1, keepdims=True)
            scale = (max_dirs - 1) / np.maximum(count - 1, 1)
            pick = np.round(np.round(rank * scale) / scale) == rank
            mask &= (count <= max_dirs) | pick
            ii, kk = np.nonzero(mask)
            rows_a.append(dirs[ii, kk])
        else:
            dirs = np.stack([cone[k] for k in task_ix])  # (I, 15, 3)
            if np.any(in_body):
                dirs = np.where(in_body[:, None, None], dirs @ r.T, dirs)
            ii = np.repeat(np.arange(n_i), dirs.shape[1])
            rows_a.append(dirs.reshape(-1, 3))
        rows_i.append(ii)
        rows_t.append(target[ii])
        rows_b.append(np.full(len(ii), bi))
    inst_ix = np.concatenate(rows_i)
    return Candidates(
        inst_ix,
        np.concatenate(rows_t),
        np.concatenate(rows_a),
        tcp[inst_ix],
        np.concatenate(rows_b),
    )


# --- loads --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotRef:
    """Actuator masses used when picking poses before the actuators are known."""

    elbow: float = 0.38
    wrist_pitch: float = 0.30
    forearm_roll: float = 0.12
    tool_roll: float = 0.10
    yaw: float = 0.38
    shoulder: float = 0.38


def fixed_loads(design: ArmDesign, mm: MassModel, include_rod: bool = True) -> list[Load]:
    """The arm's own non-actuator masses."""
    loads = [
        Load("upper", design.l1 / 2, mm.tube_kg_per_m * design.l1),
        Load(
            "fore",
            0.0,
            mm.elbow_housing_kg + (mm.rod_kg / 2 if design.arch == "parallelogram" and include_rod else 0.0),
        ),
        Load("fore", design.l2 / 2, mm.tube_kg_per_m * design.l2),
        Load("fore", design.l2, mm.wrist_housing_kg),
        Load("tool", design.wrist_len - 0.02, mm.flange_kg),
    ]
    return loads


def slot_loads(design: ArmDesign, mm: MassModel) -> dict[str, Load]:
    """Where each moving actuator sits, at unit mass."""
    slots = {
        "wrist_pitch": Load("fore", design.l2, 1.0),
        "tool_roll": Load("tool", mm.roll_offset, 1.0),
    }
    if design.arch == "direct":
        slots["elbow"] = Load("fore", 0.0, 1.0)
    if design.dof == 6:
        slots["forearm_roll"] = Load("fore", design.l2 - mm.forearm_roll_offset, 1.0)
    return slots


def static_mass_points(
    design: ArmDesign, mm: MassModel, p: Poses, slot_mass: dict[str, float], hexapod: Hexapod
) -> list[tuple[Array, Array]]:
    """(position, gravity force) of every arm mass, for the stability check."""
    out = []
    for ld in fixed_loads(design, mm):
        out.append((load_point(p, ld), ld.mass))
    for name, ld in slot_loads(design, mm).items():
        out.append((load_point(p, ld), slot_mass[name]))
    # Turret, yaw and shoulder (and parallelogram elbow) actuators sit on the mount.
    mount = p.S - design.pedestal * p.zt
    turret = mm.turret_kg + slot_mass["yaw"] + slot_mass["shoulder"] + (mm.spring_kg * 2 if design.spring else 0.0)
    if design.arch == "parallelogram":
        turret += slot_mass["elbow"] + mm.rod_kg / 2
    out.append((0.5 * (mount + p.S), turret))
    n = len(p.S)
    return [(pos, np.broadcast_to(np.array([0.0, 0.0, -G]) * m, (n, 3)) if np.isscalar(m) else m) for pos, m in out]


# --- the kinematic pass --------------------------------------------------------------


@dataclass
class KinResult:
    cand: Candidates
    poses: Poses  # only candidates that survived
    margin: Array
    feasible: Array  # (I,) any surviving candidate
    surv: Array  # indices into cand of the survivors


def task_force(task: Task, a: Array) -> Array:
    if task.force_world is not None:
        return np.broadcast_to(task.force_world, a.shape).copy()
    if task.force_axis is not None:
        return task.force_axis * a
    return np.zeros_like(a)


def kinematic_table(
    design: ArmDesign,
    tasks: list[Task],
    tools: dict[str, Tool],
    inst: Instances,
    hexapod: Hexapod,
    body_poses: list[BodyPose],
    mm: MassModel | None = None,
    slot_ref: SlotRef | None = None,
    max_dirs: int = 8,
) -> KinResult:
    mm = mm or MassModel()
    ref = slot_ref or SlotRef()
    cand = make_candidates(design, tasks, tools, inst, hexapod, body_poses, max_dirs)
    poses = solve_poses(design, hexapod, body_poses, cand.target, cand.approach, cand.tcp_len, cand.body_index)
    keep = np.nonzero(poses.ok)[0]
    poses = poses.take(keep)
    task_of = inst.task[cand.instance[keep]]
    ground_z = np.array([t.ground_z for t in tasks])[task_of]
    bank_x = np.array([t.bank_x for t in tasks])[task_of]
    free = collision_free(hexapod, body_poses, poses, cand.tcp_len[keep], ground_z, bank_x)
    keep = keep[free]
    poses = poses.take(np.nonzero(free)[0])
    task_of = task_of[free]
    # Stability with reference actuator masses, the task's tool, payload and force.
    slot_mass = {k: getattr(ref, k) for k in ("elbow", "wrist_pitch", "forearm_roll", "tool_roll", "yaw", "shoulder")}
    pts = static_mass_points(design, mm, poses, slot_mass, hexapod)
    tool_mass = np.array([tools[t.tool].mass for t in tasks])[task_of]
    tool_cg = np.array([tools[t.tool].cg for t in tasks])[task_of] + design.wrist_len
    payload = np.array([t.payload for t in tasks])[task_of]
    grav = np.array([0.0, 0.0, -G])
    pts.append((poses.W + tool_cg[:, None] * poses.a, tool_mass[:, None] * grav))
    pts.append((poses.T, payload[:, None] * grav))
    force = np.zeros_like(poses.T)
    for ti, t in enumerate(tasks):
        sel = task_of == ti
        if np.any(sel):
            force[sel] = task_force(t, poses.a[sel])
    pts.append((poses.T, force))
    margin, on_feet = stability_margin(hexapod, body_poses, poses.body_index, pts)
    need = np.where(np.array([t.duty for t in tasks])[task_of] == "hold", HOLD_MARGIN_M, PEAK_MARGIN_M)
    stable = (margin >= need) & (on_feet >= MIN_WEIGHT_ON_FEET)
    keep = keep[stable]
    poses = poses.take(np.nonzero(stable)[0])
    feasible = np.zeros(len(inst.task), bool)
    feasible[cand.instance[keep]] = True
    return KinResult(cand, poses, margin[stable], feasible, keep)


# --- the torque pass -----------------------------------------------------------------

SLOTS = ("elbow", "wrist_pitch", "forearm_roll", "tool_roll")


@dataclass
class TorqueTable:
    """Per instance, the K least-effort candidates (padded; `valid` marks real ones).

    For actuator masses m (by slot) the required torques are
        static = base_static + sum_k m_k * slot_static[k]
        peak   = |static + ext| + ext_abs + base_inert + sum_k m_k * slot_inert[k]
    and a candidate works when |static (+ ext for hold tasks)| <= rated and peak <= peak.
    """

    base_static: Array  # (I, K, 6) gravity of fixed masses, tool, payload, minus springs
    ext: Array  # (I, K, 6) task force, signed
    ext_abs: Array  # (I, K, 6) tool torque, either direction
    base_inert: Array  # (I, K, 6)
    slot_static: Array  # (S, I, K, 6)
    slot_inert: Array  # (S, I, K, 6)
    valid: Array  # (I, K)
    hold: Array  # (I,) True for sustained tasks
    margin: Array  # (I, K)
    cand_index: Array  # (I, K) index into KinResult.surv (-1 = none)


def _moment_torques(design: ArmDesign, p: Poses, moment: Array) -> Array:
    """(N, 6) holding torques for a pure moment applied to the tool."""
    tau = np.zeros((len(p.S), 6))
    tau[:, YAW] = -dot(p.zt, moment)
    tau[:, SHOULDER] = 0.0 if design.arch == "parallelogram" else -dot(p.ehat, moment)
    tau[:, ELBOW] = -dot(p.ehat, moment)
    w = wrist_split(p, moment, design.dof)
    tau[:, WRIST_PITCH] = w[:, 0]
    tau[:, FOREARM_ROLL] = w[:, 1]
    tau[:, TOOL_ROLL] = w[:, 2]
    return tau


def candidate_torques(
    design: ArmDesign,
    tasks: list[Task],
    tools: dict[str, Tool],
    task_of: Array,
    poses: Poses,
    mm: MassModel,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """base_static, ext, ext_abs, base_inert (N, 6) and slot_static, slot_inert (S, N, 6)."""
    n = len(poses.S)
    base = np.zeros((n, 6))
    inert = np.zeros((n, 6))
    gripper = tools["gripper"]
    for ld in fixed_loads(design, mm):
        base += load_torques(design, poses, ld) - spring_torques(design, poses, ld)
        inert += inertial_torques(design, poses, ld, DESIGN_ACCEL)
    # The springs are tuned for the gripper; other tools and payloads are carried.
    base -= spring_torques(design, poses, Load("tool", design.wrist_len + gripper.cg, gripper.mass))
    tool_mass = np.array([tools[t.tool].mass for t in tasks])[task_of]
    tool_s = np.array([tools[t.tool].cg for t in tasks])[task_of] + design.wrist_len
    payload = np.array([t.payload for t in tasks])[task_of]
    standoff = np.array([t.standoff for t in tasks])[task_of]
    tcp_s = np.linalg.norm(poses.T - poses.W, axis=1)
    for ld in (Load("tool", tool_s, tool_mass), Load("tool", tcp_s - standoff, payload)):
        base += load_torques(design, poses, ld)
        inert += inertial_torques(design, poses, ld, DESIGN_ACCEL)
    force = np.zeros((n, 3))
    torque = np.zeros(n)
    for ti, t in enumerate(tasks):
        sel = task_of == ti
        if np.any(sel):
            force[sel] = task_force(t, poses.a[sel])
            torque[sel] = t.tool_torque
    ext = load_torques(design, poses, Load("tool", tcp_s, 0.0, force))
    ext_abs = np.abs(_moment_torques(design, poses, torque[:, None] * poses.a))
    slots = slot_loads(design, mm)
    s_static = np.zeros((len(SLOTS), n, 6))
    s_inert = np.zeros((len(SLOTS), n, 6))
    for k, name in enumerate(SLOTS):
        if name in slots:
            ld = slots[name]
            s_static[k] = load_torques(design, poses, ld) - spring_torques(design, poses, ld)
            s_inert[k] = inertial_torques(design, poses, ld, DESIGN_ACCEL)
    return base, ext, ext_abs, inert, s_static, s_inert


REF_RATED = np.array([6.0, 9.0, 9.0, 3.0, 1.5, 1.5])


def torque_table(
    design: ArmDesign,
    tasks: list[Task],
    tools: dict[str, Tool],
    inst: Instances,
    kin: KinResult,
    mm: MassModel | None = None,
    k_best: int = 6,
) -> TorqueTable:
    mm = mm or MassModel()
    ref = SlotRef()
    inst_of = kin.cand.instance[kin.surv]
    task_of = inst.task[inst_of]
    base, ext, ext_abs, inert, s_static, s_inert = candidate_torques(design, tasks, tools, task_of, kin.poses, mm)
    ref_m = np.array([getattr(ref, s) for s in SLOTS])
    static = base + np.einsum("s,snj->nj", ref_m, s_static)
    peak = np.abs(static + ext) + ext_abs + inert + np.einsum("s,snj->nj", ref_m, s_inert)
    effort = np.max(np.maximum(np.abs(static), peak / 2.5) / REF_RATED, axis=1)
    # Rank candidates within each instance by effort, prefer the larger stability margin.
    order = np.lexsort((-kin.margin, effort, inst_of))
    inst_sorted = inst_of[order]
    first = np.searchsorted(inst_sorted, np.arange(len(inst.task)))
    rank = np.arange(len(order)) - first[inst_sorted]
    take = rank < k_best
    sel = order[take]
    ii, kk = inst_sorted[take], rank[take]
    n_i = len(inst.task)
    shape = (n_i, k_best, 6)
    tt = TorqueTable(
        base_static=np.zeros(shape),
        ext=np.zeros(shape),
        ext_abs=np.zeros(shape),
        base_inert=np.zeros(shape),
        slot_static=np.zeros((len(SLOTS), *shape)),
        slot_inert=np.zeros((len(SLOTS), *shape)),
        valid=np.zeros((n_i, k_best), bool),
        hold=np.array([tasks[t].duty == "hold" for t in inst.task]),
        margin=np.zeros((n_i, k_best)),
        cand_index=np.full((n_i, k_best), -1),
    )
    tt.base_static[ii, kk] = base[sel]
    tt.ext[ii, kk] = ext[sel]
    tt.ext_abs[ii, kk] = ext_abs[sel]
    tt.base_inert[ii, kk] = inert[sel]
    tt.slot_static[:, ii, kk] = s_static[:, sel]
    tt.slot_inert[:, ii, kk] = s_inert[:, sel]
    tt.valid[ii, kk] = True
    tt.margin[ii, kk] = kin.margin[sel]
    tt.cand_index[ii, kk] = sel
    return tt


def feasible_with(tt: TorqueTable, slot_mass: Array, rated: Array, peak: Array) -> Array:
    """(C, I) which instances each of C actuator sets can serve.

    slot_mass (C, S) by SLOTS; rated/peak (C, 6) by joint (0 where a joint is absent, which
    the caller avoids by giving absent joints infinite capacity).
    """
    static = tt.base_static[None] + np.einsum("cs,sikj->cikj", slot_mass, tt.slot_static)
    inert = tt.base_inert[None] + np.einsum("cs,sikj->cikj", slot_mass, tt.slot_inert)
    with_ext = static + tt.ext[None]
    hold = tt.hold[None, :, None, None]
    sustained = np.where(hold, np.abs(with_ext), np.abs(static))
    burst = np.where(hold, np.abs(static), np.abs(with_ext)) + tt.ext_abs[None] + inert
    ok = np.all(sustained <= RATED_DERATE * rated[:, None, None, :], axis=3)
    ok &= np.all(burst <= PEAK_DERATE * peak[:, None, None, :], axis=3)
    ok &= tt.valid[None]
    return np.any(ok, axis=2)
