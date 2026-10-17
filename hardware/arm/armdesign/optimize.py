"""Search the design space and write the cost-versus-capability study.

    uv run python -m armdesign.optimize            # full study (about 10 minutes, 4 cores)
    uv run python -m armdesign.optimize --quick    # a smaller grid for a smoke run

Stage A sweeps geometry (DOF, elbow drive, link lengths, mount position, pedestal) for
kinematic coverage: reach, approach, collisions and tip-over with reference masses.
Stage B takes the geometries on the coverage/size frontier, adds the gravity-spring
option, and enumerates every actuator assignment from data/actuators.yaml against the
joint torques each task point needs. Output: out/study.json (everything the viewer and
the docs read) and out/selected.yaml (the recommended design).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from armdesign.evaluate import (
    PEAK_DERATE,
    RATED_DERATE,
    SLOTS,
    TorqueTable,
    kinematic_table,
    load_tasks,
    load_tools,
    make_instances,
    torque_table,
)
from armdesign.model import (
    ELBOW,
    FOREARM_ROLL,
    JOINTS,
    ROOT,
    SHOULDER,
    TOOL_ROLL,
    WRIST_PITCH,
    YAW,
    Actuator,
    ArmDesign,
    Hexapod,
    MassModel,
    body_pose_grid,
    load_yaml,
)
from armdesign.parts import LEVERS, joint_pool, load_catalog, peak_available

OUT = ROOT / "out"
Array = np.ndarray


# --- catalog and costs -----------------------------------------------------------------


def load_costs() -> dict[str, float]:
    return {k: float(v) for k, v in load_yaml("costs.yaml")["usd"].items()}


def fixed_cost(design: ArmDesign, costs: dict[str, float], driver_board: bool = False) -> dict[str, float]:
    """Everything but the actuators, itemised. driver_board: some joint runs on the fleet's
    own driver board, whose shared section also hosts the servo rail."""
    c = {
        "aluminium tube": costs["tube_per_m"] * (design.l1 + design.l2 + 0.08),
        "printed parts": costs["printed_parts"],
        "bearings": costs["bearings"],
        "fasteners": costs["fasteners"],
        "tool changer": costs["tool_changer"],
        "gripper (servo + fin-ray fingers)": costs["gripper"],
        "wrist camera + ToF": costs["wrist_sensing"],
        "harness + connectors": costs["harness"],
        "field kit (seals, vents, servo pods, heat plate)": costs["seals"],
    }
    if design.arch == "parallelogram":
        c["parallelogram rod + rod ends"] = costs["parallelogram_rod"]
    if design.spring:
        c["gravity springs"] = costs["spring_each"] * (2 if design.arch == "parallelogram" else 1)
    # The gripper is a bus servo, so the bus is always there.
    if driver_board:
        c["custom driver board (shared parts)"] = costs["custom_driver_board_fixed"]
        c["servo rail on the driver board"] = costs["custom_servo_rail"]
    else:
        c["servo bus adapter"] = costs["servo_bus_adapter"]
    return c


def fixed_mass(design: ArmDesign, mm: MassModel) -> float:
    m = mm.tube_kg_per_m * (design.l1 + design.l2) + mm.turret_kg + mm.elbow_housing_kg
    m += mm.wrist_housing_kg + mm.flange_kg
    if design.arch == "parallelogram":
        m += mm.rod_kg
    if design.spring:
        m += mm.spring_kg * (2 if design.arch == "parallelogram" else 1)
    return m


# The twist the tool roll must be able to give: the valve task's 3 Nm (data/tasks.yaml).
# The coverage score weighs it like any task, so without this floor the cheapest designs
# trade valves and knobs away entirely for a lighter wrist.
MIN_TOOL_TWIST_NM = 3.0

# What a joint must be in each tier. "field" is the outdoor-robot bar: the shoulder and
# elbow can feel contact through motor current and survive being back-driven by a fall or
# a branch, the yaw is backdrivable, nothing at the base is a hobby part, the base joints
# know their output angle at power-up (no homing sweep from the stowed pose), and the
# tool roll can turn a valve.
TIERS = {
    "any": {},
    "field": {
        "yaw": lambda a: a.kind == "qdd" and a.backdrivable == "yes" and a.output_encoder == "yes",
        "shoulder": lambda a: a.kind == "qdd" and a.force_sense == "yes" and a.output_encoder == "yes",
        "elbow": lambda a: a.kind == "qdd" and a.force_sense == "yes" and a.output_encoder == "yes",
        "tool_roll": lambda a: PEAK_DERATE * a.peak >= MIN_TOOL_TWIST_NM,
    },
}
# Root-sum-square tip error from gear backlash, open loop. Micro-dosing and stamping want
# +-2-5 mm at the tool; the wrist camera closes the last few millimetres.
FIELD_MAX_BACKLASH_MM = 6.0
# The arm's own mass budget on a medium (8-20 kg) hexapod, tools excluded.
MAX_ARM_KG = 2.5
# Scores are float32 sums over ~10^3 points; a point weighs at least ~1e-5.
SCORE_EPS = 1e-6


def joint_options(
    catalog: list[Actuator],
    joint: str,
    tier: str = "any",
    lever: str = "retail",
    arch: str = "parallelogram",
    pedestal: float = math.inf,
) -> list[Actuator]:
    """Candidates for one joint, without the ones another candidate beats on every count."""
    pool = joint_pool(catalog, joint, lever, arch, pedestal)
    rule = TIERS[tier].get(joint)
    if rule is not None:
        pool = [a for a in pool if rule(a)]
    if joint in ("wrist_pitch",):
        pool = [a for a in pool if a.mass <= 0.45]
    elif joint in ("forearm_roll", "tool_roll"):
        pool = [a for a in pool if a.mass <= 0.35]

    def beats(x: Actuator, y: Actuator) -> bool:
        # A part that needs our driver board never beats one that doesn't: the board's
        # shared cost is paid once per arm, outside any one part's price.
        ge = x.needs_driver <= y.needs_driver
        ge &= x.price <= y.price and x.rated >= y.rated and x.peak >= y.peak and x.mass <= y.mass
        ge &= x.speed >= y.speed and x.backlash_deg <= y.backlash_deg and x.inertia <= y.inertia
        ge &= _rank(x.force_sense) >= _rank(y.force_sense)
        gt = x.price < y.price or x.rated > y.rated or x.peak > y.peak or x.mass < y.mass
        gt = gt or x.backlash_deg < y.backlash_deg
        return ge and gt

    return [y for y in pool if not any(beats(x, y) for x in pool if x is not y)]


def _rank(force_sense: str) -> int:
    return {"yes": 2, "partly": 1}.get(force_sense, 0)


# --- stage A ----------------------------------------------------------------------------

_CTX: dict[str, object] = {}


def _init(coarse: bool) -> None:
    hx = Hexapod.load()
    tasks = load_tasks()
    _CTX.update(
        hx=hx,
        tasks=tasks,
        tools=load_tools(),
        inst=make_instances(tasks, hx, coarse=True),
        bp=body_pose_grid(hx, fine=not coarse),
    )


def _coverage(feasible: Array) -> tuple[float, dict[str, float]]:
    tasks = _CTX["tasks"]
    inst = _CTX["inst"]
    per = {}
    for i, t in enumerate(tasks):
        sel = inst.task == i
        per[t.id] = float(feasible[sel].mean()) if np.any(sel) else 0.0
    w = np.array([t.weight for t in tasks])
    score = float(sum(per[t.id] * t.weight for t in tasks) / w.sum())
    return score, per


def _kin_job(design: ArmDesign) -> dict:
    kin = kinematic_table(design, _CTX["tasks"], _CTX["tools"], _CTX["inst"], _CTX["hx"], _CTX["bp"])
    score, per = _coverage(kin.feasible)
    return {"design": asdict(design), "kin_score": score, "kin_tasks": per}


def stage_a(quick: bool, workers: int) -> list[dict]:
    lengths = np.round(np.arange(0.22, 0.41, 0.06 if quick else 0.03), 3)
    mounts = (0.12, 0.24) if quick else (0.10, 0.18, 0.26)
    peds = (0.06, 0.14) if quick else (0.05, 0.10, 0.16)
    designs = [
        ArmDesign(arch, dof, float(l1), float(l2), m, p)
        for dof in (5, 6)
        for arch in ("parallelogram", "direct")
        for l1 in lengths
        for l2 in lengths
        for m in mounts
        for p in peds
    ]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(True,)) as ex:
        return list(ex.map(_kin_job, designs, chunksize=4))


def frontier(rows: list[dict], keys: list[tuple[str, int]]) -> list[dict]:
    """Rows not dominated on keys; sign +1 maximises, -1 minimises."""
    vals = np.array([[s * r[k] for k, s in keys] for r in rows])
    keep = []
    for i, v in enumerate(vals):
        dom = np.all(vals >= v, axis=1) & np.any(vals > v, axis=1)
        if not np.any(dom):
            keep.append(rows[i])
    return keep


# --- stage B ----------------------------------------------------------------------------


@dataclass
class Assignment:
    yaw: Actuator
    shoulder: Actuator
    elbow: Actuator
    wrist_pitch: Actuator
    forearm_roll: Actuator | None
    tool_roll: Actuator

    def by_joint(self) -> dict[str, Actuator | None]:
        return {j: getattr(self, j) for j in JOINTS}


def _joint_ok(tt: TorqueTable, slot_mass: Array, j: int, rated: Array, peak_avail: Array) -> Array:
    """(O, I, K) for O actuator options on joint j, with the given moving masses.
    peak_avail is each option's derated peak less what its own rotor needs."""
    static = tt.base_static[..., j] + np.einsum("s,sik->ik", slot_mass, tt.slot_static[..., j])
    inert = tt.base_inert[..., j] + np.einsum("s,sik->ik", slot_mass, tt.slot_inert[..., j])
    with_ext = static + tt.ext[..., j]
    hold = tt.hold[:, None]
    sustained = np.where(hold, np.abs(with_ext), np.abs(static))
    burst = np.where(hold, np.abs(static), np.abs(with_ext)) + tt.ext_abs[..., j] + inert
    ok = (sustained[None] <= RATED_DERATE * rated[:, None, None]) & (burst[None] <= peak_avail[:, None, None])
    return ok & tt.valid[None]


def _prune(opts: list[Actuator], ok: Array) -> list[int]:
    """Indices of options not beaten by another that serves a superset of the points for
    no more money, mass, backlash or rotor inertia, and without needing the driver board
    when the beaten one doesn't. Exact: swapping a beaten option for its better never
    loses capability or breaks a budget.

    "No worse on every count" is transitive, so an option is beaten exactly when some
    option ranked before it (lexicographically on the same counts) is no worse."""
    n = len(opts)
    if n <= 1:
        return list(range(n))
    flat = np.packbits(ok.reshape(n, -1), axis=1)
    ids: dict[bytes, int] = {}
    inverse = np.array([ids.setdefault(r.tobytes(), len(ids)) for r in flat])
    first = np.unique(inverse, return_index=True)[1]
    rows = flat[first]
    # covers_u[a, b]: unique coverage row b contains row a.
    covers_u = ~np.any(rows[:, None, :] & ~rows[None, :, :], axis=2)
    covers = covers_u[inverse][:, inverse]
    price = np.array([a.price for a in opts])
    mass = np.array([a.mass for a in opts])
    back = np.array([a.backlash_deg for a in opts])
    inert = np.array([a.inertia for a in opts])
    sense = np.array([_rank(a.force_sense) for a in opts])
    drv = np.array([a.needs_driver for a in opts], dtype=int)
    count = ok.reshape(n, -1).sum(axis=1)
    no_worse = (
        (drv[None, :] <= drv[:, None])
        & (price[None, :] <= price[:, None])
        & (mass[None, :] <= mass[:, None])
        & (back[None, :] <= back[:, None])
        & (inert[None, :] <= inert[:, None])
        & (sense[None, :] >= sense[:, None])
        & covers
    )
    order = np.lexsort((np.arange(n), -count, -sense, inert, back, mass, drv, price))
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    beaten = np.any(no_worse & (rank[None, :] < rank[:, None]), axis=1)
    return [int(i) for i in order if not beaten[i]]


def backlash_levers(design: ArmDesign) -> dict[str, float]:
    """Lever arms (m) from each joint to the gripper tip at 85% of full stretch."""
    tool = 0.11
    reach = 0.85 * (design.l1 + design.l2) + design.wrist_len + tool
    return {
        "yaw": reach,
        "shoulder": reach,
        "elbow": design.l2 + design.wrist_len + tool,
        "wrist_pitch": design.wrist_len + tool,
        "forearm_roll": 0.03,
        "tool_roll": 0.03,
    }


def _frontier_idx(cost: Array, score: Array, feasible: Array) -> Array:
    """Flat indices of the cost/score frontier among feasible entries."""
    idx = np.nonzero(feasible.ravel())[0]
    if not len(idx):
        return idx
    c, sc = cost.ravel()[idx], score.ravel()[idx]
    order = np.lexsort((-sc, c))
    best = np.maximum.accumulate(sc[order])
    prev = np.concatenate([[-1.0], best[:-1]])
    return idx[order[sc[order] > prev + SCORE_EPS]]


def _merge_points(e: Array, s: Array, y: Array, w: Array) -> tuple[Array, Array, Array, Array, float]:
    """Shrink the point axis of three joints' packed pose bits without changing any
    combination's score. A point some joint can never serve drops out; a point one pose
    serves for every option of every joint is a constant; points whose bit columns match
    merge into one with their weights summed."""
    always_bits = np.bitwise_and.reduce(e, 0) & np.bitwise_and.reduce(s, 0) & np.bitwise_and.reduce(y, 0)
    never = ~e.any(0) | ~s.any(0) | ~y.any(0)
    always = always_bits != 0
    keep = np.nonzero(~never & ~always)[0]
    stack = np.concatenate([e, s, y])[:, keep]
    cols = np.ascontiguousarray(stack.T).view(np.dtype((np.void, stack.shape[0]))).ravel()
    _, first, inv = np.unique(cols, return_index=True, return_inverse=True)
    w_m = np.bincount(inv.ravel(), weights=w[keep], minlength=len(first)).astype(np.float32)
    stack = stack[:, first]
    ne, ns = len(e), len(s)
    return stack[:ne], stack[ne : ne + ns], stack[ne + ns :], w_m, float(w[always].sum())


def enumerate_actuators(
    design: ArmDesign,
    tt: TorqueTable,
    weights: Array,
    catalog: list[Actuator],
    tier: str = "any",
    lever: str = "retail",
    mass_fixed: float = 0.0,
    max_mass: float = math.inf,
    max_backlash_mm: float = math.inf,
    board_fixed: float = 0.0,
) -> list[dict]:
    """The design's cost/score frontier over every actuator assignment that keeps within
    the mass and backlash budgets. Cost counts the parts and, when any part needs it, the
    custom driver board's shared section."""
    opts = {j: joint_options(catalog, j, tier, lever, design.arch, design.pedestal) for j in JOINTS}
    lev = backlash_levers(design)
    w32 = weights.astype(np.float32)

    def cap(xs: list[Actuator], joint: str) -> tuple[Array, Array]:
        return np.array([a.rated for a in xs]), np.array([peak_available(a, joint, PEAK_DERATE) for a in xs])

    def attr(xs: list[Actuator], name: str) -> Array:
        return np.array([getattr(a, name) for a in xs], dtype=float)

    def blash(xs: list[Actuator], joint: str) -> Array:
        return (np.radians(attr(xs, "backlash_deg")) * lev[joint] * 1000) ** 2

    def board(xs: list[Actuator]) -> Array:
        return np.array([a.needs_driver for a in xs])

    # Wrist parts: prune each wrist-pitch list per tool roll (the roll's mass loads the pitch).
    pairs = []
    no_mass = np.zeros(len(SLOTS))
    for tr in opts["tool_roll"]:
        m = no_mass.copy()
        m[SLOTS.index("tool_roll")] = tr.mass
        if design.dof == 6:
            m[SLOTS.index("forearm_roll")] = tr.mass
        wps = opts["wrist_pitch"]
        ok_wp = _joint_ok(tt, m, WRIST_PITCH, *cap(wps, "wrist_pitch"))
        pairs += [(wps[i], tr) for i in _prune(wps, ok_wp)]

    candidates = []
    for wp, tr in pairs:
        # A 6-DOF wrist uses the same part for both rolls.
        fr = tr if design.dof == 6 else None
        base_m = {"elbow": 0.0, "wrist_pitch": wp.mass, "forearm_roll": fr.mass if fr else 0.0, "tool_roll": tr.mass}
        m0 = np.array([base_m[s] for s in SLOTS])
        ok = tt.valid.copy()
        for j, name, a in ((WRIST_PITCH, "wrist_pitch", wp), (TOOL_ROLL, "tool_roll", tr)) + (
            ((FOREARM_ROLL, "forearm_roll", fr),) if fr else ()
        ):
            r, p = cap([a], name)
            ok &= _joint_ok(tt, m0, j, r, p)[0]
        if not np.any(ok):
            continue
        distal = [wp, tr] + ([fr] if fr else [])
        d_price = sum(a.price for a in distal)
        d_mass = sum(a.mass for a in distal)
        d_back = (
            blash([wp], "wrist_pitch")[0] + blash([tr], "tool_roll")[0] + (blash([fr], "forearm_roll")[0] if fr else 0)
        )
        d_board = any(a.needs_driver for a in distal)
        el = opts["elbow"]
        ok_el = _joint_ok(tt, m0, ELBOW, *cap(el, "elbow")) & ok[None]  # the elbow motor never loads itself
        ei = _prune(el, ok_el)
        el, ok_el = [el[i] for i in ei], ok_el[ei]
        sh_all, yw_all = opts["shoulder"], opts["yaw"]
        if design.arch == "direct":
            # The elbow motor rides the arm, so each one changes what the shoulder and yaw carry.
            groups = []
            for e_i, e in enumerate(el):
                m = m0.copy()
                m[SLOTS.index("elbow")] = e.mass
                ok_sh = _joint_ok(tt, m, SHOULDER, *cap(sh_all, "shoulder")) & ok_el[e_i][None]
                si = _prune(sh_all, ok_sh)
                ok_yw = _joint_ok(tt, m, YAW, *cap(yw_all, "yaw")) & ok_el[e_i][None]
                yi = _prune(yw_all, ok_yw)
                groups.append(([e_i], [sh_all[i] for i in si], ok_sh[si], [yw_all[i] for i in yi], ok_yw[yi]))
        else:
            ok_sh = _joint_ok(tt, m0, SHOULDER, *cap(sh_all, "shoulder")) & ok[None]
            si = _prune(sh_all, ok_sh)
            ok_yw = _joint_ok(tt, m0, YAW, *cap(yw_all, "yaw")) & ok[None]
            yi = _prune(yw_all, ok_yw)
            groups = [(list(range(len(el))), [sh_all[i] for i in si], ok_sh[si], [yw_all[i] for i in yi], ok_yw[yi])]
        # Pack each point's candidate poses into a byte: a combination serves a point when
        # some candidate pose works for every joint at once.
        el_bits = np.packbits(ok_el, axis=-1)[..., 0]
        for e_idx, sh, ok_sh, yw, ok_yw in groups:
            if not sh or not yw:
                continue
            els = [el[i] for i in e_idx]
            sh_bits = np.packbits(ok_sh, axis=-1)[..., 0]
            yw_bits = np.packbits(ok_yw, axis=-1)[..., 0]
            e_b, s_b, y_b, w_m, always = _merge_points(el_bits[e_idx], sh_bits, yw_bits, w32)
            scores = np.empty((len(els), len(sh), len(yw)), np.float32)
            for i in range(len(els)):
                scores[i] = ((e_b[i][None, None, :] & s_b[:, None, :] & y_b[None, :, :]) != 0).astype(np.float32) @ w_m
            scores += always
            E, S, Y = (slice(None), None, None), (None, slice(None), None), (None, None, slice(None))
            needs = d_board | board(els)[E] | board(sh)[S] | board(yw)[Y]
            cost = attr(els, "price")[E] + attr(sh, "price")[S] + attr(yw, "price")[Y] + d_price + board_fixed * needs
            mass = mass_fixed + d_mass + attr(els, "mass")[E] + attr(sh, "mass")[S] + attr(yw, "mass")[Y]
            back = np.sqrt(d_back + blash(els, "elbow")[E] + blash(sh, "shoulder")[S] + blash(yw, "yaw")[Y])
            feasible = (mass <= max_mass) & (back <= max_backlash_mm)
            for k in _frontier_idx(cost, scores, feasible):
                a, b, c = np.unravel_index(k, scores.shape)
                candidates.append(
                    (float(cost[a, b, c]), float(scores[a, b, c]), Assignment(yw[c], sh[b], els[a], wp, fr, tr))
                )
    # The design's own frontier across every wrist pair.
    candidates.sort(key=lambda x: (x[0], -x[1]))
    results, best = [], -1.0
    for _cost, sc, asg in candidates:
        if sc > best + SCORE_EPS:
            best = sc
            results.append({"assignment": asg, "score": sc})
    return results


def precision_speed(design: ArmDesign, acts: dict[str, Actuator]) -> tuple[float, float]:
    """Tip error from gear backlash (mm, root-sum-square over the joints: the joints'
    play does not all line up at once, and gravity preloads most of it) and the slowest
    proximal joint's tip speed (m/s), with the gripper on and the arm at 85% of stretch."""
    levers = backlash_levers(design)
    err = math.sqrt(sum((math.radians(a.backlash_deg) * levers[j]) ** 2 for j, a in acts.items()))
    speed = min(a.speed * levers[j] for j, a in acts.items() if j in ("yaw", "shoulder", "elbow"))
    return err * 1000, speed


def summarise(
    design: ArmDesign,
    asg: Assignment,
    score: float,
    costs: dict[str, float],
    mm: MassModel,
    tier: str = "any",
    lever: str = "retail",
) -> dict:
    acts = {j: a for j, a in asg.by_joint().items() if a is not None}
    backlash_mm, tip_speed = precision_speed(design, acts)
    fixed = fixed_cost(design, costs, any(a.needs_driver for a in acts.values()))
    act_cost = sum(a.price for a in acts.values())
    mass = fixed_mass(design, mm) + sum(a.mass for a in acts.values())
    return {
        "design": asdict(design),
        "label": design.label,
        "actuators": {j: a.id for j, a in acts.items()},
        "score": score,
        "cost": round(act_cost + sum(fixed.values()), 2),
        "actuator_cost": round(act_cost, 2),
        "fixed_cost": {k: round(v, 2) for k, v in fixed.items()},
        "mass": round(mass, 3),
        "tier": tier,
        "lever": lever,
        "stages": {j: a.stage for j, a in acts.items() if a.stage},
        "sources": {j: a.source for j, a in acts.items()},
        "backlash_mm": round(backlash_mm, 2),
        "tip_speed": round(tip_speed, 2),
        "force_sensing": all(acts[j].force_sense == "yes" for j in ("shoulder", "elbow")),
        "kinds": sorted({a.kind for a in acts.values()}),
    }


def _stage_b_job(args: tuple[ArmDesign, int]) -> list[dict]:
    design, _ = args
    tasks, tools, inst, hx, bp = (_CTX[k] for k in ("tasks", "tools", "inst", "hx", "bp"))
    kin = kinematic_table(design, tasks, tools, inst, hx, bp)
    tt = torque_table(design, tasks, tools, inst, kin)
    w_task = np.array([t.weight for t in tasks])
    counts = np.bincount(inst.task, minlength=len(tasks))
    weights = w_task[inst.task] / counts[inst.task] / w_task.sum()
    costs = load_costs()
    mm = MassModel()
    board = sum(fixed_cost(design, costs, True).values()) - sum(fixed_cost(design, costs).values())
    out = []
    # The geometry's torque table is shared; each lever setting only changes the parts.
    for lever in LEVERS:
        catalog = load_catalog(lever)
        for tier in TIERS:
            found = enumerate_actuators(
                design,
                tt,
                weights,
                catalog,
                tier,
                lever,
                mass_fixed=fixed_mass(design, mm),
                max_mass=MAX_ARM_KG,
                max_backlash_mm=FIELD_MAX_BACKLASH_MM if tier == "field" else math.inf,
                board_fixed=board,
            )
            out.extend(summarise(design, r["assignment"], r["score"], costs, mm, tier, lever) for r in found)
    return out


def stage_b(finalists: list[ArmDesign], workers: int) -> list[dict]:
    jobs = [(d, i) for i, d in enumerate(finalists)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(False,)) as ex:
        out = []
        for rows in ex.map(_stage_b_job, jobs):
            out.extend(rows)
    return out


def pick_finalists(rows: list[dict], per_group: int) -> list[ArmDesign]:
    finalists = []
    for dof in (5, 6):
        for arch in ("parallelogram", "direct"):
            grp = [r for r in rows if r["design"]["dof"] == dof and r["design"]["arch"] == arch]
            for r in grp:
                d = r["design"]
                r["size"] = d["l1"] + d["l2"]
                r["ped"] = d["pedestal"]
            front = frontier(grp, [("kin_score", 1), ("size", -1), ("ped", -1)])
            front.sort(key=lambda r: -r["kin_score"])
            # Spread the picks along the frontier rather than taking the top few.
            if len(front) > per_group:
                idx = np.linspace(0, len(front) - 1, per_group).round().astype(int)
                front = [front[i] for i in sorted(set(idx))]
            for r in front:
                for spring in (False, True):
                    finalists.append(ArmDesign(**{**r["design"], "spring": spring}))
    return finalists


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-group", type=int, default=7)
    ap.add_argument("--reuse-a", action="store_true", help="take stage A from the last out/study.json")
    ap.add_argument("--stage-a-only", action="store_true", help="run the geometry sweep and stop")
    ap.add_argument("--from-a", help="take stage A from this JSON file")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    if args.from_a:
        rows_a = json.loads((ROOT / args.from_a).read_text())
    elif args.reuse_a:
        rows_a = json.loads((OUT / "study.json").read_text())["stage_a"]
    elif args.stage_a_only:
        rows_a = stage_a(args.quick, args.workers)
        (OUT / "stage_a.json").write_text(json.dumps(rows_a))
        print(f"stage A: {len(rows_a)} geometries in {time.time() - t0:.0f}s")
        return
    else:
        rows_a = stage_a(args.quick, args.workers)
    print(f"stage A: {len(rows_a)} geometries in {time.time() - t0:.0f}s")
    finalists = pick_finalists(rows_a, 3 if args.quick else args.per_group)
    t1 = time.time()
    rows_b = stage_b(finalists, args.workers)
    print(f"stage B: {len(finalists)} designs, {len(rows_b)} frontier points in {time.time() - t1:.0f}s")
    fronts: dict[str, dict[str, list[dict]]] = {}
    for lever in LEVERS:
        fronts[lever] = {}
        for tier in TIERS:
            sel = [r for r in rows_b if r["tier"] == tier and r["lever"] == lever]
            f = frontier(sel, [("score", 1), ("cost", -1)]) if sel else []
            f.sort(key=lambda r: r["cost"])
            fronts[lever][tier] = f
            print(f"frontier [{lever} / {tier}]: {len(f)} designs")
            for r in f if tier == "field" else []:
                print(
                    f"  ${r['cost']:7.0f}  score {r['score']:.3f}  {r['mass']:.2f} kg  "
                    f"{r['backlash_mm']:.1f} mm  {r['label']}  {r['actuators']}"
                )
    study = {
        "generated": time.strftime("%Y-%m-%d"),
        "hexapod": load_yaml("hexapod.yaml")["name"],
        "stage_a": rows_a,
        "stage_b": rows_b,
        "frontier": fronts,
    }
    (OUT / "study.json").write_text(json.dumps(study, indent=1))


if __name__ == "__main__":
    main()
