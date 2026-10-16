"""Choose the design from the study and package everything the viewer and the docs read.

    uv run python -m armdesign.report select   # out/study.json -> design/selected.yaml
    uv run python -m armdesign.report          # selected design -> viewer/data/*, viewer/assets/*

The pick: on the "field" frontier (backdrivable, force-sensing shoulder and elbow, within
the mass budget and the backlash limit), the cheapest design within 5% of the best score
any field design reaches. Past that knee each extra point of capability costs $25-100.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import sys

import numpy as np
import yaml

from armdesign.evaluate import (
    PEAK_DERATE,
    RATED_DERATE,
    SLOTS,
    feasible_with,
    fixed_loads,
    kinematic_table,
    load_tasks,
    load_tools,
    make_instances,
    slot_loads,
    torque_table,
)
from armdesign.model import (
    JOINTS,
    ROOT,
    ArmDesign,
    BodyPose,
    Hexapod,
    Load,
    MassModel,
    body_pose_grid,
    load_torques,
    load_yaml,
    solve_poses,
    spring_torques,
)
from armdesign.optimize import fixed_cost, fixed_mass, load_costs, precision_speed

OUT = ROOT / "out"
VIEWER = ROOT / "viewer"
KNEE = 0.95


def choose(study: dict) -> dict:
    field = study["frontier"]["field"]
    best = max(r["score"] for r in field)
    ok = [r for r in field if r["score"] >= KNEE * best]
    return min(ok, key=lambda r: r["cost"])


def select() -> None:
    study = json.loads((OUT / "study.json").read_text())
    pick = choose(study)
    sel = {
        "why": f"cheapest field-tier design within {round((1 - KNEE) * 100)}% of the best field-tier score",
        "design": pick["design"],
        "actuators": pick["actuators"],
        "score": pick["score"],
        "cost": pick["cost"],
        "mass": pick["mass"],
    }
    (ROOT / "design").mkdir(exist_ok=True)
    with (ROOT / "design" / "selected.yaml").open("w") as f:
        f.write("# Written by `python -m armdesign.report select` from out/study.json.\n")
        yaml.safe_dump(sel, f, sort_keys=False)
    print(f"selected: ${pick['cost']} score {pick['score']:.3f} {pick['label']} {pick['actuators']}")


# --- metrics of the selected design -------------------------------------------------------


def payload_at_reach(design: ArmDesign, acts: dict, hx: Hexapod, mm: MassModel, frac: float = 0.95) -> dict:
    """Largest payload at the gripper with the arm stretched horizontally (wrist centre at
    frac of full reach, tool level), continuous and for a short lift."""
    tools = load_tools()
    gr = tools["gripper"]
    tcp = design.wrist_len + gr.length
    s = np.array([design.mount_x, 0.0, hx.center_height + hx.deck_z + design.pedestal])
    reach = frac * (design.l1 + design.l2)
    target = s + np.array([reach + tcp, 0.0, 0.0])
    p = solve_poses(design, hx, [BodyPose()], target[None], np.array([[1.0, 0.0, 0.0]]), np.array([tcp]), np.array([0]))
    arm = np.zeros(6)
    for ld in fixed_loads(design, mm):
        arm += load_torques(design, p, ld)[0] - spring_torques(design, p, ld)[0]
    arm += load_torques(design, p, Load("tool", design.wrist_len + gr.cg, gr.mass))[0]
    arm -= spring_torques(design, p, Load("tool", design.wrist_len + gr.cg, gr.mass))[0]
    for name, ld in slot_loads(design, mm).items():
        m = acts[name].mass
        arm += m * (load_torques(design, p, ld)[0] - spring_torques(design, p, ld)[0])
    per_kg = load_torques(design, p, Load("tool", tcp, 1.0))[0]
    out = {}
    for label, cap, derate in (("continuous_kg", "rated", RATED_DERATE), ("peak_kg", "peak", PEAK_DERATE)):
        best = math.inf
        for j, name in enumerate(JOINTS):
            if name not in acts or abs(per_kg[j]) < 1e-9:
                continue
            limit = derate * getattr(acts[name], cap)
            best = min(best, (limit - abs(arm[j])) / abs(per_kg[j]))
        out[label] = round(max(best, 0.0), 2)
    out["reach_m"] = round(reach + tcp, 3)
    return out


def leg_fit(acts: dict, hx: Hexapod) -> dict:
    """Could the hexapod's legs use the arm's shoulder actuator? Static femur torque with a
    tripod (3 feet) carrying robot + arm, foot 0.20 m out from the femur axis, x2 for gait."""
    a = acts["shoulder"]
    need_cont = (hx.mass + 2.5) * 9.81 / 3 * 0.20
    return {
        "actuator": a.name,
        "femur_need_nm": round(need_cont, 1),
        "femur_peak_need_nm": round(2 * need_cont, 1),
        "rated_nm": a.rated,
        "peak_nm": a.peak,
        "fits": a.rated >= need_cont and a.peak >= 2 * need_cont,
    }


def workspace(design: ArmDesign, acts: dict, hx: Hexapod) -> dict:
    """Every task point and whether the selected design (actuators included) serves it."""
    tasks, tools = load_tasks(), load_tools()
    inst = make_instances(tasks, hx, coarse=True)
    bp = body_pose_grid(hx, fine=True)
    kin = kinematic_table(design, tasks, tools, inst, hx, bp)
    tt = torque_table(design, tasks, tools, inst, kin)
    slot_mass = np.array([[acts[s].mass if s in acts else 0.0 for s in SLOTS]])
    rated = np.array([[acts[j].rated if j in acts else np.inf for j in JOINTS]])
    peak = np.array([[acts[j].peak if j in acts else np.inf for j in JOINTS]])
    served = feasible_with(tt, slot_mass, rated, peak)[0]
    per_task = []
    for i, t in enumerate(tasks):
        sel = inst.task == i
        per_task.append(
            {
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "weight": t.weight,
                "reach": round(float(kin.feasible[sel].mean()), 3),
                "served": round(float(served[sel].mean()), 3),
                "points": int(sel.sum()),
            }
        )
    w = np.array([t.weight for t in tasks])
    score = float(sum(p["served"] * t.weight for p, t in zip(per_task, tasks, strict=True)) / w.sum())
    pts = []
    for k in range(len(inst.task)):
        t = tasks[inst.task[k]]
        p = inst.target[k]
        if t.frame == "body":
            r, tr = hx.body_rt(BodyPose())
            p = r @ p + tr
        pts.append([round(float(v), 3) for v in p] + [int(inst.task[k]), int(kin.feasible[k]) + int(served[k])])
    return {"tasks": per_task, "score": round(score, 4), "points": pts}


def bom(design: ArmDesign, acts: dict) -> list[dict]:
    rows = []
    for j in JOINTS:
        if j in acts:
            a = acts[j]
            rows.append(
                {
                    "group": "Actuators",
                    "item": a.name,
                    "where": j.replace("_", " "),
                    "qty": 1,
                    "usd": a.price,
                    "kg": a.mass,
                }
            )
    costs = load_costs()
    fixed = fixed_cost(design, costs)
    mm = MassModel()
    masses = {
        "aluminium tube": mm.tube_kg_per_m * (design.l1 + design.l2) / 0.36 * 0.30,
        "printed parts": None,
        "parallelogram rod + rod ends": mm.rod_kg,
        "gravity springs": mm.spring_kg * 2,
    }
    for k, v in fixed.items():
        group = "Sensing" if "camera" in k else "End effector" if k.startswith(("gripper", "tool")) else "Structure"
        if k in ("harness + connectors", "seals + boots", "servo bus adapter"):
            group = "Electrical + sealing"
        rows.append({"group": group, "item": k, "where": "", "qty": 1, "usd": round(v, 2), "kg": masses.get(k)})
    return rows


def compact_sim(src: dict, hz: int = 25) -> dict:
    """Downsample a simulation log for the viewer."""
    t = np.array(src["log"]["t"])
    step = max(1, round((len(t) - 1) / (t[-1] - t[0]) / hz)) if len(t) > 1 else 1
    keep = slice(None, None, step)
    log = {k: v[keep] for k, v in src["log"].items()}
    log["pos"] = [[[round(x, 4) for x in p] for p in frame] for frame in log["pos"]]
    log["quat"] = [[[round(x, 3) for x in q] for q in frame] for frame in log["quat"]]
    return {**{k: v for k, v in src.items() if k != "log"}, "log": log}


def main(argv: list[str]) -> None:
    if argv[:1] == ["select"]:
        select()
        return
    from armdesign.sim import load_selected

    design, aa = load_selected()
    acts = {j: getattr(aa, j) for j in JOINTS if getattr(aa, j, None) is not None}
    hx = Hexapod.load()
    mm = MassModel()
    study = json.loads((OUT / "study.json").read_text())
    data = VIEWER / "data"
    data.mkdir(parents=True, exist_ok=True)
    ws = workspace(design, acts, hx)
    backlash, speed = precision_speed(design, acts)
    cad = json.loads((ROOT / "cad" / "cad.json").read_text())
    total_cost = sum(a.price for a in acts.values()) + sum(fixed_cost(design, load_costs()).values())
    selected = {
        "design": design.__dict__,
        "label": design.label,
        "actuators": {
            j: {
                k: getattr(a, k)
                for k in (
                    "id",
                    "name",
                    "kind",
                    "rated",
                    "peak",
                    "mass",
                    "speed",
                    "price",
                    "backlash_deg",
                    "force_sense",
                    "ip",
                    "interface",
                )
            }
            for j, a in acts.items()
        },
        "cost": round(total_cost, 2),
        "mass": round(fixed_mass(design, mm) + sum(a.mass for a in acts.values()), 3),
        "backlash_mm": round(backlash, 2),
        "tip_speed": round(speed, 2),
        "payload": payload_at_reach(design, acts, hx, mm),
        "legs": leg_fit(acts, hx),
        "printed_grams": cad["printed_grams"],
        "printed_parts": cad["printed"],
        "coverage": ws["tasks"],
        "score": ws["score"],
        "hexapod": load_yaml("hexapod.yaml"),
    }
    # The study, trimmed to what the charts draw.
    keep = (
        "cost",
        "score",
        "mass",
        "backlash_mm",
        "tip_speed",
        "label",
        "actuators",
        "tier",
        "design",
        "force_sensing",
        "kinds",
    )
    slim = {
        "frontier": {k: [{x: r[x] for x in keep} for r in v] for k, v in study["frontier"].items()},
        "stage_b": [{x: r[x] for x in keep} for r in study["stage_b"]],
        "stage_a": [{"d": r["design"], "s": round(r["kin_score"], 4)} for r in study["stage_a"]],
    }
    with (ROOT / "data" / "benchmarks.csv").open() as f:
        bench = [row for row in csv.DictReader(f)]
    slim["benchmarks"] = [
        {
            k: row[k]
            for k in (
                "name",
                "reach_mm",
                "payload_kg",
                "mass_kg",
                "price_usd",
                "price_kind",
                "actuators",
                "backdrivable",
                "ip",
            )
        }
        for row in bench
        if row["reach_mm"] and row["payload_kg"] and row["price_usd"]
    ]
    (data / "study.json").write_text(json.dumps(slim, separators=(",", ":")))
    (data / "selected.json").write_text(json.dumps(selected, indent=1))
    (data / "workspace.json").write_text(json.dumps(ws, separators=(",", ":")))
    (data / "bom.json").write_text(json.dumps(bom(design, acts), indent=1))
    sims = []
    for f in sorted((OUT / "sim").glob("*.json")):
        if f.name == "index.json":
            continue
        s = compact_sim(json.loads(f.read_text()))
        (data / f"sim_{f.stem}.json").write_text(json.dumps(s, separators=(",", ":")))
        entry = {
            "id": s["name"],
            "title": s["title"],
            "summary": s["summary"],
            "tool": s["tool"],
            "note": s.get("note", ""),
        }
        doses = [e["t"] for e in s["events"] if e["label"].startswith("dose weed")]
        if len(doses) > 1:
            entry["per_weed_s"] = round((doses[-1] - doses[0]) / (len(doses) - 1), 3)
        sims.append(entry)
    order = ["strawberries", "tomato", "weed_body", "weed_arm", "rock", "soil_probe", "spray"]
    sims.sort(key=lambda s: order.index(s["id"]) if s["id"] in order else 99)
    (data / "sims.json").write_text(json.dumps(sims, indent=1))
    # One JSON of base64 GLBs: artifact hosting serves JSON but not .glb.
    meshes = {g.stem: base64.b64encode(g.read_bytes()).decode() for g in sorted((ROOT / "cad" / "glb").glob("*.glb"))}
    (data / "meshes.json").write_text(json.dumps(meshes))
    print(f"viewer data written; coverage score {ws['score']:.3f}, payload {selected['payload']}")


if __name__ == "__main__":
    main(sys.argv[1:])
