"""The parts the optimizer may put at each joint, under a given set of cost levers.

Every price is landed: what the order costs delivered to the US, duty and freight
included, since nearly every part ships from China. Three levers, each switchable so the
study can show what each one is worth:

* sourcing: a part's cheapest credible landed price (duty-paid reseller, Taobao agent,
  AliExpress) instead of the original listing's, when the research found the same
  variant (`price_alt_usd` in data/actuators.yaml);
* reductions: a printed capstan or belt stage between a turret-mounted motor and its
  joint (data/transmissions.yaml). Only the yaw, the shoulder and a parallelogram's elbow
  sit on the turret, so only they can take one without adding mass to the arm;
* custom: bare motor + gearbox units (data/bare_units.yaml) driven by the fleet's own
  multi-axis driver board, priced per axis in data/costs.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache

from armdesign.model import Actuator, load_yaml

LEVERS: dict[str, dict[str, bool]] = {
    "retail": {"sourcing": False, "reductions": False, "custom": False},
    "sourcing": {"sourcing": True, "reductions": False, "custom": False},
    "reductions": {"sourcing": False, "reductions": True, "custom": False},
    "custom": {"sourcing": False, "reductions": False, "custom": True},
    "all": {"sourcing": True, "reductions": True, "custom": True},
}
LEVER_LABELS = {
    "retail": "US retail, integrated actuators",
    "sourcing": "+ cheaper sourcing",
    "reductions": "+ printed reductions",
    "custom": "+ custom driver electronics",
    "all": "all three levers",
}
TURRET_JOINTS = ("yaw", "shoulder", "elbow")  # the elbow only on a parallelogram
# A joint has to turn at least this fast for the 0.5 s spot-dosing cycle (a 0.3 m hop
# peaks near 3 rad/s at the base joints).
MIN_BASE_RPM = 30.0
ALPHA_BASE = 8.0  # rad/s^2 the base joints are sized to reach (4 m/s^2 at 0.5 m)
# Above these total ratios a joint no longer back-drives well enough to use motor
# current as a force sensor.
FORCE_SENSE_RATIO = 30.0
PARTLY_RATIO = 60.0


def _yn(v: object) -> str:
    # YAML 1.1 reads a bare yes/no as a boolean.
    return "yes" if v is True else "no" if v is False else str(v)


@dataclass(frozen=True)
class Stage:
    id: str
    kind: str
    ratio: float
    efficiency: float
    backlash_deg: float
    mass: float
    cost: float
    max_torque: float
    joints: tuple[str, ...] = TURRET_JOINTS
    drum_r: float = 0.0  # outer radius of a capstan drum about the joint axis, m
    pinion_r: float = 0.0  # effective (rope-centre) pinion radius, m
    rope: str = ""
    notes: str = ""


@cache
def load_stages() -> tuple[Stage, ...]:
    """Every stage and ratio in data/transmissions.yaml. A capstan's torque limit, drum and
    mass follow from its rope and ratio (docs/research/reductions.md)."""
    d = load_yaml("transmissions.yaml")
    out = []
    for s in d["stages"]:
        common = dict(
            kind=s["kind"],
            efficiency=float(s["efficiency"]),
            backlash_deg=float(s["backlash_deg"]),
            joints=tuple(s["joints"]),
            notes=str(s.get("notes", "")),
        )
        for n in s["ratios"]:
            cost = float(s["cost_usd"]["base"] + s["cost_usd"]["per_ratio"] * n)
            if s["kind"] == "capstan":
                for r in s["ropes"]:
                    rp = (r["pinion_core_mm"] + r["d_mm"]) / 2000
                    reff = n * rp
                    out.append(
                        Stage(
                            id=f"capstan{n:g}-{r['id']}",
                            ratio=float(n),
                            mass=s["mass_kg"]["base"] + s["mass_kg"]["per_r2"] * reff**2,
                            cost=cost,
                            max_torque=s["torque_fraction_of_mbl"] * s["bend_strength"] * r["mbl_n"] * reff,
                            drum_r=reff + (r["d_mm"] / 2 + s["flange_mm"]) / 1000,
                            pinion_r=rp,
                            rope=str(r["name"]),
                            **common,
                        )
                    )
            else:
                out.append(
                    Stage(
                        id=f"{s['kind']}{n:g}",
                        ratio=float(n),
                        mass=float(s["mass_kg"]["base"] + s["mass_kg"]["per_ratio"] * n),
                        cost=cost,
                        max_torque=float(s["max_pinion_nm"]) * n,
                        **common,
                    )
                )
    return tuple(out)


def stage_fits(s: Stage, joint: str, arch: str, pedestal: float) -> bool:
    """A stage can drive this joint on this geometry: the joint is on the turret, the
    stage is meant for it, and a capstan drum clears the deck under the shoulder axis."""
    if joint == "elbow" and arch != "parallelogram":
        return False
    if joint not in s.joints:
        return False
    return s.kind != "capstan" or s.drum_r <= pedestal


def rotor_inertia(kind: str, mass: float, dia: float) -> float:
    """Motor-side rotor inertia, kg m^2. An estimate where no datasheet gives one: an
    outrunner's bell is about a quarter of the unit's mass at 40% of its diameter."""
    if kind in ("qdd", "diy_qdd"):
        return 0.25 * mass * (0.4 * dia) ** 2
    if kind == "stepper":
        return 5e-6 if mass < 0.8 else 3e-5
    return 1e-7  # coreless micro-motor in a hobby servo


def landed(a: dict, costs: dict[str, float]) -> float:
    """What the listed order costs delivered: the researched figure, else list price plus
    US duty and freight."""
    if "landed_usd" in a:
        return float(a["landed_usd"])
    return float(a["price_usd"]) * (1 + costs["import_duty_pct"] / 100) + costs["default_shipping_usd"]


def landed_alt(a: dict) -> float | None:
    """The cheapest credible landed price for the same variant, or None."""
    alt = a.get("price_alt_usd")
    if alt is None or a.get("price_alt_confidence", "low") == "low":
        return None
    return float(alt)


def _actuator(
    a: dict, price: float, source: str, *, needs_driver: bool = False, retail: float | None = None
) -> Actuator:
    kind = a["kind"]
    mass, dia = float(a["mass_kg"]), float(a["dia_mm"]) / 1000
    ratio = float(a.get("ratio", 1.0))
    return Actuator(
        id=a["id"],
        name=a["name"],
        kind=kind,
        rated=float(a["rated_nm"]),
        peak=float(a["peak_nm"]),
        mass=mass,
        speed=float(a["speed_rpm"]) * 2 * math.pi / 60,
        price=round(price, 2),
        dia=dia,
        length=float(a["len_mm"]) / 1000,
        backdrivable=_yn(a["backdrivable"]),
        interface=str(a["interface"]),
        backlash_deg=float(a["backlash_deg"]),
        force_sense=_yn(a["force_sense"]),
        output_encoder=_yn(a["output_encoder"]),
        ip=str(a["ip"]),
        notes=str(a.get("notes", "")),
        ratio=ratio,
        needs_driver=needs_driver,
        source=source,
        retail=float(retail if retail is not None else a["price_usd"]),
        inertia=rotor_inertia(kind, mass, dia) * ratio**2,
    )


def _price(a: dict, costs: dict[str, float], sourcing: bool) -> tuple[float, str]:
    price, source = landed(a, costs), "retail"
    alt = landed_alt(a) if sourcing else None
    if alt is not None and alt < price:
        price, source = alt, str(a.get("price_alt_source", "alt"))
    return price, source


def load_catalog(lever: str = "retail") -> list[Actuator]:
    """Every whole part (motor + its own gearbox) available under a lever setting, at its
    landed price."""
    flags = LEVERS[lever]
    costs = {k: float(v) for k, v in load_yaml("costs.yaml")["usd"].items()}
    out = []
    for a in load_yaml("actuators.yaml")["actuators"]:
        if a.get("exclude"):
            continue
        out.append(_actuator(a, *_price(a, costs, flags["sourcing"])))
    if flags["custom"]:
        per_axis = costs["custom_driver_per_axis"]
        for a in load_yaml("bare_units.yaml")["units"]:
            price, source = _price(a, costs, flags["sourcing"])
            out.append(_actuator(a, price + per_axis, f"{source} + custom driver", needs_driver=True))
    return out


def with_stage(a: Actuator, s: Stage) -> Actuator | None:
    """The part driving its joint through a reduction stage, or None if the stage can't
    carry the torque or the joint would turn too slowly."""
    speed = a.speed / s.ratio
    if speed * 60 / (2 * math.pi) < MIN_BASE_RPM:
        return None
    peak = min(a.peak * s.ratio * s.efficiency, s.max_torque)
    rated = min(a.rated * s.ratio * s.efficiency, s.max_torque)
    total = a.ratio * s.ratio
    fs = a.force_sense
    if fs == "yes" and total > FORCE_SENSE_RATIO:
        fs = "partly" if total <= PARTLY_RATIO else "no"
    return replace(
        a,
        id=f"{a.id}+{s.id}",
        name=f"{a.name} + {s.ratio:g}:1 {s.kind}",
        rated=rated,
        peak=peak,
        mass=a.mass + s.mass,
        speed=speed,
        price=round(a.price + s.cost, 2),
        backlash_deg=a.backlash_deg / s.ratio + s.backlash_deg,
        force_sense=fs,
        ratio=total,
        stage=f"{s.ratio:g}:1 {s.kind}",
        inertia=a.inertia * s.ratio**2,
    )


def joint_pool(
    catalog: list[Actuator], joint: str, lever: str, arch: str = "parallelogram", pedestal: float = math.inf
) -> list[Actuator]:
    """Parts for one joint: the whole parts, plus every part with every stage that fits
    the joint where reductions are allowed. Base joints must reach MIN_BASE_RPM; bare
    motors only drive joints on the turret."""
    pool = list(catalog)
    if joint in TURRET_JOINTS:
        pool = [a for a in pool if a.speed * 60 / (2 * math.pi) >= MIN_BASE_RPM]
    # A bare motor has to live in the sealed turret box, next to its driver board.
    if joint not in ("yaw", "shoulder") and not (joint == "elbow" and arch == "parallelogram"):
        pool = [a for a in pool if not a.needs_driver]
    if LEVERS[lever]["reductions"]:
        stages = [s for s in load_stages() if stage_fits(s, joint, arch, pedestal)]
        pool += [x for a in catalog for s in stages if (x := with_stage(a, s)) is not None]
    return pool


def peak_available(a: Actuator, joint: str, derate: float) -> float:
    """Peak torque left for the task after accelerating the motor's own rotor."""
    if joint in TURRET_JOINTS:
        return derate * a.peak - a.inertia * ALPHA_BASE
    return derate * a.peak


def resolve(ids: dict[str, str], lever: str) -> dict[str, Actuator]:
    """Parts by joint from the ids a study wrote ("rs_edulite_05+capstan3" is the EduLite
    05 through a 3:1 capstan), priced under the same lever setting."""
    cat = {a.id: a for a in load_catalog(lever)}
    stages = {s.id: s for s in load_stages()}
    out = {}
    for joint, pid in ids.items():
        base, _, stage = pid.partition("+")
        part = cat[base]
        if stage:
            geared = with_stage(part, stages[stage])
            if geared is None:
                raise ValueError(f"{pid} is not a valid stage for {joint}")
            part = geared
        out[joint] = part
    return out
