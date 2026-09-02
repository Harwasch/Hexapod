# 04 — Motor and reduction options

The plan is a PCB-stator axial-flux motor with an in-plane cycloid, chosen
because both can be made in-house at low cost. This note asks whether
anything beats it on the three things that matter here — torque per kilogram,
torque per dollar, and buildability in a small shop — and what to change if
the answer is "partly". It is a first-principles and experience survey; the
figures are order-of-magnitude, marked as such, and the actuator stage must
replace them with measured and datasheet numbers before anything is cut.

What each joint needs (from [01-sizing.md](01-sizing.md) §6, both stances):
roughly 55 / 135 / 145 N·m continuous and 70 / 245 / 155 N·m peak for yaw,
femur and knee, at 9 rad/s (yaw) and ~4 rad/s (pitch) unloaded, in a
Ø170 × 42 mm pancake stacked three-high on the hip axis.

## 1. Motors

The figure of merit for a motor in a geared joint is **continuous torque per
kilogram at the speed the ratio needs**, because the ratio multiplies torque
and divides speed, and the motor's mass is in the body either way. The
airgap shear stress σ (torque per unit rotor surface per unit radius) is the
honest way to compare types before a datasheet is open.

| Motor type | σ, continuous (kPa, estimate) | Torque per kg (N·m/kg, estimate) | Cost per unit at our size | Buildable in-house | Notes |
|---|---|---|---|---|---|
| **PCB stator, ironless axial flux, dual Halbach rotors** (the plan) | 1–2 | 2–3 | ~$100–150 in parts (magnets dominate) | Yes: PCB from any fab, magnets glued into printed or machined rings, no winding | No cogging, no iron loss, very low inductance. Copper is the limit: 12 layers of 2–3 oz is ~1 mm of copper in a 3 mm gap. Heat leaves only through the board. |
| **Wound flat-coil ironless axial flux** (same rotors, coils instead of a PCB) | 3–6 | 5–8 | ~$100–150 (same magnets, cheap wire, printed winding jigs) | Yes: CNC- or hand-wound flat coils potted in a printed or epoxy-glass carrier | 3–5× the copper of a PCB in the same gap; concentrated coils are easy to wind flat. This is the DIY axial-flux generator recipe scaled down, and it is the cheapest route to 2–3× the torque in the same housing. Manual labour per unit is the cost. |
| **Iron-core axial flux, segmented stator (YASA-style)** | 15–30 | 10–20 | ~$200–400; SMC or laminated segments are the hard part | Marginal: printed bobbins yes, magnetic cores no | Best torque density of any pancake. Needs soft-magnetic composite or wound-strip laminations per tooth; cogging and iron loss; commercial only for now. |
| **Radial outrunner, drone/robot class** (T-Motor U8 / MAD / hobby 8108–9235) | 10–15 | 5–8 | ~$60–150 off the shelf | No, but nothing to build | The proven quadruped choice (MIT Cheetah, ODRI). Not a pancake: 30–40 mm tall per motor plus the reducer, and it does not want to stack coaxially with a cycloid inside its rotor. |
| **Frameless torque motor** (Kollmorgen TBM, Allied, etc.) | 15–25 | 8–12 | $400–1200 | No | Industrial pancake with a large bore that a cycloid could sit in. Too expensive at eighteen. |
| **Hub / e-bike motor cores** | 8–12 | 4–6 | $50–120 | No | Cheap torque, but 100–250 mm radial, heavy iron, low speed; wrong shape for a stack. |

Reading it against the plan:

* **The PCB stator is the cheapest and the easiest to make, and it is the
  weakest by a factor of ~3–5 against wound coils in the same magnets.** That
  is why the sizing needs two stators at 50–60:1 for the femur and knee and
  why the whole concept rests on measuring σ. It remains the right first
  prototype: it proves the rotors, the cycloid, the driver and the thermal
  path with the fewest hand-made parts.
* **The same rotor and housing accept a wound flat-coil stator.** If the
  PCB measures under ~1.5 kPa, or the two-stator stack is too tall or too
  lossy, the upgrade is a stator swap, not a redesign. Plan the housing for
  either from the start: the stator carrier should be a separate part with
  the same bolt circle and the same gap.
* **Off-the-shelf radial motors win on torque density and cost of the motor
  alone** but lose the pancake stack, and eighteen of them with a reducer
  each are the $8–11k benchmark already noted. They are the right thing to
  build the first leg with while the custom actuator is being measured.
* **Iron-core axial flux is the ceiling** and not reachable in-house today.
  Worth revisiting only if the design moves to a contract motor build.

## 2. Reductions

Each joint needs 45–60:1 between a motor happy at 4–6 krpm and a joint at
4–13 rad/s, in a plane, inside a Ø110 mm bore, with 150–250 N·m going
through it and a foot that hits things.

| Reduction | Ratio per stage | Efficiency (estimate) | Backdrivable | Shock tolerance | Buildable in-house | Cost per joint | Notes |
|---|---|---|---|---|---|---|---|
| **Single-stage cycloid** (the plan) | 10–80:1 | 85–92 % | Poorly above ~40:1 | Very good (load shared over many pins) | Yes: the disc is a 2-D profile — laser- or water-jet cut in 6–8 mm steel, or wire-EDM; pins are dowels or needle rollers; eccentric on a standard bearing | $40–120 | Torque density is the best of the buildable options. Needs two discs 180° apart or a counterweight at 5 krpm input; fine tolerances on pin pitch decide noise and life. |
| **Compound / two-stage cycloid in one plane** | 30–150:1 | 80–88 % | No | Very good | Yes, one more disc | +$30 | Lets a single stator reach the femur rating without a 100-lobe disc; the extra disc costs efficiency and a few millimetres of height. |
| **Planetary, single stage** | 4–10:1 | 95–97 % | Yes | Good | Partly: printed carriers with bought gears, or bought sets | $30–80 | The quasi-direct-drive choice (MIT Cheetah 6:1). Only works with a motor 5–10× ours; not on the table with a PCB stator. |
| **Compound planetary, two stage** | 20–60:1 | 88–93 % | Marginal | Good | Partly | $60–150 | The usual off-the-shelf actuator's reducer (AK80-64, RMD-X8). Backlash grows with two stages; can be made in-plane but not inside a rotor bore. |
| **Harmonic / strain wave** | 50–160:1 | 65–80 % | No | Poor (flexspline fatigue on impact) | No | $250–600 | Zero backlash, but the efficiency and the shock sensitivity are wrong for a foot, and eighteen of them cost more than the motors. |
| **Cable / capstan** | 5–20:1 | 95–98 % | Yes | Good (cable stretch is a spring) | Yes: drums and steel cable | $20–60 | Zero backlash and transparent. Interesting *as the coxa transmission*: a 3–5:1 capstan at the femur axis, fed by a cable run along the coxa, lets the in-body cycloid drop to 15–20:1. |
| **Timing belt** | 2–5:1 | 93–96 % | Yes | Good (compliant) | Yes | $15–40 | ODRI's answer (two belt stages, 9:1, hobby motor, ~$300 per actuator, open source). Needs the motor 5–10× ours for our torques; but a belt stage along the coxa is a plausible way to carry the femur and knee drives. |
| **Magnetic gear** | 5–15:1 | 95 %+ | Yes | Perfect (slips) | Not yet | high | Overload-proof and silent; torque density too low for this and a research project of its own. |

Reading it:

* **The cycloid is the right reducer for this joint and this shop.** Nothing
  buildable in-house matches its torque density or its shock tolerance, and
  the disc being a 2-D profile is the point: laser- or water-jet-cut steel
  is precise, cheap, and needs no gear cutting. The things to design for are
  balance (two discs or a counterweight), a real steel disc and hardened
  pins for the femur and knee, and grease sealing. Backdrivability is lost
  above ~40:1, and the sizing already assumes output-side sensing and an
  elastic element for that.
* **The compound cycloid is the fallback if a single stator must do the
  femur** — it trades ~5 % efficiency and a disc for not needing the second
  stator. Keep the housing height able to take it.
* **A capstan or belt stage along the coxa deserves a look in the
  transmission chunk (M1)**, not as a replacement for the cycloid but as
  the last 3–5:1: the drives have to cross the yaw axis and travel 150 mm to
  the femur axis anyway, and a cable does that with no backlash and some
  useful compliance. It would let the in-body cycloid run 15–20:1, well
  inside the backdrivable range.

## 3. What this means for the plan

1. **Keep PCB + in-plane cycloid as the first build.** It is the lowest
   part count and the lowest skill floor, and it proves everything except
   the copper.
2. **Design the housing for a stator swap**: the same rotors, gap and bolt
   circle take a wound flat-coil stator, which is the cheap 2–3× torque
   upgrade if the PCB measures low, and the way back to one stator per
   joint.
3. **Design the housing height for a second disc**: the compound cycloid is
   the other route to one stator.
4. **Build the first leg on bought actuators** (the AK80-64 / RMD-X8 class
   for femur and knee, a smaller unit for yaw) so the transmission and the
   controller are not waiting on the motor, and so there is a measured
   benchmark for the custom unit to beat.
5. **Put "capstan or belt as the coxa stage" on the M1 list**, because it
   could halve the in-body ratio and give back backdrivability.

The two numbers the actuator stage owes back: **σ of a PCB stator at our
gap, measured on a dyno**, and **the cycloid's measured efficiency and
noise at 50:1 with laser-cut discs**. Everything in this note is provisional
until those exist.
