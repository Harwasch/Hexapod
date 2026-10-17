# Field arm requirements

What the arm has to do, where each number comes from, and how it is encoded for the
optimizer. The task library itself is `data/tasks.yaml`; the research behind the forces,
with every source, is in [research/task_forces.md](research/task_forces.md).

## Host

The arm rides on a **medium hexapod (8–20 kg)**. No hexapod has been designed yet, so the
study uses a reference host in `data/hexapod.yaml`: 12 kg without the arm, a 0.56 × 0.30 ×
0.10 m body standing 0.24 m high, coxa-femur-tibia legs of 0.06 / 0.18 / 0.26 m, feet on a
0.80 × 0.90 m hexagon. With the feet planted, the legs can move the body ±7–12 cm in
height, −10 to +15° in pitch and ±20° in yaw. The optimizer and the planner use that motion.

| Constraint                 | Value                          | Why                                                                                  |
| -------------------------- | ------------------------------ | ------------------------------------------------------------------------------------ |
| Arm mass, tools excluded   | ≤ 2.5 kg, field kit included   | chosen with the medium-hexapod host                                                  |
| Tip-over margin, holding   | ≥ 5 cm inside the feet         | centre of pressure inside the support polygon                                        |
| Tip-over margin, pull/push | ≥ 2 cm                         | short events; the planner can lean the body                                          |
| Weight left on the feet    | ≥ 25 %                         | a push must not lift the robot off the ground                                        |
| Cost                       | minimised, delivered to the US | "incredibly low cost", traded against capability; list price + duty + freight        |
| Precision                  | ≤ 6 mm open loop (RSS)         | dosing and stamping want ±2–5 mm; the wrist camera closes the rest                   |
| Base joints                | backdrivable QDD, ≥ 30 rpm     | current as a force sensor, survives being back-driven, 0.5 s dosing cycle            |
| Base joints' output angle  | known at power-up              | no homing sweep from the stowed pose on the deck                                     |
| Tool twist                 | ≥ 3 Nm at the tool roll (peak) | valves and knobs; without it the cheapest designs drop the valve task to save weight |

## Tasks

Seventeen tasks across the six jobs asked for (weeding, dosing and planting, sampling,
close-range sensing, harvesting, field maintenance). Each task is a region of target points
around the hexapod, an approach cone, a tool, and the force the world puts on the tool.

| Task                    | Where                    | Force on the tool         | Basis                                                       |
| ----------------------- | ------------------------ | ------------------------- | ----------------------------------------------------------- |
| Pull a young weed       | ground, 0.25–0.85 m out  | 10 N up                   | seedlings 0.3–2.5 N to pull; 4× for small rosettes          |
| Pull a taprooted weed   | ground                   | 100 N up                  | dock-weeding robots design for 50–150 N                     |
| Stab or stamp a weed    | ground                   | 80 N down                 | stamping ~100 N, capped by what the robot weighs            |
| Spot-spray / micro-dose | 8 cm above ground        | none                      | 1–5 µL droplets, ±2.5 mm                                    |
| Plant a seed or plug    | ground                   | 15 N down                 | plug press-in                                               |
| Push in a soil probe    | ground, near vertical    | 80 N down                 | TEROS-class moisture probe 70–100 N; deeper cores go rotary |
| Sensor over the ground  | 15 cm standoff           | none, 0.35 kg sensor head | MicaSense-class multispectral camera                        |
| Sensor into a canopy    | 0.05–0.45 m up, sideways | none                      |                                                             |
| Clip a leaf             | 0.05–0.9 m up            | 10 N                      |                                                             |
| Strawberry              | 0.03–0.21 m up           | 15 N pull                 | 13.9 N straight pull (3.2 N bend-then-pull)                 |
| Tomato, pepper, apple   | 0.3–1.1 m up             | 30 N pull, 0.25 kg        | apple 28.3 N (95th pct), tomato 12.5–16.3 N                 |
| Rock, bottle, stick     | ground                   | 1.0 kg                    | fist-size rock 0.4–1.0 kg                                   |
| Stow on the deck        | robot's own deck         | 1.0 kg                    | samples, rocks, tool changes                                |
| Water sample            | 0.1–0.3 m below a bank   | 0.3 kg                    |                                                             |
| Gate latch or button    | 0.7–1.1 m up             | 20 N push                 | accessible latches are limited to 22 N                      |
| Small valve or knob     | 0.05–0.5 m up            | 3 Nm twist                | 1/4–1/2 in. valves; bigger ones by turning the body         |
| Prune a stem            | 0.1–1.0 m up             | 15 N push                 | 100–500 N blade force stays inside a geared shear           |

Weights in `data/tasks.yaml` favour the daily jobs: weeding, dosing and spraying count
three times as much as a gate latch.

## What the research changed

- **Robot weight caps any downward push.** An unanchored 8–20 kg hexapod can push about
  50–160 N. A 10–20 mm soil core needs 130–900 N, so coring needs a rotary auger tool, or
  the body pressing down, rather than a stronger arm. The study shows this directly: only
  points close to the body pass the tip-over check for the 80 N probe.
- **Cutting force belongs in the tool.** Pruning 3–10 mm stems takes 100–500 N at the blade.
  A geared shear (about 10 Nm at the pivot, ~0.4 kg) keeps that force off the arm.
- **Fruit needs force control, not force.** Detachment is 3–30 N, but the bruising limit is
  10–24 N of grip: a compliant fin-ray gripper and a motor-current force limit.
- **Very little in this price class is sealed, and seals cost backdrivability.** Only one
  actuator found (RobStride 02, IP67, about $248 delivered) has an ingress rating. Seal
  friction grows with diameter squared, so the plan
  ([research/rugged.md](research/rugged.md)) is:
  - V-rings and printed labyrinths at the two 20–25 mm shafts leaving the turret;
  - a drip skirt at the yaw;
  - sealed pods for the wrist servos.
  - **Do not** put O-rings, lip seals or TPU bellows on a QDD joint.
  - Realistic protection is about IPX3–4 at the turret and IPX5–6 at the wrist until a
    hose test says otherwise.
- **Sun, not rain, sets the material.** A dark horizontal surface runs about 50 K over
  ambient. PETG (HDT 69–78 °C) must not carry load in the sun. Use white ASA outside and
  PA12-CF near the motors. The motors need shaded aluminium to shed heat; the gravity
  springs keep their continuous current low.

## Environment

−10 to +45 °C (with sun-load margin), dust, rain and herbicide spray. White ASA for exterior
printed parts and PA12-CF for loaded parts near the motors, stainless fasteners,
conformal-coated electronics, and Dyneema DM20 rope kept in the shade inside the turret.
