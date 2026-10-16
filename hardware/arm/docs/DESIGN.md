# FA-1 design

FA-1 is a 5-axis arm with a parallelogram elbow and gravity springs. It rides on the front
of the medium hexapod's deck and costs **$607 in parts**. It came out of an optimization
over 1,764 geometries and 27 real actuators, scored against the 17 tasks in
[REQUIREMENTS.md](REQUIREMENTS.md), and it was then run through seven task simulations in
MuJoCo. The interactive viewer (`viewer/index.html`) shows all of it.

| Spec                     | FA-1                                                                                                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Axes                     | 5: yaw, shoulder, elbow (parallelogram), wrist pitch, tool roll; the hexapod body adds height, pitch, yaw and a forward lean |
| Reach to the gripper tip | 0.82 m from the shoulder axis                                                                                                |
| Payload                  | 1.0 kg anywhere on the ground work patch; 0.33 kg continuous / 1.17 kg peak at full horizontal stretch                       |
| Arm mass                 | 2.28 kg without the tool (budget 2.5 kg)                                                                                     |
| Parts cost               | $607 per arm at about 10 units                                                                                               |
| Backlash at the tool     | 5.7 mm open loop (root-sum-square); the wrist camera closes the rest                                                         |
| Speed                    | 0.48 s per weed when spot-dosing (simulated, ±2 mm)                                                                          |
| Force sensing            | motor current at the yaw, shoulder and elbow (backdrivable QDD)                                                              |
| Capability               | serves 84% of the weighted task points (best design found at any price: 88%)                                                 |

## Layout

```text
      tool roll ─┐   ┌─ wrist pitch (servo)
                 ▼   ▼
   gripper ◄── [flange]─[wrist]══════ forearm (Al tube) ════╗ elbow pivot (bearings only)
                                                            ║
          rod (carbon, parallel to the upper arm) ─────╮    ║ upper arm (Al tube)
                                                       │    ║
   elbow motor ── crank ───────────────────────────────╯    ║
   (GIM6010-8)     ╲                                        ║
                    ╲── turret, shoulder axis 5 cm above the deck ── shoulder motor (GIM8108-8)
                             │
                   yaw motor (EduLite 05), recessed into the deck
```

- **Both big motors sit on the turret.** The shoulder motor turns the upper arm. The elbow
  motor, on the other side of the same axis, turns a crank, and a rod carries that crank's
  angle to a lever behind the elbow. The elbow pivot is just bearings. No motor rides on
  the arm, so the moving mass is two aluminium tubes, a wrist and a tool.
- **The parallelogram controls the forearm's absolute angle.** A force at the tool is split
  between the two motors, the shoulder taking it as if it acted at the elbow pivot and
  the elbow motor taking the rest. With a direct elbow the shoulder would carry the
  whole moment. In the 60 N weed pull, the shoulder peaked at 80% of its torque and the
  elbow at 31%. A direct-drive shoulder would have needed the sum.
- **Zero-length gravity springs** (an extension spring and a Dyneema cable over a pulley)
  on the shoulder and the crank cancel the arm's own weight at every pose. This only works
  exactly because each motor sees a single absolute angle. The motors then carry only the
  payload and the task force.
- **The yaw actuator is recessed into the deck**, which lets the shoulder axis sit 5 cm
  above the deck. That keeps the centre of mass low and gets the tool under the body's
  front edge, which helps for pulling and probing close to the feet.
- **The tool changer is a passive bayonet.** The tool-roll joint twists it locked against
  a holster on the deck, so tool changes need no extra motor. Six pogo pins carry 24 V,
  CAN and two spare lines to powered tools.

## Actuators

| Joint       | Part                                 | Rated / peak Nm | Mass  | Price   | Why this one                                                            |
| ----------- | ------------------------------------ | --------------- | ----- | ------- | ----------------------------------------------------------------------- |
| Yaw         | RobStride EduLite 05                 | 1.8 / 6         | 242 g | $80     | cheapest integrated QDD with an output encoder; yaw sees little gravity |
| Shoulder    | Steadywin GIM8108-8 + output encoder | 7.5 / 22        | 396 g | $135    | best $/Nm under 450 g; backdrivable 8:1                                 |
| Elbow       | Steadywin GIM6010-8 + output encoder | 5 / 11          | 388 g | $105    | the parallelogram halves the elbow's share                              |
| Wrist pitch | Hiwonder HX-35HM bus servo           | 1.2 / 3.4       | 71 g  | $36     | light at the end of the lever; carries only the tool                    |
| Tool roll   | Hiwonder HX-65HM bus servo           | 2.2 / 6.4       | 144 g | $50     | turns a 3 Nm valve and locks the tool changer                           |
| Gripper     | Feetech STS3215 in a fin-ray gripper | 1 / 2.9         | 55 g  | $24 kit | compliant TPU fingers; current-limited grip for fruit                   |

Everything at the base is a backdrivable quasi-direct-drive (QDD) motor on CAN. Motor
current doubles as a force sensor for finding the ground, a stem or a person, and a fall
or a branch back-drives the joint instead of stripping a gear. The two wrist joints use
bus servos because they carry little torque and every gram there costs torque at the
shoulder. Actuator datasheets, prices and sources are in `data/actuators_research.yaml`.

## How it was chosen

**Stage A** swept 1,764 geometries: 5 or 6 axes, parallelogram or direct elbow, upper arm
and forearm lengths of 0.22–0.40 m, three mount positions and three pedestal heights. It
scored kinematic coverage of 1,034 task points. A point counts as covered when an
inverse-kinematics solution exists at an acceptable approach angle, using any body pose the
legs allow, without hitting the body, the legs or the ground, and without tipping the
robot.

**Stage B** took 56 geometries from that frontier, with and without springs. For each one
it computed the torque every joint needs at every task point, then tried every actuator
assignment from the catalog. Gravity, the tool, payload and task force set the static
torque, checked against each part's rated torque; a 4 m/s² bound on dynamics was added on
top and checked against peak torque.

Results on the "field grade" frontier (a backdrivable, force-sensing shoulder and elbow,
QDD at the yaw, ≤ 6 mm backlash, ≤ 2.5 kg):

| Parts cost | Best capability | Design                                                         |
| ---------- | --------------- | -------------------------------------------------------------- |
| $468       | 20%             | cheapest field-grade arm; three EduLite 05s cannot hold much   |
| $543       | 79%             | 5-axis parallelogram + springs, GIM6010s at shoulder and elbow |
| **$607**   | **84%**         | **FA-1**: a GIM8108 shoulder, servo wrist upgraded             |
| $630       | 85%             | 6-axis, three STS3215 servos in the wrist                      |
| $972       | 88%             | 6-axis, RobStride 06 shoulder: the most any design reaches     |

FA-1 is the cheapest field-grade design within 5% of the best. Past it, each extra
percentage point costs $25–100. The 6-axis options add under one point and put three
hobby servos at the end of the arm; the hexapod's body yaw already supplies the sixth axis.

What the study says about the architecture, at the same parts budget:

| At ≤ $607, 5 axes         | Best capability  |
| ------------------------- | ---------------- |
| Parallelogram, springs    | 84% (FA-1, $607) |
| Parallelogram, no springs | 83%              |
| Direct elbow, springs     | 79%              |
| Direct elbow, no springs  | 75%              |

- **The parallelogram is worth 2–5 points** of capability at the same budget, most where money is tight.
- **Springs matter most when money is tight.** At $550 they lift the best design from 75%
  to 79%, because they let smaller motors carry the arm.
- **Field grade costs almost nothing here.** The best design allowed hobby servos and
  steppers anywhere reaches 85% for $600, but with 11 mm of backlash, no force sensing and
  servo gears that strip when back-driven.

## Simulation

Seven scenarios run in MuJoCo (`armdesign/sim.py`). The parallelogram is a real closed
chain. Every motor is torque-controlled and clipped at its catalog peak. The legs are
position servos, and soil, roots and stems are force laws. The tests
(`tests/test_model.py`) check that the optimizer's statics match MuJoCo's closed chain
within 3%.

| Scenario                                   | Highest motor load (of peak) | Smallest tip-over margin | At the targets         |
| ------------------------------------------ | ---------------------------- | ------------------------ | ---------------------- |
| Three strawberries into the deck tray      | 30% (yaw)                    | 30 cm                    | 1.4–3.2 mm             |
| Tomato 0.8 m up, body raised and tilted    | 35% (yaw)                    | 27 cm                    | 1.5–6.7 mm             |
| 60 N taproot, body leans in and legs lift  | 63% (shoulder)               | 12 cm                    | 2.5–6 mm               |
| The same root, pulled by the arm alone     | 80% (shoulder)               | 19 cm                    | 1–2.7 mm               |
| 1 kg rock at full reach, stowed on deck    | 91% (yaw, swinging it round) | 26 cm                    | 2–18 mm                |
| Soil probe pressed 6 cm by the body (64 N) | 72% (shoulder)               | 20 cm                    | 2 mm                   |
| Eight weeds micro-dosed                    | 56% (yaw)                    | 28 cm                    | ~2 mm, 0.48 s per weed |

- **The legs pay off when they shorten the lever.** Lifting with the legs does not reduce
  joint torque: torque is the force times the horizontal lever, whatever moves.
  Leaning the body 12 cm over the weed first is what cut the shoulder from 80% to 63%.
  This lean is not yet part of the optimizer's body-pose set, so its taproot coverage
  (13%) is conservative.
- **The 1 kg rock is the hardest job.** The yaw (EduLite 05) peaks at 91% while
  swinging the rock to the deck, and the wrist servo at 84%. Swing slower with heavy
  payloads, or step the yaw up to a GIM6010-8 (+$25).
- **Pressing the probe runs the shoulder at about its continuous rating** for the 3 s
  press, which is fine thermally. Deeper cores need a rotary auger tool.

## Where it can work

Share of each task's points served (the viewer's Workspace mode maps them):

| Served | Tasks                                                                                                                |
| ------ | -------------------------------------------------------------------------------------------------------------------- |
| 100%   | young weeds, spot-spray, planting, ground sensing, strawberries, rocks and litter, deck stowing, water sample, valve |
| 89–95% | canopy sensing, leaf clips, fruit up to 1.1 m, gate latch, pruning                                                   |
| 13–21% | taproots (torque), stamping and soil probes (robot weight); only close to the body                                   |

The last row is the physics, not the arm. A 12 kg robot can push about 110 N, and
taproots need the body leaning in.

## Fleet fit

- **Hexapod legs.** The leg joints need about 9.5 Nm continuous and 19 Nm peak at the femur
  in a tripod gait (12 kg robot, 2.5 kg arm). The GIM8108-8 (7.5 / 22 Nm) is close but
  under on continuous torque. Use the same vendor family one size up for the femur and
  tibia, and the GIM6010/8108 for the coxa. Then the whole robot shares one CAN protocol,
  one driver stack and one spares box.
- **The tool flange is the fleet standard.** The bayonet with six pogo pins fits anything
  that wants a tool: the wheeled carrier's bigger arm, a quadcopter's lightweight gripper,
  a rack of tools on the carrier.
- **A wheeled-platform arm** is the same code with a different `data/hexapod.yaml` (a heavier,
  wider base) and a longer link range. Rerun the study for it.

## Risks and next steps

- **Not built yet.** Next: print the turret and one link, bench-test the shoulder and
  elbow on a stand, and measure backlash, holding current and the spring balance.
- **Sealing is a plan, not a design.** None of the base actuators is sealed (the GIM parts
  claim IP54). The plan is TPU gaiters over each joint, a drip skirt on the turret and a
  sealed wrist. The RobStride 02 IP67 ($240) is the drop-in if the boots fail.
- **The EduLite 05's gears are powder metal**, a shock-tolerance question at the yaw. The
  GIM6010-8 is the fallback.
- **The wrist servos are not backdrivable.** A hard hit on the tool can strip them, so
  the tool changer's magnets are the breakaway.
- **Spring cables wear.** Inspect them, and give the controller a spring-broken check
  (holding current against the model).
- **Electronics and firmware are out of scope** for this pass: a CAN-FD bus shared with
  the legs, a 12 V buck for the servos, and a wrist camera with a ToF sensor on the
  hexapod's computer.
