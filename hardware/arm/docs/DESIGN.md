# FA-2 design

> **Known errors (review 2026-10-19, not yet fixed)**
>
> - **Wrist statics.** `wrist_split` (`armdesign/model.py`, lines 378–381) decomposes the
>   wrist moment along axes that are not orthogonal, when it should project it onto each
>   joint axis (Jᵀ). Example: at a 45° bend, a 3 Nm moment about the tool axis should put
>   2.12 Nm on the forearm roll; the model puts 0. FA-2 is the 6-DOF pick, so its wrist
>   actuator sizing may be understated.
> - **Speed.** The sim (`armdesign/sim.py`, line 578) clips torque but never limits joint
>   speed. In the spray task the shoulder reaches 1.38× and the wrist pitch 1.52× their
>   catalog speed limits; in the strawberry task the wrist reaches 1.18×. So "0.48 s per
>   weed" is not backed by a feasible run.
>
> The picks and numbers below are unchanged from the study as run; treat FA-2's wrist sizing
> and cycle times as unverified until these are fixed.

FA-2 is a 6-axis arm with a parallelogram elbow. Its shoulder and elbow are cheap integrated
QDD actuators that drive their joints through 3:1 Dyneema rope capstans on the turret.

- **Cost:** $756 delivered to the US, duty and freight included, at about 10 units. That
  is 10% less than FA-1 at the same prices ($840).
- **Capability:** it serves 90% of the weighted task points; FA-1 serves 87%.

It came out of a second optimization over the same 1,764 geometries:

- every price is delivered, and checked against a live cart;
- three cost levers: duty-paid sourcing, printed reductions and a custom driver board;
- stricter field-grade rules (below).

It was then run through seven MuJoCo task simulations. The interactive viewer
(`viewer/index.html`) shows all of it.

| Spec                     | FA-2                                                                                                                               | FA-1, same rules and prices |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Axes                     | 6: yaw, shoulder, elbow (parallelogram), forearm roll, wrist pitch, tool roll; plus the hexapod body's height, pitch, yaw and lean | 5                           |
| Reach to the gripper tip | 0.85 m from the shoulder axis                                                                                                      | 0.82 m                      |
| Payload                  | 1 kg on the ground patch; 0.25 kg continuous / 1.19 kg peak at full horizontal stretch                                             | 0.33 / 1.17 kg              |
| Arm mass                 | 2.50 kg without the tool, field kit included (budget 2.5 kg)                                                                       | 2.73 kg (over budget)       |
| Parts cost, delivered    | **$756**                                                                                                                           | $840 ($607 list)            |
| Backlash at the tool     | 4.4 mm open loop (root-sum-square); the wrist camera closes the rest                                                               | 5.7 mm                      |
| Speed                    | 0.48 s per weed when spot-dosing (simulated, about 2 mm)                                                                           | 0.48 s                      |
| Force sensing            | motor current at the yaw, shoulder and elbow (backdrivable QDD, total ratio ≤ 30)                                                  | same                        |
| Capability               | **90%** of weighted task points (best at any price with bought parts: 90.5%; with the custom board: 92.6%)                         | 87%                         |

## What changed from FA-1, and why

1. **Prices are delivered, which moved the baseline.**
   - Nearly every motor ships from China with US duty unpaid. Since July 2026 that is
     40.3%: MFN 2.8% + Section 301 25% + the new forced-labour tier 12.5%. De minimis is
     suspended.
   - Freight is priced as $6 + $25/kg per part, fitted to verified per-unit quotes.
   - FA-1's $607 parts list costs $793 delivered, and $840 with the $47 field kit every
     design now carries.
   - Buying the same parts from duty-paid resellers (OpenELAB, Zennixtek, Hiwonder's
     duty-paid shipping) brings it to $721.
   - Every comparison below uses delivered prices ([research/sourcing.md](research/sourcing.md)).
2. **Rope capstans multiply cheap actuators.**
   - **The stage:** a 2 mm Dyneema DM20 rope on a helical-grooved steel pinion and a
     smooth, flanged printed drum. It is 3:1 with no backlash and about 92% efficient.
     Parts cost $11, plus $6 for an absolute encoder at the joint. The actuator's own
     encoder turns three times over the joint's range, so it cannot give the joint angle
     at power-up.
   - **Shoulder:** a $132 Damiao DM-J4310 becomes 9.7 Nm continuous / 18 Nm peak (the
     rope's limit). FA-1 paid $205 for its 7.5 / 22 Nm shoulder.
   - **Elbow:** an $82 RobStride EduLite 05 becomes 5 / 16.6 Nm.
   - The parallelogram keeps both drives on the turret, so the stages add no moving mass.
   - **Limits** ([research/reductions.md](research/reductions.md)):
     - the drum must fit under the shoulder axis, so a 5 cm pedestal caps a 2 mm rope at
       5:1 and about 31 Nm;
     - a capstan cannot do the yaw's ±200°;
     - a belt yaw on a geared actuator exceeds 30:1 and stops being backdrivable.
3. **A sixth axis at the wrist.** With the capstans paying for the base, the budget
   stretched to a forearm-roll bus servo. It lets the tool come in from the side under the
   canopy and at latches. At FA-2's price the best 6-axis design serves 90.2% and the best
   5-axis 88.7%.
4. **The field kit is real mass and money.** It costs $57 and adds 0.45 kg: 0.35 kg on the
   turret and 0.10 kg at the wrist ([research/rugged.md](research/rugged.md)). It covers:
   - V-rings and labyrinths at the two shafts leaving the turret;
   - a drip skirt at the yaw;
   - sealed servo pods with metal pins;
   - a white ASA sun shield and an aluminium heat plate.

   With the kit, FA-1 weighs 2.73 kg, over the 2.5 kg budget. FA-2 fits with it included.

5. **Field grade got stricter**, so the study cannot buy coverage with something a field
   robot can't live with:
   - the tool roll must give 3 Nm to turn a valve;
   - the base joints must know their output angle at power-up;
   - the base joints must stay backdrivable;
   - the base joints must turn at 30 rpm or more;
   - each motor's rotor inertia is charged against its peak;
   - body poses the legs cannot hold are rejected;
   - the body may lean forward 10 cm over planted feet.

## Layout

```text
      tool roll ─┐   ┌─ wrist pitch (HX-35HM)
                 ▼   ▼
   gripper ◄── [flange]─[wrist]─(forearm roll, STS3250)══ forearm (Al tube) ══╗ elbow pivot
                                                                             ║ (bearings only)
          rod (carbon, parallel to the upper arm) ──────────────╮            ║ upper arm
                                                                │            ║ (Al tube)
   elbow drive: EduLite 05 ── 3:1 capstan ── crank ─────────────╯            ║
                                     ╲                                       ║
                                      ╲── turret box, shoulder axis 16 cm above the deck
                                           shoulder drive: DM-J4310 ── 3:1 capstan
                                     │
                           yaw motor (EduLite 05), recessed into the deck
```

- **Both base drives sit on the turret.**
  - Each drive motor hangs below and in front of the shoulder axis. Its pinion engages a
    drum on the joint: the upper-arm hub for the shoulder, the crank for the elbow.
  - The motors sit outside the upper arm's sweep and the crank's sweep over the joints'
    reachable range. A swept-collision check of the CAD finds no clash at any pose the
    tasks use.
  - The elbow pivot is just bearings, and no motor rides on the arm.
- **The parallelogram controls the forearm's absolute angle.** A force at the tool is
  split between the two drives instead of stacking on the shoulder. In the 60 N weed pull,
  the shoulder peaked at 96% of its peak torque with the arm alone, and at 67% with the
  body leaning in.
- **The turret is a closed printed box on a 16 cm pedestal**, between the cheeks and under
  everything that swings round the shoulder axis.
  - It stiffens the column and houses the rope drives and electronics.
  - The ropes must stay out of the sun, which can cost Dyneema 40% of its strength in
    1–1.5 years.
  - 1.1% of candidate poses would pass through the column, but every task point has a
    clear alternative.
- **No gravity springs.** At this price the best design without springs serves 90.2%, and
  the best with them 88.7%. Within 2.5 kg, the 0.16 kg of springs and pulleys bought more
  as motor than as balance.
- **The tool changer is a passive bayonet** locked by the tool roll. Six pogo pins carry
  24 V, CAN and two spare lines to powered tools.

## Actuators and price

| Joint        | Part                                 | At the joint, rated / peak Nm | Delivered       | Source                                    |
| ------------ | ------------------------------------ | ----------------------------- | --------------- | ----------------------------------------- |
| Yaw          | RobStride EduLite 05                 | 1.8 / 6                       | $82             | OpenELAB, duty paid                       |
| Shoulder     | Damiao DM-J4310-2EC V4 + 3:1 capstan | 9.7 / 18.4 (rope limit)       | $132 + $11 + $6 | Zennixtek, duty paid (place a test order) |
| Elbow        | RobStride EduLite 05 + 3:1 capstan   | 5.0 / 16.6                    | $82 + $11 + $6  | OpenELAB, duty paid                       |
| Forearm roll | Feetech STS3250 bus servo            | 1.6 / 4.9                     | $73             | OpenELAB, duty paid                       |
| Wrist pitch  | Hiwonder HX-35HM bus servo           | 1.2 / 3.4                     | $39             | Hiwonder, duty-paid USPS                  |
| Tool roll    | Feetech STS3250 bus servo            | 1.6 / 4.9                     | $73             | OpenELAB, duty paid                       |

| Group                                              | Delivered |
| -------------------------------------------------- | --------- |
| Actuators                                          | $480      |
| Reductions (two capstans and their joint encoders) | $34       |
| Structure (tube, prints, bearings, fasteners, rod) | $69       |
| Tool changer and gripper                           | $45       |
| Wrist camera and ToF                               | $43       |
| Harness, field kit, servo bus adapter              | $85       |
| **Total**                                          | **$756**  |

Everything at the base is a backdrivable QDD on CAN with an absolute encoder at the joint.
The total ratio stays at or below 30, so motor current still works as a force sensor. Rope
drives are compliant: about 0.6–1.0 kNm/rad once the printed structure is counted. The
joint encoders close the position loop on that compliance.

## How it was chosen

**Stage A** swept 1,764 geometries:

- 5 or 6 axes;
- a parallelogram or direct elbow;
- an upper arm and forearm of 0.22–0.40 m each;
- three mount positions and three pedestal heights.

It scored kinematic coverage of 1,034 task points over every body pose the legs can hold.
A point counts when an inverse-kinematics solution exists at an acceptable approach angle,
without hitting the body, the legs or the ground, and without tipping the robot.

**Stage B** took the 56 geometries on that frontier, with and without springs. For each,
it tried every assignment of parts under six lever settings:

- retail;
- each lever alone;
- bought parts (sourcing + reductions);
- all three.

At every task point it checks:

- static torque (gravity, tool, payload, task force) against the rated torque;
- a 4 m/s² dynamic bound plus the rotor's own inertia against the peak torque;
- the mass budget and the backlash budget.

Pruning drops only parts that another part beats on every count, a driver board included.
Task points that every combination treats alike are merged before scoring. Both steps are
exact for the joint being pruned. A brute-force check without pruning gave identical
frontiers.

### What each lever is worth

Best field-grade capability (percent of weighted task points served) by delivered parts
budget:

| Lever setting                          | Cheapest ≥ FA-1 | at $650 | at $750 | at FA-1's $840 | best anywhere |
| -------------------------------------- | --------------- | ------- | ------- | -------------- | ------------- |
| US retail, integrated actuators        | $851            | –       | 62%     | 84%            | 89.6%         |
| + duty-paid sourcing                   | $700            | 74%     | 87%     | 89%            | 89.6%         |
| + printed capstans                     | $794            | –       | 82%     | 87%            | 90.5%         |
| + custom driver board                  | $791            | 5%      | 79%     | 88%            | 92.6%         |
| **Bought parts (sourcing + capstans)** | **$645**        | **87%** | **89%** | **90%**        | 90.5%         |
| All three                              | $645            | 87%     | 89%     | 91%            | 92.6%         |

- **Sourcing and capstans compound.** Alone, sourcing moves the frontier $151 left and the
  capstans $57. Together they reach FA-1's capability for $645. The cheapest parts to buy
  duty-paid (EduLite 05, DM-J4310) are exactly the ones a 3:1 capstan makes strong enough.
- **The custom board only pays at the top.** Bare Steadywin motors on our own 3-axis FOC
  driver carry the frontier above about $800. But the board costs about $133 delivered and
  16–24 or more person-weeks of engineering, which 10 arms never repay. See
  [ELECTRONICS.md](ELECTRONICS.md); it is worth doing only if the hexapod's 18 leg joints
  share it.

### Three picks

|                     | FA-1                        | Lite                                            | **FA-2**                                         | Max                                                                             |
| ------------------- | --------------------------- | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| Rule                | –                           | cheapest that matches FA-1                      | where returns diminish                           | most capability at FA-1's price                                                 |
| Delivered cost      | $840                        | $645 (−23%)                                     | **$756 (−10%)**                                  | $828 (−1%)                                                                      |
| Task points served  | 86.7%                       | 86.6%                                           | **90.2%**                                        | 91.0%                                                                           |
| Axes, elbow         | 5, para + springs           | 5, para + springs                               | 6, para                                          | 6, para                                                                         |
| Mass / backlash     | 2.73 kg / 5.7 mm            | 2.46 kg / 4.0 mm                                | 2.50 kg / 4.4 mm                                 | 2.45 kg / 4.4 mm                                                                |
| Base drives         | EduLite / GIM8108 / GIM6010 | EduLite ×3, shoulder and elbow through capstans | EduLite / DM-J4310 + capstan / EduLite + capstan | bare GWK6010-8 ×2 on our board (shoulder through a capstan) / EduLite + capstan |
| Needs our own board | no                          | no                                              | no                                               | yes                                                                             |

- **FA-2's rule:** the cheapest design on the bought-parts frontier beyond which no upgrade
  buys 2 points of capability per $100. Up to $756 the frontier climbs 5–10 points per
  $100; the next $40 buys 0.3 points.
- **Lite and FA-2 come from bought parts only.** Lite matches FA-1 within half a point,
  the sampling noise of the score. It spends its savings differently, though: it serves no
  taproot points, which FA-1 managed at 21%.

### Where FA-2 gains and gives up against FA-1

| Task                  | FA-1 | FA-2 |
| --------------------- | ---- | ---- |
| Stab or stamp a weed  | 29%  | 58%  |
| Push in a soil probe  | 42%  | 58%  |
| Look into a canopy    | 91%  | 100% |
| Clip a leaf           | 95%  | 100% |
| Gate latch or button  | 94%  | 100% |
| Prune a stem          | 96%  | 100% |
| Fruit from a bush     | 92%  | 94%  |
| Rock, bottle or stick | 100% | 96%  |
| Turn a valve or knob  | 100% | 77%  |
| Pull a taprooted weed | 21%  | 21%  |

The other seven tasks are at 100% for both. FA-2 serves fewer valve points than FA-1, and
the limit is the yaw, not the tool roll:

- its tool roll (STS3250, 4.9 Nm peak) clears the 3 Nm floor;
- with the tool vertical, the valve's reaction also runs through the yaw, whose axis is
  vertical too;
- at FA-2's longer reach, the EduLite 05 yaw runs out of margin at 23% of valve poses.

Turn those valves by stepping the body closer. If valves matter more than the study
weights them, swap the yaw for a RobStride 05 or GIM6010-8.

## Simulation

Seven scenarios run in MuJoCo (`armdesign/sim.py`):

- the parallelogram is a real closed chain;
- every motor is torque-controlled and clipped at its peak (the capstan drives at their
  rope limit);
- the forearm roll holds level when a task doesn't need it;
- the legs are position servos, and soil, roots and stems are force laws;
- released fruit and rocks land in the deck tray.

The tests check that the optimizer's statics match MuJoCo within 3%.

| Scenario                                   | Highest motor load (of peak) | Smallest tip-over margin | At the targets              |
| ------------------------------------------ | ---------------------------- | ------------------------ | --------------------------- |
| Three strawberries into the deck tray      | 40% (shoulder)               | 29 cm                    | 1.8–3.2 mm                  |
| Tomato 0.8 m up, body raised and tilted    | 38% (yaw)                    | 27 cm                    | 1.6–5.9 mm                  |
| 60 N taproot, body leans in and legs lift  | 67% (shoulder)               | 12 cm                    | 2.7–5.5 mm                  |
| The same root, pulled by the arm alone     | 96% (shoulder)               | 19 cm                    | 1.6–2.9 mm                  |
| 1 kg rock at full reach, stowed on deck    | 54% (wrist pitch)            | 26 cm                    | 2.4–7.9 mm                  |
| Soil probe pressed 6 cm by the body (72 N) | 56% (shoulder)               | 14 cm                    | 2.1 mm                      |
| Eight weeds micro-dosed                    | 62% (yaw)                    | 26 cm                    | about 2 mm, 0.48 s per weed |

- **Heavy payloads swing slower.** With the rock in hand, the half-turn to the deck tray
  takes 1.6 s. At 1.0 s the yaw saturates and the rock lands 37 mm off.
- **Let the body do the pulling.** The arm-alone taproot pull reaches 96% of the
  shoulder's rope-limited peak. Leaning the body 12 cm over the weed first cuts it to 67%.
  The planner should always lean for pulls.

## Fleet fit

- **Hexapod legs.** The femur needs about 9.5 Nm continuous and 19 Nm peak in a tripod gait
  (12 kg robot, 2.5 kg arm). FA-2's shoulder drive, a DM-J4310 + 3:1 capstan, gives
  9.7 / 18.4 Nm. With a slightly larger drum, the same drive could be the femur. Then the
  whole robot would share one actuator, one capstan design and one spares box.
- **The custom board is a fleet decision.** At 18 leg joints per hexapod, it saves enough
  against off-the-shelf drivers to be worth building; for the arm alone it is not
  ([ELECTRONICS.md](ELECTRONICS.md)).
- **The tool flange is the fleet standard.** The same bayonet with pogo pins fits the
  wheeled carrier's arm, a drone's light gripper and a tool rack.
- **A wheeled-platform arm** is the same code with a different `data/hexapod.yaml` and link
  range. Rerun the study for it.

## Risks and next steps

- **Not built yet.** Bench-test one 3:1 capstan with an EduLite 05 first
  ([research/reductions.md](research/reductions.md), §8):
  - stiffness, backlash and backdrive torque;
  - 10^5 weed-pull cycles at 40–50 °C, then with grit.

  At about 35% of rope strength per peak, life is roughly 20–35k peak cycles. Keep peaks
  rare by leaning the body in, and re-tension after about 50 and about 1,000 cycles.

- **Backdrive through 27–30:1.** Each capstan multiplies its actuator's own friction by
  three. Measure the joint backdrive torque; force sensing needs it under about 1 Nm.
- **Heat.** RobStride rates continuous torque at 25 °C on a heat sink, so the EduLite yaw
  has little margin at 45 °C in a sealed box. The turret needs about 0.2 m² of shaded
  aluminium, or the motors must stay at low continuous torque. Log motor temperature in
  the first field trials.
- **Sealing is a plan.** The yaw is an open labyrinth, so expect about IPX3–4 at the turret
  and IPX5–6 at the wrist pods. Hose-test it.
- **Suppliers.**
  - Zennixtek (the DM-J4310) has no independent reviews: place a test order.
  - Steadywin is phasing out the GDS68 driver.
  - Tariffs can change after the US–China truce expires on 2026-11-10.
- **The wrist servos are not backdrivable.** The tool changer's magnets are the breakaway.
  The gripper's STS3215 needs a firmware torque limit near 0.4 Nm, where its gears are
  known to strip.
- **Spring variants' CAD is less checked.** The CAD and the collision sweep were done for
  FA-2, which has no springs. The spring mast now stands beyond the crank's sweep, but a
  spring design such as Lite still needs the same swept check.
