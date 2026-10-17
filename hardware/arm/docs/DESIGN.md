# FA-2 design

FA-2 is a 6-axis arm with a parallelogram elbow. Its shoulder and elbow are cheap integrated
QDD actuators driving their joints through 3:1 Dyneema rope capstans on the turret. It rides
on the front of the medium hexapod's deck. It costs **$737 delivered to the US**: duty and
freight included, at about 10 units. That is 9% less than FA-1 at the same prices ($813),
and it serves 90% of the weighted task points against FA-1's 87%.

It came from a second optimization over the same 1,764 geometries:

- every price is landed (delivered with duty and freight) and checked against a live cart;
- three cost levers: duty-paid sourcing, printed reductions and a custom driver board;
- the requirements tightened, as below.

It was then run through seven MuJoCo task simulations. The interactive viewer
(`viewer/index.html`) shows all of it.

| Spec                     | FA-2                                                                                                                               | FA-1, same rules and prices |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Axes                     | 6: yaw, shoulder, elbow (parallelogram), forearm roll, wrist pitch, tool roll; plus the hexapod body's height, pitch, yaw and lean | 5                           |
| Reach to the gripper tip | 0.85 m from the shoulder axis                                                                                                      | 0.82 m                      |
| Payload                  | 1 kg on the ground patch; 0.25 kg continuous / 1.19 kg peak at full horizontal stretch                                             | 0.33 / 1.17 kg              |
| Arm mass                 | 2.50 kg without the tool, field kit included (budget 2.5 kg)                                                                       | 2.73 kg (over budget)       |
| Parts cost, delivered    | **$737**                                                                                                                           | $813 ($607 list)            |
| Backlash at the tool     | 4.4 mm open loop (root-sum-square); the wrist camera closes the rest                                                               | 5.7 mm                      |
| Speed                    | 0.48 s per weed when spot-dosing (simulated, about 2 mm)                                                                           | 0.48 s                      |
| Force sensing            | motor current at the yaw, shoulder and elbow (backdrivable QDD, total ratio ≤ 30)                                                  | same                        |
| Capability               | **90%** of weighted task points (best at any price with bought parts: 90.5%; with the custom board: 92.6%)                         | 87%                         |

## What changed from FA-1, and why

1. **Prices are landed, and that moved the baseline.** Nearly every motor ships from China
   with US duty unpaid. Since July 2026 that duty is 40.3%: MFN 2.8% + Section 301 25% + the
   new forced-labour tier 12.5%. De minimis is suspended. FA-1's $607 parts list costs
   **$813 delivered**. Buying the same parts from duty-paid resellers (OpenELAB, Zennixtek,
   Hiwonder's duty-paid shipping) brings it to $714. Every comparison below is on delivered
   prices ([research/sourcing.md](research/sourcing.md)).
2. **Rope capstans multiply cheap actuators.** A 2 mm Dyneema DM20 rope runs on a
   helical-grooved steel pinion and a smooth flanged printed drum. That makes a 3:1 stage
   with no backlash, about 92% efficiency and an $11 parts cost.
   - It turns a $132 Damiao DM-J4310 into a 9.7 Nm continuous / 18 Nm peak shoulder. The
     peak is the rope's limit.
   - It turns an $82 RobStride EduLite 05 into a 5 / 16.6 Nm elbow.
   - FA-1 paid $198 for its 7.5 / 22 Nm shoulder.
   - The parallelogram keeps both drives on the turret, so the stages add no moving mass.
   - The drum must clear the deck under the shoulder axis. On a 5 cm pedestal that limits
     a 2 mm rope to 5:1 and about 31 Nm.
   - A capstan cannot do the yaw's ±200°, so the yaw option is a belt round a fixed ring
     ([research/reductions.md](research/reductions.md)).
3. **A sixth axis at the wrist.** With the capstans paying for the base, the budget
   stretched to a forearm-roll bus servo. It lets the tool come in from the side under the
   canopy and at latches. At FA-2's price the best 6-axis design serves 90.2% and the best
   5-axis 88.7%.
4. **The field kit is real mass and money.** It is costed at $57
   ([research/rugged.md](research/rugged.md)):
   - V-rings and labyrinths at the two shafts leaving the turret, and a drip skirt at the
     yaw;
   - sealed servo pods with metal pins;
   - a white ASA sun shield and an aluminium heat plate.

   The kit adds 0.45 kg: 0.35 kg on the turret and 0.10 kg at the wrist. FA-1 was 2.28 kg
   without it; with it, it is 2.73 kg, over the 2.5 kg budget. FA-2 fits with the kit
   included.

5. **The requirements got stricter**, so the study cannot buy coverage with something a
   field robot can't live with:
   - the tool roll must give 3 Nm to turn a valve;
   - the base joints must know their output angle at power-up;
   - the base joints must turn at 30 rpm or more;
   - the motor's own rotor inertia is charged against its peak torque;
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

- **Both base drives sit on the turret.** The shoulder capstan turns the upper arm. The
  elbow capstan, on the other side of the same axis, turns a crank, and a rod carries the
  crank's angle to a lever behind the elbow. The elbow pivot is just bearings, and no motor
  rides on the arm.
- **The parallelogram controls the forearm's absolute angle.** A force at the tool is split
  between the two drives instead of stacking on the shoulder. In the 60 N weed pull, the
  shoulder peaked at 96% of its peak torque with the arm alone and 67% with the body leaning
  in.
- **The turret is a closed printed box** on a 16 cm pedestal. It stiffens the column and is
  the sealed, vented housing for the rope drives and electronics. Ropes must stay out of the
  sun, which costs Dyneema about 40% of its strength in 1–1.5 years.
- **No gravity springs.** At this price the best design without springs serves 90.2%, and
  the best with them 88.7%. Within 2.5 kg, the 0.16 kg of springs and pulleys bought more as
  motor than as balance.
- **The tool changer is a passive bayonet** locked by the tool roll. Six pogo pins carry
  24 V, CAN and two spare lines to powered tools.

## Actuators and price

| Joint        | Part                                 | At the joint, rated / peak Nm | Delivered  | Source                                    |
| ------------ | ------------------------------------ | ----------------------------- | ---------- | ----------------------------------------- |
| Yaw          | RobStride EduLite 05                 | 1.8 / 6                       | $82        | OpenELAB, duty paid                       |
| Shoulder     | Damiao DM-J4310-2EC V4 + 3:1 capstan | 9.7 / 18.4 (rope limit)       | $132 + $11 | Zennixtek, duty paid (place a test order) |
| Elbow        | RobStride EduLite 05 + 3:1 capstan   | 5.0 / 16.6                    | $82 + $11  | OpenELAB, duty paid                       |
| Forearm roll | Feetech STS3250 bus servo            | 1.6 / 4.9                     | $73        | OpenELAB, duty paid                       |
| Wrist pitch  | Hiwonder HX-35HM bus servo           | 1.2 / 3.4                     | $39        | Hiwonder, duty-paid USPS                  |
| Tool roll    | Feetech STS3250 bus servo            | 1.6 / 4.9                     | $73        | OpenELAB, duty paid                       |

| Group                                              | Delivered |
| -------------------------------------------------- | --------- |
| Actuators                                          | $480      |
| Reductions (two capstans)                          | $22       |
| Structure (tube, prints, bearings, fasteners, rod) | $69       |
| Tool changer and gripper                           | $38       |
| Wrist camera and ToF                               | $43       |
| Harness, field kit, servo bus adapter              | $85       |
| **Total**                                          | **$737**  |

Everything at the base is a backdrivable QDD on CAN with an absolute output encoder. The
capstans are inside the turret box, and the total ratio stays at or below 30, so motor
current still works as a force sensor. Rope drives are compliant: about 0.6–1.0 kNm/rad
once the printed structure is counted. The output encoders close the position loop on it.

## How it was chosen

**Stage A** swept 1,764 geometries:

- 5 or 6 axes;
- parallelogram or direct elbow;
- upper arm and forearm of 0.22–0.40 m;
- three mount positions and three pedestal heights.

It scored kinematic coverage of 1,034 task points over every body pose the legs can hold.
A point counts when an inverse-kinematics solution exists at an acceptable approach angle,
without hitting the body, the legs or the ground, and without tipping the robot.

**Stage B** took the 56 geometries on that frontier, with and without springs, and tried
every assignment of parts under six lever settings:

- retail;
- each lever alone;
- bought parts (sourcing + reductions);
- all three.

Each assignment is checked at every task point:

- **static torque** (gravity, tool, payload, task force) against the rated torque;
- **a 4 m/s² dynamic bound plus the rotor's own inertia** against the peak torque;
- the **mass and backlash budgets**.

The search is exact: pruning only drops parts that another part beats on every count.
Task points that every combination treats alike are merged before scoring.

### What each lever is worth

Best field-grade capability (percent of weighted task points served) by delivered parts
budget:

| Lever setting                          | Cheapest ≥ FA-1 | at $650 | at $750 | at $813 | best anywhere |
| -------------------------------------- | --------------- | ------- | ------- | ------- | ------------- |
| US retail, integrated actuators        | $823            | –       | 73%     | 84%     | 89.6%         |
| + duty-paid sourcing                   | $693            | 74%     | 87%     | 89%     | 89.6%         |
| + printed capstans                     | $757            | –       | 86%     | 87%     | 90.5%         |
| + custom driver board                  | $780            | 6%      | 79%     | 88%     | 92.6%         |
| **Bought parts (sourcing + capstans)** | **$626**        | **87%** | **90%** | **90%** | 90.5%         |
| All three                              | $626            | 87%     | 90%     | 91%     | 92.6%         |

- **Sourcing and capstans compound.** Alone, sourcing moves the frontier $130 left and the
  capstans $66. Together
  they reach FA-1's capability for $626, because the cheapest parts to buy duty-paid
  (EduLite 05, DM-J4310) are exactly the ones a 3:1 capstan makes strong enough.
- **The custom board only pays at the top.** Bare Steadywin motors on our own 3-axis FOC
  driver are the most capable per dollar above about $780. But the board costs about $133
  delivered and 16–24 or more person-weeks of engineering, which 10 arms never repay. See
  [ELECTRONICS.md](ELECTRONICS.md); it is worth doing only if the hexapod's 18 leg joints
  share it.

### Three picks

|                     | FA-1                        | Lite                                            | **FA-2**                                         | Max                                                  |
| ------------------- | --------------------------- | ----------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Rule                | –                           | cheapest that matches FA-1                      | where returns diminish                           | most capability at FA-1's price                      |
| Delivered cost      | $813                        | $626 (−23%)                                     | **$737 (−9%)**                                   | $810                                                 |
| Task points served  | 86.7%                       | 86.6%                                           | **90.2%**                                        | 91.3%                                                |
| Axes, elbow         | 5, para                     | 5, para + springs                               | 6, para                                          | 6, para                                              |
| Mass / backlash     | 2.73 kg / 5.7 mm            | 2.46 kg / 4.0 mm                                | 2.50 kg / 4.4 mm                                 | 2.47 kg / 4.4 mm                                     |
| Base drives         | EduLite / GIM8108 / GIM6010 | EduLite ×3, shoulder and elbow through capstans | EduLite / DM-J4310 + capstan / EduLite + capstan | bare GWK6010-8 ×3 on our board, two through capstans |
| Needs our own board | no                          | no                                              | no                                               | yes                                                  |

- Lite and FA-2 come from bought parts only.
- **FA-2's rule** is the cheapest design beyond which no upgrade buys 2 points of
  capability per $100. Up to $737 the frontier climbs 5–10 points per $100; the next $40
  buys 0.3 points.
- **Lite** matches FA-1 within half a point, the sampling noise of the score.
- **Max** is no worse than FA-1 at any task, which no bought-parts design under FA-1's
  price manages. Build it if the board gets built for the legs.

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

The other seven tasks are at 100% for both. FA-2 serves fewer valve points than FA-1. Its
tool roll (STS3250, 4.9 Nm peak) clears the 3 Nm floor. But with the tool vertical, the
valve's reaction also runs through the yaw, whose axis is vertical too, and at FA-2's
longer reach the EduLite 05 yaw runs out of margin at 23% of the valve poses. Turn those by
stepping the body closer, or swap the yaw for a RobStride 05 or GIM6010-8 if valves matter
more than the study weights them.

## Simulation

Seven scenarios run in MuJoCo (`armdesign/sim.py`):

- the parallelogram is a real closed chain;
- every motor is torque-controlled and clipped at its peak (the capstan drives at their
  rope limit);
- the forearm roll holds level when a task doesn't need it;
- the legs are position servos, and soil, roots and stems are force laws.

The tests check that the optimizer's statics match MuJoCo within 3%.

| Scenario                                   | Highest motor load (of peak) | Smallest tip-over margin | At the targets              |
| ------------------------------------------ | ---------------------------- | ------------------------ | --------------------------- |
| Three strawberries into the deck tray      | 40% (shoulder)               | 29 cm                    | 1.8–3.2 mm                  |
| Tomato 0.8 m up, body raised and tilted    | 38% (yaw)                    | 27 cm                    | 1.6–5.9 mm                  |
| 60 N taproot, body leans in and legs lift  | 67% (shoulder)               | 12 cm                    | 2.7–5.5 mm                  |
| The same root, pulled by the arm alone     | 96% (shoulder)               | 19 cm                    | 1.6–2.9 mm                  |
| 1 kg rock at full reach, stowed on deck    | 54% (wrist pitch)            | 26 cm                    | 2.3–7.9 mm                  |
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

- **Not built yet.** First, bench-test one 3:1 capstan with an EduLite 05
  ([research/reductions.md](research/reductions.md), §8):
  - stiffness, backlash and backdrive torque;
  - 10^5 weed-pull cycles at 40–50 °C, then with grit.

  Life at about 35% of rope strength per peak is about 20–35k peak cycles. Keep peaks rare
  by leaning the body in, and re-tension after about 50 and 1,000 cycles.

- **Backdrive through 27–30:1.** Each capstan multiplies its actuator's own friction by
  three. Measure the joint backdrive torque. Force sensing needs it under about 1 Nm.
- **Heat.** RobStride rates continuous torque at 25 °C on a heat sink, so the EduLite yaw
  has little margin at 45 °C in a sealed box. The turret needs about 0.2 m² of shaded
  aluminium, or the motors must stay at low continuous torque. Log motor temperature in
  the first field trials.
- **Sealing is a plan.** The yaw is an open labyrinth, so realistic protection is about
  IPX3–4 at the turret and IPX5–6 at the wrist pods. Hose-test it.
- **Suppliers.** Zennixtek (the DM-J4310) has no independent reviews: place a test order.
  Steadywin is phasing out the GDS68 driver. Tariffs can change after the US–China truce
  expires on 2026-11-10.
- **The wrist servos are not backdrivable.** The tool changer's magnets are the
  breakaway. The gripper's STS3215 needs a firmware torque limit near 0.4 Nm, where its
  gears are known to strip.
