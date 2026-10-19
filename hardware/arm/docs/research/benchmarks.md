# Robot-arm benchmark for a hexapod-mounted outdoor arm (~0.6–0.7 m reach, ~1 kg payload)

Compiled 2026-10-16 from web sources (vendor pages, docs, papers, reseller listings). The CSV version of the main table is `benchmarks.csv` in the same folder.

## How to read this

- **Units.** Reach is the vendor's "reach" or "working radius" in mm. Payload is the vendor's rated (nominal) payload in kg. Mass is arm-only mass in kg. Repeatability is in mm. Prices are in USD.
- **Derived columns, for the cost-vs-performance chart:**
  - `kg*m` = payload x reach. This is a crude "useful moment" metric.
  - `$/kg*m` = price / (payload x reach). Lower is better.
  - `Payload/mass` = payload / arm mass. Higher is better, and it is critical on a legged base.
- **Price kinds differ and are labelled in each row.** They include parts BOM, kit, assembled, reseller and whole-robot prices. Currency conversions assume 1 EUR = 1.15 USD, 1 GBP = 1.33 USD, 1 JPY = 0.0067 USD and 1 DKK = 0.155 USD.
- **Payload definitions are not comparable across vendors:**
  - OpenArm and WidowX AI rate payload at **full extension, including the end-effector**.
  - Trossen X-series arms guarantee payload only to **50 % extension**.
  - Hobby kits often quote best-case "tucked" numbers.
  - Treat the hobby/kit payloads (Arctos, ArmPi, SO-101) as optimistic.
- **Missing data.** "n/d" means not published or not found. A "-" in a derived column means an input is missing.
- **Unverified items.** Items marked † come from prior knowledge and were not re-verified in this session.
- **Volatile prices.** Reseller prices vary 2–3x for the same arm (PiPER, for example, runs from $1,999 direct to $8k at some resellers). The table uses the lowest credible new price and names its source.

## Key takeaways for positioning

1. **The frontier to beat is the 2025–26 QDD "AI-data" arm cluster.** This covers reBot B601, I2RT YAM, AgileX PiPER, Galaxea A1/A1Z, OpenArm and Trossen WidowX AI.
   - **Specs:** $1.2k–6.5k for 0.6–0.8 m reach, 1.5–4 kg payload and 4–6 kg arm mass.
   - **Cost and mass ratios:** about **$1.2k–2.6k per kg\*m**, with payload/mass of 0.3–0.75.
   - **Common recipe:** nearly all of them use Damiao, RobStride or in-house low-ratio BLDC joint modules on CAN/CAN-FD, plus machined-aluminium core links and printed covers.
2. **Hobby stepper arms look cheap per kg\*m on paper but are poor for a legged robot.**
   - The cheap end includes Arctos at $377/kg*m and AR4 at $1.6k/kg*m.
   - They are heavy (AR4 12 kg, PAROL6 5.5 kg for only 0.4 m reach) and not backdrivable.
   - Their payload claims are optimistic, and a review reports backlash and open-loop issues on the AR4.
3. **Industrial-style harmonic-drive arms are the wrong mass class for a legged base.**
   - This group is Lite 6, xArm 5, Nova 2, Fairino FR3 and igus ReBeL.
   - They weigh 8–15 kg for 0.6–3 kg payload, at $3k–13k per kg*m.
4. **Legged-robot arms set the mass benchmark: payload/mass of about 0.5–1.0.**
   - DynaArm carries about its own mass (8–9 kg arm, 6–7 kg continuous) and is IP66.
   - Spot Arm is 8 kg, with 5 kg continuous at 0.5 m and 11 kg peak.
   - Unitree Z1 is 4.3 kg for 2–3 kg.
   - **Target for our arm:** 1 kg at 0.65 m (0.65 kg*m) with arm mass of 3 kg or less, at a BOM of about $1k (about $1.5k/kg*m). That would sit on or beyond the QDD frontier.
5. **Outdoor sealing is the open gap.**
   - **Arms with outdoor ratings:**
     - DynaArm is IP66.
     - Spot Arm is "water and dust resistant"; Spot itself is IP54.
     - Kinova Gen3 lite is IP22.
     - Unitree's B2 base is IP67, but no IP rating was found for the Z1 arm.
   - **Low-cost arms:** none of them has an IP rating, and two are only rated for operating temperature (PiPER and reBot, -20 to 50 °C).
   - Sealing plus a breakaway/torque-limited joint is where an outdoor design can differentiate.
6. **Ag lesson: specialised tools beat dexterity.**
   - Small-scale ag systems mostly avoid 6-DOF arms. They use:
     - a Cartesian gantry (FarmBot),
     - a 3-DOF delta (Small Robot Co. "Dick"),
     - rotary claw heads (Harvest CROO),
     - nozzle arrays (Ecorobotix),
     - string trimmers (Tertill),
     - passive hoes (Naio).
   - Where 6-DOF arms are used (Traptic, Fieldwork, Dogtooth), they are COTS or simple arms run in parallel and protected by enclosures or a structured crop geometry.

## Benchmark table

### Low-cost / hobby & open-source

| Arm                                    | DOF | Reach mm | Payload kg | Mass kg | Repeat. mm | Tip/joint speed | Price USD (kind)                                                                 | kg*m | $/kg*m | Payload/mass | Actuators                                                                        | Backdrivable / force                                            | IP   | Open                              | Notable trick                                                                                                                                                         | Src              |
| -------------------------------------- | --- | -------- | ---------- | ------- | ---------- | --------------- | -------------------------------------------------------------------------------- | ---- | ------ | ------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| LeRobot SO-101 (SO-100 similar)        | 5+G | 400      | 0.3        | 0.8     | n/d        | n/d             | 122 – parts BOM, follower only ($230 leader+follower)                            | 0.12 | 1,017  | 0.37         | 6x Feetech STS3215 7.4 V bus servos, 1/345 (follower); leader mixes 1/147, 1/191 | follower no / leader yes; no force sensing (servo current only) | none | yes                               | Leader uses low-ratio (1/147-1/191) servos so it backdrives for teleop; single servo SKU; printed links. Reach/payload are 3rd-party estimates (~0.5 kg max at tuck). | [1], [2], [3]    |
| Koch v1.1                              | 5+G | n/d      | n/d        | n/d     | n/d        | n/d             | 278 – parts BOM, follower ($199 leader)                                          | -    | -      | -            | 2x Dynamixel XL430-W250-T + 4x XL330-M288-T (leader 6x XL330-M077)               | partial (leader backdrivable); current only                     | none | yes                               | Stronger XL430 only on the gravity-loaded proximal joints, 18 g XL330 elsewhere: torque budget follows gravity load.                                                  | [4], [5]         |
| PAROL6 (Source Robotics)               | 6   | 400      | 1          | 5.5     | 0.2        | n/d             | 2,736 – kit EUR 2,379 (vendor); DIY files free                                   | 0.4  | 6,840  | 0.18         | Stepper motors + precision planetary gearboxes / belts                           | no / no                                                         | none | yes                               | Industrial-style printed arm: steppers + planetary + belts, pneumatic lines routed through arm, CAN + isolated I/O.                                                   | [6], [7]         |
| Annin AR4 (MK5 specs; MK3/MK4 similar) | 6   | 629      | 1.9        | 12.25   | 0.25       | n/d             | 1,888 – kit: combo $1,189 + motor/driver pack $699 (gripper +$139, covers +$300) | 1.2  | 1,580  | 0.16         | Geared stepper motors (StepperOnline) + belts, encoders                          | no / no                                                         | none | yes (free CAD + software)         | Machined-aluminium links + off-the-shelf geared steppers. MK3 (RoboDK): 1 kg / 600 mm / 0.2 mm / 10 kg; review found backlash and no closed loop in firmware.         | [8], [9], [10]   |
| BCN3D Moveo                            | 5+G | n/d      | n/d        | n/d     | n/d        | n/d             | 460 – DIY materials < EUR 400                                                    | -    | -      | -            | NEMA 17 steppers (one with 5:1 gearbox), belt reductions, Arduino Mega + RAMPS   | no / no                                                         | none | yes                               | Fully 3D-printed structure with belt-reduced steppers: cheapest structure, but low stiffness/accuracy; payload/reach never formally published.                        | [11], [12], [13] |
| Thor (AngelLM)                         | 6   | n/d      | 0.75       | n/d     | n/d        | n/d             | 400 – DIY hardware < EUR 350 (excl. printer)                                     | -    | -      | -            | 7x NEMA 17 steppers (some 5:1 geared), belts + printed gears                     | no / no                                                         | none | yes (CC-BY-SA-4.0)                | Semi-differential wrist (two small gears acting on a big one) puts both wrist motors in the forearm. 625 mm tall fully extended.                                      | [14], [15]       |
| Arctos (V2 open-loop kit)              | 6   | 600      | 2          | n/d     | n/d        | n/d             | 452 – non-printed parts kit (+ CAD licence EUR 40-50)                            | 1.2  | 377    | -            | Stepper motors + belt drives + 3D-printed cycloidal gearboxes, Arduino Mega/GRBL | no / no                                                         | none | partial (paid non-commercial CAD) | Printed cycloidal reducers after a belt stage: high ratio and low backlash from printed parts. 2 kg payload is a vendor claim.                                        | [16], [17]       |
| Hiwonder ArmPi FPV                     | 5+G | n/d      | 0.5        | n/d     | n/d        | n/d             | 460 – kit w/o Raspberry Pi ($576-754 with Pi)                                    | -    | -      | -            | Hiwonder serial bus servos                                                       | no / no                                                         | none | software yes                      | Eye-in-hand camera + bus servos; education-grade. 0.5 kg 'straightened' payload is a vendor claim.                                                                    | [18], [19]       |

### Research / AI-data-collection arms (incl. QDD)

| Arm                                 | DOF | Reach mm | Payload kg | Mass kg | Repeat. mm | Tip/joint speed                | Price USD (kind)                                                                                         | kg*m | $/kg*m | Payload/mass | Actuators                                                                                                                     | Backdrivable / force                                              | IP                       | Open                           | Notable trick                                                                                                                                                  | Src              |
| ----------------------------------- | --- | -------- | ---------- | ------- | ---------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- | ---- | ------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| OpenArm (Enactic) v1 / 2.0          | 7+G | 633      | 4.1        | 5.5     | n/d        | n/d                            | 6,500 – BOM headline on openarm.dev (scope per-arm vs bimanual unclear); WowRobo OpenArm 2.0 from $6,500 | 2.6  | 2,505  | 0.75         | Damiao QDD: 2x DM-J8009P-2EC (J1-2, 20/40 Nm), DM-J4340P + DM-J4340 (J3-4, 9/27 Nm), 4x DM-J4310-2EC (J5-7 + gripper, 3/7 Nm) | yes / yes (motor current; bilateral force feedback, gravity comp) | none                     | yes                            | QDD (~9-10:1 planetary) + software gravity comp; payload rated at full extension incl. EE (6 kg peak); Al/stainless + printed covers; 1 kHz CAN-FD.            | [20], [21]       |
| AgileX PiPER                        | 6+G | 626      | 1.5        | 4.2     | 0.1        | J1 180 deg/s                   | 1,999 – assembled (AgileX store; resellers $2.5-4k)                                                      | 0.94 | 2,129  | 0.36         | 6 integrated BLDC joint modules (reducer type unpublished), CAN, integrated controller                                        | partial (drag teaching) / current-based                           | n/d (rated -20 to 50 C)  | SDK/ROS only                   | Controller integrated in base; ~0.36 payload/mass at 626 mm for ~$2k.                                                                                          | [22], [23]       |
| Trossen WidowX 250 S (discontinued) | 6+G | 650      | 0.25       | n/d     | n/d        | n/d                            | 3,550 – assembled (sale price)                                                                           | 0.16 | 21,846 | -            | Dynamixel X-series smart servos (XM430 class)                                                                                 | partial (current mode) / current only                             | none                     | yes (software, CAD)            | Payload only guaranteed to 50% extension - a reminder that hobby-servo payload specs are not at full reach.                                                    | [24], [25]       |
| Trossen ViperX 300 S                | 6+G | 750      | 0.75       | n/d     | n/d        | n/d                            | 6,130 – assembled                                                                                        | 0.56 | 10,898 | -            | Dynamixel XM540-W270 (waist, 2x shoulder, 2x elbow, forearm roll, wrist) + XM430-W350 (wrist rot., gripper)                   | partial / current only                                            | none                     | yes (software, CAD)            | 'Shadow' servos: paired motors on shoulder and elbow double torque with one SKU (base of original ALOHA).                                                      | [26], [27]       |
| Trossen WidowX AI (Trossen AI kits) | 6+G | 769      | 1.5        | 4       | 1          | J0-2 360 deg/s; J3-5 540 deg/s | 2,995 – assembled base arm (2026 guide; store $5,935); Stationary AI kit $16-24k                         | 1.15 | 2,596  | 0.38         | BLDC joint modules (27 Nm J0-2, 7 Nm J3-5), 24 V, Ethernet/UDP                                                                | yes (leader/follower teleop) / current                            | n/d                      | software yes                   | Same arm SKU used as leader and follower; 1.5 kg rated at full extension.                                                                                      | [28], [29], [30] |
| Galaxea A1                          | 6+G | 600      | 3          | 4.2     | n/d        | n/d                            | 3,999 – assembled (reseller)                                                                             | 1.8  | 2,222  | 0.71         | Galaxea force-controlled joint modules                                                                                        | yes / yes (force-controlled)                                      | n/d                      | SDK                            | 3 kg rated / 5 kg peak at ~0.6 m with ~4 kg arm (payload/mass ~0.7); same arm family on Galaxea R1 (3.5 kg @ 0.6 m).                                           | [31], [32]       |
| Galaxea A1Z                         | 6+G | 626.5    | 3          | 5       | 0.1        | n/d                            | 2,999 – assembled (reseller)                                                                             | 1.88 | 1,596  | 0.6          | Galaxea force-controlled joint modules                                                                                        | yes / yes                                                         | n/d                      | SDK                            | Cheaper A1 variant: 3 kg rated, 5 kg peak.                                                                                                                     | [33]             |
| Berkeley Blue                       | 7   | n/d      | 2          | n/d     | 3.7        | 7.5 Hz position bandwidth      | 5,000 – target cost at >1,500 units (prototype BOM $3,328 in 77-unit run)                                | -    | -      | -            | iFlight gimbal outrunner BLDC + single-stage 7.125:1 GT3 timing belt; 2-DOF plastic geared differential per link              | yes / yes (proprioceptive current)                                | none                     | partial (software)             | Belt QDD + per-link differential moves motor mass toward shoulder (~30% less gravity torque and inertia); gripper reuses arm servo; base doubles as heat sink. | [34], [35]       |
| Pollen Reachy 2 arm                 | 7+G | n/d      | 3          | n/d     | n/d        | n/d                            | 70,000 – whole robot (3rd-party reports)                                                                 | -    | -      | -            | Orbita 2D (shoulder, elbow) + Orbita 3D (wrist) parallel/differential BLDC actuators                                          | n/d                                                               | none                     | yes (software)                 | Parallel 2-/3-DOF Orbita joints share motors across axes; internal cable routing. Arm ~662 mm long; docs list 0.5 kg vs datasheet 3 kg.                        | [36], [37], [38] |
| ARX X5 / ARX5                       | 6+G | 620      | 2          | 3.3     | n/d        | n/d                            | 8,333 – reseller, backorder (no official MSRP found)                                                     | 1.24 | 6,720  | 0.61         | ARX in-house BLDC joint modules, CAN (EtherCAT-CAN adapter), 500 Hz                                                           | yes / current                                                     | n/d                      | SDK (arx5-sdk)                 | Very low mass (~3.3 kg) force-controlled arm used for UMI-style imitation learning. Payload/reach are reseller 'comparable' figures.                           | [39], [40], [41] |
| I2RT YAM (Pro/Ultra/Big variants)   | 6+G | 780      | 2          | 4.29    | n/d        | n/d                            | 2,999 – assembled follower (reseller); $3,449 YAM w/ gripper                                             | 1.56 | 1,922  | 0.47         | Damiao DM-series QDD (DM4340 / DM4310; DM4310 or DM3507 gripper), CAN                                                         | yes / current                                                     | none                     | software yes, public CAD files | Motors drop to damping mode after 400 ms comms timeout (safe failure); Pro 3 kg, Ultra 4 kg, Big YAM 4 kg @ 1.1 m.                                             | [42], [43]       |
| Seeed reBot Arm B601-DM             | 6+G | 607      | 1.5        | 4.6     | 0.2        | n/d                            | 1,197 – DIY bundle ($1,499 assembled w/ gripper)                                                         | 0.91 | 1,315  | 0.33         | 3x DM-J4340P-2EC (J1-3, 27 Nm) + 3x DM-J4310-2EC (J4-6, 7 Nm) + 1x DM-J4310 gripper                                           | yes / current (MIT mode)                                          | none (rated -20 to 50 C) | yes                            | Cheapest QDD arm found: CNC 5052-Al core links + printed covers; 1.5 kg at 70% reach w/o gripper (767 mm with gripper).                                        | [44]             |
| Seeed reBot Arm B601-RS             | 6+G | 754      | 2.5        | 6.6     | 0.1        | n/d                            | 2,300 – reseller EUR 2,380 incl. VAT (variant labelling ambiguous)                                       | 1.89 | 1,220  | 0.38         | RobStride QDD: 4x RS00 + 3x RS06, 48 V                                                                                        | yes / current                                                     | none                     | yes                            | Same open design re-actuated with RobStride 48 V modules: 2.5 kg rated / 5 kg max.                                                                             | [45], [46]       |

### Commercial desktop arms & low-cost cobots

| Arm                       | DOF | Reach mm | Payload kg | Mass kg | Repeat. mm | Tip/joint speed  | Price USD (kind)                                     | kg*m | $/kg*m | Payload/mass | Actuators                                           | Backdrivable / force                                 | IP               | Open         | Notable trick                                                                                              | Src              |
| ------------------------- | --- | -------- | ---------- | ------- | ---------- | ---------------- | ---------------------------------------------------- | ---- | ------ | ------------ | --------------------------------------------------- | ---------------------------------------------------- | ---------------- | ------------ | ---------------------------------------------------------------------------------------------------------- | ---------------- |
| UFactory Lite 6           | 6   | 440      | 0.6        | 8       | 0.5        | 0.5 m/s          | 3,349 – assembled (UFactory, DDP)                    | 0.26 | 12,686 | 0.07         | BLDC + harmonic drive                               | no (manual/teach mode) / current collision detection | IP40 (3rd-party) | SDK only     | Harmonic drives + controller in base; precise but heavy for its payload (0.075 kg/kg).                     | [47], [48]       |
| UFactory xArm 5 Lite      | 5   | 700      | 3          | 11.3    | 0.1        | 1 m/s; 180 deg/s | 7,249 – assembled (Knox Labs)                        | 2.1  | 3,452  | 0.27         | BLDC servo + harmonic reducer, 17-bit abs. encoders | no / current collision detection                     | n/d              | SDK only     | 5-DOF (one wrist axis dropped) to save cost for top-down picking.                                          | [49], [50]       |
| Dobot Magician            | 4   | 320      | 0.5        | 3.4     | 0.2        | n/d              | 1,144 – assembled (GBP 860 ex VAT)                   | 0.16 | 7,150  | 0.15         | Steppers with planetary reducers, all in base       | no / no                                              | none             | no (SDK)     | Parallelogram linkages keep the tool level and put all motors at the base.                                 | [51], [52], [53] |
| Dobot Nova 2              | 6   | 625      | 2          | 11      | 0.05       | n/d              | 10,258 – assembled (EUR 8,920 ex VAT; guide ~$12.8k) | 1.25 | 8,206  | 0.18         | Integrated BLDC + harmonic joint modules            | no / current collision detection                     | n/d              | no           | Entry-level cobot; integrated joints.                                                                      | [54], [55], [56] |
| Elephant myCobot 280 (M5) | 6   | 280      | 0.25       | 0.86    | 0.5        | n/d              | 795 – assembled (JPY 118,580 incl. tax)              | 0.07 | 11,357 | 0.29         | Small serial bus servos                             | no / no                                              | none             | software yes | All-servo, no external controller (M5Stack/RPi in base).                                                   | [57], [58]       |
| Elephant myCobot 320      | 6   | 350      | 1          | 3       | 0.5        | n/d              | n/d – not found                                      | 0.35 | -      | 0.33         | Servo joints                                        | no / no                                              | none             | software yes |                                                                                                            | [59]             |
| Kinova Gen3 lite          | 6+G | 760      | 0.5        | 5.4     | n/d        | n/d              | 13,500 – assembled w/ gripper (US retail)            | 0.38 | 35,526 | 0.09         | Kinova integrated actuators                         | n/d                                                  | IP22             | SDK (Kortex) | Low power (20 W avg / 100 W peak) - attractive for battery bases.                                          | [60], [61]       |
| igus ReBeL 6DOF           | 6   | 664      | 2          | 8.2     | n/d        | n/d              | 7,499 – assembled (igus US list)                     | 1.33 | 5,647  | 0.24         | BLDC + igus polymer strain-wave gears †             | no / no                                              | n/d              | no           | Injection-moulded plastic gears/housings make it the lightest cobot class (8.2 kg); nominal reach 400 mm.  | [62], [63]       |
| Fairino FR3               | 6   | 622      | 3          | 15      | 0.02       | n/d              | 6,099 – assembled (reseller)                         | 1.87 | 3,268  | 0.2          | BLDC + harmonic integrated joints                   | no / current collision detection                     | n/d              | no           | Price floor for an industrial-spec 3 kg cobot (~$6k), but 15 kg mass - wrong mass class for a legged base. | [64], [65]       |

### Arms on legged / mobile robots

| Arm                                                | DOF                        | Reach mm | Payload kg | Mass kg | Repeat. mm | Tip/joint speed     | Price USD (kind)                                             | kg*m | $/kg*m | Payload/mass | Actuators                                                                      | Backdrivable / force                                                     | IP                                   | Open                | Notable trick                                                                                                                                                                 | Src                    |
| -------------------------------------------------- | -------------------------- | -------- | ---------- | ------- | ---------- | ------------------- | ------------------------------------------------------------ | ---- | ------ | ------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------ | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Boston Dynamics Spot Arm                           | 6+G                        | 984      | 5          | 8       | n/d        | n/d                 | n/d – not public (Spot+Arm ~ $140k reseller)                 | 4.92 | -      | 0.62         | BD custom electric actuators                                                   | n/d / yes (force & constrained-manipulation control)                     | 'water & dust resistant' (Spot IP54) | no                  | Whole-body manipulation: body re-poses to extend workspace and add leverage; 11 kg peak lift, 25 kg drag; 5 kg continuous at 0.5 m.                                           | [66], [67], [68], [69] |
| Duatic / Dyna-Tech DynaArm (ANYmal)                | 4 (2021 research ver.)     | 1000     | 6          | 9       | 1          | 10 m/s tip (vendor) | n/d – quote only                                             | 6    | -      | 0.67         | Duatic QDD actuators, 48 V, joint torque control                               | yes / yes                                                                | IP66                                 | software stack open | Elbow actuator mounted at the shoulder drives the elbow via a belt, so reaction torque goes to the base; carries ~its own mass (8 kg arm, 6-7 kg continuous; 14 kg for 10 s). | [70], [71], [72]       |
| Unitree Z1 Air (B1/B2 arm)                         | 6+G                        | 740      | 2          | 4.3     | 0.1        | 180 deg/s per joint | 6,600 – assembled (2022 launch price; resellers now $13-21k) | 1.48 | 4,459  | 0.47         | Unitree BLDC + harmonic reducer (60+:1), 33 Nm max, 15-bit enc., RS-485, 1 kHz | low (harmonic) / yes (joint torque control ~0.2 Nm, collision detection) | n/d                                  | SDK                 | Designed to bolt onto Unitree quadrupeds; accepts torque/angle/velocity/stiffness/damping commands.                                                                           | [73], [74], [75]       |
| Unitree Z1 Pro                                     | 6+G                        | 740      | 3          | 4.5     | 0.1        | 180 deg/s per joint | 13,388 – reseller (model ambiguous)                          | 2.22 | 6,031  | 0.67         | As Z1 Air                                                                      | low / yes                                                                | n/d                                  | SDK                 | Offered as B2 (IP67 quadruped, 40 kg walking payload) manipulation add-on.                                                                                                    | [73], [75]             |
| Unitree D1 (Go2 arm)                               | 6+G                        | 550      | 0.5        | 2.37    | n/d        | n/d                 | 5,400 – assembled (reseller)                                 | 0.28 | 19,636 | 0.21         | Servo joints, Al alloy body, 60 W                                              | n/d                                                                      | n/d                                  | SDK                 | Small arm sized for 15 kg-class Go2 EDU quadruped; 670 mm with gripper.                                                                                                       | [76], [77]             |
| ALMA (ANYmal + 6-DoF torque-controlled arm)        | 6                          | n/d      | n/d        | n/d     | n/d        | n/d                 | n/d – research system                                        | -    | -      | -            | Torque-controlled arm on torque-controlled quadruped                           | yes / yes                                                                | n/d                                  | no                  | Hierarchical whole-body controller: walk/trot/pace while doing operational-space EE control; torso posture optimisation enlarges the arm workspace.                           | [78], [79]             |
| Deep Whole-Body Control (Go1 + low-cost 6-DoF arm) | 6                          | n/d      | n/d        | n/d     | n/d        | n/d                 | 6,000 – whole system (quadruped + arm)                       | -    | -      | -            | Hobby-servo arm (WidowX-250 class, †) on Unitree Go1                           | partial / no                                                             | none                                 | yes (code)          | One RL policy coordinates legs + arm; legs bend/tilt to extend reach - the legged base is used as extra DOF at ~$6k system cost.                                              | [80]                   |
| Hello Robot Stretch 3                              | lift+telescope+3-DOF wrist | 520      | 2          | n/d     | n/d        | n/d                 | 24,950 – whole mobile manipulator (24.5 kg robot)            | 1.04 | 23,990 | -            | Closed-loop steppers, belt-driven lift and telescoping arm; servo wrist †      | partial / current-based contact detection                                | none                                 | software + some HW  | Cartesian lift + telescoping arm: all heavy motors on the base, tiny moving mass; base yaw supplies the missing DOF.                                                          | [81], [82]             |
| Enchanted Tools Mirokai (arm)                      | n/d                        | n/d      | 1.5        | n/d     | n/d        | n/d                 | 30,000 – whole robot (3rd-party, unverified)                 | -    | -      | -            | n/d                                                                            | n/d                                                                      | n/d                                  | no                  | Ball-bot base + light tray-carrying arms; 1.5-3 kg per arm reported - specs unverified.                                                                                       | [83], [84]             |

### Agricultural small-scale systems

| System                                 | Manipulator               | Price USD (kind)                            | Throughput / speed                | Actuation                                                 | IP                      | Open | Design trick / lesson                                                                                                             | Src          |
| -------------------------------------- | ------------------------- | ------------------------------------------- | --------------------------------- | --------------------------------------------------------- | ----------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| FarmBot Genesis XL (Cartesian gantry)  | 3 (X/Y/Z) + tool          | 4,500 – kit (v1.8 docs)                     | n/d                               | 4x NEMA 17 steppers with rotary encoders                  | outdoor use (no IP)     | yes  | Cartesian gantry over a 3 x 6 m bed with swappable tools via a magnetic universal tool mount †: trivial kinematics, cheap.        | [85], [86]   |
| Small Robot Company 'Dick' (weeder)    | 3 (delta)                 | n/d – n/a (service model)                   | n/d                               | igus delta robot carrying RootWave electric-weeding probe | n/d                     | no   | Cheap 3-DOF delta under the chassis for vertical tool strikes; electrical 'zapping' tolerates mm-level position error.            | [87], [88]   |
| Fieldwork Robotics raspberry harvester | multi-arm (4 planned)     | n/d – n/a                                   | ~2.2 lb/h per robot (measured)    | n/d                                                       | n/d                     | no   | Several cheap arms in parallel + soft gripper; target 25,000 berries/day.                                                         | [89], [90]   |
| Dogtooth strawberry harvester          | 2 arms                    | n/d – n/a                                   | ~12 s/berry (early); faster since | n/d                                                       | n/d                     | no   | Table-top crop rows constrain workspace so arms can be simple; ~1/100 berries missed (2023).                                      | [91], [92]   |
| Traptic strawberry harvester           | COTS arm(s)               | n/d – n/a                                   | claim 100k berries/day            | Off-the-shelf robot arm + custom gripper                  | enclosed cart           | no   | Arm in an enclosed cart behind a small tractor: controlled lighting, night picking, weather protection for a non-sealed COTS arm. | [93]         |
| Harvest CROO strawberry harvester      | 16 picking heads          | n/d – n/a                                   | 8 s per plant (goal: 4 s)         | Rotating 6-claw picking heads; delta robots for packing   | n/d                     | no   | Rotary multi-claw heads instead of articulated arms; parallelism (16 heads) for throughput.                                       | [94], [95]   |
| Saga Robotics Thorvald                 | no arm (modular platform) | n/d – service / n/a                         | n/d                               | Modular drive/steer modules                               | IP67 lower / IP56 upper | no   | Modular reconfigurable chassis; IP rating split by height (submersible parts IP67). UV-C treatment instead of manipulation.       | [96], [97]   |
| Naio Oz (weeding robot)                | no arm (tool bar)         | 30,000 – whole robot excl. tools            | ~2 km/h                           | n/a                                                       | n/d                     | no   | Passive mechanical hoes on a tool bar - no manipulator needed for inter-row weeding. Robot mass 130-180 kg.                       | [98], [99]   |
| Tertill (garden weeder)                | no arm                    | 349 – consumer retail                       | n/d                               | String trimmer + drive motors, solar                      | outdoor (no IP stated)  | no   | Replaces manipulation with a spinning string trimmer under the body; solar-powered; 1.1 kg robot.                                 | [100], [101] |
| Ecorobotix ARA (spot sprayer)          | no arm (nozzle array)     | 155,000 – ARA 620 ~DKK 1M (towed implement) | 7.2 km/h                          | 156 nozzles, 6 x 6 cm resolution                          | n/d                     | no   | Selective actuation of a dense nozzle array replaces a manipulator entirely.                                                      | [102], [103] |

## Design tricks that cut cost per unit of performance (relevant to an outdoor arm on a legged robot)

Each trick gives what it buys, then the evidence, then a note for our hexapod arm. Notes marked **[analysis]** are our engineering inference, not a source claim.

### A. Move mass and torque toward the base

1. **Put the elbow motor at the shoulder and drive the elbow by belt or linkage.**
   - DynaArm drives elbow flexion from an actuator at the shoulder through a belt. The elbow's reaction torque then goes straight into the base, unloading the shoulder joint ([Sleiman et al. 2021, arXiv:2103.00946](https://arxiv.org/abs/2103.00946)).
   - Berkeley Blue moved motor mass toward the shoulder with belt transmissions and per-link differentials. The authors estimate about 30 % less gravity torque and about 30 % less "flying inertia" ([Gealy et al. 2019, arXiv:1904.03815](https://arxiv.org/abs/1904.03815)).
   - **[analysis]** For 1 kg at 0.65 m with a ~2.5 kg arm, the worst-case horizontal shoulder torque is about 13–16 N·m. Roughly 25–35 % of that comes from distal motor and wrist mass, so relocating the elbow and wrist-pitch motors is the cheapest torque reduction available.
2. **Use a parallelogram or 4-bar linkage so heavy motors sit at the base and the tool stays level.**
   - This is the Dobot Magician and MG400 architecture ([Emerald 2024 kinematics paper on parallelogram arms, validated on the Magician](https://www.emerald.com/insight/content/doi/10.1108/IR-01-2024-0046/full/html)).
   - It works well for a 4–5 DOF "pick and place / tool-down" arm, and pairs well with a hexapod body that can supply roll and pitch.
3. **Use cable drives or tension amplification to keep all actuators in the shoulder.**
   - LIMS2-AMBIDEX puts all heavy actuators in the shoulder and body and drives the arm by wire ([NAVER Labs](https://naverlabs.com/en/blogDetail?seq=33674)).
   - The classic low-cost precedent is the Stanford 7-DOF arm: steppers in the base with a series-elastic belt transmission, 2 kg payload, sub-3 mm repeatability and about a $4.1k BOM ([Quigley, Asbeck & Ng, ICRA 2011](https://andrewng.org/publications/a-low-cost-compliant-7-dof-robotic-manipulator); [Hizook](https://hizook.com/highly-anticipated-icra-2011-talks-low-cost-arms-helicopter-manipulation-and-ball-ca/)).
   - **Cost:** wrist complexity and cable maintenance, which matter outdoors.
4. **Use a telescoping or prismatic arm on a lift, so all motors stay on the base.**
   - Hello Robot Stretch 3 uses this layout: 2 kg payload, 0.52 m reach, and the mobile base provides yaw ([Stretch 3 hardware guide](https://docs.hello-robot.com/0.3/hardware/hardware_guide_stretch_3)).
   - A good fit for "reach under a canopy / into a row" tasks.

### B. Get force sensing for free

5. **Use quasi-direct-drive (QDD) actuators from the legged-robot supply chain.**
   - Low gear ratios (about 6–10:1) give backdrivability, impact tolerance and proprioceptive force sensing from motor current, with no F/T sensor ([Wensing et al., IEEE T-RO 2017, MIT Cheetah actuators](https://dspace.mit.edu/handle/1721.1/119863)).
   - The 2025–26 low-cost arms that use them:
     - OpenArm: Damiao DM-J8009P / 4340P / 4310, 4.1 kg nominal payload, gravity comp, bilateral force feedback ([openarm.dev](https://openarm.dev/); [CNX](https://www.cnx-software.com/2026/09/30/openarm-2-0-an-open-source-7-dof-robot-arm-with-qdd-joints-bilateral-force-feedback-in-hand-camera/)).
     - reBot B601-DM: $1,197 DIY bundle with seven Damiao actuators ([CNX](https://www.cnx-software.com/2026/04/17/rebot-arm-b601-dm-an-open-source-61-dof-robotic-arm-for-embodied-ai-and-teleoperation-applications/)).
     - reBot B601-RS: RobStride RS00/RS06 ([OpenELAB](https://openelab.io/blogs/learn/seeed-rebot-b601-rs-robstride-motors)).
     - I2RT YAM: Damiao DM series ([i2rt GitHub](https://github.com/i2rt-robotics/i2rt)).
   - **[analysis]** If the hexapod legs already use Damiao or RobStride-class actuators, using the same family in the arm gives shared spares, one CAN-FD/MIT-mode driver stack and volume pricing.
6. **Size each joint to its gravity load and keep few motor SKUs.**
   - OpenArm uses its big DM-J8009P (20 N·m rated / 40 N·m peak) only on J1–J2, the 4340 class on J3–J4, and the 4310 class (3 N·m rated) on the wrist.
   - Koch v1.1 uses stronger XL430s only on the first two joints and 18 g XL330s elsewhere ([Koch v1.1](https://github.com/jess-moss/koch-v1-1)).
   - Trossen ViperX pairs "shadow" servos on the shoulder and elbow to double torque with one SKU ([Interbotix docs](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html)).
   - **[analysis]** A 4340P-class joint (9 N·m rated / 27 N·m peak) is fine for the elbow, at about 5–6 N·m worst case. At the shoulder it needs one of:
     - a 2:1 belt stage,
     - an 8009P-class motor,
     - a gravity spring,
     - avoiding sustained full horizontal extension by using body height.

### C. Counterbalance gravity, mostly in software

7. **Counterbalance gravity with springs, or in software.**
   - **Springs:**
     - Zero-length-spring balancing can cut gravity torque to under 5 % of an uncompensated arm ([KoreaScience 2000](https://koreascience.or.kr/article/JAKO200000618832738.page)).
     - Counterbalancing lets motors be sized for dynamic torque, cutting motor mass by 2x or more ([US 9,314,934 "Gravity-counterbalanced robot arm"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9314934)).
     - A compliant-beam version avoids expensive zero-length springs ([IEEE 7281250](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000007281250)).
   - **Software:** Blue and OpenArm use software gravity comp only (feed-forward torque).
   - **[analysis]** A spring balancer is exact only for a fixed gravity direction. On a walking hexapod with body pitch and roll on slopes, use a _partial_ spring (about 50–70 % of the static shoulder load) plus model-based software compensation that reads the body IMU.

### D. Share motors and simplify reductions

8. **Use differential or parallel joints so two motors share two DOF.**
   - Examples:
     - Blue: plastic geared differential per link.
     - Reachy 2: Orbita 2D/3D parallel actuators ([Pollen docs](https://docs.pollen-robotics.com/hardware-guide/specifications/general/)).
     - Thor: semi-differential wrist ([Thor GitHub](https://github.com/AngelLM/Thor)).
   - Both motors can push the same axis, so peak torque doubles for single-axis moves, and fewer unique parts are needed.
9. **Use cheap reductions where backdrivability is not needed.**
   - Options:
     - A single-stage timing belt (Blue: GT3, 7.125:1, above 95 % efficient, low backlash).
     - 3D-printed cycloidal gearboxes (Arctos; Faze4 uses printed 11/15/27:1 cycloidal stages) ([Faze4](https://github.com/PCrnjak/Faze4-Robotic-arm)).
     - Anti-backlash printed cycloids that trade play for friction ([U. Twente thesis 2024](https://essay.utwente.nl/fileshare/file/103595/103595_Volbeda_MA_EEMCS.pdf)).
     - Geared steppers (AR4, PAROL6).
   - **Downsides:** steppers are heavy, inefficient and not backdrivable, and open-loop stepper arms show backlash and drift ([Hackster AR4 review](https://www.hackster.io/news/is-the-annin-ar4-the-workhorse-robot-arm-for-you-6a26487fb517)).
   - Polymer strain-wave gears (igus ReBeL†) cut mass and cost compared with steel harmonics.
10. **Simplify the structure and electronics.**
    - Blue's cost tricks:
      - injection-moulded repeated shells,
      - commodity bike/3D-printer/drone parts,
      - a gripper built from the same servo module,
      - sensors co-located on each driver board,
      - the base used as a heat sink.
    - The reBot and OpenArm cost recipe is CNC aluminium core links plus printed covers.
    - PiPER and Lite 6 integrate the controller into the base.

### E. Use the legged base

11. **Use the legged base as extra DOF (whole-body loco-manipulation).**
    - **ALMA (ANYmal plus a 6-DOF arm):** walks, trots or paces while controlling the end-effector, and optimises torso posture to enlarge the arm workspace ([Bellicoso et al., ICRA 2019](https://research-collection.ethz.ch/handle/20.500.11850/384187)).
    - **Unified MPC on ANYmal plus DynaArm** ([arXiv:2103.00946](https://arxiv.org/abs/2103.00946)).
    - **Spot:** re-poses its whole body to extend reach and add leverage, and can drag 25 kg ([BD Spot Arm](https://bostondynamics.com/products/spot/arm/); [IEEE Spectrum](https://spectrum.ieee.org/boston-dynamics-spot-robot-arm)).
    - **Deep Whole-Body Control:** one RL policy runs a Go1 plus a low-cost arm, with legs bending to extend reach, for about $6k of hardware ([Fu, Cheng & Pathak, CoRL 2022](https://arxiv.org/abs/2210.10044)).
    - **Hexapod-specific:**
      - ALLOMAN (Beihang) uses leg-arm integration instead of adding an arm, and manipulates while walking in a pentapod gait ([TechXplore](https://techxplore.com/news/2022-04-alloman-hexapod-robot-multifunctional-platform.amp); [Springer FME 2022](https://link.springer.com/article/10.1007/s11465-021-0664-0)).
      - HeLoM learns whole-body hexapod loco-manipulation ([arXiv:2509.23651](https://arxiv.org/abs/2509.23651)).
    - **[analysis]** A hexapod standing on 3–6 legs can servo its body in 6-DOF over a useful range while remaining statically stable. That allows:
      - a **4–5 DOF arm**, using body roll or yaw instead of wrist roll or base yaw;
      - crouching to reach the ground instead of adding reach;
      - avoiding sustained horizontal arm poses, which shrinks the shoulder actuator;
      - using a front leg as a temporary second manipulator, for example to hold a branch.
12. **Cut DOF when the base supplies them.**
    - xArm 5 drops one wrist axis.
    - DynaArm on ANYmal was a 4-DOF arm.
    - Stretch gets yaw from its base.

### F. Survive outdoors

13. **Use a passive tool changer with a magnetic breakaway.**
    - A magnetic breakaway combined with a tool-changer interface (pneumatic and electrical) on a mobile manipulator protects the arm in collisions and reduces mass, cost and complexity ([US 11,904,458](https://patents.google.com/patent/US11904458)).
    - Racks of passive tools (brushes and similar) picked up as needed ([Berkeley CASE 2018 tool-changer paper](https://goldberg.berkeley.edu/pubs/CASE2018-ron-tool-changer-submitted.pdf)).
    - FarmBot's universal tool mount swaps seeder, watering and weeder tools on a cheap gantry† ([FarmBot Genesis](https://genesis.farm.bot/v1.8/assembly/intro/high-level-overview)).
    - **[analysis]** For land management (cutting, weeding, sampling, spraying), a breakaway tool interface is the cheapest protection against branch snags and robot falls.
14. **Seal the arm (IP rating) without killing backdrivability.**
    - DynaArm shows a QDD arm can be IP66 at 48 V ([Duatic flyer 2026](<https://www.hannovermesse.de/apollo/hannover_messe_2026/obs/Binary/A1508335/Duatic%20Flyer%202026%20(1)-compressed.pdf>)).
    - Thorvald splits its rating by height: IP67 for submersible lower parts, IP56 above ([Develop3D](https://develop3d.com/cad/saga-robotics-fields-of-the-future/)).
    - Traptic sidesteps sealing by putting a COTS arm in an enclosed cart, which also controls lighting for night picking ([TechCrunch](https://techcrunch.com/2019/10/02/traptic-uses-3d-vision-and-robotic-arms-to-harvest-ripe-strawberries)).
    - **[analysis]** Lip seals add friction that hurts current-based force sensing. Prefer labyrinth or V-ring seals on QDD output shafts, internal cable routing (Reachy and PAROL6 route lines internally), and a stow pose over the body that also keeps the CoM inside the support polygon while walking.
15. **Prefer task-specific end-effectors over dexterity.**
    - Small Robot Co. "Dick" uses an igus delta plus an electric weeding probe, which tolerates position error ([Agri-TechE](https://www.agri-tech-e.co.uk/small-robot-company-zaps-weeds-with-worlds-first-non-chemical-robotic-weeder/)).
    - Tertill uses a string trimmer ([Bob Vila](https://www.bobvila.com/articles/tertill-weeding-robot-review/)).
    - Ecorobotix uses 156 nozzles at 6x6 cm resolution ([Landbrugsavisen](https://landbrugsavisen.dk/traktortech/dobbelt-op-seks-meter-praecisionssproejte-faar-storebror-paa-12-meter-330569)).
    - Harvest CROO uses rotary claw heads ([Farm Show](https://www.farmshow.com/a_article.php?aid=34280)).
    - Multi-arm parallelism improves throughput (Fieldwork, CROO).
    - Relaxed repeatability (a few mm, Blue-class) is acceptable when the tool tolerates error, and it saves gearbox cost.
16. **Add a camera in the hand or on the elbow.**
    - Examples: OpenArm 2.0 in-hand RGB, DynaArm elbow RealSense ([arXiv:2405.07445](https://arxiv.org/abs/2405.07445)), ArmPi FPV.
    - This reduces the need for absolute accuracy, so cheaper, lower-stiffness arms work with visual servoing.

## Rough hexapod-arm design targets implied by the benchmark [analysis]

| Item            | Target                                                                                                                                       |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Reach / payload | 650 mm / 1.0 kg rated at full extension, incl. tool. That is 0.65 kg*m, the same class as reBot B601-DM (0.91 kg*m) and PiPER (0.94 kg*m).   |
| Arm mass        | 3.0 kg or less including wrist. That gives payload/mass of 0.33 or more; aim for 0.5, the legged benchmark (Z1 0.47, DynaArm about 0.7–0.9). |
| BOM             | Under $1,000 (under $1,540/kg*m), to sit at or beyond the reBot/YAM/PiPER frontier.                                                          |
| DOF             | 5 + gripper/tool, with body roll and yaw from the hexapod. Optionally 6.                                                                     |
| Actuation       | QDD (same family as legs) with an 8009P-class or belt-reduced shoulder, a 4340-class elbow driven from the shoulder, and a 4310-class wrist. |
| Protection      | IP54–IP65 target, breakaway tool interface, stow cradle on the body.                                                                         |

## Table references

Numbers in the `Src` column refer to this list. The CSV `source` column holds the same URLs.

1. https://github.com/TheRobotStudio/SO-ARM100
2. https://huggingface.co/docs/lerobot/so101
3. https://roboticscenter.ai/hardware/so-101/specs
4. https://github.com/jess-moss/koch-v1-1
5. https://github.com/AlexanderKoch-Koch/low_cost_robot
6. https://source-robotics.com/products/parol6-robotic-arm-kit
7. https://hackaday.io/project/191860-parol6-desktop-robotic-arm
8. https://www.anninrobotics.com/
9. https://robodk.com/robot/Annin-Robotics/AR4
10. https://www.hackster.io/news/is-the-annin-ar4-the-workhorse-robot-arm-for-you-6a26487fb517
11. https://github.com/NordThing/BCN3D-Moveo
12. https://www.createeducation.com/wp-content/uploads/2024/08/Use-Case-Education-Department-of-Catalonia-BCN3D-Technologies-1.pdf
13. https://forum.makerforums.info/t/we-have-released-the-the-bcn3d-moveo/51853
14. https://github.com/AngelLM/Thor
15. https://hackaday.io/project/12989-thor
16. https://www.robotshop.com/products/arctos-robotics-arctos-6-axis-diy-robotic-arm-hardware-v2-open-loop-kit-complete-non-printed-components-arduino-ros-ai-robotics-projects
17. https://blog.arduino.cc/?p=34636
18. https://www.robotshop.com/products/hiwonder-hiwonder-armpi-fpv-ai-vision-raspberry-pi-5-ros-robotic-arm-with-python-open-source-developer-kit-without-raspberry-pi
19. https://probots.co.in/hiwonder-armpi-fpv-ai-vision-raspberry-pi-5-ros-robotic-arm-with-python-open-source-standard-kit.html
20. https://openarm.dev/
21. https://www.cnx-software.com/2026/09/30/openarm-2-0-an-open-source-7-dof-robot-arm-with-qdd-joints-bilateral-force-feedback-in-hand-camera/
22. https://global.agilex.ai/products/piper
23. https://www.mybotshop.de/AgileX-Piper_6
24. https://www.trossenrobotics.com/widowx-250
25. https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250.html
26. https://trossenrobotics.com/viperx-300
27. https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html
28. https://docs.trossenrobotics.com/trossen_arm/main/specifications/wxai.html
29. https://store.trossenrobotics.com/products/widowx-ai
30. https://grabarobot.com/blog/trossen-robotics-price-guide-2026/
31. https://www.roboticscenter.ai/ar/store/product/galaxea-robotics-galaxea-a1
32. https://aiwiki.ai/wiki/galaxea_dynamics_r1
33. https://aifitlab.com/products/galaxea-a1z-robotic-arm
34. https://arxiv.org/abs/1904.03815
35. https://news.berkeley.edu/2019/04/09/meet-blue-the-low-cost-human-friendly-robot-designed-for-ai/
36. https://docs.pollen-robotics.com/hardware-guide/specifications/general/
37. https://pollen-robotics.com/reachy-2
38. https://www.foxnews.com/tech/can-70000-robot-transform-ai-research.amp
39. https://www.roboticscenter.ai/store/product/arx-arm
40. https://www.roboticscenter.ai/store/product/arx-arm-x5
41. https://github.com/real-stanford/arx5-sdk
42. https://github.com/i2rt-robotics/i2rt
43. https://www.roboticscenter.ai/store/product/i2rt-yam-6dof/
44. https://www.cnx-software.com/2026/04/17/rebot-arm-b601-dm-an-open-source-61-dof-robotic-arm-for-embodied-ai-and-teleoperation-applications/
45. https://openelab.io/blogs/learn/seeed-rebot-b601-rs-robstride-motors
46. https://www.mybotshop.de/Seeed-reBot-Arm-B601-RS-Assembled-Kit-with-Gripper-w-o-Power-supply_1
47. https://www.ufactory.cc/product-page/ufactory-lite-6/
48. https://encycam.com/component-library/ufactory-lite-6-li10xx
49. https://docs.xarm.ufactory.cc/8.technical_specifications.html
50. https://www.knoxlabs.com/products/ufactory-xarm-5-lite-collaborative-robot-arm
51. https://techpowerup.com/reviews/Dobot/Magician
52. https://www.rapidonline.com/70-0480
53. https://www.emerald.com/insight/content/doi/10.1108/IR-01-2024-0046/full/html
54. https://robodk.com/robot/Dobot/Nova-2
55. https://unchainedrobotics.de/en/products/roboter/cobot/dobot-nova-2
56. https://grabarobot.com/blog/dobot-robot-price-guide-2026/
57. https://qviro.com/product/elephant-robotics/mycobot-280-pi/specifications
58. https://www.switch-science.com/products/8974
59. https://qviro.com/product/elephant-robotics/mycobot-320-2022-pi/specifications
60. https://docs.clearpathrobotics.com/docs_robots/legacy/ros1_robots/accessories/manipulators/kinova_gen3_lite
61. https://www.robotshop.com/products/gen3-lite-6-dof-educational-professional-robot-arm-05kg-payload
62. https://www.igus.eu/robolink/rebel-cobot
63. https://www.robotshop.com/products/rbtx-igus-rebel-cobot-6-dof
64. https://top3dshop.com/product/fairino-fr3
65. https://robodk.com/robot/ko/FAIR-Innovation/INO-FR3
66. https://bostondynamics.com/wp-content/uploads/2020/10/spot-arm.pdf
67. https://bostondynamics.com/products/spot/arm/
68. https://spectrum.ieee.org/boston-dynamics-spot-robot-arm
69. https://www.roboticscenter.ai/ar/store/product/boston-dynamics-spot-arm
70. https://arxiv.org/abs/2103.00946
71. https://arxiv.org/abs/2405.07445
72. https://www.hannovermesse.de/apollo/hannover_messe_2026/obs/Binary/A1508335/Duatic%20Flyer%202026%20(1)-compressed.pdf
73. https://www.unitree.com/z1
74. https://spectrum.ieee.org/unitree-robot-arm
75. https://stemfinity.com/products/unitree-z1-robotic-arms
76. https://www.elektor.com/products/unitree-go2-d1-servo-robotic-arm
77. https://stemfinity.com/products/d1-servo-arm
78. https://research-collection.ethz.ch/handle/20.500.11850/384187
79. https://infoscience.epfl.ch/record/271845
80. https://arxiv.org/abs/2210.10044
81. https://docs.hello-robot.com/0.3/hardware/hardware_guide_stretch_3
82. https://www.therobotreport.com/stretch-3-mobile-manipulator-hello-robot-designed-open-source-development/
83. https://robotsguide.com/robots/mirokai
84. https://humanoid.guide/?p=3844
85. https://genesis.farm.bot/v1.8/assembly/intro/high-level-overview
86. https://farm.bot/pages/genesis
87. https://www.agri-tech-e.co.uk/small-robot-company-zaps-weeds-with-worlds-first-non-chemical-robotic-weeder/
88. https://cleantechnica.com/2021/04/29/meet-dick-the-weed-killing-robot-with-a-tesla-heart/
89. https://www.freethink.com/hard-tech/harvesting-robots
90. https://www.hortidaily.com/article/9426260/autonomous-robots-can-pick-up-to-25-000-raspberries-per-day/
91. https://theengineer.co.uk/red-harvest-robot-farmers-find-sweet-spot-for-strawberries
92. https://futurefarming.com/tech-in-focus/field-robots/strawberry-picking-robots-contribute-to-peace-of-mind-for-tasmanian-grower
93. https://techcrunch.com/2019/10/02/traptic-uses-3d-vision-and-robotic-arms-to-harvest-ripe-strawberries
94. https://www.farmshow.com/a_article.php?aid=34280
95. https://www.growingproduce.com/?p=134654
96. https://develop3d.com/cad/saga-robotics-fields-of-the-future/
97. https://futurefarming.com/tech-in-focus/field-robots/robot-combats-powdery-mildew-in-strawberries-using-uv-c-treatment
98. https://www.futurefarming.com/naio-oz-mechanical-weeding-robot-and-farmhand/
99. https://robotomated.com/explore/agricultural/naio-oz-v4
100. https://www.bobvila.com/articles/tertill-weeding-robot-review/
101. https://newatlas.com/tertill-garden-weeding-robot/49619
102. https://landbrugsavisen.dk/traktortech/dobbelt-op-seks-meter-praecisionssproejte-faar-storebror-paa-12-meter-330569
103. https://futurefarming.com/crop-solutions/weed-pest-control/ecorobotix-introduces-ultra-high-precision-sprayer-to-north-american-market
