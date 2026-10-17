# Printed and cheap reduction stages for the FA-1 turret motors

Research date: 2026-10-17. Scope: the reduction between a turret-mounted motor and the yaw, the
shoulder, and the parallelogram crank (elbow). Every number has a source link, or is marked
**estimate** with its reasoning. Prices are for about 10 arms.

FA-1 context used here (from `hardware/arm/docs/DESIGN.md`, `design/selected.yaml`, `armdesign/mjcf.py`):

- The shoulder axis sits **5 cm above the deck**, with l1 = 0.37 m and l2 = 0.28 m.
- Joint ranges in the simulation are yaw ±3.5 rad (±200°), shoulder −0.7…2.7 rad (195° span)
  and elbow −2.95…0.1 rad.
- Needs (continuous / peak) are yaw 1.5/5 Nm, shoulder 6/18 Nm, elbow 4/10 Nm.

---

## 1. Answer first

**Shoulder (20 Nm outdoor): use a single-stage Dyneema DM20 capstan at about 7-8:1, driven by a bare
81xx outrunner, with an output encoder at the joint.**

- It has zero backlash and about 95% efficiency both ways. Sandia measured 95-96% on a synthetic-rope
  capstan ([Mazumdar 2017](https://www.osti.gov/servlets/purl/1340266)).
- Estimated stiffness is 1.3-3 kNm/rad. That is as stiff as or stiffer than the best printed cycloid
  (1.27-1.47 kNm/rad, [Roozing 2024](https://files.energiazero.org/cartelle/meccanica/riduttori/2023_cycloidal_drive_nonpinwheel.pdf))
  and Blue's belt (1.3 kNm/rad, [Gealy 2019](https://ar5iv.labs.arxiv.org/html/1904.03815)).
- The reduction parts cost about $8-16 per joint.

| Option                | Rope             | Pinion core | Drum core | Ratio     | Peak strand tension (with pretension) | Fits the 5 cm axis?                                     |
| --------------------- | ---------------- | ----------- | --------- | --------- | ------------------------------------- | ------------------------------------------------------- |
| Fits today's envelope | 1.5 mm Dyneema   | 10.5-12 mm  | Ø96 mm    | 7.2-8.1:1 | ≈ 720 N, 32% of a 2.26 kN rope        | yes, about the same envelope as the GIM8108 it replaces |
| Longer life           | 2 mm DM20 (4 kN) | 16 mm       | Ø128 mm   | 7.2:1     | 13-16% of breaking load               | no: raise the axis to about 7 cm or recess the turret   |

Pretension is set to 0.75 × the torque-induced tension. The derivation is in Section 6.

**Motor pairing.** The capstan pairs with the cheapest torque per dollar found by the sibling electronics
research: the Steadywin **WK8115_1** rotor and stator kit at $37 (1.1 / 5.17 Nm motor-side, Ø90 × 19 mm;
see `custom_electronics.md` §1.2).

- At 7.2-8:1 and η 0.95 that gives about 7.5-8.4 Nm rated and 35-39 Nm peak at the output (**estimate**).
- The current part, a GIM8108-8, gives 7.5 / 22 Nm with backlash.
- The cheapest geared alternative, the GWK8108-8 core, costs $70 with 15 arcmin of backlash
  (`sourcing.md`).

**Elbow (parallelogram crank).** Use the same capstan at 7-9:1 with 1.1-1.5 mm rope on a Ø90-96 mm drum.
A GB6010 ($36, 0.8 / 1.59 Nm) gives about 6 / 12 Nm (**estimate**). Before committing, check that the
crank's absolute-angle range stays under about 270° (Section 6.2).

**Yaw.** There are two good options.

- **(a) Capstan around a fixed ring**, Ø140-150 mm, with 1-1.1 mm rope and an 8-9 mm pinion, at about
  15:1. This needs the yaw range cut to about ±150°.
- **(b) Closed timing belt** (2GT 6 mm or 3M 9 mm) around a fixed toothed ring, at 7-10:1. This keeps
  the simulation's ±200° yaw.
- A GB4310 ($22, 0.2 / 0.49 Nm) at 15:1 gives about 2.9 / 7 Nm (**estimate**).
- Either way the turret then needs its own thin-section or slewing bearing.

**Fallbacks.**

- **Two-stage belt (3:1 × 3:1, 3MGT 15 mm)** if a joint needs more than about 300° of rotation or the
  drum will not package.
- **Printed PA-CF cycloid (11:1, Roozing design)** if the capstan does not fit.
- **Reject printed strain-wave.** The best documented printable unit peaks at 3.5 Nm with 48-58%
  efficiency ([HD20](https://github.com/robrotics/hd20/releases)).
- **Reject steel cable at compact sizes.** Steel failed after 3,381 cycles at D/d = 8
  ([Mazumdar 2017](https://www.osti.gov/servlets/purl/1340266)). It broke within 5 hours on Aaed Musa's
  stand and failed under jumping loads on Stanley.

### 1.1 Corrections for the in-progress model (`data/transmissions.yaml`, `armdesign/cad.py`)

- **Capstan ratios.** The capstan is good to about 10:1 in a single stage; 12-16:1 works on a big yaw
  ring. The placeholder list stops at 6:1. Use about 7-8:1 for the shoulder and crank, so the motor
  stays quasi-direct-drive.
- **Drum size grows with the ratio.**
  - Drum diameter = ratio × (pinion core + d) − d.
  - The CAD's 2 mm cable on a Ø20 mm pinion gives a Ø160 mm drum at 8:1 by its radius × ratio rule,
    or Ø174 mm measured to the rope centreline. Either collides with a deck 5 cm below the axis.
  - Use 1.5 mm rope on a Ø10.5-12 mm pinion (D/d = 7-8), or raise the axis.
- **Drum grooves.** The CAD cuts concentric ring grooves at 1.15 d pitch on the drum.
  - As the pinion turns, its wraps walk along the axis by one pitch per turn. Ring grooves would force a
    growing fleet angle.
  - Make the drum smooth and flanged, or give it one helical groove with lead = N × pinion pitch. Aaed
    Musa did the latter, and it pushed his ratio to 8.55:1
    ([capstan page](https://www.aaedmusa.com/projects/capstandrive)).
  - Only the pinion needs a close-pitched helix.
- **Cable grade.** The CAD names SK78. Prefer DM20 or SK99. SK78 creeps about 85× faster than DM20 in one
  secondary report, so pretension would decay outdoors in the sun (Section 5).
- **Belt backlash.** The placeholder's 0.1° is optimistic for printed pulleys. HowToMechatronics measured
  25-45 arcmin (0.4-0.75°)
  ([test](https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/)).
- **Belt ratings.** HTD 5M 15 mm on a 24T pinion is rated about 8 Nm at the pinion (Gates, Section 3.3).
  At 3:1 that is about 24 Nm output for genuine Gates belts, and roughly half that for generic belts on
  printed pulleys (**estimate**).
- **Capstan mass.** About 0.045 kg + 0.0009 kg × N² (**estimate**: drum mass grows with R², and R grows
  with N). That is 0.06 kg at 4:1, 0.10 kg at 8:1 and 0.135 kg at 10:1. The drum can double as the link
  hub.
- **Capstan cost.** About $8 + $1 × N is fine (**estimate**, Section 6).
- **Capstan torque limit.** τ_max ≈ 0.2 × MBL × R_eff, which keeps the peak strand at 35% of MBL with
  pretension = 0.75 × the torque-induced tension. Examples: 1.5 mm rope at R = 49 mm gives 22 Nm; 2 mm DM20
  at R = 65 mm gives 52 Nm.

---

## 2. Comparison of stage types (for about 20 Nm output)

| Stage                                                     | Ratio (1 stage)                     | Fwd eff.                                                                                           | Back-drive eff.    | Backlash                                                                                  | Torsional stiffness                                                              | Added mass for 20 Nm                                   | Added cost (10 units)                                              | Life / outdoor                                                                                                                                    |
| --------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Capstan, Dyneema DM20/SK99 on printed PC/PET-CF drums** | 3-10 (12-16 on a yaw ring)          | 0.95-0.96 (Sandia, Vectran rope); ETH Capler-Leg mechanical losses ~4.5%                           | ≈ same (symmetric) | 0 while pretensioned                                                                      | **est. 1.1-3.1 kNm/rad** (Section 4.6); Capler-Leg calculated 1,614 Nm/rad       | **est. 0.08-0.14 kg**                                  | **est. $8-16**                                                     | 247k cycles (Vectran, Sandia); 500k hops (DM20, Capler). Dyneema working limit about 60-70 °C. Needs a dust cover and re-tensioning after run-in. |
| Capstan, stainless 7x19 cable on aluminium drums          | 2-6 (pinion ≥ 18-24 d)              | 0.95 (Sandia steel test)                                                                           | ≈ same             | 0                                                                                         | higher per mm of cable, but drums must be 2.25× larger                           | est. 0.25-0.4 kg                                       | est. $25-45 (machined drums)                                       | **3,381 cycles at D/d 8** (Sandia); broke in under 5 h (Aaed Musa). Corrosion: use 316.                                                           |
| Timing belt, 1 stage (GT3/2GT/HTD 3M-5M)                  | 2-7 (Blue 7.125, Doggo 3, Woofer 4) | >0.95 (Blue); up to 98% (Gates); 95-97% printed when tight (HowToMechatronics)                     | ≈ 0.93-0.95 (est.) | 0.4-0.75° on printed pulleys (25-45 arcmin measured); est. 0.1-0.25° on moulded aluminium | **1.3 kNm/rad** (Blue, 3MGT 15 mm, 16T→114T)                                     | est. 0.12-0.2 kg (output pulley must be a full circle) | est. $10-20                                                        | Rubber belt rated −25 to +100 °C. Keep out of sun, ozone and moisture (Gates storage note). Grit in the teeth causes tooth jump (est.).           |
| Timing belt, 2 stages                                     | 6-16                                | 0.95-0.97 measured on a printed 16:1 when tight; 0.70 when loose                                   | est. 0.9           | 0.4-0.6° measured (25-35 arcmin after tensioning; wear 25→35)                             | est. 0.5-1 kNm/rad (the final stage dominates)                                   | est. 0.2-0.35 kg                                       | est. $15-30 ($7.5 measured for a NEMA17-size unit)                 | As above; more tensioners to maintain.                                                                                                            |
| 3D-printed cycloid (PA6-CF, steel rollers)                | 8-25 (11 in Roozing)                | ~0.90 (BHL, PLA); 0.60-0.66 (PLA 15:1, HowToMechatronics); Coulomb friction 0.53-0.62 Nm (Roozing) | est. 0.75-0.85     | **0.20-0.25° typ., 0.35° max** new (PA-CF); doubled after 10 h; BHL PLA max 1.31°         | **1.27-1.47 kNm/rad** (PA-CF); 0.32 (BHL PLA); 0.63 (Roozing earlier PLA design) | **0.36-0.38 kg** (Ø104 × 23 mm, 36 Nm nominal)         | €65-98 BOM (Roozing, EU retail); est. $35-60 with Chinese bearings | Wear; PA-CF absorbs water; needs a sealed, greased housing outdoors. BHL: 60 h pendulum test OK.                                                  |
| 3D-printed planetary (nylon / PA-CF)                      | 4-8 (1 stage), 16+ (2 stages)       | 0.78-0.95 (HowToMechatronics, 16:1); 0.60-0.80 (COMPAct)                                           | est. 0.7-0.85      | 0.5-1° (30-60 arcmin); 0.59° single stage, 2.6° compound (COMPAct)                        | **0.20-0.24 kNm/rad** (COMPAct)                                                  | est. 0.3-0.5 kg                                        | est. $15-35 ($23 excluding prints for a NEMA17 16:1)               | Tooth wear and creep; same outdoor issues as the cycloid. OpenTorque 6:1 nylon claims 28 Nm continuous (unverified).                              |
| 3D-printed strain-wave (PETG/TPU flexspline)              | 20-50                               | 0.48-0.58 (HD20); ≈ 0.65 (HowToMechatronics)                                                       | poor, est. 0.3     | ≈ 0 at light load, **2.7-3.1° under load**                                                | very low (no measured data; est. < 0.1 kNm/rad)                                  | n/a: 20 Nm not demonstrated                            | est. $5-20                                                         | Flexspline breaks and takes a permanent set overnight. **Reject.**                                                                                |

Sources for the table:

- Sandia synthetic-rope capstan: [Mazumdar 2017](https://www.osti.gov/servlets/purl/1340266)
- ETH Capler-Leg: [Hwangbo 2018](https://ar5iv.labs.arxiv.org/html/1806.10632)
- Blue arm: [Gealy 2019](https://ar5iv.labs.arxiv.org/html/1904.03815)
- Stanford Doggo: [Kau 2019](https://arxiv.org/pdf/1905.04254)
- Stanford Woofer: [Dynamic Walking 2020](https://www.seas.upenn.edu/~posa/DynamicWalking2020/695-1037-1-RV.pdf) (search snippet only)
- Gates: [design manual E2/20099](https://chiaravalli.com/assets/d51d3c7e-4c6c-450d-a27e-9716381d3365)
- HowToMechatronics: [belt vs planetary vs cycloid](https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/) and [harmonic vs cycloid](https://howtomechatronics.com/how-it-works/harmonic-vs-cycloidal-drive-designing-3d-printing-testing)
- Roozing: [ICRA 2024 PDF](https://files.energiazero.org/cartelle/meccanica/riduttori/2023_cycloidal_drive_nonpinwheel.pdf)
- Berkeley Humanoid Lite: [arXiv 2504.17249](https://arxiv.org/html/2504.17249)
- COMPAct: [arXiv 2510.07197](https://arxiv.org/abs/2510.07197)
- HD20: [robrotics/hd20](https://github.com/robrotics/hd20/releases)
- OpenTorque: [RobotShop forum](https://community.robotshop.com/forum/t/3d-printed-opentorque-actuator) (search snippet only)
- Aaed Musa: [capstan page](https://www.aaedmusa.com/projects/capstandrive)
- Stanley: [Hackaday.io](https://hackaday.io/project/176726-stanley-the-capstan-based-quadruped-kit)

---

## 3. Evidence by stage type

### 3.1 Capstan with synthetic rope

**Sandia, "Synthetic Fiber Capstan Drives…", IEEE RA-L 2017** ([OSTI PDF](https://www.osti.gov/servlets/purl/1340266)).
This is the best primary source for design rules.

Configuration:

- Single-stage 5-10:1 drives on the WANDERER and STEPPR bipeds.
- 150-200 Nm output, using 3 mm New England Ropes V12 Vectran (9,400 N break).
- 25.4 mm input sheave and two ropes wrapped in a figure-eight.

Measured performance:

- 95% efficiency unlubricated and 96% lubricated.
- 28 Hz torque bandwidth and about 60 dBA.
- Survived 247,614 cycles on a realistic gait trajectory.
- Rope stretch reached up to 0.1 rad at the output, so they measure output position.

Design rules taken from the paper:

- Sheave diameter Ds = 8 dr (the rope maker's minimum). Pulley diameter Dp = N × Ds.
- Rope tension Tr = 2τ/Dp + Tpre, with Tpre taken equal to the torque-induced tension. Peak rope load
  was 42% of break.
- Termination load F_term = T_torque / e^(μψ) + Tpre, with μ = 0.3 for Vectran on aluminium and ψ ≥ 2π
  on the sheave and π on the pulley.
- The termination is the weak link. Epoxy-potted ferrules failed at 2,982 N, 32% of rope strength.
- Wear-in takes 10-50 cycles. Re-tension after it, and occasionally up to about 1,000 cycles.
- Aggressive pretension raised life from 22,000 to 32,500 cycles.
- 20% less peak load gave about 3× the life.
- Krytox grease added about 20% life. Grease near the terminations cut the capstan friction and caused a
  termination failure at 21,000 cycles.
- A jacketed composite rope could not be tensioned and failed at 2,000 cycles.
- Unjacketed 4 mm Zylon reached 113k cycles but loses more than 70% of its strength after 144 h of UV.
- Belt comparison: a 6:1, 150 Nm belt needs a 40 mm wide belt, and the output pulley must be a full circle.
- Mass comparison for 150 Nm: the rope drive weighed 0.46 kg against 1.0 kg for spur gears and 1.2 kg for
  a planetary.

**ETH Capler-Leg, "Cable-Driven Actuation for Highly Dynamic Robotic Systems", IROS 2018**
([ar5iv](https://ar5iv.labs.arxiv.org/html/1806.10632)).

- Uses Dyneema DM20, chosen over steel, SK70/SK95, Vectran, Kevlar and Zylon.
- Ratios: hip 5:1, knee 5.33:1. The knee uses a 3 mm cable for stiffness.
- Calculated stiffness is 1,614 Nm/rad, with about 2° deflection at maximum torque.
- DM20 crept less than 0.3% in 100 days at 70 °C.
- The cable survived 500,000 hop cycles.
- Lessons:
  - Knots are unreliable, and clamps are weak because Dyneema's friction coefficient is only about
    0.05-0.08. They used multiple eye-splices on pins.
  - Use UV-resistant, unjacketed cable.
  - Stay below 80 °C.
  - Wind the two cables as helices so cable length stays constant.

**Aaed Musa capstan test stand and CARA quadruped**
([capstan page](https://www.aaedmusa.com/projects/capstandrive), [CARA](https://www.aaedmusa.com/projects/cara),
[GitHub](https://github.com/aaedmusa/Capstan-Drive), [BOM](https://raw.githubusercontent.com/aaedmusa/Capstan-Drive/main/Capstan%20Drive%20Test%20Stand%20BOM.xlsx),
[The Amp Hour #712](https://theamphour.com/712-robots-everywhere-with-aaed-musa/)).

Test stand:

- PLA, 8.55:1, 120° of travel, 852 g.
- Eaglepower 8308 90KV motor and ODrive S1.
- Helical grooves on both drums and a lead-screw tensioner.
- Stainless wire broke after about 4-5 h, three times; he blamed D/d, not strength.
- He chose DM20 over Vectran after a two-week endurance test.

CARA quadruped:

- 8:1 drives with 2 mm DM20.
- First builds measured 7.912 and 8.213:1. Designing to the rope centreline ("effective diameter") gave
  8.0006:1.
- Small drums are PET-CF at 100% infill; the rest is PC.
- CARA 2.0 uses 1 mm DM20, at about $1,300 per robot. He gives about 300° as the practical capstan travel.

BOM prices (US retail):

- 8308 motor $61.
- 2 mm DM20 $26.49 per 100-unit spool. The unit is not stated; at 100 ft that is about $0.87/m
  (**estimate**).
- 40×50×6 bearings $8.59 each.
- Mechanical parts excluding motor, ODrive and magnet total about $31, of which bearings are about $21.

**Stanley capstan quadruped kit** ([Hackaday.io](https://hackaday.io/project/176726-stanley-the-capstan-based-quadruped-kit)).

- Stainless cables broke repeatedly once the robot jumped, so the builds use custom DM20 rope.
- PETG small drums failed. Nylon and its CF/GF blends wore grooves.
- **Prusa PC Blend** drums show no breakage or significant wear.

**U. Toronto MASc, Ridgway 2023** ([PDF](https://www.dynsyslab.org/wp-content/papercite-data/theses/MASc-Ridgway-2023.pdf)).

- A 31:1 single-stage capstan with a 1-2 mm radius pinion and 8-strand Spectra fishing braid (8 lb,
  0.15 mm, flattening to about 0.35 mm wide).
- Rules from it:
  - Wrap the rope in a helix with a gap between turns.
  - Wraps walk one cable width per pinion turn, so a smaller pinion needs a wider drum.
  - A stiff spring in the tensioner absorbs creep.
  - The test failed when the capstan slipped on the motor shaft: key or clamp the pinion.

**Haptics and Barrett WAM (steel and tungsten cable)**

- SensAble ([US6985133](https://patents.google.com/patent/US6985133)):
  - Steel cables need D/d ≈ 15-50 and "fail rapidly at … about 15 or lower".
  - Stainless frays after about 750,000 cycles. Tungsten 0.014" on a 0.175" capstan (12.5:1) showed no
    wear after 5.5M cycles.
  - Cable tension is about 10 lbf.
- Barrett/Townsend ([US8858374](https://patents.google.com/patent/US8858374)):
  - Pretensioned pairs are twice as stiff as non-pretensioned ones.
  - Pretension must be at least half the maximum operating tension.
  - μ = 0.2-0.5 for stainless on metal.
  - Pretension spreads poorly past wraps, so tensioning is iterative.

**MIT Cheetah.** I found **no** capstan drive on any MIT Cheetah.

- Cheetah 1 used Kevlar tendons as a passive structure, with coaxial planetary actuators
  ([NBC](https://www.nbcnews.com/news/amp/wbna51105575), [Spectrum](https://spectrum.ieee.org/mit-cheetah-robot-running)).
- Cheetah 3's knee uses a roller chain with an extra 1.15× reduction
  ([MIT DSpace](https://dspace.mit.edu/bitstream/1721.1/126619/2/IROS.pdf); rate-limited, seen as a search snippet).
- Mini-Cheetah-style legs drive the knee through a 14/9 timing belt
  ([arXiv 2210.03713](https://arxiv.org/pdf/2210.03713); [Katz thesis](https://dspace.mit.edu/handle/1721.1/118671)).
- The closest well-documented capstan leg is ETH's Capler-Leg.

**Stanford Doggo and Pupper.**

- Doggo uses two 3:1 belt drives per leg (16T→48T, 3D-printed pulleys, T-Motor MN5212): 1.51 Nm continuous
  and 4.8 Nm peak at the output ([Kau 2019](https://arxiv.org/pdf/1905.04254)).
- Woofer uses 4:1 belts
  ([Dynamic Walking 2020](https://www.seas.upenn.edu/~posa/DynamicWalking2020/695-1037-1-RV.pdf); search snippet).
- Pupper uses servos or geared hub motors (from memory, not checked). It has no printed reduction worth
  copying.

### 3.2 Capstan with steel cable

- Wire-rope makers set minimum tread diameters for long life:
  - 7x19: 24:1 desirable, 18:1 critical.
  - 7x7: 42:1 desirable, 28:1 critical.
  - Source: [Loos](https://loosco.com/?p=1283) chart as summarised in search. In the
    [Lexco chart PDF](https://www.lexcocable.com/resources/downloads/wire-rope-guidelines/), 1/16" 7x19
    needs a 1.5" tread and 1/16" 7x7 needs 2-5/8".
- At those ratios a 20 Nm, 8:1 steel capstan needs a Ø29-38 mm pinion and a Ø230-300 mm drum. That is far
  too large for the turret.
- Undersized steel fails fast. Sandia's 3.2 mm steel at 8:1 lasted 3,381 cycles against Vectran's 98-247k
  ([Mazumdar 2017](https://www.osti.gov/servlets/purl/1340266)). Aaed Musa and Stanley saw the same.
- Use steel only where D/d ≥ 24 fits easily: for example a large yaw ring with a Ø25-40 mm pinion and a
  very low ratio.

### 3.3 Timing belts (HTD / GT2 / GT3) on printed or aluminium pulleys

Rated torque at the small pulley, derived from Gates power tables (P/ω) in the
[Gates design manual E2/20099](https://chiaravalli.com/assets/d51d3c7e-4c6c-450d-a27e-9716381d3365).
These apply to genuine Gates belts on proper pulleys. Derate for generic belts and printed pulleys,
perhaps to 50-70% (**estimate**).

| Belt (width)          | Small pulley     | Rated torque @ 20 rpm | @ 100 rpm |
| --------------------- | ---------------- | --------------------- | --------- |
| 3MGT (GT3) 15 mm      | 16T (PD 15.3 mm) | 6.3 Nm                | 4.9 Nm    |
| 3MGT 15 mm            | 22T (PD 21.0 mm) | 9.0 Nm                | -         |
| 5MGT 15 mm            | 24T (PD 38.2 mm) | 17.1 Nm               | 14.6 Nm   |
| HTD 5M 15 mm          | 24T (PD 38.2 mm) | 8.1 Nm                | 7.9 Nm    |
| 2MGT (GT2-class) 6 mm | 20T (PD 12.7 mm) | ≈ 1.5 Nm @ 10 rpm     | 1.1 Nm    |

- Ratings assume at least 6 teeth in mesh. Width factors: 15 mm = 1.89 × 9 mm, 6 mm = 0.62 × 9 mm.
- Gates also gives:
  - efficiency up to 98%
  - −25 to +100 °C operation
  - misalignment ≤ 0.25°
  - flanges required
  - store away from moisture, sunlight and ozone.

Real data:

- **Blue arm** ([ar5iv](https://ar5iv.labs.arxiv.org/html/1904.03815)):
  - GT3, 3 mm pitch, 15 mm wide with fiberglass cords, 16T→114T (7.125:1).
  - Efficiency above 95% and stiffness 1.3 kNm/rad.
  - The shoulder makes about 20 Nm continuous and 35 Nm in bursts; that is 367-643 N of belt tension on
    the 109 mm pulley (my arithmetic).
  - Back-driving torque is 0.89 Nm per stage, of which 0.47 Nm is cogging.
  - Belt skips are listed as future work, and belts cut the motors' heat path.
- **HowToMechatronics printed GT2 two-stage 16:1**
  ([link](https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/)):
  95-97% efficient when tight and 70% when loose (it skipped). Backlash was 25-45 arcmin, and after about
  10 h it went from 25 to 35 arcmin. Cost $7.5.
- **Packaging:** a single-stage 7:1 needs a Ø110 mm full-circle output pulley; a two-stage needs about
  Ø65 mm. Only the two-stage fits under the 5 cm axis.
- **Cheap belts:**
  - Generic HTD-5M 15 mm closed belts from a Shenzhen seller cost about £2.93
    ([eBay](https://www.ebay.de/itm/286585507239)).
  - ContiTech 300-5M-15 is $3.83 and Optibelt $6.57
    ([Acorn](https://acorn-ind.co.uk/p/contitech/metric-timing-belts/300-5m-15-contitech),
    [Acorn](https://www.acorn-ind.co.uk/p/optibelt/metric-timing-belts/300-5m-15-optibelt/)).
  - A GT2 6 mm reseller lists 516 N breaking strength and suggests a working load of 1/15 of that
    ([minibearings](https://distributors.minibearings.com.au/collections/belts-timing-2-000mm-pitch-gt-clean-6mm-wide)).
    That is much more conservative than the Gates tables.

### 3.4 3D-printed cycloidal

- **Roozing & Roozing, ICRA 2024**, pinwheel vs non-pinwheel
  ([PDF](https://files.energiazero.org/cartelle/meccanica/riduttori/2023_cycloidal_drive_nonpinwheel.pdf),
  [DBLP](https://dblp1.uni-trier.de/pid/158/9262.html)):
  - 11:1 in PA6-CF (Fibertree F3 PA-CF Pro) with off-the-shelf steel parts, printed on a Prusa MK3S+.
  - T-Motor Antigravity 8012 gives 36.4 Nm nominal at 40 A.
  - Gearbox 360-377 g, Ø104 × 23 mm. BOM €65 (non-pinwheel) or €98 (pinwheel).
  - Play 0.20-0.25° typical, 0.30-0.35° maximum. RMS transmission error 0.38-0.56°.
  - Coulomb friction 0.53-0.62 Nm. Stiffness 1,271-1,468 Nm/rad, up from 633 for their earlier PLA design.
  - After the 10 h endurance run (1,390 cycles) play roughly doubled, from elephant's-foot wear on the
    discs.
  - The design is open source.
- **Berkeley Humanoid Lite** ([arXiv 2504.17249](https://arxiv.org/html/2504.17249)):
  - PLA cycloid on a MAD M6C12 150KV motor.
  - Stiffness 319 Nm/rad. Maximum backlash 0.0229 rad (1.31°) across six units.
  - About 90% gearbox efficiency over most conditions.
  - Survived a 60 h pendulum test (0.5 Hz, about 108k cycles by my arithmetic).
  - Actuator about $188 (US) or $157 (China).
- **HowToMechatronics, PLA**
  ([25:1](https://howtomechatronics.com/how-it-works/harmonic-vs-cycloidal-drive-designing-3d-printing-testing),
  [15:1](https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/)):
  - 25:1: about 65% efficient, about 21.5 Nm before skipping, 46-149 arcmin of backlash.
  - 15:1: 60-66% efficient, 60-150 arcmin of backlash, and failed after about 2 h of reversing load.
  - Verdict: avoid in PLA.
- Related work:
  - [Volbeda MSc, Twente](https://essay.utwente.nl/fileshare/file/103595/103595_Volbeda_MA_EEMCS.pdf):
    anti-backlash preload trades play for friction.
  - [Bezzini 2024, Robotics 13:168](https://www.mdpi.com/2218-6581/13/11/168): 1:30 and 1:80 printed
    cycloid and planetary units for exoskeletons.

### 3.5 3D-printed planetary

- **HowToMechatronics** 16:1 printed planetary
  ([link](https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/)):
  78-95% efficient, 30-60 arcmin of backlash, little wear in about 10 h, $23 excluding printed parts.
- **COMPAct** ([arXiv 2510.07197](https://arxiv.org/abs/2510.07197)):
  - Single stage: 60-80% efficient, 0.59° backlash, 242.7 Nm/rad.
  - Compound: 60%, 2.6°, 201.6 Nm/rad.
  - Tested up to 18 Nm.
- **OpenTorque** ([GitHub](https://github.com/G-Levine/OpenTorque-Actuator),
  [Hackaday](https://hackaday.io/project/159404/instructions/discussion-161983)):
  - X8318S 100KV motor, nylon sun and planets, THK RA8008 cross-roller bearing (repo BOM).
  - One builder reports 6:1, about 28 Nm continuous and 56 Nm theoretical, about 1.2 kg and about $150
    ([RobotShop forum](https://community.robotshop.com/forum/t/3d-printed-opentorque-actuator); unverified).
- Planetaries carry 20 Nm, but measured printed planetaries are about 5× softer than a capstan or cycloid
  and have 0.5-1° of backlash.

### 3.6 3D-printed strain-wave

- **HD20** (20:1, PETG flexspline) ([releases](https://github.com/robrotics/hd20/releases)):
  - 2.1-3.3 Nm measured, 3.5 Nm peak.
  - 48-58% efficient; torque drops about 10% after run-in, with 18% unit-to-unit spread.
- **HowToMechatronics 25:1 PLA**
  ([link](https://howtomechatronics.com/how-it-works/harmonic-vs-cycloidal-drive-designing-3d-printing-testing)):
  - About 65% efficient.
  - 2.7-3.1° of play under load.
  - The flexspline took a permanent set overnight and failed on a NEMA23.
- Others:
  - [Open HD (Hackaday)](https://hackaday.io/project/160378-open-hd) uses a TPU flexspline that ratchets
    under reverse torque.
  - [Pacana 2019](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-10214446-99d3-441b-9655-1f8b96ca17ef/c/07_Pacana_03_2019.pdf)
    tested printed flexspline materials.
- Sandia notes that strain-wave gearing below about 30:1 is impractical
  ([Mazumdar 2017](https://www.osti.gov/servlets/purl/1340266)).
- **Not credible for a 20 Nm outdoor shoulder.**

---

## 4. Capstan design rules (FA-1 checklist)

### 4.1 Pinion (sheave) diameter, drum-to-cable

- **Dyneema and Vectran**
  - D/d ≥ 8 keeps 100% of the rope's strength
    ([RYA](https://www.rya.org.uk/knowledge/safety/look-after-yourself/hmpe-rope); Sandia uses Ds = 8 d).
  - For comparison: D/d 5:1 keeps about 80% ([KWRS sling data](https://www.kwrs.com/wp-content/uploads/2025/11/HMPE-Section.pdf)),
    2:1 about 65% and 1:1 about 50% (RYA).
  - Fatigue: doubling D/d roughly triples fatigue life, and 35% more load halves it
    ([Seahorse](https://www.seahorsemagazine.com/article/december-2016/fatigue)).
  - Use **8-10 for FA-1**. Accept 7 at the shoulder if packaging forces it. By the Seahorse rule, life
    goes as about (D/d)^1.58, so D/d 7 gives about 80% of the life at D/d 8 (**estimate**).
- **Steel 7x19**
  - 24 desirable, 18 critical; 7x7 needs 42 and 28 ([Loos](https://loosco.com/?p=1283),
    [Lexco](https://www.lexcocable.com/resources/downloads/wire-rope-guidelines/)).
  - Haptics practice is 15-50 ([US6985133](https://patents.google.com/patent/US6985133)).
- **Ratio from effective diameters** (to the rope centreline): N = (D_drum + d) / (D_pinion + d).
  - Design to these, not to the outer diameters, or the ratio will be off by 2-3%
    ([CARA](https://www.aaedmusa.com/projects/cara)).
  - Calibrate the final ratio with the output encoder.

### 4.2 Wraps and travel

- 3-5 working wraps on the pinion
  ([Aaed Musa](https://www.aaedmusa.com/projects/capstandrive); Ridgway gives 3-6).
- Each pinion turn walks the wrap set one groove pitch along the pinion.
  - Pinion groove length ≈ p × (wraps + pinion turns over the range of motion) + 2 flanges.
  - Example: 1.5 mm rope, p = 1.8 mm, 4 wraps + 4.5 turns gives about 15 mm of groove and a 21 mm pinion
    (**estimate**).
- **Dyneema's friction is low** (μ ≈ 0.05-0.08, [Capler-Leg](https://ar5iv.labs.arxiv.org/html/1806.10632)).
  - The tension ratio e^(μψ) is only 2.6-4.5 for 3 wraps and 4.8-12 for 5. Steel at μ 0.3 reaches about
    12,000 for 5 wraps ([US8858374](https://patents.google.com/patent/US8858374) example).
  - So **anchor the rope mechanically to the pinion**, for example a cross-hole plus a clamp or knot
    pocket at mid-wrap. Do not rely on friction, especially when wet.
- **Drum arc** = range of motion + 2 × θ_extra.
  - Sandia keeps ψ ≥ 2π on the sheave and π on the pulley.
  - A patent family recommends 2-3 extra wraps to shield the terminations
    ([CA 2551514](https://brevets-patents.ic.gc.ca/opic-cipo/cpd/fra/brevet/2551514/sommaire.html)).
  - With Dyneema the extra drum arc gives little relief (μψ is small), so size the terminations for full
    rope tension.
  - Practical range-of-motion ceiling for one sector is about 270-300°
    ([Sandia](https://www.osti.gov/servlets/purl/1340266) about 270°; [Aaed Musa](https://theamphour.com/712-robots-everywhere-with-aaed-musa/) about 300°).

### 4.3 Pretension

- **Minimum:** T_pre ≥ ½ × τ_peak / R. Below that the slack side goes loose, stiffness halves and shock
  loads start ([US8858374](https://patents.google.com/patent/US8858374)).
- **Conservative:** T_pre = τ_peak / R (Sandia, "treat Tpre as equal to" the torque term).
- **FA-1:** T_pre ≈ 0.75 × τ_peak / R. Size the rope so that T_pre + τ_peak / R ≤ 30-35% of MBL at peak
  and ≤ 10-20% at continuous torque.
  - That keeps DM20 in its "near-zero creep" zone: below 20% of break and below 40 °C (vendor claims).
  - It also follows Sandia's load-life finding: 20% less load gave about 3× the life.
- Pretension aggressively. It raised life by about 48% in Sandia's tests.
- Re-tension after 10-50 wear-in cycles and again within the first 1,000.
- Pre-stretch new rope by loading it to about 30-50% of MBL for a few minutes before fitting
  (**estimate**, normal rigging practice; Sandia's first pull left 5 mm of permanent stretch on 305 mm).
- **Radial load.** Pretension loads the pinion with about 2 × T_pre + τ / R: about 1 kN at the 96 mm
  shoulder option.
  - Straddle-mount the pinion: one bearing in the motor and one outboard. Do not cantilever it on the
    motor shaft (**estimate**).
  - Aaed Musa's BOM includes a 15×28×7 bearing, likely for this.

### 4.4 Anchoring

- **Splices:** use eye-splices (locked Brummel or long bury) over steel pins, and use several. Capler-Leg
  found a single splice failed its creep requirement ([ar5iv](https://ar5iv.labs.arxiv.org/html/1806.10632)).
- **Knots** roughly halve rope strength
  ([Horigome & Endo 2016](https://www.researchgate.net/publication/294107411_Basic_study_for_drive_mechanism_with_synthetic_fiber_rope_-_investigation_of_strength_reduction_by_bending_and_terminal_fixation_method)).
  RYA reports up to 80% loss for tight bends.
- **Clamps** keep high strength only with enough pressing force. Sewing and caulking reach 85-90%, and a
  grooved-pulley fixation reaches 91.3% (same group, RA-L 2018, via search).
- **Epoxy-potted ferrules** were Sandia's weak link: 32% minimum and about 46% average of rope strength.
- **Tensioner:** put it at a drum anchor (lead screw or turnbuckle) or use a sliding tensioning pulley
  (Sandia).
  - A stack of disc springs much stiffer than the rope span (rope ≈ EA/L ≈ 200 N/mm) holds preload
    against creep and costs only a little stiffness (**estimate**; the idea is from Ridgway 2023).

### 4.5 Keeping the rope on the drum

- Cut a **helical groove on the pinion**.
  - Pitch ≈ 1.15-1.3 d, or up to 1.5 d for flattening braid. Groove radius about 0.55 d, depth about
    0.3-0.4 d (**estimate**; Ridgway saw a 0.15 mm braid flatten to 0.35 mm).
  - The groove keeps wraps from rubbing on each other and stops the rope flattening (Sandia).
- Make the **drum smooth with flanges**, or give it one helical groove with lead = N × p. Never use
  concentric ring grooves (Section 1.1).
- **Polish** contact surfaces (Sandia). Printed parts need sanding or vapour smoothing, or an aluminium or
  steel pinion.
- Keep the rope **always tensioned**. Slack is what lets it jump a groove.
- Fit a printed **guard** over the pinion with less than d/2 clearance, so a slack rope cannot lift out
  (**estimate**).
- Keep the **fleet angle** at or below about 1.5-2° (**estimate**; Gates allows 0.25° for belts).
- Hold the pinion and drum axes parallel to well under a millimetre over the width. A builder report
  says sub-mm misalignment caused binding and tension loss
  ([LinkedIn anecdote](https://eg.linkedin.com/in/hossamyasser)).

### 4.6 Stiffness versus pretension

- With both strands taut: K ≈ EA × R² × (1/L1 + 1/L2).
  - Each L is the free span plus the rope lying on the drum between its anchor and the tangent point.
    With low-friction Dyneema that rope carries nearly full tension.
  - K is lowest at mid-travel.
- **Pretension does three things:**
  - It keeps both strands active, which doubles stiffness ([US8858374](https://patents.google.com/patent/US8858374)).
  - It beds the braid in: Sandia's Vectran went from 287 to 618 kN/m after first loading.
  - Beyond the no-slack level it adds little stiffness, but more bearing load, friction and creep.
- **FA-1 estimates** (EA guessed from Sandia's 3 mm Vectran at 188 kN, scaled by area and modulus to
  about 40 kN for 1.5 mm and 70 kN for 2 mm Dyneema):

  | Option                 | Stiffness     | Deflection at 20 Nm | Tip deflection at 0.37 m |
  | ---------------------- | ------------- | ------------------- | ------------------------ |
  | 1.5 mm rope, R = 49 mm | ≈ 1.3 kNm/rad | 0.9°                | ≈ 6 mm                   |
  | 2 mm rope, R = 65 mm   | ≈ 3.1 kNm/rad | 0.37°               | ≈ 2.4 mm                 |

- Stiffness rises with R (because L also grows with R) and with d². Measure it on the bench.
- Use the joint encoder and the motor encoder together. The difference between them gives a cheap
  torque estimate.

---

## 5. Cable options

| Cable                                                                                                                                                                                               | Dia (mm) | Breaking load (N)   | Min pinion Ø (mm)                            | Creep                                                                                                                                                               | $/m                                                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| ROBLINE Ocean STAT20 XG, Dyneema DM20, UV coat ([Lindemann](https://www.lindemann-kg.de/en/products/1/Tauwerk%20ROBLINE/Single%20Braids/Ocean%20STAT20%20XG))                                       | 2        | 4,000 (400 daN)     | 16 (8 d); 20 for long life                   | "zero creep" (vendor); DM20 crept <0.3% in 100 d at 70 °C ([Capler](https://ar5iv.labs.arxiv.org/html/1806.10632))                                                  | €1.60 ≈ **$1.85** (est. FX 1.15)                                                                                                                   |
| ROBLINE Ocean STAT20 XG DM20                                                                                                                                                                        | 3        | 8,000               | 24                                           | as above                                                                                                                                                            | €2.30 ≈ $2.65                                                                                                                                      |
| Mastrant-M 2 mm, DM20 core with polyester cover ([GigaParts](https://www.gigaparts.com/100-meter-330ft-mastrant-m-2mm-3-32-braided-rope-pes-dy-2-5kn-550lb-strength.html))                          | 2        | 2,500 (title claim) | 16                                           | DM20 core; **the cover is a risk**: Sandia's jacketed rope slackened and failed at 2k cycles. Strip it or avoid.                                                    | $54.95 / 100 m = **$0.55**                                                                                                                         |
| LIROS D-Pro 01505, SK78, PU-coated ([Jimmy Green](https://jimmygreen.com/?p=11138), [Flints](https://www.flints.co.uk/product/pg_dyneema%20d%20pro/ROP150502))                                      | 2        | 4,020 (410 kg)      | 16                                           | SK78: about 0.006%/day at 30 °C, about 85× DM20 ([Rigging Doctor](https://www.riggingdoctor.com/life-aboard/2020/4/12/not-all-dyneema-is-created-equal), secondary) | £1.00-1.04 ex VAT ≈ $1.35                                                                                                                          |
| LIROS D-Pro SK78 ([Skyman](https://www.skyman.cz/en/p/liros-d-pro-1-5-mm-dyneema-2-3-kn-red-60-m))                                                                                                  | 1.5      | 2,256 (230 kg)      | 12                                           | as SK78                                                                                                                                                             | est. $1.2                                                                                                                                          |
| LIROS D-Pro SK78 ([Shelby Outdoor](https://shelbyoutdoor.com/cs/p109684/liros-dyneema-d-pro-cord-1-mm))                                                                                             | 1.1      | 1,950 (195 daN)     | 9                                            | as SK78                                                                                                                                                             | est. $1.0                                                                                                                                          |
| Extremtextil 12-strand 100% SK99 ([listing](https://www.extremtextil.de/en/dyneema-cord-extra-strong-2mm-100-dyneema.html), now 410 Gone; values from search snippet)                               | 2        | 7,300 (730 daN)     | 16                                           | SK99 creeps like SK78 and is about 20% stronger ([Marlow via Scuttlebutt](https://www.sailingscuttlebutt.com/2014/11/06/much-know-rope/))                           | €2.20 ≈ $2.5                                                                                                                                       |
| Generic "Dyneema/HMPE" 2 mm, unknown grade ([RopeServicesUK datasheet](https://www.ropeservicesuk.com/wp-content/uploads/downloads/datasheets/hmpe/Dyneema_HMPE_Rope_Specifications_Datasheet.pdf)) | 2        | 2,940 (300 kg)      | 16                                           | unknown grade; assume SK75-like (2.3%/yr at 20% MBL per forum quote): re-tension often                                                                              | est. $0.2-0.6 (AliExpress)                                                                                                                         |
| 8-strand UHMWPE fishing braid, about 100 lb ([Bass Pro, PowerPro 0.46 mm](https://www.basspro.com/shop/en/77484))                                                                                   | ≈ 0.46   | ≈ 445               | 4                                            | unknown grade, flattens (Ridgway); fine for wrist or yaw tests only                                                                                                 | est. $0.05-0.15                                                                                                                                    |
| New England Ropes V12 Vectran (Sandia) ([Mazumdar](https://www.osti.gov/servlets/purl/1340266))                                                                                                     | 3        | 9,400               | 24 (about 85% strength at 8:1)               | no cold creep; UV-sensitive unless coated (Vectran T97 loses about 30% after 600 h UV)                                                                              | est. $2-4                                                                                                                                          |
| Stainless 7x19, type 302/304, 1/16" ([Loos](https://loosco.com/product/cable/stainless-steel-cable-bare-7x19-commercial), [Certex](https://www.certex.com/product/stainless-steel-aircraft-cable/)) | 1.6      | 2,135 (480 lb)      | 38 desirable (24:1; Lexco 1.5"), 29 critical | zero cold creep; bend-fatigue limited                                                                                                                               | £0.31-0.37 for 1.5 mm 316 ([GS Products](https://gsproducts.co.uk/1-5mm-7x19-stainless-steel-wire-rope-cable), which rates it only 130 kg) ≈ $0.45 |
| Stainless 7x19, 3/32"                                                                                                                                                                               | 2.4      | 4,092 (920 lb)      | 57 (Lexco 2.25")                             | as above                                                                                                                                                            | est. $0.5-0.8                                                                                                                                      |
| Stainless 7x7, 1/16" ([stagelightingstore](https://www.stagelightingstore.com/1-16in-Stainless-Steel-Cable-7x7-500ft-Reel))                                                                         | 1.6      | 2,135 (480 lb)      | 67 (42:1, Lexco 2-5/8")                      | as above; stiff construction                                                                                                                                        | est. $0.3                                                                                                                                          |
| Braided tungsten (SensAble haptics) ([US6985133](https://patents.google.com/patent/US6985133))                                                                                                      | 0.36     | 165 (37 lb)         | 4.5 (12.5:1 used)                            | none; 5.5M cycles without wear                                                                                                                                      | est. > $5 (niche)                                                                                                                                  |

**Pick:** DM20 first (Robline STAT20 XG or another named-DM20 braid, uncovered), and SK99 second. Avoid
SK78 and unknown-grade "Dyneema" for preloaded drives outdoors, and avoid jacketed ropes.

Notes on the data:

- Grade matters more than diameter.
- DM20 is slightly weaker than SK78 but "virtually zero creep"
  ([Marlow via Scuttlebutt](https://www.sailingscuttlebutt.com/2014/11/06/much-know-rope/),
  [Dynamica](https://dynamica-ropes.com/products/dm20/): 3-4% elongation at break, very good UV, max
  working temperature 60-65 °C).
- One forum quote gives creep at 20% of break as SK75 2.3%/yr, SK78 0.5%/yr and DM20 0.0%
  ([catsailor](https://www.catsailor.com/forum/ubbthreads.php/topics/161734), secondary).
- At 70 °C, SK75 and SK78 ruptured within days, while DM20 survived 6 months
  ([Rigging Doctor](https://www.riggingdoctor.com/life-aboard/2020/4/12/not-all-dyneema-is-created-equal),
  unattributed tests).
- Generic HMPE datasheets give −40 to +70 °C working range and minimal water absorption.

---

## 6. Sized options for FA-1 (all **estimates** from the rules above)

Assumptions:

- Pretension = 0.75 × τ_peak / R. Peak strand uses the conservative Sandia form (T_pre + τ / R).
- EA = 22 kN for 1.1 mm, 40 kN for 1.5 mm and 70 kN for 2 mm Dyneema.
- Free span 20 mm.

| Joint                          | Rope                 | Pinion core / D/d | Drum core | N   | T_torque @ peak | T_pre | Peak strand (% MBL) | Pinion turns / walk | K (est.)    | Radial load on pinion |
| ------------------------------ | -------------------- | ----------------- | --------- | --- | --------------- | ----- | ------------------- | ------------------- | ----------- | --------------------- |
| Shoulder A (fits Ø96 envelope) | 1.5 mm D-Pro or DM20 | 10.5 mm / 7       | 96 mm     | 8.1 | 410 N @ 20 Nm   | 308 N | 718 N (32%)         | 4.5 / 8 mm          | 1.3 kNm/rad | ≈ 1.0 kN              |
| Shoulder A2                    | 1.5 mm               | 12 mm / 8         | 96 mm     | 7.2 | 410 N           | 308 N | 718 N (32%)         | 4.0 / 7 mm          | 1.3 kNm/rad | ≈ 1.0 kN              |
| Shoulder B (raised axis)       | 2 mm DM20 (4 kN)     | 16 mm / 8         | 128 mm    | 7.2 | 308 N           | 231 N | 538 N (13%)         | 4.0 / 10 mm         | 3.1 kNm/rad | ≈ 0.8 kN              |
| Elbow crank                    | 1.1 mm D-Pro         | 9 mm / 8.2        | 90 mm     | 9.0 | 220 N @ 10 Nm   | 165 N | 384 N (20%)         | 6.0 / 8 mm          | 0.6 kNm/rad | ≈ 0.55 kN             |
| Elbow crank (stiffer)          | 1.5 mm               | 12 mm / 8         | 96 mm     | 7.2 | 205 N           | 154 N | 359 N (16%)         | 4.8 / 9 mm          | 1.2 kNm/rad | ≈ 0.5 kN              |
| Yaw ring                       | 1.1 mm D-Pro         | 9 mm / 8.2        | 150 mm    | 15  | 66 N @ 5 Nm     | 50 N  | 116 N (6%)          | 12.5 / 17 mm        | 1.0 kNm/rad | ≈ 0.17 kN             |

### 6.1 Shoulder

- The drum sweeps nearly a full circle about the shoulder axis: its arc is the range of motion plus the
  end allowances, and it moves through the range of motion.
- So the drum radius must stay under the axis height of 50 mm, minus clearance, unless the turret has a
  well. Option A keeps today's envelope.
- WK8115_1 ($37, 5.17 Nm stall at 33 A) at 8:1 × 0.95 gives about 39 Nm peak. That is about 2× the current
  GIM8108-8's 22 Nm, with zero backlash.
- Holding 6 Nm needs about 0.8 Nm at the motor, about 4.2 A at Kt 0.19. That is about 6 W in copper at
  0.35 Ω (**estimate**; check the winding resistance convention).

### 6.2 Elbow crank

- The parallelogram crank turns with the forearm's absolute angle.
- The shoulder and elbow simulation ranges would allow up to about 370° in total. The real reachable
  span is smaller but must be checked: a capstan sector needs it under about 270°.
- If it is over, use a two-stage belt for the crank.

### 6.3 Yaw

- The simulation allows ±200°, which a capstan cannot do.
- Either cap yaw at about ±150° for the ring capstan, or use a closed belt around a fixed toothed ring.
  - Example: 3M 9 mm, a 150T ring (PD 143 mm) and a 15T pinion give 10:1. Continuous rotation is
    possible, with backlash about 0.3-0.6° (**estimate**).
  - A 20T 2MGT 6 mm pinion is rated about 1.1-1.5 Nm (Gates). At 7.5-10:1 that is 8-15 Nm, against the
    5 Nm needed.
- The turret then needs its own bearing: a thin-section 618xx pair or a 4-point slewing ring.
  - Simply Bearings lists a 61808-2RS at £6.84 each at 9+ units
    ([Simply Bearings](https://www.simplybearings.co.uk/products/61808-2rs)).
  - Chinese equivalents cost about $1-3 (**estimate**).

### 6.4 Cost per capstan joint

**Estimate: $8-16.**

| Part                                                             | Cost | Basis                                                 |
| ---------------------------------------------------------------- | ---- | ----------------------------------------------------- |
| Rope, 2-3 m                                                      | $1-5 | listings above                                        |
| 2-3 bearings                                                     | $3-6 | Chinese thin-section or 688/6800-class (**estimate**) |
| 60-150 g of PC, PET-CF or ASA                                    | $1-3 | ~$20/kg (**estimate**)                                |
| Steel dowel or pinion shaft, M4 lead screw, disc springs, screws | $2-3 | **estimate**                                          |
| Output-encoder magnet                                            | $0.5 | **estimate**                                          |

For comparison, Aaed Musa's US-retail mechanical BOM is about $31.

---

## 7. Outdoor behaviour

**Dust and grit.** Abrasion between fibres is the main failure mode of synthetic rope (Sandia). Grit
also wears printed gear teeth and cycloid lobes, and belt teeth.

- Enclose each stage in a printed cover with a lip seal or TPU boot. The design already plans TPU gaiters.
- A capstan is the easiest stage to enclose: its contact zone is small and it needs no oil bath.
- Outdoors, prefer a dry PTFE rope treatment over grease, which holds grit (**estimate**). Keep any
  lubricant off the extra wraps and terminations (Sandia).

**UV.**

- DM20 is "very good" ([Dynamica](https://dynamica-ropes.com/products/dm20/)). Robline's coating is
  optimised for UV.
- Vectran and Zylon need coatings or jackets (Sandia).
- Rubber belts should be kept from sunlight and ozone (Gates).
- Printed parts: ASA or PC; avoid PLA.

**Water.**

- HMPE has minimal water absorption and floats. Wet rope lowers friction further, which is another reason
  to anchor the pinion mechanically.
- PA-CF cycloid parts absorb water and change size and stiffness.
- Fiberglass-cord belts should be kept dry (Gates storage note).
- Steel: use 316.

**Temperature.**

- DM20 working limit is 60-65 °C ([Dynamica](https://dynamica-ropes.com/products/dm20/)); Capler stays
  below 80 °C. Generic HMPE: −40 to +70 °C.
- A dark printed drum in summer sun can reach 60-70 °C (**estimate**). Use a light-coloured or white
  shroud and keep motors from heat-soaking the drum.
- PLA softens around 55-60 °C (**estimate**). Use PC or PC-blend (Stanley: no wear) or PET-CF (CARA's
  small drums).
- Belts are fine from −25 to +100 °C (Gates).

**Creep.**

- Of the ropes, DM20 creeps least: about 85× less than SK78 (secondary source).
- Printed drums creep under constant pretension too. Put the tension anchors in metal inserts or steel
  pins, and avoid PLA (**estimate**).
- Schedule re-tensioning at assembly, after about 50 cycles, after about 1,000 cycles, then at service
  intervals (Sandia).
- Add a controller check: compare the motor and joint encoders at a known load to flag lost pretension
  (**estimate**).

---

## 8. Test plan before committing (cheap, about 1 week)

1. Print Shoulder A2 (1.5 mm DM20 or SK99, Ø12 pinion, Ø96 PC drum) on a bench stand with a WK8115_1, or
   any 81xx, and an output encoder.
2. Measure stiffness with the output locked and current ramped (Roozing's method). Measure backlash and
   hysteresis, and back-drive torque.
3. Run 10^5 cycles of the weed-pull torque profile at 40-50 °C in a heated box, logging pretension
   through the encoder difference.
4. Repeat the cycle test with grit (sand in a bag over the stage) to set the cover requirement.
5. If the crank's absolute range comes out over about 270°, build the two-stage 3MGT 15 mm belt
   alternative for it.

---

## 9. All sources

Capstan and rope:

- Mazumdar et al., Synthetic Fiber Capstan Drives (Sandia, RA-L 2017): https://www.osti.gov/servlets/purl/1340266
- ResearchGate entry for the same: https://www.researchgate.net/publication/312253704_Synthetic_Fiber_Capstan_Drives_For_Highly_Efficient_Torque_Controlled_Robotic_Applications
- Hwangbo et al., Cable-Driven Actuation for Highly Dynamic Robotic Systems (ETH Capler-Leg): https://ar5iv.labs.arxiv.org/html/1806.10632 , https://arxiv.org/pdf/1806.10632
- Aaed Musa capstan drive: https://www.aaedmusa.com/projects/capstandrive
- Aaed Musa CARA: https://www.aaedmusa.com/projects/cara
- Capstan-Drive GitHub: https://github.com/aaedmusa/Capstan-Drive
- Capstan-Drive README: https://raw.githubusercontent.com/aaedmusa/Capstan-Drive/main/README.md
- Capstan-Drive BOM: https://raw.githubusercontent.com/aaedmusa/Capstan-Drive/main/Capstan%20Drive%20Test%20Stand%20BOM.xlsx
- The Amp Hour #712: https://theamphour.com/712-robots-everywhere-with-aaed-musa/
- CARA 2.0 write-up (secondary): https://sesamedisk.com/affordable-quadruped-robotics-innovation/
- Hackaday capstan tag (fetch blocked): https://hackaday.com/tag/capstan
- Stanley capstan quadruped: https://hackaday.io/project/176726-stanley-the-capstan-based-quadruped-kit
- Ridgway MASc 2023: https://www.dynsyslab.org/wp-content/papercite-data/theses/MASc-Ridgway-2023.pdf
- MIT CBA lug-capstans (HTTP 503, snippet only): https://023.cba.mit.edu/jakeread/lug-capstans/-/tree/8ad6d1a41caaf2ab2496feb7ef136ad3fd16a7fc
- Barrett / Townsend pretension patent: https://patents.google.com/patent/US8858374
- SensAble haptic interface patent: https://patents.google.com/patent/US6985133
- Capstan patent family, extra wraps: https://brevets-patents.ic.gc.ca/opic-cipo/cpd/fra/brevet/2551514/sommaire.html
- Werkmeister & Slocum 2007, capstan stiffness (catalogue record): https://ibl.kb.nl/articles/0000000010267/31/1
- Baser & Konukseven 2010, capstan slip error: https://avesis.metu.edu.tr/yayin/1bb3f2be-2599-4d97-9ef2-569ca7aeb5d6/theoretical-and-experimental-determination-of-capstan-drive-slip-error
- Horigome & Endo 2016, rope bending and terminations: https://www.researchgate.net/publication/294107411_Basic_study_for_drive_mechanism_with_synthetic_fiber_rope_-_investigation_of_strength_reduction_by_bending_and_terminal_fixation_method
- Endo lab page: https://www.robotics.mech.e.titech.ac.jp/gendo/en/archives/t03
- Capstan drives, secondary blog: https://aaronnewmanb.substack.com/p/capstan-drives-affordable-precision
- RYA, HMPE rope: https://www.rya.org.uk/knowledge/safety/look-after-yourself/hmpe-rope
- Seahorse, rope fatigue: https://www.seahorsemagazine.com/article/december-2016/fatigue
- KWRS HMPE section: https://www.kwrs.com/wp-content/uploads/2025/11/HMPE-Section.pdf
- IFREMER SK78 bending study: https://archimer.ifremer.fr/doc/00470/58128/112881.pdf
- Dynamica DM20: https://dynamica-ropes.com/products/dm20/
- ROBLINE Ocean STAT20 XG: https://www.lindemann-kg.de/en/products/1/Tauwerk%20ROBLINE/Single%20Braids/Ocean%20STAT20%20XG
- Mastrant-M 2 mm: https://www.gigaparts.com/100-meter-330ft-mastrant-m-2mm-3-32-braided-rope-pes-dy-2-5kn-550lb-strength.html
- LIROS D-Pro (Jimmy Green): https://jimmygreen.com/?p=11138
- LIROS D-Pro (Flints): https://www.flints.co.uk/product/pg_dyneema%20d%20pro/ROP150502
- LIROS D-Pro 1.5 mm (Skyman): https://www.skyman.cz/en/p/liros-d-pro-1-5-mm-dyneema-2-3-kn-red-60-m
- LIROS D-Pro 1.1 mm (Shelby Outdoor): https://shelbyoutdoor.com/cs/p109684/liros-dyneema-d-pro-cord-1-mm
- Extremtextil SK99 2 mm: https://www.extremtextil.de/en/dyneema-cord-extra-strong-2mm-100-dyneema.html
- RopeServicesUK HMPE datasheet: https://www.ropeservicesuk.com/wp-content/uploads/downloads/datasheets/hmpe/Dyneema_HMPE_Rope_Specifications_Datasheet.pdf
- Rigging Doctor, Dyneema grades: https://www.riggingdoctor.com/life-aboard/2020/4/12/not-all-dyneema-is-created-equal
- Marlow via Scuttlebutt: https://www.sailingscuttlebutt.com/2014/11/06/much-know-rope/
- Catsailor forum, creep figures: https://www.catsailor.com/forum/ubbthreads.php/topics/161734
- DM20 / Dyneema Max press (not retrievable): https://www.offshore-energy.biz/dsm-dyneema-improves-dm20-with-dyneema-max-technology/
- PowerPro braid (Bass Pro): https://www.basspro.com/shop/en/77484
- Cable-robot rope behaviour study: https://hal.uca.fr/hal-03135509v1
- Miermeister, elastic cable model with hysteresis: https://www.springerprofessional.de/en/an-elastic-cable-model-for-cable-driven-parallel-robots-includin/2220182

Steel cable:

- Loos, cable vs pulley diameter: https://loosco.com/?p=1283
- Loos, wire rope bend radius: https://loosco.com/recommended-wire-rope-bend-radius/
- Lexco wire rope guidelines (PDF): https://www.lexcocable.com/resources/downloads/wire-rope-guidelines/
- Loos 7x19 commercial: https://loosco.com/product/cable/stainless-steel-cable-bare-7x19-commercial
- Certex stainless aircraft cable: https://www.certex.com/product/stainless-steel-aircraft-cable/
- Lexco 7x19 type 316: https://catalog.lexcocable.com/printitems/cable-stainless-steel-galvanized/7-x-19-stainless-steel-aircraft-cable-type-316?plpver=10
- GS Products 1.5 mm 7x19: https://gsproducts.co.uk/1-5mm-7x19-stainless-steel-wire-rope-cable
- Stage Lighting Store 1/16" 7x7: https://www.stagelightingstore.com/1-16in-Stainless-Steel-Cable-7x7-500ft-Reel
- Unirope sheave dimensions: https://www.unirope.com/?p=2029

Belts:

- Gates design manual E2/20099 (PowerGrip GT3/HTD power tables, pp. 128-134; engineering data p. 157-158): https://chiaravalli.com/assets/d51d3c7e-4c6c-450d-a27e-9716381d3365
- Gealy et al., Blue arm: https://ar5iv.labs.arxiv.org/html/1904.03815 , https://arxiv.org/pdf/1904.03815
- IEEE Spectrum on Blue: https://spectrum.ieee.org/uc-berkeley-blue-low-cost-force-controlled-robot-arm
- Kau et al., Stanford Doggo: https://arxiv.org/pdf/1905.04254
- Stanford Woofer (Dynamic Walking 2020): https://www.seas.upenn.edu/~posa/DynamicWalking2020/695-1037-1-RV.pdf
- HowToMechatronics, best printed drive: https://howtomechatronics.com/how-it-works/what-is-the-best-3d-printed-drive-for-your-next-robotic-project/
- GT2 6 mm belt reseller: https://distributors.minibearings.com.au/collections/belts-timing-2-000mm-pitch-gt-clean-6mm-wide
- Generic HTD-5M belts (eBay): https://www.ebay.de/itm/286585507239
- ContiTech 300-5M-15 (Acorn): https://acorn-ind.co.uk/p/contitech/metric-timing-belts/300-5m-15-contitech
- Optibelt 300-5M-15 (Acorn): https://www.acorn-ind.co.uk/p/optibelt/metric-timing-belts/300-5m-15-optibelt/
- Pat biped, knee belt: https://arxiv.org/pdf/2210.03713
- Katz thesis, Mini Cheetah actuator: https://dspace.mit.edu/handle/1721.1/118671
- MIT Cheetah 3 paper: https://dspace.mit.edu/bitstream/1721.1/126619/2/IROS.pdf
- MIT Cheetah coverage: https://spectrum.ieee.org/mit-cheetah-robot-running , https://www.nbcnews.com/news/amp/wbna51105575

Printed gears:

- Roozing & Roozing ICRA 2024 (PDF): https://files.energiazero.org/cartelle/meccanica/riduttori/2023_cycloidal_drive_nonpinwheel.pdf
- Roozing (DBLP): https://dblp1.uni-trier.de/pid/158/9262.html
- Berkeley Humanoid Lite: https://arxiv.org/html/2504.17249 , https://arxiv.org/pdf/2504.17249
- HowToMechatronics, harmonic vs cycloid: https://howtomechatronics.com/how-it-works/harmonic-vs-cycloidal-drive-designing-3d-printing-testing
- Volbeda MSc, Twente: https://essay.utwente.nl/fileshare/file/103595/103595_Volbeda_MA_EEMCS.pdf
- Bezzini et al. 2024: https://www.mdpi.com/2218-6581/13/11/168
- COMPAct: https://arxiv.org/abs/2510.07197 , https://arxiv.org/html/2510.07197v1
- OpenTorque GitHub: https://github.com/G-Levine/OpenTorque-Actuator
- OpenTorque BOM: https://raw.githubusercontent.com/G-Levine/OpenTorque-Actuator/master/Bill%20of%20Materials.csv
- OpenTorque Hackaday: https://hackaday.io/project/159404/instructions/discussion-161983
- OpenTorque build report (RobotShop): https://community.robotshop.com/forum/t/3d-printed-opentorque-actuator
- HD20 printed strain-wave: https://github.com/robrotics/hd20/releases
- Open HD: https://hackaday.io/project/160378-open-hd
- Pacana 2019, printed harmonic drive: https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-10214446-99d3-441b-9655-1f8b96ca17ef/c/07_Pacana_03_2019.pdf
- 3K-H-V robot drive: https://arxiv.org/pdf/2210.02046

Motors and bearings (context):

- Eaglepower LA8308 KV90 (Robu): https://robu.in/product/eaglepower-la-8308-kv90-motor/
- Eaglepower 8308 (Accio aggregator): https://www.accio.ai/find-product/eaglepower-8308-90kv-40-poles
- Eaglepower 8308 (AliExpress listing from Aaed Musa): https://www.aliexpress.com/i/2255800377511662.html
- 61808-2RS (Simply Bearings): https://www.simplybearings.co.uk/products/61808-2rs
- Sibling reports: `research2/sourcing.md` (GWK8108-8 $70.36, GWK6010-8 $46.82) and `research2/custom_electronics.md` §1.2 (WK8115_1 $37, GB6010 $36, GB4310 $22).
