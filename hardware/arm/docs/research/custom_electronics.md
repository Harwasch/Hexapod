# FA-1 lever 2: bare actuators + our own multi-axis FOC driver

Research date 2026-10-17. Quantity basis: 10 arms (12 boards with 2 spares, 30-36 driven axes).
Prices are live reads on 2026-10-17 unless dated otherwise: the aifitlab.com, OpenELAB, CubeMars, mjbots
and ODrive Shopify catalogues (`/products.json`), and the JLCPCB parts API with LCSC price ladders.
Anything I could not read from a primary source is marked **estimate**, with the reasoning next to it.
"Vendor" means a number copied from the seller's spec table that I could not check independently.

---

## 0. Bottom line

1. **Per-axis electronics cost:** a 3-axis turret board costs about **$27.6 per axis**, or **$83 per board**.
   That includes two magnetic encoders per axis (motor and output side), cables, PCB fab and assembly at
   JLCPCB, an aluminium heat spreader and a 10% contingency. A 6-axis board comes to about **$24 per axis**
   ($146 per board). For comparison:
   - moteus-c1 retails at $69 per axis and peaks at 20 A.
   - Steadywin charges $20-40 for its own driver ($34 bought separately) and $10 more for the output encoder.
   - ODrive Micro costs $84-89 and peaks at 7 A, too small for this arm.
   - A Chinese ODrive-3.6 clone (ODESC dual) works out to $27.50 per axis, but it has classic CAN only,
     one encoder per axis and current ratings nobody guarantees.
2. **On parts cost it roughly breaks even.** It does not pay for its engineering at 10 arms.
   - Against today's FA-1 (EduLite 05 + GIM8108-8 with output encoder + GIM6010-8 with output encoder, $320,
     plus a $10 servo bus adapter), a bare-motor build with this board saves **about $27-32 per arm**, roughly
     5% of the $607 parts cost.
   - The engineering is about **14-24 person-weeks** (section 5). At 10 arms that never pays back on cost.
   - It pays back only if the same board also drives the hexapod's legs: 18 joints per robot, 180 for 10 robots.
3. **The real gain is capability:**
   - **Peak current:** 40 A per axis instead of the 22-23 A in Steadywin's bundled drivers. The same $70
     GIM6010-8 elbow goes from 11 Nm to about 17 Nm peak (**estimate**: Kt × current, linear). Steadywin's own
     48 V spec for this gearbox lists 17.9 Nm stall, so the gears have been rated near that.
   - **Cheaper shoulder:** it opens up the driverless GWK7015-8 ($90, vendor 11.4 / 31.7 Nm) as the shoulder.
   - **One bus and protocol:** CAN-FD at 5 Mbps on every axis with one protocol, instead of Steadywin's
     GDS/GDZ/GDK variants that each use a different protocol.
   - **Two encoders per axis:** true motor-side plus output-side control on every axis. Steadywin's tables list
     "Separate Encoder Support: NO".
   - **One sealed box:** a single conformal-coated, sealed electronics box on the turret instead of four
     driver boards out in the weather.
   - **Servo bus built in:** the board can host the 12 V servo rail and the TTL bus master for the wrist servos,
     which removes the $10 servo bus adapter and a separate buck.
4. **Recommendation:**
   - **Prototype arm:** use moteus-c1 boards ($69) on bare Steadywin units now.
   - **Production, arm only:** fork the open moteus-c1/n1 design into a 3-axis board (Path M, about $42 per
     axis). The proven moteus firmware runs unchanged apart from a new board definition.
   - **Arm and legs:** build the cheaper STM32G431 + DRV8300 board (Path C, about $28 per axis) and port the
     MIT-licensed Recoil or RoboMotor firmware, but only if the board will also drive the legs.

---

## 1. (a) Bare units a custom FOC driver can run (1-30 Nm output)

### 1.1 Geared bare units (motor + planetary, no driver)

All prices are single-unit US retail on 2026-10-17. None of these stores publishes a 10-piece price break, so
the qty-10 price is assumed equal to the unit price (**estimate**). Kt_out is the vendor's output torque
constant. "Peak @ our 40 A" is Kt_out × min(40 A, current at vendor stall), linear, ignoring magnetic
saturation (**estimate**). Never exceed the vendor stall torque, because the gearbox limits are unpublished.

| Unit                                             | Ratio | Rated Nm @ A   | Peak (vendor) Nm @ A | Kt_out Nm/A | R (vendor)  | Mass   | Size (mm)    | Price                    | Fit                                                                         |
| ------------------------------------------------ | ----- | -------------- | -------------------- | ----------- | ----------- | ------ | ------------ | ------------------------ | --------------------------------------------------------------------------- |
| Steadywin GIM6010-8, 24-slot delta, no driver    | 8     | 5 @ 10.5       | 11 @ 23.4            | 0.47        | 0.44 Ω l-l  | 370 g  | Ø80 × 29.5   | $70 ($80 w/ 2nd encoder) | elbow; about 17 Nm at 37 A (est.)                                           |
| Steadywin GIM8108-8, 48-slot star, no driver     | 8     | 7.5 @ 7        | 22 @ 22 (48 V)       | 1.0         | 0.67 Ω      | 378 g  | Ø92 × 44     | $85 ($95 w/ 2nd encoder) | shoulder, as in FA-1                                                        |
| Steadywin GIM4315-8, no driver                   | 8     | 1.6 @ 1.86     | 14.25 @ 19.5         | 0.96        | 1.5 Ω       | 343 g  | Ø57 × 47     | $58 ($68)                | **yaw** (needs 1.5/5 Nm)                                                    |
| Steadywin GWK7015-8 (new flat, driver-less)      | 8     | 11.4 @ 7 (sic) | 31.71 @ ">60"        | 1.2         | 0.371 Ω l-l | 375 g  | Ø78 × 22.5   | **$90**                  | **shoulder upgrade**: 31.7 Nm needs ~26 A                                   |
| Steadywin GWK7010-8                              | 8     | 7.6 @ 7.1      | 23.24 @ ">60"        | 0.65        | 0.27 Ω l-l  | 334 g  | Ø78 × 22.5   | $80                      | elbow/shoulder; 23 Nm needs ~36 A                                           |
| Steadywin GWK6010-8                              | 8     | 5.68 @ 12.3    | 16.89 @ ">30"        | 0.39        | 0.85 Ω l-l  | 260 g  | Ø67.5 × 30.5 | $60                      | elbow/yaw; 16.9 Nm needs ~43 A (our 40 A gives ~15.6 Nm)                    |
| Steadywin GWK8108-8                              | 8     | 10.2 @ 17.1    | 24.78 @ ">60"        | 0.6         | 0.06 Ω l-l  | 327 g  | Ø87 × 25.5   | $90                      | needs ~41 A; low-resistance winding                                         |
| Steadywin GWK6013-9                              | 9     | 9.8 @ 13.8     | 30.83 @ ">30"        | 0.47        | 0.074 Ω l-l | 319 g  | Ø69 × 26.5   | $95                      | 30.8 Nm needs ~66 A: n1-class driver only                                   |
| HAITAI HWG7015-J8                                | 8     | 11.4           | 31.71                | 1.2         | 0.371 Ω     | 375 g  | Ø78 × 22.5   | $110                     | identical table to GWK7015-8 (same factory part?); Steadywin is $20 cheaper |
| Steadywin GIM3510-8, no driver                   | 8     | 1 @ 3.3        | 6.28 @ 28.9          | 0.26        | 0.52 Ω      | 248 g  | Ø46 × 46.5   | $75 ($85)                | small yaw/wrist                                                             |
| Steadywin GIM4310-10, no driver                  | 10    | 2.05 @ 2.3     | 5.6 @ 6.8 (GDS34)    | 0.96        | 1.89 Ω      | 217 g  | Ø53 × 32     | $82                      | yaw (light)                                                                 |
| Steadywin GIM8108-9 / 8115-9 / 8108-6, no driver | 9/9/6 | -              | -                    | -           | -           | -      | -            | $130 / $155 / $135       | listed for completeness; more expensive than the GWK parts                  |
| CubeMars AK40-10 KV170, no driver                | 10    | 1.3            | 4.1                  | -           | -           | ~200 g | -            | $99.90                   | expensive                                                                   |
| CubeMars AK45-10 KV75, no driver                 | 10    | 2.5            | 7                    | -           | -           | ~260 g | -            | $119.90                  | expensive                                                                   |
| CubeMars AK45-36 KV80, no driver                 | 36    | 8              | 24                   | -           | -           | ~340 g | -            | $149.90                  | only partly backdrivable                                                    |
| Damiao DM-S3519 (M3508-style), no driver         | 19.2  | 3.5 @ 9.2      | 7.8 @ 20.5           | ~0.38       | 0.2 Ω       | 396 g  | Ø42 × 91.5   | $75                      | long and thin; 2 incremental encoders                                       |
| MyActuator RMD-X6 1:6 V2, no driver              | 6     | -              | -                    | -           | -           | -      | -            | $230                     | not competitive                                                             |

Observations:

- **The GWK series is the most interesting 2025-26 part here.** It is a flat 8:1 steel planetary sold without
  a driver ($60-95), meant for an external controller. The best pick is the GWK7015-8: high Kt, so it needs
  only ~26 A for 31.7 Nm, and it is $90 against $135 for today's shoulder.
  - **Verify:** the vendor's "rated 11.4 Nm @ 7 A" does not match its own Kt of 1.2 Nm/A (that would be
    8.4 Nm). Treat rated torque as unverified until a bench test.
  - **Verify:** the GWK listings offer no "secondary encoder" option, so the output encoder is ours. Check that
    the rotor carries a magnet for the motor-side encoder.
- **Winding choice decides how much driver you need.** High-Kt windings (GWK7015-8, GIM8108-8, GIM4315-8) reach
  their stall torque at 15-26 A. Low-resistance ones (GWK8108-8, GWK6013-9) need 40-66 A, which is a moteus-n1
  or x1 class driver. Specify the board for **40 A peak**, and pick motors that reach their peak below that.
- **Cheaper sourcing (lever 1):** an eBay (Germany) listing showed GIM6010-8 at US$49.90 + $16 shipping. I saw
  this only in a search snippet; the page returned 403, so it is unverified. Robotics Center (San Francisco)
  lists the bare units at $70 / $85, the same as aifitlab. OpenELAB is 30-60% dearer.
- **What "no driver" includes:** on Steadywin GIM units the motor-side encoder sits on the driver PCB, so a bare
  unit needs our own encoder board at the back. The open RoboMotor-GIM6010-V2 driver does exactly this for the
  bare GIM6010-8, reading the rotor with an MA732 and adding an MA730 output-encoder PCB. That shows it can be
  done, but check the magnet on the parts you receive.

### 1.2 Motor-only units (gimbal / outrunner / frameless) that need a printed reduction (lever 3)

Output with an 8:1 printed stage is Kt × I × 8 × η, with η = 0.85 for a printed planetary or 0.95 for a capstan
(**estimate**). The FDM cycloid in Berkeley Humanoid Lite measured about 90% mechanical efficiency over most of
its range (arXiv 2504.17249).

| Motor                                                    | Rated / stall Nm @ A     | Kt Nm/A                | R       | Mass        | Size (mm) | Price                                         | Output at 8:1, η 0.85 (est.)     |
| -------------------------------------------------------- | ------------------------ | ---------------------- | ------- | ----------- | --------- | --------------------------------------------- | -------------------------------- |
| **Steadywin WK8115_1** (rotor + stator kit, magnet disc) | 1.1 @ 5.8 / 5.17 @ 33.1  | 0.19                   | 0.35 Ω  | 313 g       | Ø90 × 19  | **$37**                                       | 7.5 rated / 35 peak              |
| Steadywin GB8115-1 (housed gimbal)                       | 1.1 / 5.17 @ 33.1        | 0.19                   | 0.33 Ω  | 518 g       | -         | $78                                           | same as WK8115, heavier          |
| Steadywin GB8115-4 (36 V winding)                        | 1.26 @ 2.1 / 4.11 @ 6.6  | 0.67                   | 3.89 Ω  | 518 g       | -         | $78                                           | 8.6 / 28                         |
| Steadywin WK10025                                        | 4 @ 3.8 / 8.5 @ 16       | 1.03                   | 3.06 Ω  | 576 g       | Ø109 × 33 | $75                                           | 27 / 58 (gear-limited)           |
| Steadywin GB7215                                         | 1.2 @ 2.9 / 3.7 @ 16     | 0.41                   | 1 Ω     | 390 g       | -         | $50                                           | 8 / 25                           |
| Steadywin GB6025                                         | 1.77 @ 7.4 / 2.68 @ 10.7 | 0.23                   | 3.5 Ω   | 478 g       | -         | $49                                           | 12 / 18                          |
| Steadywin GB5225                                         | 1.26 @ 1.28 / 1.9 @ 1.94 | 0.98                   | 11.8 Ω  | 372 g       | -         | $39                                           | 8.6 / 13                         |
| Steadywin GB6010                                         | 0.8 @ 1.3 / 1.59 @ 2.7   | 0.61                   | 10.58 Ω | 242 g       | -         | $36                                           | 5.4 / 10.8                       |
| Steadywin GB4310                                         | 0.2 / 0.49 @ 1.9         | 0.23                   | 10.3 Ω  | 112 g       | -         | $22                                           | wrist only                       |
| HAITAI HT8115 / HT7215 / HT6010 / HT5210                 | -                        | -                      | -       | -           | -         | $85 / $75 / $60 / $35 (no encoder)            | -                                |
| CubeMars RO80 KV105                                      | 1.3 @ 15 / 4             | -                      | -       | 352 g       | -         | $104.90                                       | 8.8 / 27                         |
| CubeMars RO60 KV115                                      | 0.8 @ 8.5 / 2.4          | -                      | -       | 248 g       | -         | $77.90                                        | 5.4 / 16                         |
| CubeMars GL60 KV25 / GL80 KV60 (gimbal)                  | 0.6/1.75 ; 1/2.9         | -                      | -       | 230 / 315 g | -         | $108.99 / $181.99                             | too expensive                    |
| CubeMars RI50 / RI60 / RI80 frameless inrunners          | -                        | -                      | -       | -           | -         | $61.90 / $98.90 / $132.90 (no hall)           | need machined housing            |
| mjbots mj5208                                            | 1.7 peak                 | -                      | -       | 193 g       | -         | $74                                           | 11.6 peak                        |
| MAD M6C12 150KV (BHL 6512 actuator)                      | ~1.1 @ 20 A (est.)       | ~0.055 (8.27/Kv, est.) | -       | 257 g       | -         | $129 US / $124 CN (BHL BOM, 2025)             | BHL 15:1 cycloid ramped to 20 Nm |
| MAD 5010 110KV (BHL 5010 actuator)                       | -                        | -                      | -       | -           | -         | $84 US / $62 CN (BHL BOM)                     | -                                |
| Eaglepower LA8308 KV90                                   | -                        | ~0.092 (8.27/Kv, est.) | -       | -           | -         | $42.90-56.40 (Alibaba aggregator, unverified) | -                                |
| Eaglepower 8318 100KV                                    | -                        | -                      | -       | -           | -         | $62-70 (aggregator, unverified)               | heavy (OpenTorque-class)         |
| T-Motor GB54-1 KV33                                      | -                        | -                      | -       | -           | -         | $102.19 (rcdrone, sold out)                   | not competitive                  |
| ODrive M8325s 100KV / D6374 150KV / D5065 270KV          | -                        | -                      | -       | -           | -         | $149 / $119 / $89                             | not competitive                  |

**Cheapest torque per dollar of any motor found: the WK8115_1 at $37.** It has the same electromagnetic table
as the GB8115-1, comes as a rotor + stator kit at 313 g, and suits a turret-mounted capstan or belt stage
(lever 3).

- The catch is current: about 33 A for 5.2 Nm motor torque, which is fine for a 40 A board but not for a
  moteus-c1.
- The high-turn variants (GB8115-4, GB5225, GB6010) need only 2-7 A but give up speed.

---

## 2. (b) Custom multi-axis FOC driver: "FTD-3" (FA-1 Turret Driver, 3 axes)

### 2.1 What the arm asks of it

| Joint                     | Need (cont. / peak) | Candidate motor                                             | Current cont. / peak         | Speed headroom at peak current (est.) 24 V / 48 V |
| ------------------------- | ------------------- | ----------------------------------------------------------- | ---------------------------- | ------------------------------------------------- |
| Yaw                       | 1.5 / 5 Nm          | GIM4315-8                                                   | 1.6 / 5.2 A                  | 233 / 598 rpm                                     |
| Shoulder                  | 6 / 18 Nm           | GIM8108-8                                                   | 6 / 18 A                     | 58 / 188 rpm                                      |
| Shoulder (upgrade)        | 6 / 18 Nm           | GWK7015-8                                                   | 5 / 15 A (26 A for 31.7 Nm)  | 99 / 232 rpm                                      |
| Elbow                     | 4 / 10 Nm           | GIM6010-8 delta                                             | 8.5 / 21 A (37 A for ~17 Nm) | 143 / 392 rpm                                     |
| Wrist, tool roll, gripper | -                   | keep the Hiwonder/Feetech servos; board provides 12 V + bus | -                            | -                                                 |

How the speed estimate was made: speed = Kv_out × (0.9 × Vbus − I_peak × R_l-l), using the vendor's Kv and R,
with R read as line-to-line. If the vendor's 0.67 Ω for the GIM8108-8 is per phase, the 24 V figure roughly
halves.

**Sizing: 40 A peak (about 2 s) and 12 A continuous per channel**, with the heat spreader bolted to an aluminium
turret plate. Bus current stays small: peak electrical power per axis is a few hundred watts, and continuous is
under 5 A at 48 V. One XT30 (15 A) feeds the board.

### 2.2 48 V or 24 V?

- **Board parts cost the same either way.** 100 V FETs (BSC070N10NS5, $0.48) and the 100 V DRV8300 gate driver
  ($0.44) cost what 60 V parts do. The only 48 V premium is the 100 V buck (LM5164, $1.79 instead of $0.54 for a
  60 V TPS54360B) and 100 V capacitors. In total that is about $2 per board.
- **24 V works for this arm.** The shoulder can still deliver peak torque up to about 58 rpm (6 rad/s), which
  covers task motions.
- **48 V gives about 3× the speed headroom at peak torque and half the bus current.** Half the current means
  thinner wires across the yaw joint, and voltage-limited torque roll-off stays out of the operating range.
- **Decision:** design for **12-58 V input** (13S max; DRV8300 absolute maximum is 100 V, SMBJ58A TVS) and run
  at **48 V nominal (12S Li-ion)** if the legs share the bus. Run it unchanged on a 24 V (6S/7S) hexapod if
  that is what the legs use.
- **Regeneration:** a back-driven arm pumps energy into the bus. Feed the board from the battery, not from a
  bench supply without a brake. Steadywin sells a $45 "overvoltage bleeder module" for exactly this case.

### 2.3 Architecture

- **One STM32G431 per axis, all on one 4-layer board on the turret.** This beats one MCU for two or three axes
  because of today's prices:
  - STM32G474CEU6 (the only G4 with two advanced timers, HRTIM and 6 op-amps): **$13.79** at 30+.
  - STM32G431CBT6: **$4.17** (LCSC, 2026-10-17).
  - Three G431s cost less than one G474. Single-axis firmware is far simpler, and every axis runs the same
    binary with its own CAN node ID.
- **Where it goes:** the parallelogram puts the shoulder and elbow motors on the turret, and the yaw motor sits
  just below it.
  - Mount the board on the turret, and mount the **yaw motor's housing on the turret with its output flange on
    the deck**.
  - Then only 4 wires cross the yaw joint (48 V, GND, CAN-H, CAN-L).
  - Motor phase leads are 10-25 cm, and encoder cables are short JST-GH.
- **Per channel:**
  - Gate driver and FETs: DRV8300DRGER with 6 × BSC070N10NS5 (100 V, 7 mΩ, SuperSO8).
  - Current sensing: three low-side 1 mΩ 2512 shunts into the G431's three internal op-amps (PGA), biased the way
    the B-G431B-ESC1 does it. Two ADCs sample in pairs.
  - Hardware overcurrent trip: G431 COMP + DAC into the TIM1 break input. The DRV8300 has no fault or
    current-sense function of its own.
  - Other: NTC, bus-voltage divider, 8 MHz crystal (needed for CAN-FD bit timing), TCAN1044AV CAN-FD
    transceiver, two JST-GH-6 encoder ports (SPI/SSI), SWD pads.
- **Shared section:**
  - XT30 in and out (daisy-chain).
  - MX74700 ideal-diode controller with 2 × BSC070 for reverse-polarity protection.
  - SMBJ58A TVS and bulk electrolytics.
  - LM5164 100 V buck to 12 V (gate drive, 1 A), then TPS563201 to 5 V (transceivers, encoders), then a 3.3 V
    LDO per MCU.
  - CAN-FD in/out JST-GH-4 with ESD diode and switchable 120 Ω termination.
  - Optional **12 V / 5 A servo rail** (TPS54560B) and a 74LVC1G125 half-duplex buffer. One G431's spare UART
    then becomes the Hiwonder/Feetech bus master behind CAN-FD.
- **Encoder satellites:** a 15 × 15 mm two-layer PCB per encoder, with an MT6701 ($1.52, 14-bit, SSI/ABZ/I2C) or
  an AS5047P ($4.12) if moteus compatibility is wanted. One sits behind each rotor and one at each output flange.
  6 per arm.
- **PCB:**
  - About 100 × 70 mm, 4 layers, 1.6 mm. 2 oz outer copper is preferred for 40 A peaks; 1 oz works for 12 A
    continuous with power on two layers.
  - Power stages sit along one edge, against the heat spreader.
  - Conformal coat. The board sits in a sealed enclosure on the turret (IP54 target).

```
 48 V ──XT30──┬─[MX74700 + 2×FET]─┬─ TVS ─ bulk ─┬──────────────┬──────────────┐
              │                    │              │              │              │
 CAN-FD ══════╪════════════════════╪══╦═══════════╪══╦═══════════╪══╦═══ out    │
              │   LM5164→12 V ─ TPS563201→5 V     │  ║           │  ║           │
              │                    │       ┌──────┴──╨──┐ ┌──────┴──╨──┐ ┌──────┴──╨──┐
              │                    │       │ G431 #1    │ │ G431 #2    │ │ G431 #3    │
              │                    │       │ DRV8300    │ │ DRV8300    │ │ DRV8300    │
              │                    │       │ 6×BSC070   │ │ 6×BSC070   │ │ 6×BSC070   │
              │                    │       │ 3×1mΩ+PGA  │ │ 3×1mΩ+PGA  │ │ 3×1mΩ+PGA  │
              │                    │       └─┬──┬──┬────┘ └─┬──┬──┬────┘ └─┬──┬──┬────┘
              │  (opt) TPS54560B→12 V servo  │  M  E1 E2    │  M  E1 E2    │  M  E1 E2
              └─ 74LVC1G125 → servo TTL bus  yaw            shoulder       elbow
```

### 2.4 Component choices (why these)

| Function                   | Chosen (LCSC #, 30+ price)                                                    | Alternatives considered                                                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MCU                        | STM32G431CBT6 (C529355) $4.17                                                 | G474CEU6 $13.79 (needed for moteus firmware); GD32C103 $1.44 (CAN-FD, no op-amps or HRTIM; JLC stock 0); AT32F403A $1.87 (no FD); STSPIN32G4 $4.87 (G431 + gate driver + buck in one package, used by RoboMotor; check its voltage rating) |
| Gate driver                | DRV8300DRGER (C3655801) $0.44: 100 V, 0.75/1.5 A, integrated bootstrap diodes | DRV8323S $3.17 (60 V, SPI, 3 CSAs; used on moteus-c1); DRV8353S/RS $3.72-4.93 (100 V, CSAs; needed for moteus firmware); FD6288Q/JSM6288Q $0.34; EG2133 $0.34                                                                              |
| FETs                       | Infineon BSC070N10NS5 (C534364) $0.475, 100 V 7 mΩ, 27 k in JLC stock         | NCEP0140AG $0.42 (18 mΩ); AON6290 $0.62 (3.6 mΩ); Toshiba TPN19008QM $0.65 (80 V, moteus-c1's FET)                                                                                                                                         |
| Current sensing            | 3 × 1 mΩ 2512 (Ralec C154668) $0.097 + G431 internal PGA                      | INA181A3 $0.35 (or $0.15 clone) per phase; INA240A1 $2.14 inline                                                                                                                                                                           |
| Motor-side encoder         | MT6701QT (C2913974) $1.52                                                     | AS5047P (C962063) $4.12, supported by moteus; MA732 $4.78, supported by moteus; MT6835 $3.93 (21-bit)                                                                                                                                      |
| Output encoder             | MT6701 (same part)                                                            | AS5600 $1.31 (JLC stock ~0), I2C, supported by moteus as a reference source                                                                                                                                                                |
| CAN-FD transceiver         | TCAN1044AVDRQ1 (C3234993) $0.30, 5 Mbps                                       | SIT1044T/3 $0.32; TJA1044GT/3 $0.37; TCAN1057AEV $0.97 (moteus-c1)                                                                                                                                                                         |
| 48 V → 12 V (gate/logic)   | LM5164DDAR (C477928) $1.79, 100 V 1 A                                         | TPS54360B $0.54 (60 V: too close to 13S + regen); MAX15062 $1.32 (60 V, moteus-c1)                                                                                                                                                         |
| 12 V servo rail (optional) | TPS54560B (C1850354) $0.77, 60 V 5 A + SS56                                   | LM5146 controller (100 V) if a 13S pack is used                                                                                                                                                                                            |
| Reverse polarity           | MX74700T (C22468156) $0.35 + 2 FETs                                           | TI LM74700-Q1 $2.06                                                                                                                                                                                                                        |

### 2.5 BOM at qty 10 arms (12 boards incl. 2 spares; 36 channels)

Prices are the JLCPCB/LCSC 30+ ladder on 2026-10-17, which is what JLCPCB charges for parts it places.

**Per channel (Path C)**

| Part                                            | Qty | Unit $ | Line $    |
| ----------------------------------------------- | --- | ------ | --------- |
| STM32G431CBT6                                   | 1   | 4.165  | 4.17      |
| DRV8300DRGER                                    | 1   | 0.443  | 0.44      |
| BSC070N10NS5                                    | 6   | 0.475  | 2.85      |
| 1 mΩ 2512 shunt                                 | 3   | 0.097  | 0.29      |
| TCAN1044AVDRQ1                                  | 1   | 0.301  | 0.30      |
| TLV70233 LDO                                    | 1   | 0.214  | 0.21      |
| 8 MHz crystal                                   | 1   | 0.302  | 0.30      |
| 100 V 4.7 µF 1210                               | 3   | 0.348  | 1.04      |
| 100 V 2.2 µF 1206                               | 2   | 0.134  | 0.27      |
| 63 V 220 µF electrolytic                        | 1   | 0.251  | 0.25      |
| NTC 10k                                         | 1   | 0.075  | 0.08      |
| GH-compatible 6-pin (encoders)                  | 2   | 0.071  | 0.14      |
| Misc. 0402/0603 R/C/LED, ~45 pcs (**estimate**) | 1   | 0.60   | 0.60      |
| **Channel subtotal**                            |     |        | **10.95** |

**Shared per board:** XT30 × 2, ideal diode and 2 FETs, TVS × 2, 2 × 220 µF, LM5164 + inductor, TPS563201 +
inductor, CAN connectors and ESD, servo-bus buffer, misc. **Subtotal $5.21.**
The optional 12 V / 5 A servo rail adds **$2.27**.

**Encoder satellite (each):** MT6701 $1.52, connector $0.07, passives $0.02, panelised two-layer PCB about $0.25
(**estimate**), economic SMT share about $0.25 (**estimate**), diametric magnet about $0.35 (AliExpress,
**estimate**), 30 cm GH cable about $0.40 (**estimate**). **About $2.86 each, 6 per 3-axis arm.**

**Fab and assembly per order of 12 boards:**

| Item                         | Cost                       | Basis                   |
| ---------------------------- | -------------------------- | ----------------------- |
| 4-layer PCB fab, 2 oz outer  | ~$90 for the batch         | **estimate**; see below |
| Shipping                     | $25                        | **estimate**            |
| JLCPCB Standard PCBA setup   | $25.56                     | JLC price page          |
| Stencil                      | $8.21                      | JLC price page          |
| Extended-part feeder loading | 22 unique × $1.53 = $33.66 | JLC price page          |
| SMT joints                   | 620 joints/board × $0.0016 | JLC price page          |
| **Total per board**          | **~$16.2**                 |                         |

The PCB fab figure is an estimate from:

- JLC's published small-batch 4-layer rate of $70.6/m² plus fixed charges (a 2024 notice; an August 2026 notice
  dropped volume surcharges);
- a third-party 2026 estimate of $12-25 for 10 pieces at 100 × 100 mm with 1 oz copper;
- a 15-30% uplift for 2 oz copper.

**Totals (Path C):**

| Variant                                          | Parts per board          | Fab/PCBA per board | Encoders         | Heat spreader + pad (est.) | +10% contingency | **Per board** | **Per axis** |
| ------------------------------------------------ | ------------------------ | ------------------ | ---------------- | -------------------------- | ---------------- | ------------- | ------------ |
| FTD-3 (3 axes)                                   | 3 × 10.95 + 5.21 = 38.05 | 16.19              | 6 × 2.86 = 17.14 | 4                          | yes              | **$82.9**     | **$27.6**    |
| FTD-6 (6 axes, ~140 × 100 mm)                    | 70.88                    | 21.18              | 34.28            | 6                          | yes              | **$145.6**    | **$24.3**    |
| FTD-3M (Path M, moteus-derived: G474 + DRV8353S) | 3 × 23.94 + 5.21 = 77.04 | 16.29              | 17.14            | 4                          | yes              | **$125.9**    | **$42.0**    |

FTD-6 is for a 6-axis arm variant or for a hexapod leg pair. The wrist joints should stay on servos: a BLDC
wrist would add 150-250 g at the tip.

**Prototype / NRE (not included above):** expect two spins of 5 boards each, about $300 per spin (fab, setup,
stencil, extended fees, parts). Bench equipment (current probe, dyno with a load cell) is extra. **Estimate.**

---

## 3. Comparison with off-the-shelf and open designs

| Driver                                             | Price (date)                                                                               | Bus                   | Peak / cont. A                                         | Bus link                                      | Output encoder                                                | Firmware                                  | Fit for FA-1                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------- | ------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **FTD-3, this spec (Path C)**                      | **$27.6/axis** (est., qty 36)                                                              | 12-58 V               | 40 / 12 (with spreader)                                | CAN-FD 5 Mbps                                 | yes, 2 ports per axis                                         | ours (Recoil/RoboMotor/SimpleFOC-derived) | full torque of all candidates                             |
| FTD-3M (moteus-derived)                            | $42/axis (est.)                                                                            | 12-58 V               | 40 / 12                                                | CAN-FD 5 Mbps                                 | yes (moteus `motor_position.sources`)                         | moteus, Apache-2.0                        | same, lowest firmware risk                                |
| mjbots moteus-c1                                   | $69 (2026-10-16)                                                                           | 10-51 V               | 20 / 5 (14 cooled)                                     | CAN-FD                                        | yes, aux SPI/I2C/UART                                         | moteus, open HW + FW                      | elbow capped at ~9.4 Nm (0.47 × 20 A); shoulder ~20 Nm    |
| moteus-n1 / x1 / r4.11                             | $149 / $175 / $94                                                                          | 10-54 V (r4.11: 44 V) | 100 / 9-26 (n1); 120 / 25-62 (x1); 100 / 12-32 (r4.11) | CAN-FD                                        | yes                                                           | moteus                                    | overkill and costly                                       |
| ODrive Micro                                       | $89 ($84 at 10)                                                                            | ≤28-31 V              | 7 / 3.5                                                | CAN / CAN-FD (fw ≥ 0.6.10)                    | SPI (no MOSI), quadrature, Hall                               | closed (0.6.x)                            | far too small                                             |
| ODrive S1 / Pro                                    | $149 / $229                                                                                | ≤50.5 / 58 V          | 80 / 40; 120 / 70                                      | CAN-FD                                        | RS-485, SPI, …                                                | closed                                    | costly                                                    |
| ODESC 3.6 dual (ODrive v3.6 clone)                 | $54.99 for 2 axes (SEQURE)                                                                 | 8-56 V                | 120 / 50 claimed                                       | classic CAN                                   | one encoder per axis (verify load-encoder support)            | ODrive 0.5.x, MIT, frozen                 | cheapest buy option; clone quality and ratings unverified |
| ODESC v4.2 single                                  | $28.99 (SEQURE)                                                                            | 8-56 V                | 120 / 70 claimed                                       | classic CAN                                   | one                                                           | ODrive 0.5.x                              | as above                                                  |
| Steadywin GDS68 (bought alone / as bundle premium) | $33.99 (OpenELAB) / +$20-40 at aifitlab                                                    | 12-56 V               | ~22-23 (vendor stall current)                          | CAN (GDZ: + RS-485)                           | "Separate Encoder Support: NO"; +$10 secondary-encoder option | closed, protocol varies by driver family  | today's baseline                                          |
| B-G431B-ESC1 (Berkeley Humanoid Lite's driver)     | $39.26 (ST eStore, 2026); $19 US / $23 CN in the BHL paper (2025)                          | 3-6S (≤25 V)          | 40 (forced air)                                        | CAN (G431 FDCAN; check the transceiver)       | needs firmware                                                | Recoil (MIT) or ST MCSDK                  | good at 24 V; not 48 V                                    |
| MKS ESP32 FOC V2.0 (dual)                          | $135.99 Newegg; AliExpress price not readable (**estimate $30-40**)                        | 12-24 V               | 6 per channel (12 total)                               | none built in (ESP32 TWAI = classic CAN only) | SPI/I2C/ABI                                                   | SimpleFOC                                 | too small; forum reports of boards burning                |
| SimpleFOC Mini (DRV8313)                           | $5.90-6.20 (Elecrow); €10-15 (v1.1)                                                        | 8-30 V (docs: 8-24 V) | 2.5                                                    | none                                          | via host MCU                                                  | SimpleFOC                                 | gimbal-motor wrist only                                   |
| RoboMotor-GIM6010-V2 (open HW)                     | not sold; parts ~$20-25/axis (**estimate**: STSPIN32G4 $4.87 + FETs + MA732 + 4-layer PCB) | 24 V                  | 40 A                                                   | CAN (DJI protocol)                            | yes (MA730 PCB)                                               | RoboMotor, MIT                            | best match for a bolt-on GIM6010-8 driver                 |

**Who wins where:**

- **FTD-3 is the cheapest per axis that meets the arm's current, CAN-FD and dual-encoder needs.** The ODESC
  clone matches it on price but has classic CAN, a single encoder per axis, frozen firmware, and ratings nobody
  stands behind.
- **moteus-c1 is the best buy-today option** for the first arm. It costs about $41 more per axis than FTD-3, is
  limited to 20 A, and runs the same firmware family as Path M.

---

## 4. Open firmware options

| Firmware                            | License                                | Hardware it targets                                                           | What we get                                                                                                                                                                                                                                                                                                                                                               | Port effort to FTD                                                                                 |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **moteus** (mjbots)                 | Apache-2.0, firmware and PCBs          | STM32G474 + DRV8323/DRV8353 boards (r4, c1, n1, x1)                           | CAN-FD 5 Mbps; position, velocity and torque modes; up to three encoder sources with output-side reference (AS5047, MA600/MA732, AS5600/AS5048B, AksIM-2, AMT21, BiSS-C, quadrature, Hall, sin/cos); mature tooling. Memory map puts the app at 0x08010000-0x0807F000, so it needs the **512 KB G474** and will not fit a 128 KB G431. MT6701 is not on its encoder list. | Low for a pin-compatible clone; moderate to add a board family (Path M)                            |
| **Recoil** (Berkeley Humanoid Lite) | MIT                                    | B-G431B-ESC1 (STM32G431, 3 shunts, internal op-amps)                          | Proven on BHL's 22 actuators; I2C AS5600 (the I2C read limits the loop to ~12 kHz per BHL's timing notes); CAN.                                                                                                                                                                                                                                                           | Moderate: same MCU and sensing topology as FTD Path C; add CAN-FD, SPI encoders, dual-encoder loop |
| **RoboMotor**                       | MIT (2026)                             | STM32G4 / STSPIN32G4, GIM6010-8 with output encoder                           | 17 kHz current loop, 4 kHz speed, 1 kHz position; auto-calibration; protections; DJI-style CAN protocol                                                                                                                                                                                                                                                                   | Moderate; single-author project                                                                    |
| **SimpleFOC**                       | MIT                                    | Arduino/PlatformIO; STM32G4 low-side sensing at ~20-25 kHz, one motor per MCU | Broad encoder support, including MT6701 SSI in the SimpleFOC drivers library (verify)                                                                                                                                                                                                                                                                                     | Moderate-high: no CAN-FD stack or production safety layer built in                                 |
| **ST Motor Control SDK**            | free under ST's licence (ST MCUs only) | G431 3-shunt with internal op-amps (B-G431B-ESC1 reference)                   | Production-grade FOC; code generator                                                                                                                                                                                                                                                                                                                                      | Moderate: add our own CAN-FD protocol and dual-encoder position loop                               |
| **ODrive 0.5.x**                    | MIT                                    | ODrive v3.x (STM32F405 + DRV8301): official boards are NRND; ODESC clones     | Mature, classic CAN; newer ODrive firmware is closed                                                                                                                                                                                                                                                                                                                      | Only sensible if buying ODESC boards                                                               |
| **VESC** (bldc)                     | GPL-3.0 (**not checked this session**) | STM32F4 VESC hardware                                                         | High-power ESC firmware                                                                                                                                                                                                                                                                                                                                                   | Poor fit for a servo arm                                                                           |

---

## 5. Effort and risks

### Effort

Person-weeks for one competent engineer (**estimate**, from typical bring-up of 48 V FOC boards):

| Task                                                                                                 | Path M (moteus-derived) | Path C (G431 + DRV8300) |
| ---------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------- |
| Schematic + layout (3 channels + encoder satellite)                                                  | 3                       | 3-4                     |
| Spin 1 fab/assembly (calendar), bring-up: dead time, ringing, current-sense calibration, thermal     | 3-4                     | 3-4                     |
| Firmware: board family / port, CAN-FD protocol, dual-encoder loop, calibration, faults, host library | 2-4                     | 6-10                    |
| Spin 2 + dyno validation + production of 12                                                          | 4-5                     | 4-6                     |
| **Total**                                                                                            | **~12-17 weeks**        | **~16-24 weeks**        |

At 10 arms, the saving is about $30 per arm against Steadywin bundles, or about $40 per axis against moteus-c1.
Neither covers this effort. **Reusing the board for the 18 leg joints per hexapod** changes the arithmetic: about
$7k saved against moteus-c1 across 10 robots.

### Risks

- **48 V power stage:** a shoot-through or ringing fault can burn the board. Mitigate with a hardware
  comparator trip, a per-board fuse, conformal coat, and bench limits during bring-up.
- **Regenerative overvoltage** when the arm is back-driven or falls: TVS, battery-fed bus, firmware bus-voltage
  braking.
- **Gearbox overload:** 40 A lets the firmware exceed the vendor stall torques (GIM8108-8 22 Nm, GIM6010-8
  11 Nm @ 24 V). Steadywin does not publish gearbox ratings. Clamp the limits to the vendor stall figures until
  a test proves more.
- **Unknown bare-unit interfaces:** the rotor-magnet position and size, and whether GWK units can carry a
  motor-side encoder at all. Vendor GWK tables are internally inconsistent (rated 11.4 Nm @ 7 A against Kt 1.2).
- **Encoder cabling:** SPI/SSI over 10-25 cm on a turret next to PWM phase leads. Use twisted pairs with ground,
  a slower SSI clock, and CRC checks.
- **Calibration per joint:** phase offset, encoder eccentricity, and output-encoder alignment.
- **Firmware bugs can drop the arm.** The gravity springs are a real safety margin here; add a watchdog and
  torque-off-on-CAN-loss.
- **Supply volatility:** STM32G4 prices have roughly doubled to tripled (G474 now $13.79), and some parts
  (AS5600, MR30, GD32C103) show near-zero JLC stock. Pick a second source for each part.
- **Thermal in a sealed box:** 12 A continuous needs the spreader bolted to aluminium. The printed turret alone
  will not do.
- **No vendor support:** it all lands on the team.

---

## 6. What it does to FA-1 (parts cost, BLDC axes only)

| Variant                        | Yaw / shoulder / elbow                     | Motors | Drivers / encoders          | Integration                       | Total            | Peak Nm (yaw / sh / el)          |
| ------------------------------ | ------------------------------------------ | ------ | --------------------------- | --------------------------------- | ---------------- | -------------------------------- |
| **FA-1 today**                 | EduLite 05 / GIM8108-8+enc / GIM6010-8+enc | $320   | integrated                  | + $10 servo adapter               | **$330**         | 6 / 22 / 11                      |
| **CE-A** (same families, bare) | GIM4315-8 / GIM8108-8 / GIM6010-8          | $213   | FTD-3 $82.9                 | servo rail $2.27; adapter removed | **$298** (−$32)  | ~14 vendor / 22 / ~17 est.       |
| **CE-B** (torque upgrade)      | GIM4315-8 / **GWK7015-8** / GIM6010-8      | $218   | $82.9                       | $2.27                             | **$303** (−$27)  | ~14 / **31.7 vendor** / ~17 est. |
| CE-A on moteus-c1 instead      | same as CE-A                               | $213   | 3 × $69 + 6 encoders ≈ $224 | -                                 | **$437** (+$107) | 14 / 20 / 9.4                    |

What changes beyond cost:

- **Continuous torque goes up.** The shoulder rises from 7.5 Nm rated to 11.4 Nm (vendor) with the GWK7015-8.
- **Mass:** about 15-18 g of driver per motor moves off the arm and onto the turret, which is negligible.
- **Cheaper sourcing (lever 1)** stacks on top. The unverified eBay price for the GIM6010-8 ($49.90 + $16)
  would take another ~$5-20 off.

---

## 7. To verify before committing

1. Buy one each of the GWK7015-8, GIM6010-8 bare and GIM4315-8 bare, and check:
   - the rotor magnet (diametric? size? axial gap for an MT6701 or AS5047P);
   - room for an output-flange magnet;
   - the measured Kt and R of each motor.
2. Stall-test the gearboxes at 1.0×, 1.3× and 1.6× vendor stall torque for 2 s, 1,000 cycles each.
3. Decide the hexapod bus voltage (24 V or 48 V) jointly with the legs.
4. Get a real JLCPCB quote for the 4-layer 2 oz board. My fab figure is an estimate.
5. Check that the moteus firmware's board-family detection and the DRV8353 setup run on a G474 + DRV8353S
   3-channel layout (Path M), or that Recoil builds and runs for the new pin map (Path C).
6. Confirm the AliExpress price of the MKS ESP32 FOC V2.0. It does not affect the decision.

---

## Sources

**Motors and actuators**

- aifitlab Steadywin bare units:
  - https://aifitlab.com/products/steadywin-gim6010-8-planetary-reducer-servo-motor
  - https://aifitlab.com/products/steadywin-gim8108-8-planetary-reducer-servo-motor
  - https://aifitlab.com/products/steadywin-gim4315-8-planetary-reducer-servo-motor
  - https://aifitlab.com/products/steadywin-gwk7015-8-motor
  - https://aifitlab.com/products/steadywin-gwk7010-8-motor
  - https://aifitlab.com/products/steadywin-gwk6010-8-motor
  - https://aifitlab.com/products/steadywin-gwk8108-8-motor
  - https://aifitlab.com/products/steadywin-gwk6013-9-motor
  - https://aifitlab.com/products/steadywin-gim3510-8-planetary-reducer-servo-motor
  - https://aifitlab.com/products/steadywin-gim4310-10-planetary-reducer-servo-motor
  - https://aifitlab.com/products/steadywin-gim8108-9-planetary-reducer-servo-motor
- aifitlab Steadywin motor-only units:
  - https://aifitlab.com/products/steadywin-wk8115-motor
  - https://aifitlab.com/products/steadywin-wk10025-motor
  - https://aifitlab.com/products/steadywin-gb8115-motor
  - https://aifitlab.com/products/steadywin-gb7215-motor
  - https://aifitlab.com/products/steadywin-gb6025-motor
  - https://aifitlab.com/products/steadywin-gb5225-motor
  - https://aifitlab.com/products/steadywin-gb6010-motor
  - https://aifitlab.com/products/steadywin-gb4310-motor
- aifitlab, other brands:
  - https://aifitlab.com/products/haitai-hwg7015-j8-motor
  - https://aifitlab.com/products/damiao-dm-s3519-1ec-servo-motor
  - https://aifitlab.com/products/haitai-ht8115-motor
- Steadywin official site: https://steadywin-motor.com
- Other resellers:
  - https://www.roboticscenter.ai/store/product/steadywin-gim6010-8-planetary-reducer-servo-motor/
  - https://www.ebay.de/itm/375143033892 (unverified; page returned 403)
- OpenELAB:
  - https://openelab.com/products/steadywin-gim6010-8-brushless-gearmotor
  - https://openelab.com/products/steadywin-motor-driver-can-485 (GDS68 $33.99)
- CubeMars:
  - https://store.cubemars.com/products/ak40-10-kv170
  - https://store.cubemars.com/products/ak45-10-kv75
  - https://store.cubemars.com/products/ak45-36-kv80
  - https://store.cubemars.com/products/ro60
  - https://store.cubemars.com/products/ro80
  - https://store.cubemars.com/products/ri50
  - https://store.cubemars.com/products/ri60
  - https://store.cubemars.com/products/ri80
  - https://store.cubemars.com/products/gl60-kv25
  - https://store.cubemars.com/products/gl80-kv60
- mjbots: https://mjbots.com/products/mj5208
- MAD Components:
  - https://openelab.io/products/mad-m6c12-eee-kv150-motor
  - https://openelab.io/products/mad-5010-eee-110kv-motor
  - Berkeley Humanoid Lite BOM tables in https://arxiv.org/abs/2504.17249 and https://arxiv.org/html/2504.17249v1
- Eaglepower:
  - https://fr.accio.com/plp/eaglepower-90kv-bldc-motor (aggregator)
  - https://robu.in/product/eaglepower-la-8308-kv90-motor/
- T-Motor GB54-1: https://rcdrone.top/products/t-motor-high-torque-low-speed-gimbal-motor-gb54-1-kv33
- ODrive motors:
  - https://shop.odriverobotics.com/products/m8325s
  - https://shop.odriverobotics.com/products/odrive-custom-motor-d6374-150kv

**Drivers**

- moteus:
  - https://mjbots.com/products/moteus-c1
  - https://mjbots.com/products/moteus-n1
  - https://mjbots.com/products/moteus-x1
  - https://mjbots.com/products/moteus-r4-11
  - specs and license: https://github.com/mjbots/moteus
  - c1 KiCad design and BOM: https://github.com/mjbots/moteus/tree/main/hw/c1
  - memory map: https://github.com/mjbots/moteus/blob/main/fw/stm32g474.ld
  - encoder support: https://mjbots.github.io/moteus/reference/encoders/
- ODrive:
  - https://shop.odriverobotics.com/products/odrive-micro
  - https://docs.odriverobotics.com/v/latest/hardware/micro-datasheet.html
  - https://docs.odriverobotics.com/v/latest/hardware/odrive-comparison.html
  - https://shop.odriverobotics.com/products/odrive-s1
  - https://shop.odriverobotics.com/products/odrive-pro
  - legacy open firmware: https://github.com/odriverobotics/ODrive
- ODESC clones:
  - https://sequremall.com/products/odesc-dual-drive
  - https://sequremall.com/products/odesc-v4-2-single-drive-high-current-high-precision-brushless-servo-motor-controller-based-on-odrive3-6-upgrade-software-configuration-compatible-with-odrivetool-foc-bldc
- B-G431B-ESC1: https://estore.st.com/en/b-g431b-esc1-cpn.html
- Berkeley Humanoid Lite:
  - https://arxiv.org/abs/2504.17249
  - https://github.com/HybridRobotics/Berkeley-Humanoid-Lite
  - https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/flashing-the-motor-controllers
  - https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/motor-controller-firmware-execution-timing-information
- Recoil firmware: https://github.com/T-K-233/Recoil-Motor-Controller-BESC
- RoboMotor:
  - https://github.com/baoqi-zhong/RoboMotor
  - https://github.com/baoqi-zhong/RoboMotor/tree/main/Hardware
- MKS ESP32 FOC:
  - https://github.com/makerbase-motor/MKS-ESP32FOC/tree/MKS-ESP32-FOC-V2.0
  - https://www.newegg.com/p/0J2-0788-00977
  - https://community.simplefoc.com/t/do-not-buy-the-mks-dual-foc-plus/8165
- SimpleFOC:
  - https://docs.simplefoc.com/simplefocmini
  - https://github.com/simplefoc/SimpleFOCMini
  - https://elecrow.com/simple-foc-mini-v1-0-drv8313-dc-brushless-motor-driver-board-foc-control-driver-svpwm-control-dc-8-30v-module.html
  - https://docs.simplefoc.com/low_side_current_sense
- Other open designs:
  - SF-Motion / FOC_KING: https://github.com/nordstream3/FOC
  - SojournerST: https://github.com/ckoern/SojournerST-hardware

**Parts and fabrication**

- LCSC part pages: `https://www.lcsc.com/product-detail/<C#>.html`. The C numbers are in section 2.4: C529355,
  C1235412, C3655801, C701784, C701785, C534364, C154668, C3234993, C2913974, C962063, C477928, C1850354,
  C22468156, C123134, C57131, C329955, C170101, C350177, C428721, C116592, C51940119, C5278878, C52140430,
  C7434864.
- TI DRV8300: https://www.ti.com/product/DRV8300
- JLCPCB:
  - assembly prices: https://jlcpcb.com/help/article/pcb-assembly-price
  - assembly FAQ: https://jlcpcb.com/help/article/pcb-assembly-faqs
  - 4-layer rate: https://jlcpcb.com/news/discount-on-quality-4-layer-pcbs
  - August 2026 price adjustment: https://jlcpcb.com/news/jlcpcb-fr4-pcb-price-adjustment-august2026
- Third-party PCB cost estimate: https://atlaspcb.com/blog/multilayer-pcb-cost-2026-pricing-analysis
