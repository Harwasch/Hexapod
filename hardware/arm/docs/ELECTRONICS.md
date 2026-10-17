# FTD-3 turret driver (custom-electronics lever)

FTD-3 is a 3-axis field-oriented-control (FOC) driver for the motors on the turret: the yaw,
the shoulder and the parallelogram's elbow. It lets the arm buy bare motor and gearbox units
instead of integrated actuators, and it hosts the wrist servo bus. The research and its
verification are in [research/custom_electronics.md](research/custom_electronics.md) and
[research/verification.md](research/verification.md).

**Status: specified, not chosen for FA-2.**

- The study priced the board in as one of three cost levers. With it, bare Steadywin units
  reach the highest capability per dollar above about $780 delivered. That is the "Max"
  pick: $810, 91% of task points, three GWK6010-8 units, two of them through rope capstans.
- FA-2 ($737) uses bought integrated actuators instead. The board's engineering is 16–24
  or more person-weeks, and 10 arms never repay it.
- Build the board when the hexapod's legs will use it too.

## What it has to do

| Joint    | Continuous / peak at the joint | Phase current per channel                               |
| -------- | ------------------------------ | ------------------------------------------------------- |
| Yaw      | 1.5 / 5 Nm                     | a few A                                                 |
| Shoulder | 6 / 18 Nm                      | 6 / 18 A on a GIM8108-8-class unit (Kt ≈ 1 Nm/A at 8:1) |
| Elbow    | 4 / 10 Nm                      | up to ~37 A for a GIM6010-8 delta at 15 Nm              |

**Sizing:** each channel delivers 40 A for about 2 s and 12 A continuous, with its heat
spreader bolted to aluminium. The bus runs at 48 V nominal, with 12–58 V accepted.

## Architecture

```text
 48 V ─XT30─ fuse ─ ideal diode ─ TVS ─ 100 V bulk ─┬─ brake chopper
                                                    │
 CAN-FD ════════════╦══════════════╦══════════════╦═══ out
                    ║              ║              ║
               ┌────╨─────┐   ┌────╨─────┐   ┌────╨─────┐
               │ G431     │   │ G431     │   │ G431     │   one MCU per axis,
               │ DRV8300  │   │ DRV8300  │   │ DRV8300  │   same firmware,
               │ 6×100 V  │   │ 6×100 V  │   │ 6×100 V  │   own CAN ID
               │ 3 shunts │   │ 3 shunts │   │ 3 shunts │
               └─┬──┬──┬──┘   └─┬──┬──┬──┘   └─┬──┬──┬──┘
                 M  E1 E2       M  E1 E2       M  E1 E2     E1 motor, E2 output (MT6701)
                 yaw            shoulder       elbow
 100 V buck → 12 V servo rail + half-duplex TTL bus → wrist and gripper servos
```

- **One STM32G431 per axis.** Single-axis firmware is simpler, and every axis runs the same
  binary.
  - It costs $4.13 at 30+.
  - A 128 KB G474CB costs $3.52 and would also do.
  - Only Path M (the moteus port) needs the 512 KB G474CE, at $13.79.
- **Two encoders per axis.** An MT6701 sits behind the rotor and another at the output
  flange. Use the SOP-8 MT6701CT ($1.04, in stock), which avoids X-ray inspection.
- **Where it sits.** The board is in the turret next to the motors. Only 48 V, ground and
  CAN cross the yaw.
- **Corrections from the verification, to apply before layout:**
  - 63 V bulk capacitors sit below the SMBJ58A's breakdown voltage, so the TVS never
    protects them. Use 80–100 V capacitors (100 V 100 µF is $0.10–0.24).
  - Add a regenerative brake chopper. A back-driven or falling arm pumps the bus.
  - The servo-rail buck must be a 100 V part, not the 60 V TPS54560B.
  - Add a fuse and MR30 phase connectors.
  - Size the bulk capacitance for 12 A per channel. One 220 µF can is too small.

## Cost, landed in the US, about 10 arms

| Item                                                                      | Per board  |
| ------------------------------------------------------------------------- | ---------- |
| Parts: 3 power channels ($10.95 each) + shared section + 6 encoder boards | ≈ $60      |
| JLCPCB fab and assembly (feeder loading, X-ray, through-hole, packing)    | ≈ $22–28   |
| Fuse, phase connectors, brake chopper, sealed box, contingency            | ≈ $15–30   |
| US duty (35% on PCB and SMT at JLC's duty-paid rates)                     | ≈ $25–30   |
| **Total**                                                                 | **≈ $133** |

The study charges $27.50 per axis plus $50.50 per board, and $3.20 for the servo rail, which
replaces the $10 servo-bus adapter. For comparison:

- moteus-c1 is $69 per axis but limited to 20 A.
- Steadywin's own GDS68 adds $20–40 per actuator.

## Firmware paths

| Path                  | Board                     | Per axis (ex duty) | Firmware                                                               | Effort (estimate)  |
| --------------------- | ------------------------- | ------------------ | ---------------------------------------------------------------------- | ------------------ |
| **M**: moteus-derived | G474CE + DRV8353S, 3 axes | ≈ $42              | moteus (Apache-2.0), new board definition only                         | 12–17 person-weeks |
| **C**: low-cost       | G431 + DRV8300, 3 axes    | ≈ $32–40           | port Recoil (MIT, Berkeley Humanoid Lite) or RoboMotor (MIT, immature) | 16–24 person-weeks |

Both effort figures have no primary source and are likely optimistic by 1.5–2× for a team
without motor-control experience. They leave out:

- the production test fixture;
- enclosure and thermal work;
- gearbox stall tests;
- EMC and encoder-noise debugging;
- integration with the arm controller.

## When it pays

- **At 10 arms it never pays back.** Even on parts alone, the saving per arm is below the
  cost of two prototype spins (about $600), before any labour.
- **It pays only if the hexapod's 18 leg joints use it too.** Against moteus-c1, that saves
  about $3.8–5.2k across 10 robots, which is still below the paid cost of the engineering.
  Choose it to own the motor stack across the fleet, not to save money on the arm.

## Before committing

1. **Buy one of each bare unit** (GWK6010-8, GIM6010-8, GIM8108-8) and check:
   - whether the rotor magnet can carry an MT6701 behind it;
   - whether there is room for an output-flange magnet;
   - the measured Kt and R.
2. **Stall-test the gearboxes** at 1.0×, 1.3× and 1.6× the vendor stall torque for 2 s,
   1,000 cycles each. Clamp the firmware to vendor stall until this passes.
3. **Decide the bus voltage with the legs.** The board works on 24 V or 48 V; at 48 V the
   star-wound GIM6010-8 reaches 17.9 Nm on its own driver.
4. **Get a real JLCPCB quote** for the 4-layer, 2 oz board.
