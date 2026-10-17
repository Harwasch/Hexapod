# Verification of the v2 cost-lever research

Each of the four reports in this folder (`sourcing.md`, `custom_electronics.md`,
`reductions.md`, `rugged.md`) was checked by a separate agent that tried to refute
it from primary sources: live store carts and shipping quotes, datasheets, papers and
the JLCPCB and LCSC part APIs. All checks were made on 2026-10-17. The reports are kept
as written. This page lists what the checks changed and where each correction now sits
in the model.

## Sourcing

| Finding                                                                                                                                                     | Applied in                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| The 40.3% duty stack on Chinese motors is confirmed (MFN 2.8% + Section 301 25% + the July 2026 forced-labour tier 12.5%).                                  | `costs.yaml: import_duty_pct`                           |
| Every price is compared landed, because the "retail" stores ship from China with duty unpaid.                                                               | `parts.landed`, `actuators.yaml: landed_usd`            |
| Zennixtek's real shipping at qty 10 is $7–21 per unit, not $5–15.                                                                                           | `price_alt_usd` for every Zennixtek part                |
| Hiwonder offers USPS duty-paid shipping: HX-35HM lands at about $39, not $52, and HX-65HM at about $54, not $73.                                            | `landed_usd`                                            |
| Taobao-agent and AliExpress prices could not be verified. Only verified prices are used, for example EduLite 05 at $82 from OpenELAB rather than about $71. | `price_alt_*`; the unverified figures stay in the notes |
| Steadywin's $70.36 "GWK8108-8" is a GIM8108-8 core (5 in stock). The 10.2 / 24.8 Nm table belongs to the WGG8108-8.                                         | `bare_units.yaml`                                       |
| OpenELAB sells a housed, driverless GIM8108-8 at $113.99 duty-paid.                                                                                         | `bare_gim8108_8: price_alt_usd`                         |
| Express shipping from China at qty 10 is about $15–16 per unit.                                                                                             | `costs.yaml: default_shipping_usd`                      |

## Custom electronics

| Finding                                                                                                                                                                          | Applied in                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| The FTD-3 board is about $121–145 landed, not $83. The original figure left out US duty, JLC feeder and X-ray fees, a fuse, phase connectors, a brake chopper and the enclosure. | `costs.yaml: custom_driver_*` (≈ $133 per board)       |
| GIM4315-8: the vendor table contradicts itself. Peak is about 6–10 Nm, not 14.25.                                                                                                | `bare_units.yaml`                                      |
| GIM6010-8 at 37 A gives about 15 Nm, not 17. GWK6010-8 gives about 12.7 Nm, limited by its resistance.                                                                           | `bare_units.yaml`                                      |
| Off-the-shelf alternatives: the GIM6010-8 48-slot star winding with GDS68 and output encoder ($105, 17.9 Nm at 48 V), and the GWK7015-8 with GDS68 ($115, no output encoder).    | `actuators.yaml: sw_gim6010_8_s48, sw_gwk7015_8_gds68` |
| Design errors to fix before layout: the 63 V bulk capacitors sit below the TVS breakdown voltage; the servo-rail buck must be a 100 V part; add a brake chopper.                 | `docs/ELECTRONICS.md`                                  |
| The engineering effort (12–24 person-weeks) has no primary source and is likely optimistic.                                                                                      | `docs/ELECTRONICS.md`                                  |

## Printed reductions

| Finding                                                                                                                                                     | Applied in                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Capstan efficiency of 95% was measured on polished metal in the forward direction only. Printed drums and backdrive are unmeasured, so the model uses 0.92. | `transmissions.yaml`                    |
| At D/d 8 the rope keeps about 85% of its strength, so the torque limit uses that bent strength.                                                             | `transmissions.yaml: bend_strength`     |
| The 1.5 mm rope that fits a small drum is SK78, which creeps. No 1.5 mm DM20 source was found. Only 2 mm DM20 is used.                                      | `transmissions.yaml: ropes`             |
| At about 35% of strength at every peak, life is about 20–35k peak cycles. Sandia's 247k cycles corresponds to about 17%.                                    | `DESIGN.md` risks                       |
| Joint stiffness is more likely 0.6–1.0 kNm/rad once the pinion wraps and the printed structure are counted. The output encoder closes the loop on it.       | `DESIGN.md`                             |
| A WK8115 behind a 1.5 mm capstan cannot use its 35–39 Nm. The rope limits it to about 20 Nm.                                                                | rope limit in `parts.load_stages`       |
| Printed belt pulleys: 25–45 arcmin of backlash; efficiency is lower when the belt is taut.                                                                  | `transmissions.yaml: belt` (0.5°, 0.93) |

## Ruggedization

| Finding                                                                                                                                                                                               | Applied in                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| The kit costs about $50–65 once the servo pods have metal output pins and the heat plate is big enough.                                                                                               | `costs.yaml: seals` ($57)          |
| The turret is not a sealed box. The yaw interface is an open labyrinth, so realistic protection is about IPX3–4 and IP5X. Either accept that or add a contact seal at the yaw (0.17–0.30 Nm of drag). | `DESIGN.md` risks                  |
| A 0.1 kg plate cannot shed 30–50 W. The motors need about 0.2 m² of exposed aluminium, or a low continuous torque, which the springs provide.                                                         | `MassModel.turret_kg`, `DESIGN.md` |
| RobStride rates continuous torque at 25 °C on a heat sink. The EduLite 05's 1.8 Nm leaves little margin over a 1.5 Nm yaw need at 45 °C.                                                              | `DESIGN.md` risks                  |
| STS3215 gears strip at about 0.44 Nm, so set the gripper clutch and firmware limit near 0.4 Nm, not at 60–70% of stall.                                                                               | `DESIGN.md`                        |
| In Miami-like sun, Dyneema loses about 40% of its strength in 1–1.5 years. Keep the ropes in the shade, inside the turret.                                                                            | CAD and `DESIGN.md`                |
