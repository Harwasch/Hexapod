# 03 — Early architecture levers: performance, efficiency, cost

Decisions that are cheap now and expensive later, in the order they change
the numbers in [01-sizing.md](01-sizing.md). Each one names what it buys and
what it costs; none is decided here.

## 1. Mass is the master lever

Every joint torque, the battery, the gearbox pin loads and the structure scale
with the robot's mass, and the actuators are 40 % of it. Per-DOF sizing (§6
of the sizing document) already takes the yaw and knee actuators below the
femur's rating. Holding the actuator at 1.1 kg and the robot near 49 kg is
worth more than any single clever mechanism; letting it drift to 60 kg costs
20 % on everything downstream.

## 2. One motor, three ratios

The requirement per DOF is different but the motor need not be: the same
stator PCB, magnets and rotors with a different cycloid disc and pin ring give
a yaw, a femur and a knee actuator from one production line. That keeps the
expensive parts (magnets, PCB tooling, drivers, encoders) at one part number
and puts the variation in the cheapest part, the machined disc. A smaller yaw
motor saves ~0.3 kg per leg but adds a second stator design; do it only after
the first actuator has been measured.

## 3. Standing should cost nothing

At the neutral stance the femur holds ~55–65 N·m with zero motion, and with
no help that is I²R in the copper for the whole time the robot stands, waits
or looks around. Three ways to make it free, in order of preference:

* **Gravity-compensation spring** on the femur axis: a torsion or gas spring
  sized to the neutral-stance torque cancels most of the holding current and
  halves the motor's RMS duty on a walk. Costs a spring per leg and a little
  swing-phase torque against it.
* **Non-backdrivable ratio.** A cycloid above ~40:1 with grease holds a load
  with little or no current, which the per-DOF ratios already give the femur.
  The cost is impact: a foot that lands hard drives the reducer backwards
  through its pins. Pair it with the next lever.
* **Series elasticity in the tibia or the foot**, a few millimetres of travel
  under peak load. It protects the cycloid from impact, gives a clean contact
  signal, and stores and returns a small part of each step. It is also the
  torque sensor a high-ratio joint otherwise lacks.

## 4. Where the femur and knee motors sit

The "motors in the body" goal and the sprawl leg together say: both pitch
drives cross the yaw axis concentrically, run along the coxa, and reach the
femur axis. From there the knee drive has two options:

* a **linkage or belt along the femur** to the knee (serial leg, as drawn);
* a **five-bar** where the second drive acts on a link from the femur axis and
  the knee is a passive pivot. Both pitch motors then sit coaxially at the
  femur axis and nothing moves along the femur. The five-bar also reshapes the
  torque split between the two motors across the workspace, which is the
  knee-versus-femur problem §3 of the topology document raised.

This is the transmission chunk (M1) and it decides more than the actuator:
losses (budget 10 %), reflected inertia, sealing of the yaw crossing, and
whether the hip stack really can be three pancakes on one axis.

## 5. Sensing that does not depend on backdrivability

With 40–60:1 in the loop, motor current says little about foot force. Plan
for an absolute encoder on every joint output (the cycloid has backlash and
the yaw crossing adds compliance), foot contact sensing (the elastic element
above, or a simple switch), and a body IMU. Cheap in parts, expensive to add
after the housings are cut.

## 6. Electrical architecture

* **48 V bus**, two hot-swap packs each able to carry the whole peak.
  Commodity FETs, connectors and chargers; low enough to work on safely.
* **One driver board per hip pod, three axes**, not eighteen boards. The
  motors are in the body and 100 mm from each other; a six-board system has
  a third of the connectors and a shared bus capacitor bank. CAN-FD or
  EtherCAT to a single compute module.
* **Regenerative braking into the bus** is free with FOC and matters on a
  walk down a slope; size the bus capacitance and the pack's charge current
  for it rather than a brake resistor.
* **Thermal path**: a PCB stator's only heat sink is its own copper unless it
  is bonded to something. Stack the pancakes on aluminium spacers tied into
  the hip pod and the floor plate, so the body is the heat sink. The solar
  skin on the top deck argues against using the deck for it.

## 7. Gait and control

A hexapod's static stability is its efficiency lever: a wave or ripple gait
carries a payload with no balance control at all, and a tripod gait at 1 m/s
needs a 1.25 Hz cycle and 9 rad/s at the yaw joint, nothing dynamic. Design
the controller around foot-placement and posture (the femur angle trades
torque against stance width at run time) rather than around agility, and
the actuators can stay on the ratio side of the transparency trade-off.

## 8. Cost

The magnets and the machined cycloid are the cost of the actuator; the stator
PCB and the driver are cheap in volume. Six identical hip pods, one motor
part number, three disc variants, one driver board, and carbon-tube legs with
printed or cast nodes is the bill of materials to aim for. The benchmark to
beat is eighteen off-the-shelf actuators at roughly $8–11k, and the first
prototype leg should probably use them so the transmission and software are
not waiting on the motor.

## 9. Not decided, deliberately

Four legs versus six (12 actuators and dynamic balance versus 18 and static
stability), a fourth joint per leg, wheels on the feet for flat ground, and
whether the trash-pickup tool is a deck arm or a free leg. Each is a real
option; each is a requirements question before it is a design one.
