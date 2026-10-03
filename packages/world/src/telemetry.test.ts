import { describe, expect, it } from "vitest";

import {
  applyMotion,
  ecefToGeodetic,
  enuToEcefRotation,
  fadeTowardRest,
  geodeticToEcef,
  pathPose,
  poseFromScan,
  poseToScan,
  PoseTrack,
  quatFromHeadingPitchRoll,
  quatFromYaw,
  quatSlerp,
  relativeMotion,
  scanFrame,
  syntheticReading,
  translationAbout,
  type Pose,
  type SyntheticTelemetry,
} from "./telemetry";
import { QUAT_IDENTITY, quatRotate, type Quat, type Vec3 } from "./vec";

/** The synthetic yard's root transform (data/tiles/synthetic-yard/splat/tileset.json). */
const YARD_ROOT = [
  0.8302069242949459, -0.5574553460616607, 0.0, 0.0, 0.4019010052271058, 0.5985430039875085,
  0.6929804141352418, 0.0, -0.38630563657571415, -0.575317138215857, 0.7209564103501326, 0.0,
  -2468208.195224001, -3675852.3328428925, 4575543.044746802, 1.0,
];

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i += 1) {
    expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0))).toBeLessThan(tol);
  }
}

/** Same rotation (q and −q are one rotation). */
function sameRotation(a: Quat, b: Quat, tol: number): void {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  expect(1 - dot).toBeLessThan(tol);
}

const pose = (position: Vec3, yawDeg = 0): Pose => ({
  position,
  orientation: quatFromYaw((yawDeg * Math.PI) / 180),
});

describe("frames", () => {
  it("geodetic ↔ ECEF round-trips to a tenth of a millimetre", () => {
    for (const g of [
      [-93.4, 44.9, 280],
      [0, 0, 0],
      [179.9, -89.5, 4000],
      [12.5, 60.1, -30],
    ] as Vec3[]) {
      close(ecefToGeodetic(geodeticToEcef(g)), g, 1e-7);
      const back = geodeticToEcef(ecefToGeodetic(geodeticToEcef(g)));
      close(back, geodeticToEcef(g), 1e-4);
    }
    // The equator at the prime meridian is a semi-major axis away.
    close(geodeticToEcef([0, 0, 0]), [6378137, 0, 0], 1e-6);
  });

  it("the ENU rotation points north along the meridian and up along the normal", () => {
    const q = enuToEcefRotation(0, 0);
    close(quatRotate(q, [1, 0, 0]), [0, 1, 0], 1e-12);
    close(quatRotate(q, [0, 1, 0]), [0, 0, 1], 1e-12);
    close(quatRotate(q, [0, 0, 1]), [1, 0, 0], 1e-12);
  });

  it("brings ECEF and geodetic poses into the scan frame and back", () => {
    const scan = scanFrame(YARD_ROOT);
    if (!scan) throw new Error("singular");
    const local = pose([20, 6.2, -0.1], 37);
    for (const frame of ["scan", "ecef", "geodetic"] as const) {
      const out = poseFromScan(local, frame, scan);
      if (!out) throw new Error("no pose");
      const back = poseToScan(out, frame, scan);
      if (!back) throw new Error("no pose");
      close(back.position, local.position, 1e-6);
      sameRotation(back.orientation, local.orientation, 1e-12);
    }
    // ECEF: the scan origin is the root's translation.
    const origin = poseFromScan(pose([0, 0, 0]), "ecef", scan);
    close(origin?.position ?? [], YARD_ROOT.slice(12, 15), 1e-6);
    // Without a placement only the scan frame can be read.
    expect(poseToScan(local, "ecef", undefined)).toBeUndefined();
    expect(poseToScan(local, "scan", undefined)?.position).toEqual(local.position);
  });

  it("the yard's root is a local level frame: a heading in ENU is the same yaw in the scan", () => {
    const scan = scanFrame(YARD_ROOT);
    if (!scan) throw new Error("singular");
    const at: Vec3 = [16, 12, 0];
    const geo = poseFromScan(pose(at), "geodetic", scan);
    if (!geo) throw new Error("no pose");
    // Heading 0 (north) is +y in a level frame, so a body facing north has yaw 90°.
    const north = poseToScan(
      { position: geo.position, orientation: quatFromHeadingPitchRoll(0) },
      "geodetic",
      scan,
    );
    sameRotation(north?.orientation ?? QUAT_IDENTITY, quatFromYaw(Math.PI / 2), 1e-9);
  });

  it("heading, pitch and roll follow their conventions", () => {
    // Heading 90° faces east: the body x axis stays east.
    close(quatRotate(quatFromHeadingPitchRoll(90), [1, 0, 0]), [1, 0, 0], 1e-12);
    // Nose up 30°: forward tilts up.
    const up = quatRotate(quatFromHeadingPitchRoll(90, 30), [1, 0, 0]);
    close(up, [Math.cos(Math.PI / 6), 0, Math.sin(Math.PI / 6)], 1e-12);
    // Roll 90° right side down: the left axis points up.
    close(quatRotate(quatFromHeadingPitchRoll(90, 0, 90), [0, 1, 0]), [0, 0, 1], 1e-12);
  });
});

describe("rigid motion", () => {
  it("is the identity at rest and carries the rest pose onto the reading", () => {
    const rest = pose([20, 6, 0], 90);
    const still = relativeMotion(rest, rest);
    close(still.translation, [0, 0, 0], 1e-12);
    sameRotation(still.rotation, QUAT_IDENTITY, 1e-15);
    const now = pose([22, 9, 0.5], 135);
    const m = relativeMotion(rest, now);
    close(applyMotion(m, rest.position), now.position, 1e-12);
    // A point a metre ahead of the body at rest is a metre ahead of it now.
    const ahead = applyMotion(m, [20, 7, 0]);
    close(ahead, [22 - Math.SQRT1_2, 9 + Math.SQRT1_2, 0.5], 1e-12);
  });

  it("about an origin, the same motion as a constant skin handle", () => {
    const m = relativeMotion(pose([3, 4, 0], 10), pose([5, 1, 0.2], 70));
    const o: Vec3 = [4, 4, 0];
    const t = translationAbout(m, o);
    for (const x of [
      [0, 0, 0],
      [4, 4, 0],
      [7, -2, 3],
    ] as Vec3[]) {
      // x + (R − I)(x − o) + t_o
      const r = quatRotate(m.rotation, [x[0] - o[0], x[1] - o[1], x[2] - o[2]]);
      const viaHandle: Vec3 = [r[0] + o[0] + t[0], r[1] + o[1] + t[1], r[2] + o[2] + t[2]];
      close(viaHandle, applyMotion(m, x), 1e-12);
    }
  });

  it("slerps the short way and extrapolates along the same turn", () => {
    const a = quatFromYaw(0);
    const b = quatFromYaw(Math.PI / 2);
    sameRotation(quatSlerp(a, b, 0.5), quatFromYaw(Math.PI / 4), 1e-12);
    sameRotation(quatSlerp(a, b, 1.5), quatFromYaw((3 * Math.PI) / 4), 1e-12);
    const negB: Quat = [-b[0], -b[1], -b[2], -b[3]];
    sameRotation(quatSlerp(a, negB, 0.5), quatFromYaw(Math.PI / 4), 1e-12);
  });

  it("fades toward rest linearly in position", () => {
    const rest = pose([0, 0, 0]);
    const p = pose([4, 0, 0], 90);
    expect(fadeTowardRest(rest, p, 1)).toBe(p);
    close(fadeTowardRest(rest, p, 0.25).position, [1, 0, 0], 1e-12);
    sameRotation(fadeTowardRest(rest, p, 0).orientation, QUAT_IDENTITY, 1e-15);
  });
});

describe("the playout track", () => {
  /** A body driving east at 2 m/s, a reading every `period` ms of source time. */
  const east = (t: number): Pose => pose([(2 * t) / 1000, 0, 0]);

  it("is at rest before any reading", () => {
    const track = new PoseTrack();
    const r = track.read(1000);
    expect(r.state).toBe("none");
    expect(r.pose).toBeNull();
  });

  it("interpolates between readings at the playout delay, whatever the source clock", () => {
    // The source's clock runs 1 h ahead; readings take 80 ms to arrive.
    const skew = 3_600_000;
    const track = new PoseTrack({ latencyMs: 200 });
    for (let t = 0; t <= 2000; t += 500) track.push(t + skew, east(t), t + 80);
    expect(track.offsetMs).toBe(80 - skew);
    // Local 1330 → source 1330 − 80 − 200 = 1050 (+ skew): between 1000 and 1500.
    const r = track.read(1330);
    expect(r.state).toBe("live");
    close(r.pose?.position ?? [], [2.1, 0, 0], 1e-9);
    expect(r.playoutMs).toBe(1050 + skew);
  });

  it("takes late and out-of-order readings into place", () => {
    const track = new PoseTrack({ latencyMs: 0 });
    track.push(0, east(0), 10);
    track.push(1000, east(1000), 1010);
    track.push(500, pose([5, 5, 0]), 1200); // late and odd, but inside the window
    const r = track.read(760);
    expect(r.state).toBe("live");
    // Between 500 (5, 5) and 1000 (2, 0) at 0.5.
    close(r.pose?.position ?? [], [3.5, 2.5, 0], 1e-9);
  });

  it("dead-reckons past the last reading, holds at the limit, then fades to rest", () => {
    const track = new PoseTrack({
      latencyMs: 0,
      extrapolateMs: 1000,
      staleMs: 3000,
      stale: "rest",
      fadeMs: 2000,
    });
    track.push(0, east(0), 0);
    track.push(500, east(500), 500);
    const ahead = track.read(1300);
    expect(ahead.state).toBe("extrapolated");
    close(ahead.pose?.position ?? [], [2.6, 0, 0], 1e-9);
    const held = track.read(2500);
    expect(held.state).toBe("held");
    close(held.pose?.position ?? [], [3, 0, 0], 1e-9);
    const fading = track.read(4500);
    expect(fading.state).toBe("stale");
    expect(fading.weight).toBeCloseTo(0.5, 12);
    const rested = track.read(5600);
    expect(rested.state).toBe("rest");
    expect(rested.pose).toBeNull();
    // A new reading brings it back.
    track.push(5600, east(5600), 5600);
    expect(track.read(5600).state).toBe("live");
  });

  it("freezes when told to", () => {
    const track = new PoseTrack({ latencyMs: 0, extrapolateMs: 0, staleMs: 1000, stale: "freeze" });
    track.push(0, east(0), 0);
    track.push(500, east(500), 500);
    const r = track.read(60_000);
    expect(r.state).toBe("stale");
    expect(r.weight).toBe(1);
    close(r.pose?.position ?? [], [1, 0, 0], 1e-12);
  });

  it("does not extrapolate across a long gap", () => {
    const track = new PoseTrack({ latencyMs: 0, extrapolateMs: 1000 });
    track.push(0, east(0), 0);
    track.push(10_000, east(10_000), 10_000);
    close(track.read(10_500).pose?.position ?? [], [20, 0, 0], 1e-12);
  });
});

describe("synthetic telemetry", () => {
  const loop: SyntheticTelemetry = {
    path: { kind: "circle", centre: [17.5, 6.22, -0.11], radius: 2.5, periodS: 24 },
    rateHz: 2,
    delayMs: 40,
    jitterMs: 60,
    dropouts: [[10_000, 12_000]],
  };

  it("drives round a circle facing along it, and is the same every run", () => {
    const p0 = pathPose(loop.path, 0);
    close(p0.position, [20, 6.22, -0.11], 1e-12);
    sameRotation(p0.orientation, quatFromYaw(Math.PI / 2), 1e-15);
    const quarter = pathPose(loop.path, 6);
    close(quarter.position, [17.5, 8.72, -0.11], 1e-12);
    sameRotation(quarter.orientation, quatFromYaw(Math.PI), 1e-12);
    // The object turns with the circle: a full lap brings it back.
    close(pathPose(loop.path, 24).position, p0.position, 1e-9);
    const a = syntheticReading(loop, 7);
    const b = syntheticReading(loop, 7);
    expect(a).toEqual(b);
    expect(a?.t).toBe(3500);
    expect((a?.arrival ?? 0) - 3500).toBeGreaterThanOrEqual(40);
    expect((a?.arrival ?? 0) - 3500).toBeLessThan(100);
    // Nothing in the dropout.
    expect(syntheticReading(loop, 20)).toBeNull();
    expect(syntheticReading(loop, 24)).not.toBeNull();
  });

  it("follows a polyline at its speed, heading along each leg", () => {
    const square = {
      kind: "polyline" as const,
      points: [
        [0, 0, 0],
        [4, 0, 0],
        [4, 4, 0],
      ] as Vec3[],
      speedMps: 2,
    };
    close(pathPose(square, 1).position, [2, 0, 0], 1e-12);
    const second = pathPose(square, 3);
    close(second.position, [4, 2, 0], 1e-12);
    sameRotation(second.orientation, quatFromYaw(Math.PI / 2), 1e-12);
    // Closed: the diagonal home, then round again.
    close(pathPose(square, (8 + Math.hypot(4, 4)) / 2).position, [0, 0, 0], 1e-9);
    close(pathPose({ ...square, closed: false }, 100).position, [4, 4, 0], 1e-12);
  });

  it("played out through a track, follows the path within a few centimetres", () => {
    // A playout delay longer than the reading interval plus the jitter: always between two.
    const track = new PoseTrack({ latencyMs: 600 });
    let k = 0;
    let worst = 0;
    for (let now = 0; now <= 9000; now += 16) {
      for (;;) {
        const r = syntheticReading(loop, k);
        if (!r || r.arrival > now) break;
        track.push(r.t, r.pose, r.arrival);
        k += 1;
      }
      const shown = track.read(now);
      // The first second has one reading to go on: held, not yet on the path.
      if (shown.pose === null || now < 1000) continue;
      expect(shown.state).toBe("live");
      const truth = pathPose(loop.path, shown.playoutMs / 1000);
      worst = Math.max(
        worst,
        Math.hypot(
          shown.pose.position[0] - truth.position[0],
          shown.pose.position[1] - truth.position[1],
        ),
      );
    }
    // A chord of a 2.5 m circle at 0.5 s steps (0.65 m apart) sags 2 cm at most.
    expect(worst).toBeLessThan(0.025);
  });
});
