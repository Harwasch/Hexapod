/**
 * A site fly-to is one glide (cesium/glide.ts): one path from the click to the landing, whose
 * destination moves smoothly when a better pose arrives on the way.
 *
 * It used to be a chain of Cesium flights, each correction a new leg: on production, a flight
 * from orbit to a phone scan aimed at a terrain placeholder 30 km under the sea, landed there
 * (black), then climbed 7.5 km out of the ground and came back down to the scan. These fly the
 * glide frame by frame at 60 fps and measure the path the camera takes: no jump between
 * frames when a correction arrives, early or late; never under the ground; always ending
 * exactly where it was last pointed.
 */
import { Cartesian3 } from "cesium";
import { describe, expect, it } from "vitest";

import { surfacedHeight } from "@/cesium/CameraController";
import type { ArrivalPose } from "@/cesium/flightRetarget";
import {
  Glide,
  criticallyDamped,
  easeInOut,
  glideDuration,
  glidePose,
  plausible,
  zoomPath,
  type GlidePose,
  type GlideScene,
} from "@/cesium/glide";

const FRAME_MS = 1000 / 60;

/** Orbit, where the app starts: 18,000 km over the southern United States, looking down. */
const ORBIT = { longitude: -95, latitude: 32, height: 18_000_000, heading: 0, pitch: -90 };

/** The Camp scan's arrival, on the Oregon coast: 190 m over ground at sea level. */
const CAMP: GlidePose = {
  longitude: -123.8836,
  latitude: 46.1339,
  height: 190,
  heading: 100,
  pitch: -45,
  ground: { height: 0, measured: false },
};

const flat: GlideScene = { ground: () => 0 };

interface Frame {
  t: number;
  pose: ArrivalPose;
  position: Cartesian3;
}

/** Flies a glide to the end at 60 fps, calling `at(frame index, glide)` before each frame. */
function fly(
  glide: Glide,
  at: (frame: number, glide: Glide) => void = () => undefined,
  maxFrames = 60 * 30,
): Frame[] {
  const frames: Frame[] = [];
  for (let i = 1; i <= maxFrames; i++) {
    at(i, glide);
    const t = i * FRAME_MS;
    const pose = glide.step(t);
    frames.push({
      t,
      pose,
      position: Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height),
    });
    if (glide.done) break;
  }
  return frames;
}

/** Metres moved in each frame. */
function steps(frames: Frame[]): number[] {
  const moved: number[] = [];
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1];
    const b = frames[i];
    if (a && b) moved.push(Cartesian3.distance(a.position, b.position));
  }
  return moved;
}

/**
 * The largest kick between frames: how far each frame lands from where the two before it
 * would have put the camera at a steady velocity (the second difference of its position), as
 * a share of its height above the ground -- what a jump looks like on screen. A smooth flight
 * at 60 fps stays within a few thousandths; a leg that restarts from rest, or a destination
 * that jumps, kicks by a large share of the view.
 */
function worstKick(frames: Frame[], ground: (pose: ArrivalPose) => number = () => 0): number {
  let worst = 0;
  for (let i = 1; i + 1 < frames.length; i++) {
    const [a, b, c] = [frames[i - 1], frames[i], frames[i + 1]];
    if (!a || !b || !c) continue;
    const kick = Cartesian3.subtract(
      Cartesian3.add(a.position, c.position, new Cartesian3()),
      Cartesian3.multiplyByScalar(b.position, 2, new Cartesian3()),
      new Cartesian3(),
    );
    const altitude = Math.max(b.pose.height - ground(b.pose), 1);
    worst = Math.max(worst, Cartesian3.magnitude(kick) / altitude);
  }
  return worst;
}

/** What a frame-by-frame flight may kick by, as a share of its altitude (`worstKick`). */
const SMOOTH = 0.01;

describe("the zoom path", () => {
  it("runs exactly from one end to the other, rising over a long flight and coming back down", () => {
    // Pumpkin to Camp: 900 km between two scans, both 190 m up.
    const path = zoomPath(190, 190, 900_000);
    expect(path.at(0).along).toBe(0);
    expect(path.at(0).altitude).toBeCloseTo(190, 6);
    expect(path.at(1).along).toBe(1);
    expect(path.at(1).altitude).toBeCloseTo(190, 6);
    // High enough to see both ends, and highest in the middle.
    expect(path.peak.altitude).toBeGreaterThan(300_000);
    expect(path.peak.altitude).toBeLessThan(2_000_000);
    expect(path.peak.at).toBeCloseTo(0.5, 2);
    let previous = 0;
    for (let s = 0.05; s <= 1; s += 0.05) {
      const along = path.at(s).along;
      expect(along).toBeGreaterThanOrEqual(previous);
      previous = along;
    }
  });

  it("zooms on the spot without moving along the ground, and takes longer the further it goes", () => {
    const zoom = zoomPath(18_000_000, 190, 0);
    expect(zoom.at(0.5).altitude).toBeCloseTo(Math.sqrt(18_000_000 * 190), 0);
    const short = glideDuration(zoomPath(190, 190, 1_000));
    const long = glideDuration(zoomPath(190, 190, 900_000));
    expect(short).toBeGreaterThanOrEqual(1.2);
    expect(long).toBeGreaterThan(short);
    expect(long).toBeLessThanOrEqual(5.5);
  });

  it("looks straight down while high above both ends, and holds each end's pitch near it", () => {
    const start = { ...CAMP, longitude: -122.6998, latitude: 38.0637, ground: 0 };
    const end = { ...CAMP, ground: 0 };
    expect(glidePose(start, end, 0).pitch).toBe(-45);
    expect(glidePose(start, end, 1).pitch).toBe(-45);
    expect(glidePose(start, end, 0.5).pitch).toBeCloseTo(-90, 3);
    // A hop of a few metres turns from one pitch to the other.
    const near = { ...end, longitude: end.longitude + 0.0001, pitch: -25 };
    expect(glidePose(end, near, 0.5).pitch).toBeCloseTo(-35, 0);
  });

  it("eases in and out", () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.5)).toBeCloseTo(0.5, 9);
    expect(easeInOut(0.001)).toBeLessThan(0.0001);
  });
});

describe("the critically damped destination", () => {
  it("reaches a new target without overshooting, at any frame rate", () => {
    let fine = { value: 0, velocity: 0 };
    for (let i = 0; i < 60; i++) fine = criticallyDamped(fine.value, fine.velocity, 100, 5, 1 / 60);
    const coarse = criticallyDamped(0, 0, 100, 5, 1);
    expect(fine.value).toBeCloseTo(coarse.value, 6);
    expect(fine.value).toBeGreaterThan(95);
    expect(fine.value).toBeLessThan(100);
  });
});

describe("a glide", () => {
  it("lands exactly where it was pointed, at rest", () => {
    const glide = new Glide(ORBIT, CAMP, flat, 0);
    const frames = fly(glide);
    const last = frames.at(-1)?.pose;
    expect(glide.done).toBe(true);
    expect(last?.longitude).toBeCloseTo(CAMP.longitude, 9);
    expect(last?.latitude).toBeCloseTo(CAMP.latitude, 9);
    expect(last?.height).toBeCloseTo(CAMP.height, 6);
    expect(last?.pitch).toBeCloseTo(-45, 6);
    expect(frames.at(-1)?.t ?? 0).toBeCloseTo(glide.durationS * 1000, -2);
    expect(worstKick(frames)).toBeLessThan(SMOOTH);
  });

  it("moves its destination on the way without a jump in the camera's path", () => {
    // The record's footprint, a kilometre off the summary's guess, at 40 % of the flight; then
    // the model's own bounds, 40 m off that, at 80 %.
    const glide = new Glide(ORBIT, CAMP, flat, 0);
    const n = Math.round((glide.durationS * 60) / 1);
    const record = { ...CAMP, longitude: CAMP.longitude + 0.013 };
    const model = { ...record, height: 230, ground: { height: 0, measured: true } };
    const frames = fly(glide, (i, g) => {
      if (i === Math.round(n * 0.4)) g.retarget(record);
      if (i === Math.round(n * 0.8)) g.retarget(model);
    });
    const last = frames.at(-1)?.pose;
    expect(last?.longitude).toBeCloseTo(model.longitude, 9);
    expect(last?.height).toBeCloseTo(230, 6);
    expect(worstKick(frames)).toBeLessThan(SMOOTH);
  });

  it("would show a restart from rest as the kick it is (the measure is not blind)", () => {
    // What a correction used to be: a new flight from wherever the camera was, from rest.
    const first = new Glide(ORBIT, CAMP, flat, 0);
    const frames = fly(first, () => undefined, Math.round(first.durationS * 60 * 0.6));
    const last = frames.at(-1)?.pose;
    if (!last) throw new Error("no frames");
    const second = new Glide(last, { ...CAMP, longitude: CAMP.longitude + 0.013 }, flat, 0);
    const restarted = fly(second).map((f) => ({ ...f, t: f.t + (frames.at(-1)?.t ?? 0) }));
    expect(worstKick([...frames, ...restarted])).toBeGreaterThan(SMOOTH * 3);
  });

  it("takes over a flight still moving -- another site picked on the way -- without a kick", () => {
    // On the way down to Camp, Pumpkin is picked instead: a new glide from where the camera
    // is, carrying the motion it has.
    const first = new Glide(ORBIT, CAMP, flat, 0);
    const before = fly(first, () => undefined, Math.round(first.durationS * 60 * 0.6));
    const [a, b] = before.slice(-2);
    if (!a || !b) throw new Error("no frames");
    const dt = (b.t - a.t) / 1000;
    const carry = {
      velocity: {
        x: (b.position.x - a.position.x) / dt,
        y: (b.position.y - a.position.y) / dt,
        z: (b.position.z - a.position.z) / dt,
      },
      heading: (b.pose.heading - a.pose.heading) / dt,
      pitch: (b.pose.pitch - a.pose.pitch) / dt,
    };
    const pumpkin: GlidePose = { ...CAMP, longitude: -122.6998, latitude: 38.0637 };
    const second = new Glide(b.pose, pumpkin, flat, b.t, { carry });
    const after: Frame[] = [];
    for (let i = 1; i <= 60 * 30; i++) {
      const t = b.t + i * FRAME_MS;
      const pose = second.step(t);
      after.push({
        t,
        pose,
        position: Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height),
      });
      if (second.done) break;
    }
    expect(after.at(-1)?.pose.longitude).toBeCloseTo(pumpkin.longitude, 9);
    expect(after.at(-1)?.pose.height).toBeCloseTo(pumpkin.height, 6);
    expect(worstKick([...before, ...after])).toBeLessThan(SMOOTH);
    // Taken over from rest instead, the hand-over is the kick it used to be.
    const still = new Glide(b.pose, pumpkin, flat, b.t);
    const stopped = [
      b,
      ...[1, 2, 3].map((i) => {
        const t = b.t + i * FRAME_MS;
        const pose = still.step(t);
        return {
          t,
          pose,
          position: Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height),
        };
      }),
    ];
    expect(worstKick([a, ...stopped])).toBeGreaterThan(SMOOTH * 3);
  });

  it("glides on from where it is to a correction that comes after it has run out of time", () => {
    const glide = new Glide(ORBIT, CAMP, flat, 0);
    const end = Math.ceil(glide.durationS * 60);
    const late = { ...CAMP, height: 260, ground: { height: 0, measured: true } };
    // At the last frame of the clock, before it has landed: the camera is at rest there.
    const frames = fly(glide, (i, g) => {
      if (i === end - 1) g.retarget(late);
    });
    expect(frames.at(-1)?.pose.height).toBeCloseTo(260, 6);
    expect(worstKick(frames)).toBeLessThan(SMOOTH);
    // It moves on the way up, not in one step.
    expect(Math.max(...steps(frames.slice(end - 2)))).toBeLessThan(5);
  });

  it("never goes under the ground, and re-aims a guessed end at the terrain as it loads", () => {
    // Spool, in the Montana mountains: the terrain is at 2,000 m, the first aim at the
    // ellipsoid, and the terrain under the destination appears once the descent is halfway.
    const spool: GlidePose = {
      longitude: -111.1346,
      latitude: 44.7965,
      height: 190,
      heading: 100,
      pitch: -45,
      ground: { height: 0, measured: false },
    };
    let now = 0;
    const scene: GlideScene = {
      ground: (longitude) =>
        now > 2_000 || Math.abs(longitude - spool.longitude) > 1 ? 2_000 : undefined,
    };
    const glide = new Glide(ORBIT, spool, scene, 0);
    const frames = fly(glide, (i) => {
      now = i * FRAME_MS;
    });
    const lowest = Math.min(...frames.map((f) => f.pose.height));
    expect(lowest).toBeGreaterThan(2_000);
    expect(frames.at(-1)?.pose.height).toBeCloseTo(2_190, 6);
    expect(worstKick(frames, () => 2_000)).toBeLessThan(SMOOTH);
  });

  it("trusts a measured end over the coarse terrain the globe loads on the way down", () => {
    // Spool, measured at 2,001 m; the globe's tiles under it on the way down said 2,968 m.
    const spool: GlidePose = {
      longitude: -111.1346,
      latitude: 44.7965,
      height: 2_069,
      heading: 100,
      pitch: -45,
      ground: { height: 2_001, measured: true },
    };
    const coarse: GlideScene = { ground: () => 2_968 };
    expect(fly(new Glide(ORBIT, spool, coarse, 0)).at(-1)?.pose.height).toBeCloseTo(2_069, 6);
  });

  it("hops between two poses in the mountains without arcing, where the globe knows no ground", () => {
    // A settle after landing at Spool, 2,000 m up: the globe had no tile worth believing under
    // the camera, the start's ground was taken as sea level, and the hop of 40 m arced 400 m.
    const unknown: GlideScene = { ground: () => undefined };
    const start = {
      longitude: -111.13556,
      latitude: 44.79663,
      height: 2_084,
      heading: 100,
      pitch: -45,
    };
    const end: GlidePose = {
      longitude: -111.13506,
      latitude: 44.79663,
      height: 2_094,
      heading: 100,
      pitch: -45,
      ground: { height: 2_002, measured: true },
    };
    const frames = fly(new Glide(start, end, unknown, 0));
    expect(Math.max(...frames.map((f) => f.pose.height))).toBeLessThan(2_110);
    expect(frames.at(-1)?.pose.height).toBeCloseTo(2_094, 6);
    expect(worstKick(frames, () => 2_002)).toBeLessThan(SMOOTH);
  });

  it("ignores the globe's placeholder heights, thirty kilometres under the sea", () => {
    expect(plausible(-31_017)).toBeUndefined();
    expect(plausible(24.7)).toBe(24.7);
    const placeholder: GlideScene = { ground: () => -31_017 };
    const frames = fly(new Glide(ORBIT, CAMP, placeholder, 0));
    expect(Math.min(...frames.map((f) => f.pose.height))).toBeGreaterThan(0);
    expect(frames.at(-1)?.pose.height).toBeCloseTo(190, 6);
  });

  it("is steady at a low frame rate too: the path depends on time, not on frames", () => {
    const fast = new Glide(ORBIT, CAMP, flat, 0);
    const slow = new Glide(ORBIT, CAMP, flat, 0);
    const half = (fast.durationS * 1000) / 2;
    for (let t = FRAME_MS; t < half; t += FRAME_MS) fast.step(t);
    for (let t = 250; t < half; t += 250) slow.step(t);
    const a = fast.step(half);
    const b = slow.step(half);
    const gap = Cartesian3.distance(
      Cartesian3.fromDegrees(a.longitude, a.latitude, a.height),
      Cartesian3.fromDegrees(b.longitude, b.latitude, b.height),
    );
    expect(gap).toBeLessThan(1);
  });
});

describe("a camera that landed underground", () => {
  it("is brought up over the terrain sampled at full detail, and only when it is underground", () => {
    // Production's first leg to Camp: 30 km under the Oregon coast.
    expect(surfacedHeight(-30_817, -10)).toBe(20);
    // A scan's own ground a few metres under the terrain model is not underground.
    expect(surfacedHeight(1_995, 2_001)).toBeNull();
    expect(surfacedHeight(2_069, 2_001)).toBeNull();
    // No terrain worth believing: nothing to go by.
    expect(surfacedHeight(-30_817, -31_017)).toBeNull();
    expect(surfacedHeight(-30_817, undefined)).toBeNull();
  });
});
