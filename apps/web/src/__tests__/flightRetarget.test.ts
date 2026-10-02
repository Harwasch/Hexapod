import { describe, expect, it } from "vitest";

import {
  continuingEasing,
  easingSlope,
  quadraticInOut,
  retarget,
  samePose,
  type ArrivalPose,
} from "@/cesium/flightRetarget";

describe("continuingEasing", () => {
  it("leaves at the given slope and arrives at rest, at the ends", () => {
    for (const k of [0, 0.5, 1, 2, 3]) {
      const easing = continuingEasing(k);
      expect(easing(0)).toBeCloseTo(0, 9);
      expect(easing(1)).toBeCloseTo(1, 9);
      expect(easingSlope(easing, 0)).toBeCloseTo(k, 2);
      expect(easingSlope(easing, 1)).toBeCloseTo(0, 2);
    }
  });

  it("never runs backwards or past the destination, however fast the camera already is", () => {
    for (const k of [0, 1.5, 3, 7, Number.POSITIVE_INFINITY, Number.NaN, -2]) {
      const easing = continuingEasing(k);
      let previous = 0;
      for (let i = 1; i <= 100; i += 1) {
        const value = easing(i / 100);
        expect(value).toBeGreaterThanOrEqual(previous - 1e-12);
        expect(value).toBeLessThanOrEqual(1 + 1e-12);
        previous = value;
      }
    }
  });
});

describe("retarget", () => {
  const flight = { durationS: 4, easing: quadraticInOut, lengthM: 10_000 };

  it("keeps the camera's speed through the hand-over", () => {
    // Half-way through a quadratic in-out flight the camera moves at twice the average speed.
    const elapsedS = 2;
    const speedBefore = (flight.lengthM * easingSlope(flight.easing, 0.5)) / flight.durationS;
    const next = retarget({ ...flight, elapsedS }, 5_200, 1.2);
    expect(next).not.toBeNull();
    if (!next) return;
    const speedAfter = (5_200 * easingSlope(next.easing, 0)) / next.durationS;
    expect(speedAfter).toBeCloseTo(speedBefore, 0);
    // The arrival is not postponed: the replacement takes the time the old flight had left.
    expect(next.durationS).toBeCloseTo(2, 6);
  });

  it("starts from rest when the flight had barely left, as a fresh flight would", () => {
    const next = retarget({ ...flight, elapsedS: 0 }, 9_000, 1.2);
    expect(next && easingSlope(next.easing, 0)).toBeCloseTo(0, 2);
  });

  it("gives a destination that moved late a readable approach, not a snap", () => {
    const next = retarget({ ...flight, elapsedS: 3.9 }, 400, 1.2);
    expect(next?.durationS).toBe(1.2);
  });

  it("does nothing for a flight that is over", () => {
    expect(retarget({ ...flight, elapsedS: 4 }, 100, 1.2)).toBeNull();
    expect(retarget({ ...flight, elapsedS: 9 }, 100, 1.2)).toBeNull();
    expect(retarget({ ...flight, durationS: 0, elapsedS: 0 }, 100, 1.2)).toBeNull();
  });
});

describe("samePose", () => {
  const pose: ArrivalPose = { longitude: 0, latitude: 0, height: 100, heading: 100, pitch: -45 };

  it("is the same pose within the distance tolerance and two degrees", () => {
    expect(samePose(pose, { ...pose, heading: 101.5 }, 3, 5)).toBe(true);
    expect(samePose(pose, { ...pose, heading: -258.5 }, 0, 5)).toBe(true);
  });

  it("is a different pose past either", () => {
    expect(samePose(pose, pose, 6, 5)).toBe(false);
    expect(samePose(pose, { ...pose, heading: 104 }, 0, 5)).toBe(false);
    expect(samePose(pose, { ...pose, pitch: -40 }, 0, 5)).toBe(false);
  });
});
