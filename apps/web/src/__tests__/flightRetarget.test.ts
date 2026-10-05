import { describe, expect, it } from "vitest";

import { samePose, type ArrivalPose } from "@/cesium/flightRetarget";

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
