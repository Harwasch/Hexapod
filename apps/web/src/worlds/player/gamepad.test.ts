import { describe, expect, it } from "vitest";
import { gamepadMotion } from "./gamepad";

describe("analog gamepad controls", () => {
  it("filters stick drift and missing or invalid device axes", () => {
    const values = gamepadMotion([0.1, Number.NaN, 0.14]);
    expect(Object.values(values).every((value) => value === 0)).toBe(true);
  });
  it("preserves simultaneous move/look axes with bounded magnitudes", () => {
    const motion = gamepadMotion([1, -1, 0.575, -2]);
    expect(motion).toMatchObject({ forward: 1, right: 1, up: 0, pitch: 1 });
    expect(motion.yaw).toBeCloseTo(0.5);
  });
});
