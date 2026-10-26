import { afterEach, describe, expect, it } from "vitest";

import {
  lightCopyGlsl,
  LIGHT_MAGIC,
  lightStreamData,
  LIGHT_HOOK,
} from "@/cesium/scanView/lightHook";
import { useAsCaptured } from "@/state/asCaptured";
import { useLivingLab } from "@/state/livingLab";
import { effectiveLightArm, lightSun, useLivingLight } from "@/state/livingLight";

afterEach(() => {
  useAsCaptured.setState({ on: false });
  useLivingLab.setState({ open: false });
  useLivingLight.setState({ arm: "off" });
});
describe("Living Light", () => {
  it("requires the lab flag and is disabled by As captured without losing the choice", () => {
    useLivingLight.getState().set({ arm: "b" });
    expect(effectiveLightArm()).toBe("off");
    useLivingLab.setState({ open: true });
    expect(effectiveLightArm()).toBe("b");
    useAsCaptured.setState({ on: true });
    expect(effectiveLightArm()).toBe("off");
    useAsCaptured.setState({ on: false });
    expect(effectiveLightArm()).toBe("b");
  });
  it("maps sidecars through Morton order without changing a byte", () => {
    const words = Uint32Array.from({ length: 24 }, (_, i) => i);
    words[10] = words[22] = LIGHT_MAGIC;
    const streams = lightStreamData(words.buffer, new Uint32Array([1, 0]));
    expect([...(streams[0] ?? [])]).toEqual([12, 13, 14, 15, 0, 1, 2, 3]);
    expect([...(streams[2] ?? [])]).toEqual([20, 21, LIGHT_MAGIC, 23, 8, 9, LIGHT_MAGIC, 11]);
    expect(() => lightStreamData(new ArrayBuffer(0), new Uint32Array([0]))).toThrow();
    words[10] = 0;
    expect(() => lightStreamData(words.buffer, new Uint32Array([0, 1]))).toThrow();
  });
  it("writes unbound tiles as zero and preserves the existing copy colour", () => {
    const code = lightCopyGlsl(undefined, false);
    expect(code).toContain("writeHexapodLight0(uvec4(0u))");
    expect(code).toContain("lightPreviousCopyColor(center, color)");
    expect(lightCopyGlsl(undefined, true)).toContain("loadSplatLight0()");
    expect(LIGHT_HOOK).not.toContain("view_position");
    expect(LIGHT_HOOK).not.toContain("color.rgb *=");
  });
  it("uses a unit ENU sun along the baker's equinox path", () => {
    expect(lightSun(0)[0]).toBeCloseTo(Math.sin((75 * Math.PI) / 180));
    expect(lightSun(1)[0]).toBeCloseTo(-lightSun(0)[0]);
    expect(Math.hypot(...lightSun(0.5))).toBeCloseTo(1);
    expect(lightSun(-1)).toEqual(lightSun(0));
  });
});
