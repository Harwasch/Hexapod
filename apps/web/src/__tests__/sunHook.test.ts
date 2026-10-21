/**
 * Sunlight under PlayCanvas (cesium/scanView/sunHook.ts): the per-frame hook's arithmetic is
 * CesiumJS's relight (lib/splatNormals.ts `sunlightFactor`) on a splat kept as rotation and
 * scales; the word the copy pass leaves beside each splat says whether it is lit (an inferred
 * layer's is not) and what the cell look drew for a black splat, so that relighting the drawn
 * colour afterwards is the look of the relit colour -- CesiumJS's order, sun before look.
 */
import { describe, expect, it } from "vitest";

import {
  hookSunFactor,
  packSunWord,
  SUN_HOOK_VS,
  SUN_WORD_GLSL,
  unpackSunWord,
} from "@/cesium/scanView/sunHook";
import { covarianceOf, SUN_AMBIENT, sunlightFactor, type Vec3 } from "@/lib/splatNormals";

function about(axis: Vec3, angle: number): [number, number, number, number] {
  const n = Math.hypot(...axis);
  const s = Math.sin(angle / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const IDENTITY3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

describe("the hook's relight", () => {
  it("is CesiumJS's factor for the same splat, eye and sun, on 5 000 random splats", () => {
    const next = random(19);
    for (let i = 0; i < 5_000; i += 1) {
      const quat = about([next() - 0.5, next() - 0.5, next() - 0.5], next() * 6.3);
      const big = 0.05 + next() * 0.3;
      const scale: Vec3 = [big, big * (0.1 + next() * 0.9), big * next() ** 2];
      const centre: Vec3 = [next() * 10 - 5, next() * 10 - 5, next() * 3];
      const eye: Vec3 = [next() * 30 - 15, next() * 30 - 15, 1 + next() * 10];
      const theta = next() * 2 * Math.PI;
      const sun: Vec3 = [Math.cos(theta) * 0.8, Math.sin(theta) * 0.8, 0.6];
      const sunUp = next();
      const strength = next();
      // The scan's frame as CesiumJS's eye coordinates with an identity rotation: the eye at
      // the origin, the splat at centre - eye.
      const cesium = sunlightFactor({
        covariance: covarianceOf(scale, quat),
        modelView3: IDENTITY3,
        positionEC: [centre[0] - eye[0], centre[1] - eye[1], centre[2] - eye[2]],
        sunEC: sun,
        sunUp,
        strength,
      });
      const playcanvas = hookSunFactor({ scale, quat, centre, eye, sun, sunUp, strength });
      expect(playcanvas).toBeCloseTo(cesium, 6);
    }
  });

  it("leaves a round splat as photographed by day and draws the photograph at strength 0", () => {
    const round = { scale: [0.1, 0.1, 0.1] as Vec3, quat: [0, 0, 0, 1] as const };
    const at = { centre: [0, 0, 0] as Vec3, eye: [5, 0, 0] as Vec3, sun: [-1, 0, 0] as Vec3 };
    expect(hookSunFactor({ ...round, ...at, sunUp: 1, strength: 1 })).toBe(1);
    expect(hookSunFactor({ ...round, ...at, sunUp: 0, strength: 1 })).toBeCloseTo(SUN_AMBIENT);
    const disc = { scale: [0.004, 0.09, 0.09] as Vec3, quat: [0, 0, 0, 1] as const };
    expect(hookSunFactor({ ...disc, ...at, sunUp: 1, strength: 0 })).toBe(1);
    // The disc faces the eye (east); the sun behind it: the ambient floor.
    expect(hookSunFactor({ ...disc, ...at, sunUp: 1, strength: 1 })).toBeCloseTo(SUN_AMBIENT);
  });
});

describe("the word beside each splat", () => {
  it("round-trips the look's black within half a step of 1/1023, and the skip bit", () => {
    for (const black of [
      [0, 0, 0],
      [1, 1, 1],
      [0.65, 0.2, 0.913],
      [0.0004, 0.9996, 0.5],
    ] as Vec3[]) {
      const word = packSunWord(black, false);
      expect(word).toBeGreaterThanOrEqual(0);
      expect(word).toBeLessThan(2 ** 30);
      const back = unpackSunWord(word);
      expect(back.skip).toBe(false);
      for (let c = 0; c < 3; c += 1)
        expect(Math.abs(back.black[c]! - black[c]!)).toBeLessThan(0.5 / 1023 + 1e-12);
    }
    expect(unpackSunWord(packSunWord([0.3, 0.3, 0.3], true)).skip).toBe(true);
    // A word nobody wrote (zero) is a lit splat with no look: the photograph relit.
    expect(unpackSunWord(0)).toEqual({ skip: false, black: [0, 0, 0] });
  });

  it("is what the GLSL packs and reads: ten bits a channel, the skip bit 30", () => {
    expect(SUN_WORD_GLSL).toContain("1023.0");
    expect(SUN_WORD_GLSL).toContain("<< 10u");
    expect(SUN_WORD_GLSL).toContain("<< 20u");
    expect(SUN_WORD_GLSL).toContain(`${String(2 ** 30)}u`);
    expect(SUN_HOOK_VS).toContain("loadHexapodSun().r");
    expect(SUN_HOOK_VS).toContain("hexapodSunSkipped");
  });
});
