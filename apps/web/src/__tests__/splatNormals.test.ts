/**
 * The relight's arithmetic (lib/splatNormals.ts), which the shader in cesium/splatSunlight.ts
 * transcribes: the closed-form eigenvalues, the normal only for a clearly flat splat (round,
 * needle-like and degenerate splats are neutral), the normal still found after the engine's
 * half-float storage, and the lambert term's sign -- the side facing the eye is the side lit.
 */
import { describe, expect, it } from "vitest";

import {
  covarianceOf,
  FLAT_RATIO,
  halfCovariance,
  splatNormal,
  SUN_AMBIENT,
  sunlightFactor,
  sunUpFactor,
  symmetricEigenvalues,
  toHalf,
  type Sym3,
  type Vec3,
} from "@/lib/splatNormals";

/** A unit quaternion (xyzw) turning `angle` radians about `axis`. */
function about(axis: Vec3, angle: number): [number, number, number, number] {
  const n = Math.hypot(...axis);
  const s = Math.sin(angle / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/** `v` turned by the unit quaternion `q`. */
function rotate(q: readonly number[], v: Vec3): Vec3 {
  const [x, y, z, w] = q as [number, number, number, number];
  const r = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ] as const;
  return [0, 1, 2].map(
    (i) => r[i]![0] * v[0] + r[i]![1] * v[1] + r[i]![2] * v[2],
  ) as unknown as Vec3;
}

/** |cos| of the angle between two unit vectors: 1 when parallel, either sign. */
const alignment = (a: Vec3, b: Vec3) => Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);

const IDENTITY3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Splats with a known answer: the test vectors the shader must agree with. */
const VECTORS: {
  name: string;
  scale: Vec3;
  quat: [number, number, number, number];
  normal: Vec3 | null;
}[] = [
  {
    name: "a disc in the xy plane",
    scale: [0.3, 0.3, 0.003],
    quat: [0, 0, 0, 1],
    normal: [0, 0, 1],
  },
  {
    name: "an east-facing disc (turned 90° about north)",
    scale: [0.3, 0.3, 0.003],
    quat: about([0, 1, 0], Math.PI / 2),
    normal: [1, 0, 0],
  },
  {
    name: "an elliptical disc at an odd angle",
    scale: [0.4, 0.15, 0.01],
    quat: about([0.3, -0.5, 0.8], 1.1),
    normal: rotate(about([0.3, -0.5, 0.8], 1.1), [0, 0, 1]),
  },
  {
    name: "just flat enough (ratio 0.2025)",
    scale: [0.1, 0.1, 0.045],
    quat: about([1, 1, 0], 0.7),
    normal: rotate(about([1, 1, 0], 0.7), [0, 0, 1]),
  },
  {
    name: "not flat enough (ratio 0.36)",
    scale: [0.1, 0.1, 0.06],
    quat: [0, 0, 0, 1],
    normal: null,
  },
  { name: "round", scale: [0.1, 0.1, 0.1], quat: about([0.2, 0.9, 0.1], 0.4), normal: null },
  {
    name: "a needle (no preferred side)",
    scale: [0.3, 0.01, 0.01],
    quat: [0, 0, 0, 1],
    normal: null,
  },
  { name: "empty", scale: [0, 0, 0], quat: [0, 0, 0, 1], normal: null },
];

describe("the eigenvalues", () => {
  it("are the squared scales, largest first, for any rotation", () => {
    const q = about([0.6, 0.2, -0.7], 2.3);
    const values = symmetricEigenvalues(covarianceOf([0.5, 0.2, 0.05], q));
    expect(values).not.toBeNull();
    const [l1, l2, l3] = values!;
    expect(l1).toBeCloseTo(0.25, 9);
    expect(l2).toBeCloseTo(0.04, 9);
    expect(l3).toBeCloseTo(0.0025, 9);
  });

  it("hold for a disc's two equal axes (the acos at its end)", () => {
    const [l1, l2, l3] = symmetricEigenvalues(
      covarianceOf([0.2, 0.2, 0.01], about([1, 0, 0], 0.3)),
    )!;
    expect(l1).toBeCloseTo(0.04, 9);
    expect(l2).toBeCloseTo(0.04, 9);
    expect(l3).toBeCloseTo(0.0001, 9);
  });

  it("are null for a round, an empty or a broken covariance", () => {
    expect(symmetricEigenvalues(covarianceOf([0.1, 0.1, 0.1], [0, 0, 0, 1]))).toBeNull();
    expect(symmetricEigenvalues([0, 0, 0, 0, 0, 0])).toBeNull();
    expect(symmetricEigenvalues([Number.NaN, 0, 0, 1, 0, 1])).toBeNull();
    expect(symmetricEigenvalues([-1, 0, 0, -1, 0, -1])).toBeNull();
  });
});

describe("the normal", () => {
  for (const v of VECTORS) {
    it(`of ${v.name}: ${v.normal ? "its smallest axis" : "none (neutral)"}`, () => {
      const n = splatNormal(covarianceOf(v.scale, v.quat));
      if (v.normal === null) {
        expect(n).toBeNull();
        return;
      }
      expect(n).not.toBeNull();
      expect(Math.hypot(...n!)).toBeCloseTo(1, 9);
      expect(alignment(n!, v.normal)).toBeGreaterThan(1 - 1e-9);
    });
  }

  it("needs the smallest variance under FLAT_RATIO of the middle one", () => {
    const at = (ratio: number) =>
      splatNormal(covarianceOf([0.1, 0.1, 0.1 * Math.sqrt(ratio)], [0, 0, 0, 1]));
    expect(at(FLAT_RATIO * 0.98)).not.toBeNull();
    expect(at(FLAT_RATIO * 1.02)).toBeNull();
  });

  it("survives the engine's half-float storage, down to a 2 cm splat 1 mm thick", () => {
    for (const v of VECTORS.filter((x) => x.normal !== null)) {
      const small: Vec3 = [v.scale[0] / 15, v.scale[1] / 15, v.scale[2] / 15];
      const half = halfCovariance(covarianceOf(small, v.quat));
      const n = splatNormal(half);
      expect(n, v.name).not.toBeNull();
      expect(alignment(n!, v.normal!), v.name).toBeGreaterThan(0.999);
    }
  });

  it("is not invented for a round splat whose halves are not quite equal", () => {
    const half = halfCovariance(covarianceOf([0.031, 0.03, 0.0305], about([0.4, 0.4, 0.2], 0.9)));
    expect(splatNormal(half)).toBeNull();
  });

  it("rounds like the GPU's half: 10 fraction bits, subnormals below 2^-14", () => {
    expect(toHalf(1)).toBe(1);
    expect(toHalf(1 + 2 ** -11)).toBe(1); // a tie, to even
    expect(toHalf(1 + 2 ** -10)).toBe(1 + 2 ** -10);
    expect(toHalf(2 ** -20)).toBe(2 ** -20);
    expect(toHalf(3 * 2 ** -25)).toBe(2 ** -23);
    expect(toHalf(1e6)).toBe(Infinity);
  });
});

describe("the relight", () => {
  // In front of the eye, which looks down -z; a disc facing the eye (+z), or away (either way
  // the side drawn is the side facing the eye).
  const facing: Sym3 = covarianceOf([0.3, 0.3, 0.003], [0, 0, 0, 1]);
  const at: Vec3 = [0, 0, -5];
  const factor = (sunEC: Vec3, extra: Partial<Parameters<typeof sunlightFactor>[0]> = {}) =>
    sunlightFactor({
      covariance: facing,
      modelView3: IDENTITY3,
      positionEC: at,
      sunEC,
      sunUp: 1,
      strength: 1,
      ...extra,
    });

  it("is whole where the sun is behind the eye, on the side the eye sees", () => {
    expect(factor([0, 0, 1])).toBeCloseTo(1, 12);
  });

  it("is the ambient floor where the sun is behind the splat: lambert never goes negative", () => {
    expect(factor([0, 0, -1])).toBeCloseTo(SUN_AMBIENT, 12);
  });

  it("follows the cosine in between", () => {
    const s = Math.sin(Math.PI / 3);
    expect(factor([s, 0, 0.5])).toBeCloseTo(SUN_AMBIENT + (1 - SUN_AMBIENT) * 0.5, 12);
  });

  it("lights the side facing the eye whichever way the disc was stored or seen from", () => {
    // Seen from behind (the splat behind the eye's back is not drawn, but the sign test is
    // what matters): the same sun, the normal flipped toward the eye.
    expect(factor([0, 0, -1], { positionEC: [0, 0, 5] })).toBeCloseTo(1, 12);
    const flipped = covarianceOf([0.3, 0.3, 0.003], about([1, 0, 0], Math.PI));
    expect(factor([0, 0, 1], { covariance: flipped })).toBeCloseTo(1, 12);
  });

  it("turns the normal into eye coordinates with the model-view rotation", () => {
    // The model's +z is the eye's +x (a quarter turn about y): an x-facing sun lights it.
    const quarter = [0, 0, -1, 0, 1, 0, 1, 0, 0];
    expect(factor([1, 0, 0], { modelView3: quarter, positionEC: [-5, 0, 0] })).toBeCloseTo(1, 12);
    expect(factor([0, 0, 1], { modelView3: quarter, positionEC: [-5, 0, 0] })).toBeCloseTo(
      SUN_AMBIENT,
      12,
    );
  });

  it("leaves a round splat as photographed by day and at the floor by night", () => {
    const round = covarianceOf([0.1, 0.1, 0.1], [0, 0, 0, 1]);
    expect(factor([0, 0, -1], { covariance: round })).toBe(1);
    expect(factor([0, 0, -1], { covariance: round, sunUp: 0 })).toBeCloseTo(SUN_AMBIENT, 12);
  });

  it("draws everything at the ambient floor with the sun below the horizon", () => {
    expect(factor([0, 0, 1], { sunUp: 0 })).toBeCloseTo(SUN_AMBIENT, 12);
  });

  it("draws the photograph at strength 0", () => {
    expect(factor([0, 0, -1], { strength: 0 })).toBe(1);
  });

  it("fades in over the first degrees of the sun's elevation", () => {
    const sine = (deg: number) => Math.sin((deg * Math.PI) / 180);
    expect(sunUpFactor(sine(-10))).toBe(0);
    expect(sunUpFactor(0)).toBe(0);
    expect(sunUpFactor(sine(2.5))).toBeCloseTo(sine(2.5) / sine(5), 12);
    expect(sunUpFactor(sine(5))).toBeCloseTo(1, 12);
    expect(sunUpFactor(sine(60))).toBe(1);
  });
});
