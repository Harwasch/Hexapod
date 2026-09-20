import { describe, expect, it } from "vitest";

import {
  deform,
  magnitude,
  maxDeformSpeed,
  maxDisplacement,
  maxNodeDisplacements,
  maxNodeSpeeds,
  medianGaussianScale,
  nodeDisplacement,
  sortStaleness,
  syntheticTreeRig,
  SORT_STALENESS_NOTICEABLE,
  WIND_CALM,
  type WindSettings,
} from "./index";

const rig = syntheticTreeRig();
const GALE: WindSettings = { strength: 1, bearingDeg: 90 };

describe("displacement bounds", () => {
  it("is zero at calm and zero at the anchor", () => {
    expect(maxDisplacement(rig, WIND_CALM)).toBe(0);
    expect(maxNodeDisplacements(rig, GALE)[0]).toBe(0);
    expect(maxNodeDisplacements(rig, WIND_CALM).every((b) => b === 0)).toBe(true);
  });

  it("grows strictly with strength", () => {
    let previous = -1;
    for (const strength of [0, 0.1, 0.25, 0.5, 0.75, 1]) {
      const bound = maxDisplacement(rig, { strength, bearingDeg: 0 });
      expect(bound).toBeGreaterThan(previous);
      previous = bound;
    }
  });

  it("does not depend on bearing: the tree is rotationally symmetric in its bound", () => {
    const first = maxDisplacement(rig, { strength: 0.6, bearingDeg: 0 });
    for (const bearingDeg of [45, 90, 180, 271]) {
      expect(maxDisplacement(rig, { strength: 0.6, bearingDeg })).toBe(first);
    }
  });

  it("grows with height: the canopy bound exceeds the trunk bound", () => {
    const bounds = maxNodeDisplacements(rig, GALE);
    const trunk = Math.max(...rig.nodes.map((n, i) => (n.band === "trunk" ? (bounds[i] ?? 0) : 0)));
    const leaf = Math.max(...rig.nodes.map((n, i) => (n.band === "leaf" ? (bounds[i] ?? 0) : 0)));
    expect(leaf).toBeGreaterThan(trunk);
  });

  it("is an upper bound on what deform actually produces", () => {
    const bounds = maxNodeDisplacements(rig, GALE);
    for (let k = 0; k <= 120 * 60; k += 11) {
      const transforms = deform(rig, k / 60, GALE);
      for (let i = 0; i < rig.nodes.length; i += 1) {
        expect(magnitude(nodeDisplacement(rig, transforms, i))).toBeLessThanOrEqual(
          (bounds[i] ?? 0) + 1e-12,
        );
      }
    }
  });
});

describe("speed bounds", () => {
  it("is zero at calm and grows with strength", () => {
    expect(maxDeformSpeed(rig, WIND_CALM)).toBe(0);
    expect(maxNodeSpeeds(rig, WIND_CALM).every((b) => b === 0)).toBe(true);
    expect(maxDeformSpeed(rig, { strength: 1, bearingDeg: 0 })).toBeGreaterThan(
      maxDeformSpeed(rig, { strength: 0.3, bearingDeg: 0 }),
    );
  });
});

describe("medianGaussianScale", () => {
  it("takes the largest component per splat, then the median over splats", () => {
    // Per-splat extents: 3, 1, 9, 5, 7 -> sorted 1, 3, 5, 7, 9 -> median 5.
    const scales = new Float32Array([1, 2, 3, 1, 1, 1, 9, 0, 2, 5, 4, 1, 2, 7, 3]);
    expect(medianGaussianScale(scales)).toBe(5);
  });

  it("averages the middle pair for an even count", () => {
    expect(medianGaussianScale(new Float32Array([1, 2, 3, 4]), 1)).toBe(2.5);
  });

  it("uses magnitudes, so a negative log-scale does not read as tiny", () => {
    expect(medianGaussianScale(new Float32Array([-4, 1, 1]), 3)).toBe(4);
  });

  it("returns zero for no splats", () => {
    expect(medianGaussianScale(new Float32Array(0))).toBe(0);
    expect(medianGaussianScale(new Float32Array([1, 2]))).toBe(0);
  });
});

describe("sortStaleness", () => {
  const median = 0.02;

  it("is the displacement measured in splat radii", () => {
    expect(sortStaleness(rig, GALE, median)).toBeCloseTo(maxDisplacement(rig, GALE) / median, 12);
  });

  it("is exactly zero at calm: a still tree has a perfect sort", () => {
    expect(sortStaleness(rig, WIND_CALM, median)).toBe(0);
    expect(sortStaleness(rig, WIND_CALM, median)).toBeLessThan(SORT_STALENESS_NOTICEABLE);
  });

  it("rises with strength, so the artifact is a number rather than a vibe", () => {
    const gentle = sortStaleness(rig, { strength: 0.05, bearingDeg: 0 }, median);
    const gale = sortStaleness(rig, GALE, median);
    expect(gale).toBeGreaterThan(gentle);
    // A 6 m tree at full strength moves many splat radii: this is expected to be visible.
    expect(gale).toBeGreaterThan(SORT_STALENESS_NOTICEABLE);
  });

  it("refuses to invent a number for a non-positive scale", () => {
    expect(sortStaleness(rig, GALE, 0)).toBeNaN();
    expect(sortStaleness(rig, GALE, -1)).toBeNaN();
    expect(sortStaleness(rig, GALE, Number.NaN)).toBeNaN();
  });
});
