import { describe, expect, it } from "vitest";

import { groundAt, measuredClamp, median, type MeasuredGround } from "@/cesium/placement";

/**
 * A capture on a slope, which is the case the bounding-box clamp gets wrong.
 *
 * Fifteen metres of drop across the capture. The capture's own ground and the viewer's ground
 * are the *same surface*, offset by a constant 4.2 m — the capture was reconstructed from EXIF
 * altitudes that sit above the ellipsoid the globe uses, which is exactly the situation
 * `tools/captures/ground_samples.py` was written for.
 */
const slope = (dropM: number, offsetM: number) => {
  const samples: MeasuredGround[] = [];
  const ground: number[] = [];
  for (let i = 0; i < 5; i += 1) {
    const fall = (dropM * i) / 4;
    samples.push({ lon: -82.6966 + i * 0.0001, lat: 28.0389, height: 100 - fall });
    ground.push(100 - fall - offsetM);
  }
  return { samples, ground };
};

/** Metres. Far below anything that is visible on a globe, and not a measured value. */
const MM = 1e-3;

describe("measured clamp", () => {
  it("cancels the slope: the lift is the offset, not the drop", () => {
    const { samples, ground } = slope(15, 4.2);
    const clamp = measuredClamp(samples, ground);
    expect(clamp).not.toBeNull();
    expect(clamp?.liftM).toBeCloseTo(-4.2, 6);
    expect(clamp?.cells).toBe(5);
    // Same surface, so nothing is left once the median is removed.
    expect(clamp?.spreadM ?? 1).toBeLessThan(MM);
  });

  it("is not dragged by one cell over a pond", () => {
    const { samples, ground } = slope(15, 4.2);
    // One cell whose low percentile fell through a water surface by 30 m.
    ground[2] = (ground[2] ?? 0) - 30;
    const clamp = measuredClamp(samples, ground);
    expect(clamp?.liftM).toBeCloseTo(-4.2, 6);
    // The median absolute deviation is robust in the same way the median is, so one bad
    // cell neither moves the lift nor inflates the reported spread. What the spread is for
    // is the case below: two surfaces that are not the same shape at all.
    expect(clamp?.spreadM ?? 1).toBeLessThan(MM);
  });

  it("reports a spread when the capture's ground is not the shape of the viewer's", () => {
    // The capture thinks it is flat; the viewer's ground falls 15 m across it. No single
    // lift makes those agree, and the spread is what says so.
    const samples: MeasuredGround[] = [];
    const ground: number[] = [];
    for (let i = 0; i < 5; i += 1) {
      samples.push({ lon: -82.6966 + i * 0.0001, lat: 28.0389, height: 100 });
      ground.push(100 - (15 * i) / 4);
    }
    const clamp = measuredClamp(samples, ground);
    expect(clamp?.liftM).toBeCloseTo(-7.5, 6);
    expect(clamp?.spreadM ?? 0).toBeGreaterThan(3);
  });

  it("counts only the cells that had ground under them", () => {
    const { samples, ground } = slope(15, 4.2);
    const partial: (number | undefined)[] = [...ground];
    partial[0] = undefined;
    partial[4] = undefined;
    const clamp = measuredClamp(samples, partial);
    expect(clamp?.cells).toBe(3);
    expect(clamp?.liftM).toBeCloseTo(-4.2, 6);
  });

  it("returns null when nothing could be sampled, rather than a lift of zero", () => {
    const { samples } = slope(15, 4.2);
    expect(
      measuredClamp(
        samples,
        samples.map(() => undefined),
      ),
    ).toBeNull();
    expect(measuredClamp([], [])).toBeNull();
  });

  it("ignores a cell whose height is not a number", () => {
    const samples: MeasuredGround[] = [
      { lon: 0, lat: 0, height: Number.NaN },
      { lon: 0.0001, lat: 0, height: 10 },
    ];
    const clamp = measuredClamp(samples, [5, 5]);
    expect(clamp?.cells).toBe(1);
    expect(clamp?.liftM).toBeCloseTo(-5, 6);
  });
});

describe("ground under a point", () => {
  it("prefers the drawn surface when it is plausibly the same ground", () => {
    expect(groundAt(100, 103, 60)).toBe(103);
  });

  it("keeps the terrain when the drawn surface is a roof", () => {
    expect(groundAt(100, 180, 60)).toBe(100);
  });

  it("takes whichever one exists", () => {
    expect(groundAt(undefined, 180, 60)).toBe(180);
    expect(groundAt(100, undefined, 60)).toBe(100);
    expect(groundAt(undefined, undefined, 60)).toBeUndefined();
    expect(groundAt(Number.NaN, undefined, 60)).toBeUndefined();
  });
});

describe("median", () => {
  it("averages the middle two of an even count", () => {
    expect(median([4, 1, 3, 2])).toBeCloseTo(2.5, 12);
  });

  it("takes the middle of an odd count", () => {
    expect(median([9, 1, 5])).toBe(5);
  });
});
