import { describe, expect, it } from "vitest";

import {
  groundAt,
  measuredClamp,
  median,
  robustArrivalSphere,
  type ArrivalSphere,
  type GeoPoint,
  type MeasuredGround,
} from "@/cesium/placement";

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
    // Placeholders and chords of coarse tiles, kilometres under the sea, are not ground.
    expect(groundAt(-31_017, undefined, 60)).toBeUndefined();
    expect(groundAt(undefined, -3_640, 60)).toBeUndefined();
    expect(groundAt(12, -3_640, 60)).toBe(12);
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

/**
 * A placed scan as the pipeline packs one: the origin at the middle of its footprint, on its
 * own ground (here 1,500 m up, a Montana field), with 64 dense ground cells around it some
 * 20 m out, and a tileset whose bounding sphere the floaters have dragged 40 m east and 25 m
 * up and blown up to 130 m -- the shape of the Camp scan's.
 */
const ORIGIN: GeoPoint = { lon: -111.1346, lat: 44.7965, height: 1500 };
const M_PER_DEG = 111_320;
const east = (m: number) => m / (M_PER_DEG * Math.cos((ORIGIN.lat * Math.PI) / 180));
const north = (m: number) => m / M_PER_DEG;

function cells(reachM: number, count = 64): MeasuredGround[] {
  const out: MeasuredGround[] = [];
  for (let i = 0; i < count; i += 1) {
    // A disc filled evenly: radius grows with the square root of the index.
    const r = reachM * Math.sqrt((i + 0.5) / count);
    const a = i * 2.399963; // the golden angle, so the cells do not line up
    out.push({
      lon: ORIGIN.lon + east(r * Math.cos(a)),
      lat: ORIGIN.lat + north(r * Math.sin(a)),
      height: 0.1,
    });
  }
  return out;
}

const floaterBounds: ArrivalSphere = {
  lon: ORIGIN.lon + east(40),
  lat: ORIGIN.lat,
  height: ORIGIN.height + 25,
  radiusM: 130,
};

/** Horizontal metres between two points, for checking where a sphere was put. */
function apartM(a: GeoPoint, b: GeoPoint): number {
  return Math.hypot((a.lon - b.lon) / east(1), (a.lat - b.lat) / north(1));
}

describe("robust arrival sphere", () => {
  it("frames the placement origin, wherever the floaters put the bounding sphere", () => {
    const framed = robustArrivalSphere(ORIGIN, floaterBounds, cells(20), 105);
    expect(apartM(framed, ORIGIN)).toBeLessThan(MM);
    expect(framed.height).toBeCloseTo(ORIGIN.height, 9);
    // The same scan with its floaters elsewhere: the same sphere.
    const other = { ...floaterBounds, lon: ORIGIN.lon - east(60), height: ORIGIN.height + 5 };
    expect(robustArrivalSphere(ORIGIN, other, cells(20), 105)).toEqual(framed);
  });

  it("sizes it by the scan's own ground, not by its floaters or its footprint", () => {
    const framed = robustArrivalSphere(ORIGIN, floaterBounds, cells(20), 105);
    // Ninety per cent of a disc of cells 20 m in radius is within about 19 m of its middle.
    expect(framed.radiusM).toBeGreaterThan(15);
    expect(framed.radiusM).toBeLessThan(20);
    // A few dense patches of background 80 m out are the tenth that does not count.
    const patches = [
      ...cells(20, 60),
      ...cells(0, 4).map((c) => ({ ...c, lon: c.lon + east(80) })),
    ];
    const withPatches = robustArrivalSphere(ORIGIN, floaterBounds, patches, 105);
    expect(withPatches.radiusM).toBeLessThan(21);
  });

  it("falls back to the smaller of the footprint and the bounds with too few cells to spread", () => {
    const one = cells(0, 1);
    expect(robustArrivalSphere(ORIGIN, floaterBounds, one, 35).radiusM).toBe(35);
    const tight = { ...floaterBounds, lon: ORIGIN.lon, height: ORIGIN.height + 1, radiusM: 6.3 };
    expect(robustArrivalSphere(ORIGIN, tight, one, 35).radiusM).toBe(6.3);
    expect(robustArrivalSphere(ORIGIN, tight, [], undefined).radiusM).toBe(6.3);
  });

  it("never frames more than the tiles hold, nor less than a metre", () => {
    const tight = { ...floaterBounds, lon: ORIGIN.lon, height: ORIGIN.height + 1, radiusM: 7.6 };
    // Ground cells 40 m out around tiles that reach 8 m disagree with them.
    expect(robustArrivalSphere(ORIGIN, tight, cells(40), 2).radiusM).toBe(7.6);
    expect(robustArrivalSphere(ORIGIN, floaterBounds, cells(0, 8), 105).radiusM).toBe(1);
  });

  it("keeps the bounding sphere when the origin is not in the scan", () => {
    // A capture never recentred: its frame starts where the phone app did, 200 m away.
    const astray = { ...ORIGIN, lon: ORIGIN.lon + east(200) };
    expect(robustArrivalSphere(astray, floaterBounds, cells(20), 105)).toBe(floaterBounds);
    expect(robustArrivalSphere({ ...ORIGIN, height: Number.NaN }, floaterBounds, [], 5)).toBe(
      floaterBounds,
    );
  });
});
