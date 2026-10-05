/**
 * A scan drawn at its runtime scale (`renderConfig.scale`, docs/DATA_MODEL.md "Runtime
 * scale"): the model matrix the viewer composes, and what changes and what must not.
 *
 * The contract is the API's. Every globe position the catalog gives -- the boundary, the
 * footprint, the ground samples -- already describes the scaled model, so the viewer scales only
 * what it draws in the tileset's own frame, about the root transform's origin. These pin the
 * arithmetic: the origin stays put, the model's lowest point scales with it, the ground samples
 * are never rescaled by the saved scale, and the inverse of a scaled frame is a true inverse.
 */
import { BoundingSphere, Cartesian3, Cartographic, Matrix4, Transforms } from "cesium";
import { describe, expect, it } from "vitest";

import {
  rescaleFootprint,
  rescaleGround,
  scaledHeight,
  scaledModelMatrix,
  uniformScale,
  type MeasuredGround,
} from "@/cesium/placement";
import {
  groundAtScale,
  inverseScaledTransformation,
  liftAt,
  placedMatrix,
  placementFrame,
  samePlacement,
  scaledBottom,
  scaledCenter,
} from "@/cesium/tilesetScale";
import type { SiteAsset } from "@twin/contracts";

const LON = -82.6966;
const LAT = 28.0389;
const ORIGIN_HEIGHT = 12;
const ORIGIN = Cartesian3.fromDegrees(LON, LAT, ORIGIN_HEIGHT);
const ROOT = Transforms.eastNorthUpToFixedFrame(ORIGIN);

/** Earth-fixed position of a point `enu` metres east/north/up of the origin, through `model`. */
function drawn(model: Matrix4, enu: [number, number, number]): Cartesian3 {
  const computed = Matrix4.multiply(model, ROOT, new Matrix4());
  return Matrix4.multiplyByPoint(computed, Cartesian3.fromArray(enu), new Cartesian3());
}

/** A tileset as `placementFrame` reads one: an east/north/up root, a sphere about its box. */
function tileset(boxCentre: [number, number, number], radius: number) {
  const centre = Matrix4.multiplyByPoint(ROOT, Cartesian3.fromArray(boxCentre), new Cartesian3());
  return {
    boundingSphere: new BoundingSphere(centre, radius),
    root: { transform: ROOT },
  } as never;
}

describe("the model matrix of a scaled scan", () => {
  it("keeps the root's origin where it is and draws every other point scale times as far", () => {
    const model = Matrix4.fromArray(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], 0.5));
    expect(Cartesian3.distance(drawn(model, [0, 0, 0]), ORIGIN)).toBeLessThan(1e-6);
    const point = drawn(model, [8, -6, 3]);
    // 8, 6, 3 metres registered: half of sqrt(109) on the globe, in the same direction.
    expect(Cartesian3.distance(point, ORIGIN)).toBeCloseTo(Math.sqrt(109) / 2, 6);
    const unscaled = drawn(Matrix4.IDENTITY, [8, -6, 3]);
    const along = Cartesian3.normalize(
      Cartesian3.subtract(unscaled, ORIGIN, new Cartesian3()),
      new Cartesian3(),
    );
    const now = Cartesian3.normalize(
      Cartesian3.subtract(point, ORIGIN, new Cartesian3()),
      new Cartesian3(),
    );
    expect(Cartesian3.dot(along, now)).toBeCloseTo(1, 12);
    expect(uniformScale(Matrix4.multiply(model, ROOT, new Matrix4()))).toBeCloseTo(0.5, 12);
  });

  it("is T(lift) · R · S · R⁻¹: the lift is added after the scale", () => {
    const lift: [number, number, number] = [1, 2, 3];
    const composed = Matrix4.multiply(
      Matrix4.fromTranslation(Cartesian3.fromArray(lift)),
      Matrix4.multiply(
        Matrix4.multiply(ROOT, Matrix4.fromUniformScale(0.3), new Matrix4()),
        Matrix4.inverse(ROOT, new Matrix4()),
        new Matrix4(),
      ),
      new Matrix4(),
    );
    const ours = Matrix4.fromArray(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], 0.3, lift));
    expect(Matrix4.equalsEpsilon(ours, composed, 1e-6)).toBe(true);
  });

  it("is exactly the clamp's old translation at scale 1, so an unscaled scan does not move", () => {
    const lift: [number, number, number] = [0.25, -1.5, 4.125];
    expect(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], 1, lift)).toEqual(
      Matrix4.toArray(Matrix4.fromTranslation(Cartesian3.fromArray(lift))),
    );
    // And through the frame, with a lift worked out at the registered centre exactly as the
    // clamp has always done it.
    const frame = placementFrame(tileset([2, 1, 3], 5), undefined);
    const centre = Cartographic.fromCartesian(frame.center);
    const from = Cartesian3.fromRadians(centre.longitude, centre.latitude, centre.height);
    const to = Cartesian3.fromRadians(centre.longitude, centre.latitude, centre.height - 7.5);
    expect(placedMatrix(frame, 1, -7.5)).toEqual(
      Matrix4.fromTranslation(Cartesian3.subtract(to, from, new Cartesian3())),
    );
    expect(placedMatrix(frame, 1, 0)).toEqual(Matrix4.IDENTITY);
  });

  it("scales about the root origin, not the bounding sphere's centre", () => {
    const frame = placementFrame(tileset([10, 0, 4], 6), undefined);
    expect(Cartesian3.distance(frame.origin, ORIGIN)).toBeLessThan(1e-6);
    expect(frame.ground.lon).toBeCloseTo(LON, 9);
    expect(frame.ground.lat).toBeCloseTo(LAT, 9);
    expect(frame.ground.height).toBeCloseTo(ORIGIN_HEIGHT, 6);
    // The centre moves towards the origin; drawn there by the placed matrix, too.
    const centre = scaledCenter(frame, 0.5);
    expect(Cartesian3.distance(centre, drawn(Matrix4.IDENTITY, [5, 0, 2]))).toBeLessThan(1e-6);
    const model = placedMatrix(frame, 0.5, 0);
    expect(Cartesian3.distance(drawn(model, [10, 0, 4]), centre)).toBeLessThan(1e-6);
  });

  it("falls back to the tileset's centre for a root that places nothing on the globe", () => {
    const centre = Cartesian3.fromDegrees(LON, LAT, 40);
    const frame = placementFrame(
      {
        boundingSphere: new BoundingSphere(centre, 3),
        root: { transform: Matrix4.IDENTITY },
      } as never,
      undefined,
    );
    expect(Cartesian3.distance(frame.origin, centre)).toBe(0);
  });
});

describe("the clamp under a runtime scale", () => {
  it("rests the lowest point, which scales with the model about the origin", () => {
    // The root box's lowest corner, 2 m under the origin: a metre under it at half size.
    const frame = placementFrame(tileset([0, 0, 3], 5), ORIGIN_HEIGHT - 2);
    expect(scaledBottom(frame, 1)).toBe(ORIGIN_HEIGHT - 2);
    expect(scaledBottom(frame, 0.5)).toBeCloseTo(ORIGIN_HEIGHT - 1, 9);
    expect(scaledHeight(100, 104, 0.25)).toBe(101);
    // With no box, the bottom of the sphere, which scales the same way.
    const sphere = placementFrame(tileset([0, 0, 3], 5), undefined);
    expect(scaledBottom(sphere, 0.5)).toBeCloseTo(ORIGIN_HEIGHT + (3 - 5) / 2, 4);
    // On the ground under the centre: under 30 m, the lowest point a metre below the origin.
    const rested = liftAt(frame, { scale: 1, samples: [], ground: [], under: 30 }, 0.5, 0.25);
    expect(rested?.liftM).toBeCloseTo(30 + 0.25 - (ORIGIN_HEIGHT - 1), 9);
  });

  it("never rescales the catalog's ground samples by the scale they already describe", () => {
    const samples: MeasuredGround[] = [
      { lon: LON + 0.0001, lat: LAT, height: ORIGIN_HEIGHT - 0.4 },
      { lon: LON, lat: LAT + 0.0002, height: ORIGIN_HEIGHT - 0.6 },
    ];
    const frame = placementFrame(tileset([0, 0, 3], 5), undefined);
    // No preview: the very same cells, untouched.
    expect(groundAtScale(frame, samples, 0.5 / 0.5)).toBe(samples);
    expect(rescaleGround(samples, frame.ground, 1)).toBe(samples);
    // A preview of a quarter against a saved half: half as far from the origin.
    const moved = groundAtScale(frame, samples, 0.25 / 0.5);
    expect(moved[0]?.lon).toBeCloseTo(LON + 0.00005, 12);
    expect(moved[1]?.lat).toBeCloseTo(LAT + 0.0001, 12);
    expect(moved[1]?.height).toBeCloseTo(ORIGIN_HEIGHT - 0.3, 9);
  });

  it("rests a previewed scale on the ground already sampled, before sampling it again", () => {
    const frame = placementFrame(tileset([0, 0, 3], 5), undefined);
    // Cells sampled at scale 0.5: their own ground 0.4 m under the origin, the terrain 3 m.
    const samples: MeasuredGround[] = [
      { lon: LON + 0.0001, lat: LAT, height: ORIGIN_HEIGHT - 0.4 },
      { lon: LON - 0.0001, lat: LAT, height: ORIGIN_HEIGHT - 0.4 },
    ];
    const sampled = { scale: 0.5, samples, ground: [3, 3], under: 3 };
    // At the sampled scale: the clamp's own answer.
    expect(liftAt(frame, sampled, 0.5, 0)?.liftM).toBeCloseTo(3 - (ORIGIN_HEIGHT - 0.4), 6);
    // At twice it the cells are twice as far under the origin, so the model goes up less.
    expect(liftAt(frame, sampled, 1, 0)?.liftM).toBeCloseTo(3 - (ORIGIN_HEIGHT - 0.8), 6);
    expect(liftAt(frame, undefined, 1, 0)).toBeNull();
  });
});

describe("a scaled frame's inverse", () => {
  it("is the true inverse, where the rigid one would be off by the scale squared", () => {
    const model = Matrix4.fromArray(
      scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], 0.4, [0, 0, 2]),
    );
    const toWorld = Matrix4.multiply(model, ROOT, new Matrix4());
    const toLocal = inverseScaledTransformation(toWorld, new Matrix4());
    const local = new Cartesian3(3, -4, 1.5);
    const back = Matrix4.multiplyByPoint(
      toLocal,
      Matrix4.multiplyByPoint(toWorld, local, new Cartesian3()),
      new Cartesian3(),
    );
    expect(Cartesian3.distance(back, local)).toBeLessThan(1e-6);
    expect(uniformScale(toLocal)).toBeCloseTo(2.5, 9);
    const rigid = Matrix4.inverseTransformation(toWorld, new Matrix4());
    const wrong = Matrix4.multiplyByPoint(
      rigid,
      Matrix4.multiplyByPoint(toWorld, local, new Cartesian3()),
      new Cartesian3(),
    );
    expect(Cartesian3.distance(wrong, local)).toBeGreaterThan(1);
  });

  it("is exactly inverseTransformation for a rigid frame", () => {
    expect(inverseScaledTransformation(ROOT, new Matrix4())).toEqual(
      Matrix4.inverseTransformation(ROOT, new Matrix4()),
    );
  });
});

describe("resizing a footprint for a preview", () => {
  it("moves each corner factor times as far from the origin, linearly in degrees", () => {
    const square = {
      type: "Polygon" as const,
      coordinates: [
        [
          [LON - 0.001, LAT - 0.001],
          [LON + 0.001, LAT - 0.001],
          [LON + 0.001, LAT + 0.001],
          [LON - 0.001, LAT - 0.001],
        ],
      ],
    };
    const half = rescaleFootprint(square, { lon: LON, lat: LAT }, 0.5);
    expect(half.coordinates[0]?.[1]?.[0]).toBeCloseTo(LON + 0.0005, 12);
    expect(half.coordinates[0]?.[1]?.[1]).toBeCloseTo(LAT - 0.0005, 12);
    expect(rescaleFootprint(square, { lon: LON, lat: LAT }, 1)).toBe(square);
    const multi = rescaleFootprint(
      { type: "MultiPolygon" as const, coordinates: [square.coordinates] },
      { lon: LON, lat: LAT },
      2,
    );
    expect(multi.coordinates[0]?.[0]?.[2]?.[0]).toBeCloseTo(LON + 0.002, 12);
  });
});

describe("whether a fresh record moved a scan", () => {
  const asset = (renderConfig: Record<string, unknown>, footprint: unknown = null) =>
    ({ renderConfig, footprint }) as unknown as SiteAsset;
  it("is a change of scale, ground, offset, clamp or footprint, and nothing else", () => {
    const base = {
      scale: 0.5,
      groundSamples: [{ lon: 1, lat: 2, height: 3 }],
      clampToGround: true,
    };
    expect(samePlacement(asset(base), asset({ ...base, maximumScreenSpaceError: 4 }))).toBe(true);
    expect(
      samePlacement(asset({ clampToGround: true }), asset({ clampToGround: true, scale: 1 })),
    ).toBe(true);
    expect(samePlacement(asset(base), asset({ ...base, scale: 0.25 }))).toBe(false);
    expect(samePlacement(asset(base), asset({ ...base, groundSamples: [] }))).toBe(false);
    expect(samePlacement(asset(base), asset(base, { type: "Polygon", coordinates: [] }))).toBe(
      false,
    );
  });
});
