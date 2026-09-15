import { describe, expect, it } from "vitest";

import {
  bearing,
  boundingRadiusM,
  boundsOf,
  circleFootprint,
  destination,
  footprintAreaM2,
  formatAltitude,
  formatArea,
  formatLatLon,
  formatLength,
  formatResolution,
  haversineDistance,
  metersPerPixel,
  normalizeLongitude,
  scaleBandForAltitude,
  validateFootprint,
} from "./index";

const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-122.139, 47.644],
      [-122.137, 47.644],
      [-122.137, 47.645],
      [-122.139, 47.645],
      [-122.139, 47.644],
    ],
  ],
};

describe("units", () => {
  it("formats metric lengths across scales", () => {
    expect(formatLength(0.004)).toBe("4 mm");
    expect(formatLength(0.000045)).toBe("0.05 mm");
    expect(formatAltitude(0.161)).toBe("16.1 cm");
    expect(formatAltitude(4.26)).toBe("4.3 m");
    expect(formatLength(0.253)).toBe("25.3 cm");
    expect(formatLength(12.345)).toBe("12.35 m");
    expect(formatLength(2500)).toBe("2.5 km");
  });
  it("formats imperial lengths", () => {
    expect(formatLength(0.1, "imperial")).toBe("3.94 in");
    expect(formatLength(10, "imperial")).toBe("32.81 ft");
    expect(formatLength(5000, "imperial")).toBe("3.11 mi");
  });
  it("formats areas in both systems", () => {
    expect(formatArea(50)).toBe("50 m²");
    expect(formatArea(25_000)).toBe("2.5 ha");
    expect(formatArea(3_000_000)).toBe("3 km²");
    expect(formatArea(40_469, "imperial")).toBe("10 ac");
  });
  it("formats altitude and resolution", () => {
    expect(formatAltitude(12_345)).toBe("12 km");
    expect(formatAltitude(3.14)).toBe("3.1 m");
    expect(formatResolution(0.032)).toBe("3.2 cm/px");
    expect(formatResolution(Number.NaN)).toBe("—");
  });
});

describe("coordinates", () => {
  it("formats lat/lon with hemispheres", () => {
    expect(formatLatLon({ longitude: -122.138, latitude: 47.644 }, 3)).toBe(
      "47.644° N, 122.138° W",
    );
  });
  it("computes haversine distance and bearing", () => {
    const paris = { longitude: 2.3522, latitude: 48.8566 };
    const london = { longitude: -0.1276, latitude: 51.5072 };
    expect(haversineDistance(paris, london)).toBeCloseTo(343_500, -3);
    expect(bearing(paris, london)).toBeCloseTo(330, 0);
  });
  it("destination inverts distance and bearing", () => {
    const start = { longitude: 10, latitude: 50 };
    const end = destination(start, 45, 1000);
    expect(haversineDistance(start, end)).toBeCloseTo(1000, 3);
    expect(bearing(start, end)).toBeCloseTo(45, 1);
  });
  it("normalizes longitude", () => {
    expect(normalizeLongitude(190)).toBe(-170);
    expect(normalizeLongitude(-190)).toBe(170);
  });
});

describe("footprint", () => {
  it("computes bounds, radius and area", () => {
    expect(boundsOf(square)).toEqual({
      west: -122.139,
      south: 47.644,
      east: -122.137,
      north: 47.645,
    });
    expect(boundingRadiusM(square)).toBeGreaterThan(90);
    expect(footprintAreaM2(square)).toBeCloseTo(16_700, -3);
  });
  it("subtracts holes and handles multipolygons", () => {
    const withHole = {
      type: "Polygon" as const,
      coordinates: [
        square.coordinates[0]!,
        [
          [-122.1385, 47.6442],
          [-122.1385, 47.6448],
          [-122.1375, 47.6448],
          [-122.1375, 47.6442],
          [-122.1385, 47.6442],
        ],
      ],
    };
    expect(footprintAreaM2(withHole)).toBeLessThan(footprintAreaM2(square));
    const multi = {
      type: "MultiPolygon" as const,
      coordinates: [square.coordinates, square.coordinates],
    };
    expect(footprintAreaM2(multi)).toBeCloseTo(2 * footprintAreaM2(square), 0);
  });
  it("validates footprints and unwraps features", () => {
    expect(validateFootprint(square).ok).toBe(true);
    expect(validateFootprint({ type: "Feature", geometry: square, properties: {} }).ok).toBe(true);
    expect(validateFootprint({ type: "Point", coordinates: [0, 0] })).toMatchObject({ ok: false });
    const open = {
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ],
      ],
    };
    expect(validateFootprint(open)).toMatchObject({ ok: false, error: "Rings must be closed" });
    expect(
      validateFootprint({
        type: "Polygon",
        coordinates: [
          [
            [0, 0],
            [200, 0],
            [1, 1],
            [0, 0],
          ],
        ],
      }),
    ).toMatchObject({
      ok: false,
      error: "Coordinates out of range",
    });
  });
  it("builds a closed circle footprint", () => {
    const circle = circleFootprint({ longitude: 0, latitude: 0 }, 100, 64);
    const ring = circle.coordinates[0]!;
    expect(ring).toHaveLength(65);
    expect(ring[0]).toEqual(ring[64]);
    expect(footprintAreaM2(circle)).toBeCloseTo(Math.PI * 100 * 100, -3);
  });
});

describe("scale", () => {
  it("classifies altitude bands", () => {
    expect(scaleBandForAltitude(10_000_000)).toBe("planet");
    expect(scaleBandForAltitude(100_000)).toBe("region");
    expect(scaleBandForAltitude(500)).toBe("site");
    expect(scaleBandForAltitude(80)).toBe("object");
    expect(scaleBandForAltitude(1)).toBe("detail");
  });
  it("computes metres per pixel", () => {
    expect(metersPerPixel(1000, Math.PI / 3, 1000)).toBeCloseTo(1.1547, 3);
    expect(metersPerPixel(0, 1, 100)).toBeNaN();
  });
});
