import {
  Cartesian3,
  Cartographic,
  Matrix4,
  Matrix3,
  Transforms,
  EllipsoidTerrainProvider,
} from "cesium";
import type { Cesium3DTileset } from "cesium";
import { expect, it } from "vitest";
import { ScanResolution, scanRange, seatScan } from "./scanView";

it.each([
  [44.7965, -111.1346, 2010],
  [38.0637, -122.6998, 55],
])(
  "seats a scan at terrain height at %s,%s without changing its local frame",
  async (lat, lon, height) => {
    const origin = Cartesian3.fromDegrees(lon, lat, 0);
    const root = { transform: Transforms.eastNorthUpToFixedFrame(origin) };
    const tiles = {
      root,
      boundingSphere: { center: origin, radius: 2 },
      modelMatrix: Matrix4.IDENTITY,
    } as unknown as Cesium3DTileset;
    await seatScan(tiles, new EllipsoidTerrainProvider(), (_provider, positions) => {
      // 0.1 micrometre allows ECEF double roundoff, well below scan precision.
      expect(Cartesian3.distance(Cartographic.toCartesian(positions[0]!), origin)).toBeLessThan(
        1e-7,
      );
      positions[0]!.height = height;
      return Promise.resolve(positions);
    });
    const seated = Matrix4.multiplyByPoint(tiles.modelMatrix, origin, new Cartesian3());
    expect(Cartesian3.distance(seated, Cartesian3.fromDegrees(lon, lat, height))).toBeLessThan(
      1e-7,
    );
    expect(root.transform).toEqual(Transforms.eastNorthUpToFixedFrame(origin));
  },
);

it("frames small scans closely and scales range with radius", () => {
  expect(scanRange(0.1)).toBe(0.5);
  expect(scanRange(4)).toBe(10);
  expect(scanRange(100)).toBe(250);
});

it("seats offline scans on the ellipsoid without terrain availability or a network sample", async () => {
  const terrain = new EllipsoidTerrainProvider();
  expect(terrain.availability).toBeUndefined();
  const origin = Cartesian3.fromDegrees(-111.1346, 44.7965, 30);
  const tiles = {
    root: { transform: Transforms.eastNorthUpToFixedFrame(origin) },
    boundingSphere: { center: origin, radius: 2 },
    modelMatrix: Matrix4.IDENTITY,
  } as unknown as Cesium3DTileset;
  // The real sampleTerrainMostDetailed rejects providers without availability.
  // Use the default sampler here to exercise the offline bypass, not an injected fake.
  await seatScan(tiles, terrain);
  const seated = Matrix4.multiplyByPoint(tiles.modelMatrix, origin, new Cartesian3());
  expect(Cartesian3.distance(seated, Cartesian3.fromDegrees(-111.1346, 44.7965, 0))).toBeLessThan(
    1e-7,
  );
});

it("starts at full device resolution capped at 3 and steps down only after sustained slow frames", () => {
  const quality = new ScanResolution(3);
  expect(quality.scale).toBe(3);
  expect(new ScanResolution(4).scale).toBe(3);
  for (let i = 0; i < 59; i++) quality.frame(50);
  expect(quality.scale).toBe(3);
  quality.frame(50);
  expect(quality.scale).toBe(2.5);
  // An idle request-render gap is not a slow frame.
  for (let i = 0; i < 120; i++) quality.frame(2000);
  expect(quality.scale).toBe(2.5);
});

it("does not pretend a failed terrain sample is zero height", async () => {
  const origin = Cartesian3.fromDegrees(-111, 44, 30);
  const tiles = {
    root: { transform: Transforms.eastNorthUpToFixedFrame(origin) },
    boundingSphere: { center: origin, radius: 2 },
    modelMatrix: Matrix4.IDENTITY,
  } as unknown as Cesium3DTileset;
  await expect(
    seatScan(tiles, new EllipsoidTerrainProvider(), (_provider, positions) => {
      positions[0]!.height = Number.NaN;
      return Promise.resolve(positions);
    }),
  ).rejects.toThrow("Terrain height unavailable");
  expect(tiles.modelMatrix).toBe(Matrix4.IDENTITY);
});

it("seats the root box's ground instead of burying geometry below the scan origin", async () => {
  const origin = Cartesian3.fromDegrees(0, 0, 0);
  const tiles = {
    root: {
      transform: Transforms.eastNorthUpToFixedFrame(origin),
      boundingVolume: {
        boundingVolume: {
          center: Cartesian3.fromDegrees(0, 0, 1),
          halfAxes: Matrix3.fromScale(new Cartesian3(2, 1, 1)),
        },
      },
    },
    boundingSphere: { center: Cartesian3.fromDegrees(0, 0, 1), radius: 3 },
    modelMatrix: Matrix4.IDENTITY,
  } as unknown as Cesium3DTileset;
  await seatScan(tiles, new EllipsoidTerrainProvider(), (_provider, positions) => {
    positions[0]!.height = 55;
    return Promise.resolve(positions);
  });
  const bottom = new Cartesian3(origin.x - 1, 1, 1);
  const seated = Matrix4.multiplyByPoint(tiles.modelMatrix, bottom, new Cartesian3());
  // Same ECEF double-roundoff bound as the origin tests, not a placement tolerance.
  expect(Math.abs(Cartographic.fromCartesian(seated).height - 55)).toBeLessThan(1e-7);
});
