import {
  Cartesian3,
  ClippingPolygonCollection,
  Math as CesiumMath,
  type Cesium3DTileset,
  type Rectangle,
  type Scene,
} from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Footprint } from "@twin/contracts";

import { ClippingManager } from "@/cesium/ClippingManager";

const SQUARE: Footprint = {
  type: "Polygon",
  coordinates: [
    [
      [-123.881, 46.133],
      [-123.88, 46.133],
      [-123.88, 46.134],
      [-123.881, 46.134],
      [-123.881, 46.133],
    ],
  ],
};

function stubScene(): Scene & { globe: { clippingPolygons: ClippingPolygonCollection } } {
  return {
    globe: { show: true, clippingPolygons: undefined, cartographicLimitRectangle: undefined },
    requestRender: vi.fn(),
  } as unknown as Scene & { globe: { clippingPolygons: ClippingPolygonCollection } };
}

describe("the terrain floor under a splat scan", () => {
  it("is kept from above and dropped from inside, in the photorealistic world", () => {
    vi.spyOn(ClippingPolygonCollection, "isSupported").mockReturnValue(true);
    const scene = stubScene();
    const clipping = new ClippingManager(scene);
    clipping.setWorldMode(true);
    // A splat clips the world tileset only; the globe is kept (inverse clip) inside it.
    clipping.setFootprint("fort", SQUARE, { globe: false, world: true });
    const globe = scene.globe.clippingPolygons;
    const withFloor = globe.length;
    expect(withFloor).toBeGreaterThan(1); // the sentinel and the site
    clipping.setFloorless("fort");
    expect(globe.length).toBe(withFloor - 1); // the sentinel alone: no terrain anywhere
    clipping.setFloorless(null);
    expect(globe.length).toBe(withFloor);
  });

  it("is cut like a mesh site's in the open world", () => {
    vi.spyOn(ClippingPolygonCollection, "isSupported").mockReturnValue(true);
    const scene = stubScene();
    const clipping = new ClippingManager(scene);
    clipping.setFootprint("fort", SQUARE, { globe: false, world: true });
    const globe = scene.globe.clippingPolygons;
    expect(globe.length).toBe(0);
    clipping.setFloorless("fort");
    expect(globe.length).toBe(1);
    expect(globe.enabled).toBe(true);
    clipping.setFloorless(null);
    expect(globe.length).toBe(0);
  });
});

describe("the Time Lantern's ground window", () => {
  const ring = Cartesian3.fromDegreesArray([
    -120.041, 34.697, -120.039, 34.697, -120.039, 34.699, -120.041, 34.699,
  ]);

  function photorealistic() {
    vi.spyOn(ClippingPolygonCollection, "isSupported").mockReturnValue(true);
    const scene = stubScene();
    const clipping = new ClippingManager(scene);
    const world = { clippingPolygons: undefined } as unknown as Cesium3DTileset & {
      clippingPolygons: ClippingPolygonCollection;
    };
    clipping.setWorldTileset(world);
    clipping.setWorldMode(true);
    clipping.setFootprint("fort", SQUARE, { globe: false, world: true });
    return { scene, clipping, world };
  }

  it("adds one polygon to the globe and none to the world tileset, and is not a footprint", () => {
    const { scene, clipping, world } = photorealistic();
    const globe = scene.globe.clippingPolygons;
    const before = { globe: globe.length, world: world.clippingPolygons.length };
    const keys = clipping.activeKeys;
    clipping.setLanternWindow(ring);
    expect(globe.length).toBe(before.globe + 1);
    expect(world.clippingPolygons.length).toBe(before.world);
    expect(clipping.activeKeys).toEqual(keys);
    // Swapped as the lens moves, never added twice.
    clipping.setLanternWindow(ring);
    expect(globe.length).toBe(before.globe + 1);
    clipping.setLanternWindow(null);
    expect(globe.length).toBe(before.globe);
  });

  it("widens the globe's limit to the reach while the lantern is open", () => {
    const { scene, clipping } = photorealistic();
    const limit = (): Rectangle => scene.globe.cartographicLimitRectangle;
    expect(CesiumMath.toDegrees(limit().west)).toBeGreaterThan(-123.9);
    clipping.setLanternReach({ west: -120.06, south: 34.68, east: -120.02, north: 34.72 });
    expect(CesiumMath.toDegrees(limit().west)).toBeCloseTo(-123.883, 2);
    expect(CesiumMath.toDegrees(limit().east)).toBeCloseTo(-120.02, 5);
    expect(CesiumMath.toDegrees(limit().south)).toBeCloseTo(34.68, 5);
    clipping.setLanternReach(null);
    expect(CesiumMath.toDegrees(limit().east)).toBeLessThan(-123);
  });

  it("is put back when the world is rebuilt", () => {
    const { scene, clipping } = photorealistic();
    clipping.setLanternWindow(ring);
    const withWindow = scene.globe.clippingPolygons.length;
    clipping.setWorldMode(false);
    clipping.setWorldMode(true);
    expect(scene.globe.clippingPolygons.length).toBe(withWindow);
  });

  it("does nothing in the open world or from orbit", () => {
    vi.spyOn(ClippingPolygonCollection, "isSupported").mockReturnValue(true);
    const scene = stubScene();
    const clipping = new ClippingManager(scene);
    expect(clipping.lanternSupported).toBe(false);
    clipping.setLanternWindow(ring);
    expect(scene.globe.clippingPolygons.length).toBe(0);
    clipping.setWorldMode(true);
    clipping.setOrbit(true);
    const inOrbit = scene.globe.clippingPolygons.length;
    clipping.setLanternWindow(ring);
    expect(scene.globe.clippingPolygons.length).toBe(inOrbit);
    expect(clipping.lanternSupported).toBe(false);
  });
});
