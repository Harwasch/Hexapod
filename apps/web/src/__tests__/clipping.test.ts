import { ClippingPolygonCollection, type Scene } from "cesium";
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
