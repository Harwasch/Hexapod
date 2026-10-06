/**
 * From orbit the earth is the globe, not the photorealistic world (LayerManager.setCameraAltitude,
 * ClippingManager.setOrbit).
 *
 * The world tileset's coarsest tiles are what it shows from out there, and while they load --
 * at startup, and through the middle of every long fly-to -- production drew the planet as a
 * polyhedron: straight segments for a horizon, and pieces missing. The globe is a smooth
 * ellipsoid with its atmosphere from the first frame. These pin the hand-over both ways: the
 * world goes above 400 km; it comes back below 250 km once its tiles for the view are in, or
 * after 1.5 s; and it keeps loading, hidden, on the way down.
 */
import { ClippingPolygonCollection, Rectangle, type Scene } from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Footprint } from "@twin/contracts";

import { ClippingManager } from "@/cesium/ClippingManager";
import { LayerManager } from "@/cesium/LayerManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

const SITE: Footprint = {
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

function stubScene() {
  return {
    globe: {
      show: true,
      clippingPolygons: undefined as unknown as ClippingPolygonCollection,
      cartographicLimitRectangle: undefined as unknown as Rectangle,
    },
    requestRender: vi.fn(),
  };
}

describe("the globe seen from orbit", () => {
  it("is drawn whole and unclipped in the photorealistic world, and clipped again below", () => {
    vi.spyOn(ClippingPolygonCollection, "isSupported").mockReturnValue(true);
    const scene = stubScene();
    const clipping = new ClippingManager(scene as unknown as Scene);
    clipping.setWorldMode(true);
    clipping.setFootprint("camp", SITE, { globe: false, world: true });
    const globe = scene.globe.clippingPolygons;
    expect(globe.enabled).toBe(true);
    expect(Rectangle.equals(scene.globe.cartographicLimitRectangle, Rectangle.MAX_VALUE)).toBe(
      false,
    );

    clipping.setOrbit(true);
    expect(scene.globe.show).toBe(true);
    expect(globe.enabled).toBe(false);
    expect(Rectangle.equals(scene.globe.cartographicLimitRectangle, Rectangle.MAX_VALUE)).toBe(
      true,
    );

    clipping.setOrbit(false);
    expect(globe.enabled).toBe(true);
    expect(Rectangle.equals(scene.globe.cartographicLimitRectangle, Rectangle.MAX_VALUE)).toBe(
      false,
    );
  });

  it("takes over from the world above 400 km, and hands back once the world's view is in", () => {
    const scene = stubScene();
    const setOrbit = vi.fn();
    const layers = new LayerManager({ scene } as never, new Emitter<SceneEvents>(), {
      setOrbit,
    } as unknown as ClippingManager);
    const world = {
      show: true,
      preloadWhenHidden: false,
      tilesLoaded: false,
      isDestroyed: () => false,
    };
    Object.assign(layers as unknown as Record<string, unknown>, {
      worldTilesetId: "google",
      worldTilesetRef: world,
    });
    (layers as unknown as { entries: Map<string, unknown> }).entries.set("google", {
      visible: true,
    });

    layers.setCameraAltitude(18_000_000, 0);
    expect(layers.orbitView).toBe(true);
    expect(world.show).toBe(false);
    expect(setOrbit).toHaveBeenLastCalledWith(true);
    // Too far out to be worth loading the world for.
    expect(world.preloadWhenHidden).toBe(false);

    // On the way down: loading, hidden, and still the globe within the hysteresis band.
    layers.setCameraAltitude(1_000_000, 100);
    expect(world.preloadWhenHidden).toBe(true);
    layers.setCameraAltitude(300_000, 200);
    expect(layers.orbitView).toBe(true);

    // Below 250 km, the world's tiles for the view not all in: the globe for 1.5 s more.
    layers.setCameraAltitude(200_000, 300);
    expect(layers.orbitView).toBe(true);
    layers.setCameraAltitude(150_000, 1_000);
    expect(layers.orbitView).toBe(true);
    layers.setCameraAltitude(100_000, 1_900);
    expect(layers.orbitView).toBe(false);
    expect(world.show).toBe(true);
    expect(setOrbit).toHaveBeenLastCalledWith(false);

    // Back up and down again, with the world's tiles in this time: at once.
    layers.setCameraAltitude(500_000, 3_000);
    expect(layers.orbitView).toBe(true);
    world.tilesLoaded = true;
    layers.setCameraAltitude(200_000, 3_100);
    expect(layers.orbitView).toBe(false);
    layers.destroy();
  });
});
