import type { CesiumWidget } from "cesium";
import { Event } from "cesium";
import { beforeEach, expect, it, vi } from "vitest";
import { installBasemap } from "./basemap";

const hooks = vi.hoisted(() => ({
  natural: vi.fn(),
  imagery: vi.fn(),
  terrain: vi.fn(),
  osm: vi.fn(),
}));
vi.mock("@/cesium/providers/imagery", () => ({ createNaturalEarthProvider: hooks.natural }));
vi.mock("cesium", async (original) => ({
  ...(await original<Record<string, unknown>>()),
  ImageryLayer: class {
    show = true;
    constructor(public provider: unknown) {}
  },
  IonImageryProvider: { fromAssetId: hooks.imagery },
  CesiumTerrainProvider: { fromIonAssetId: hooks.terrain },
  OpenStreetMapImageryProvider: class {
    errorEvent = new Event();
    constructor(options: unknown) {
      hooks.osm(options);
    }
  },
}));
function setup() {
  const add = vi.fn((layer: unknown) => layer);
  return {
    viewer: {
      imageryLayers: { add },
      scene: { requestRender: vi.fn() },
    } as unknown as CesiumWidget,
    add,
    status: vi.fn<(message: string) => void>(),
  };
}
beforeEach(() => {
  vi.clearAllMocks();
  hooks.natural.mockResolvedValue({ errorEvent: new Event() });
});
it("keeps a bundled map and uses OSM without a token; never tries ion's evaluation token", async () => {
  const { viewer, add, status } = setup();
  installBasemap(viewer, "", status);
  await Promise.resolve();
  expect(add).toHaveBeenCalledTimes(2);
  expect(add.mock.calls[0]?.[0]).toBeDefined();
  expect(hooks.osm).toHaveBeenCalledWith(
    expect.objectContaining({ url: "https://tile.openstreetmap.org/" }),
  );
  expect(hooks.imagery).not.toHaveBeenCalled();
  expect(hooks.terrain).not.toHaveBeenCalled();
  expect(status).toHaveBeenCalledWith("Terrain unavailable");
});
it("uses the deployed public token for imagery and terrain and shows network failures", async () => {
  hooks.imagery.mockRejectedValue(new Error("offline"));
  hooks.terrain.mockRejectedValue(new Error("offline"));
  const { viewer, add, status } = setup();
  installBasemap(viewer, " public-test-token ", status);
  await Promise.resolve();
  expect(hooks.imagery).toHaveBeenCalledWith(2, { accessToken: "public-test-token" });
  expect(hooks.terrain).toHaveBeenCalledWith(1, { requestVertexNormals: true });
  expect(add).toHaveBeenCalledOnce();
  expect(status).toHaveBeenCalledWith("Map offline");
  expect(status).toHaveBeenCalledWith("Terrain unavailable");
});
it("discards late resources after the page closes", async () => {
  const { viewer, add, status } = setup();
  const dispose = installBasemap(viewer, undefined, status);
  dispose();
  await Promise.resolve();
  expect(add).not.toHaveBeenCalled();
});

it("announces terrain readiness after installation, and releases waiters on disposal", async () => {
  const { viewer, status } = setup();
  const terrain = { errorEvent: new Event() };
  hooks.imagery.mockResolvedValue({ errorEvent: new Event() });
  hooks.terrain.mockResolvedValue(terrain);
  const ready = vi.fn(() => expect(viewer.terrainProvider).toBe(terrain));
  const remove = installBasemap(viewer, "public-test-token", status, ready);
  expect(ready).not.toHaveBeenCalled();
  await Promise.resolve();
  expect(ready).toHaveBeenCalledOnce();
  remove();
  expect(ready).toHaveBeenCalledTimes(2);
});
