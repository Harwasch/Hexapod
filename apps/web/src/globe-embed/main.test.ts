import type * as Cesium from "cesium";
import { EntityCollection, Cartesian2 } from "cesium";
import { afterEach, expect, it, vi } from "vitest";
import fixtures from "../../../ios/Modules/Globe/Tests/GlobeTests/Fixtures/messages.json";

const hooks = vi.hoisted(() => ({
  click: (_event: { position: unknown }): void => {
    /* initialized by handler */
  },
  widget: vi.fn(),
  load: vi.fn(),
}));
vi.mock("cesium", async (original) => ({
  ...(await original<typeof Cesium>()),
  ImageryLayer: { fromProviderAsync: vi.fn() },
  ScreenSpaceEventHandler: class {
    setInputAction(action: typeof hooks.click): void {
      hooks.click = action;
    }
    destroy(): void {
      /* fixture owns no browser handler */
    }
  },
}));
vi.mock("@/cesium/createGlobeWidget", () => ({ createGlobeWidget: hooks.widget }));
vi.mock("@/cesium/providers/tiles", () => ({ createEmbedScanTileset: hooks.load }));
vi.mock("@/cesium/providers/imagery", () => ({ createNaturalEarthProvider: vi.fn() }));
afterEach(() => {
  window.dispatchEvent(new Event("pagehide"));
  delete window.webkit;
});
it("boots offline, shares pins, sends native tap IDs, flies to loaded geometry and cleans up", async () => {
  document.body.innerHTML = '<div id="globe"></div><div id="credits"></div><p id="status"></p>';
  const postMessage = vi.fn();
  window.webkit = { messageHandlers: { globe: { postMessage } } };
  const entities = new EntityCollection();
  const viewer = {
    entities,
    resolutionScale: 1,
    canvas: document.createElement("canvas"),
    scene: {
      globe: {},
      requestRender: vi.fn(),
      pick: vi.fn(),
      primitives: { add: vi.fn(), remove: vi.fn() },
    },
    imageryLayers: { add: vi.fn() },
    camera: { flyTo: vi.fn(), flyToBoundingSphere: vi.fn() },
    destroy: vi.fn(),
  };
  hooks.widget.mockReturnValue(viewer);
  const tile = { boundingSphere: { radius: 4 }, destroy: vi.fn() };
  hooks.load.mockResolvedValue(tile);
  await import("./main");
  expect(postMessage).toHaveBeenCalledWith({ version: 1, type: "ready" });
  const bridge = window.globeBridge!;
  fixtures.slice(0, 3).forEach((fixture) => bridge.receive(fixture));
  await Promise.resolve();
  expect(entities.values).toHaveLength(3);
  expect(hooks.load).toHaveBeenCalledWith("./fixtures/globe-splat/tileset.json");
  expect(viewer.camera.flyToBoundingSphere).toHaveBeenCalledWith(
    tile.boundingSphere,
    expect.any(Object),
  );
  bridge.receive(fixtures[1]);
  expect(hooks.load).toHaveBeenCalledOnce();
  const pin = fixtures[0]!.snapshot!.pins[0]!;
  viewer.scene.pick.mockReturnValue({
    id: entities.getById(`property-answer:${pin.siteId}:${pin.id}`),
  });
  hooks.click({ position: new Cartesian2() });
  expect(postMessage).toHaveBeenCalledWith({ version: 1, type: "pinTapped", pinId: pin.id });
  bridge.receive({ version: 1, type: "selectScan", scanId: "absent" });
  expect(postMessage).toHaveBeenCalledWith({
    version: 1,
    type: "failure",
    message: "Invalid globe command",
  });
  bridge.receive({ version: 1, type: "snapshot", snapshot: { pins: [], scans: [], boundary: [] } });
  expect(entities.values).toHaveLength(0);
  expect(viewer.scene.primitives.remove).toHaveBeenCalledWith(tile);
});
