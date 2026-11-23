import type * as Cesium from "cesium";
import {
  EntityCollection,
  Cartesian2,
  Cartesian3,
  Transforms,
  HeadingPitchRange,
  EllipsoidTerrainProvider,
  Event as CesiumEvent,
} from "cesium";
import { afterEach, expect, it, vi } from "vitest";
import fixtures from "../../../ios/Modules/Globe/Tests/GlobeTests/Fixtures/messages.json";
import { installBasemap } from "./basemap";

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
vi.mock("./basemap", () => ({
  installBasemap: vi.fn<typeof installBasemap>((_v, _token, _status, ready) => {
    ready?.();
    return vi.fn();
  }),
}));
afterEach(() => {
  window.dispatchEvent(new Event("pagehide"));
  delete window.webkit;
});
it("boots offline, shares pins, sends native tap IDs, flies to loaded geometry and cleans up", async () => {
  document.body.innerHTML = '<div id="globe"></div><div id="credits"></div><p id="status"></p>';
  Object.defineProperty(window, "devicePixelRatio", { value: 3, configurable: true });
  const postMessage = vi.fn();
  window.webkit = { messageHandlers: { globe: { postMessage } } };
  const entities = new EntityCollection();
  const viewer = {
    entities,
    terrainProvider: new EllipsoidTerrainProvider(),
    resolutionScale: 1,
    canvas: document.createElement("canvas"),
    scene: {
      globe: {},
      screenSpaceCameraController: { minimumZoomDistance: 1, enableCollisionDetection: true },
      postRender: new CesiumEvent(),
      renderError: new CesiumEvent(),
      requestRender: vi.fn(),
      pick: vi.fn(),
      primitives: { add: vi.fn(), remove: vi.fn() },
    },
    imageryLayers: { add: vi.fn() },
    camera: { flyTo: vi.fn(), flyToBoundingSphere: vi.fn() },
    destroy: vi.fn(),
  };
  hooks.widget.mockReturnValue(viewer);
  const origin = Cartesian3.fromDegrees(-122, 38);
  const tile = {
    root: { transform: Transforms.eastNorthUpToFixedFrame(origin) },
    tileFailed: new CesiumEvent(),
    boundingSphere: { center: origin, radius: 4 },
    destroy: vi.fn(),
  };
  hooks.load.mockResolvedValue(tile);
  await import("./main");
  expect(viewer.resolutionScale).toBe(3);
  expect(viewer.scene.screenSpaceCameraController.minimumZoomDistance).toBe(0.05);
  expect(postMessage).toHaveBeenCalledWith({ version: 1, type: "ready" });
  vi.mocked(installBasemap).mock.calls[0]![2]("No imagery");
  expect(postMessage).toHaveBeenCalledWith({ version: 1, type: "failure", message: "No imagery" });
  const bridge = window.globeBridge!;
  fixtures.slice(0, 3).forEach((fixture) => bridge.receive(fixture));
  await vi.waitFor(() => expect(viewer.scene.primitives.add).toHaveBeenCalledWith(tile));
  expect(entities.values).toHaveLength(3);
  expect(viewer.scene.screenSpaceCameraController.enableCollisionDetection).toBe(false);
  expect(hooks.load).toHaveBeenCalledWith("./fixtures/globe-splat/tileset.json");
  expect(viewer.camera.flyToBoundingSphere).toHaveBeenCalledWith(
    tile.boundingSphere,
    expect.objectContaining({ offset: new HeadingPitchRange(0, -0.7, 10) }),
  );
  bridge.receive(fixtures[1]);
  expect(hooks.load).toHaveBeenCalledOnce();
  viewer.scene.pick.mockReturnValue({
    id: entities.getById(`scan:${fixtures[0]!.snapshot!.scans[0]!.id}:0`),
  });
  viewer.camera.flyToBoundingSphere.mockClear();
  hooks.click({ position: new Cartesian2() });
  expect(viewer.camera.flyToBoundingSphere).toHaveBeenCalledOnce();
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
  tile.tileFailed.raiseEvent({ url: "private", message: "network" });
  expect(postMessage).toHaveBeenCalledWith({
    version: 1,
    type: "failure",
    message: "Scan could not load",
  });
  viewer.scene.renderError.raiseEvent();
  expect(postMessage).toHaveBeenCalledWith({
    version: 1,
    type: "failure",
    message: "Globe rendering failed",
  });
  bridge.receive({ version: 1, type: "snapshot", snapshot: { pins: [], scans: [], boundary: [] } });
  expect(entities.values).toHaveLength(0);
  expect(viewer.scene.screenSpaceCameraController.enableCollisionDetection).toBe(true);
  expect(viewer.scene.primitives.remove).toHaveBeenCalledWith(tile);
});
