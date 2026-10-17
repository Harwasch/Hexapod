/**
 * The Time Lantern in the engine (cesium/TimeLantern.ts), with the widget, the clipping and the
 * layers faked as clipping.test.ts does: what it adds to the globe and in what order, when it
 * takes the wheel, and what closes it.
 */
import type * as Cesium from "cesium";
import { Cartesian3, Ellipsoid, Matrix4, Ray, type CesiumWidget, type ImageryLayer } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { LanternFrame, LanternSource } from "@/cesium/lanternTypes";
import type { LayerManager } from "@/cesium/LayerManager";
import { TimeLantern } from "@/cesium/TimeLantern";
import type { SceneEvents } from "@/cesium/types";
import type { WorldShader } from "@/cesium/worldShader";
import { Emitter } from "@/lib/emitter";
import { useSceneSelect } from "@/state/sceneSelect";

/** The "no record" grid draws a canvas, which jsdom has not: a stand-in that says what it is. */
vi.mock("cesium", async (importOriginal) => {
  const actual = await importOriginal<typeof Cesium>();
  class GridImageryProvider {
    readonly grid = true;
    readonly tilingScheme = new actual.GeographicTilingScheme();
    readonly rectangle = actual.Rectangle.MAX_VALUE;
    readonly minimumLevel = 0;
    readonly maximumLevel = 18;
    constructor(readonly options: unknown) {}
  }
  const SceneTransforms = { worldToWindowCoordinates: () => undefined };
  return { ...actual, GridImageryProvider, SceneTransforms };
});

const AREA = { west: -120.06, south: 34.68, east: -120.02, north: 34.72 };

function source(year: number): LanternSource {
  return {
    id: `naip-${year}`,
    kind: "aerial-photo",
    title: `NAIP ${year} aerial photograph`,
    dateLabel: `Flown ${year}`,
    dateMeaning: "Flight dates",
    sortDate: `${year}-06-01`,
    resolutionM: 0.6,
    mapScale: null,
    license: "U.S. public domain (USDA Farm Service Agency)",
    licenseUrl: null,
    attribution: "USDA FSA NAIP",
    accuracyLabel: "accuracy not stated · not checked against today's 3D world",
    bounds: [-120.1, 34.6, -119.99, 34.75],
    parts: [],
    tiles: {
      scheme: "web-mercator",
      urlTemplate: `https://planetarycomputer.microsoft.com/api/data/v1/mosaic/${year}/tiles/WebMercatorQuad/{z}/{x}/{y}@1x`,
      maximumLevel: 18,
    },
    warnings: [],
  };
}
const SOURCES = [2022, 2020, 2018, 2016, 2014].map(source);

/** An imagery collection as the widget's: ordered bottom to top. */
class Layers {
  readonly order: ImageryLayer[] = [];
  add(layer: ImageryLayer) {
    this.order.push(layer);
  }
  remove(layer: ImageryLayer) {
    const i = this.order.indexOf(layer);
    if (i >= 0) this.order.splice(i, 1);
    return i >= 0;
  }
  raiseToTop(layer: ImageryLayer) {
    this.remove(layer);
    this.order.push(layer);
  }
  get length() {
    return this.order.length;
  }
}

function harness() {
  const canvas = document.createElement("canvas");
  Object.defineProperty(canvas, "clientWidth", { value: 1000 });
  Object.defineProperty(canvas, "clientHeight", { value: 800 });
  // Cesium's credit box, on body as CesiumSceneManager puts it: a role="dialog" that is on the
  // page whether shown or not.
  const container = document.createElement("div");
  container.innerHTML = `<div class="cesium-credit-lightbox-overlay" style="display: none"><div class="cesium-credit-lightbox" role="dialog" aria-modal="true"></div></div>`;
  container.appendChild(canvas);
  document.body.appendChild(container);
  const preRender = new Set<() => void>();
  const loading = new Set<() => void>();
  const imageryLayers = new Layers();
  const scene = {
    preRender: {
      addEventListener: (listener: () => void) => {
        preRender.add(listener);
        return () => preRender.delete(listener);
      },
    },
    requestRender: vi.fn(),
    drawingBufferWidth: 1000,
    drawingBufferHeight: 800,
    pickPositionSupported: false,
    globe: {
      getHeight: (() => undefined) as (at: Cesium.Cartographic) => number | undefined,
      tileLoadProgressEvent: {
        addEventListener: (listener: () => void) => {
          loading.add(listener);
          return () => loading.delete(listener);
        },
      },
    },
  };
  const viewer = {
    scene,
    canvas,
    camera: {
      viewMatrix: Matrix4.clone(Matrix4.IDENTITY),
      // Straight down from 1 km over the ranch, whatever the pixel.
      getPickRay: (_window: unknown, result = new Ray()) => {
        result.origin = Cartesian3.fromDegrees(-120.04, 34.698, 1300);
        result.direction = Cartesian3.negate(
          Ellipsoid.WGS84.geodeticSurfaceNormal(result.origin, new Cartesian3()),
          new Cartesian3(),
        );
        return result;
      },
    },
    imageryLayers,
    isDestroyed: () => false,
  } as unknown as CesiumWidget;
  const events = new Emitter<SceneEvents>();
  const clipping = {
    supported: true,
    photorealisticWorld: true,
    isEnabled: true,
    orbitView: false,
    lanternSupported: true,
    setLanternWindow: vi.fn(),
    setLanternReach: vi.fn(),
  };
  let basemapDone: () => void = () => undefined;
  const basemap = new Promise<void>((resolve) => {
    basemapDone = resolve;
  });
  const layers = { ensureFallbackBasemap: vi.fn(() => basemap) };
  const worldShader = { setLantern: vi.fn() };
  const camera = { groundHeightAt: vi.fn(() => Promise.resolve(300)) };
  const lantern = new TimeLantern(
    viewer,
    events,
    clipping as unknown as ClippingManager,
    layers as unknown as LayerManager,
    worldShader as unknown as WorldShader,
    camera as unknown as CameraController,
  );
  const frames: LanternFrame[] = [];
  events.on("lantern", (frame) => frames.push(frame));
  // Near the ground in the photorealistic world: the lens may show.
  events.emit("camera", { altitude: 900 } as SceneEvents["camera"]);
  return {
    lantern,
    scene,
    frame: () => {
      for (const listener of preRender) listener();
    },
    tilesLoaded: () => {
      for (const listener of loading) listener();
    },
    canvas,
    events,
    clipping,
    layers,
    worldShader,
    imageryLayers,
    frames,
    basemapDone,
    basemap,
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

let current: ReturnType<typeof harness> | null = null;
beforeEach(() => {
  current = harness();
});
afterEach(() => {
  current?.lantern.destroy();
  current?.canvas.parentElement?.remove();
  current = null;
  useSceneSelect.setState({ mode: "select", index: -1, candidates: [] } as never);
});

function h() {
  if (!current) throw new Error("no harness");
  return current;
}

describe("the Time Lantern's imagery", () => {
  it("waits for a basemap before it adds anything to the globe", async () => {
    const { lantern, layers, imageryLayers, basemapDone, basemap } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    expect(layers.ensureFallbackBasemap).toHaveBeenCalled();
    expect(imageryLayers.length).toBe(0);
    basemapDone();
    await basemap;
    await flush();
    expect(imageryLayers.length).toBeGreaterThan(0);
  });

  it("puts the no-record grid right under the current source, with two neighbours loading", async () => {
    const { lantern, imageryLayers, basemapDone } = h();
    basemapDone();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    await flush();
    lantern.select(2);
    const order = imageryLayers.order;
    const top = order.at(-1);
    const below = order.at(-2);
    expect((below?.imageryProvider as unknown as { grid?: boolean }).grid).toBe(true);
    expect(top?.alpha).toBe(1);
    const sources = order.filter((layer) => !(layer.imageryProvider as { grid?: boolean }).grid);
    expect(sources).toHaveLength(3);
    expect(sources.filter((layer) => layer.alpha === 0)).toHaveLength(2);
    expect(sources.every((layer) => layer.show)).toBe(true);
    // The reach preloads while the lens is open, and only then.
    const reach = h().clipping.setLanternReach.mock.calls.at(-1)?.[0] as { west: number } | null;
    expect(reach?.west).toBeLessThan(AREA.west);
    lantern.setMode("off");
    expect(h().clipping.setLanternReach).toHaveBeenLastCalledWith(null);
    expect(order.every((layer) => !layer.show)).toBe(true);
  });
});

describe("the wheel and the keys", () => {
  it("listens to the wheel only while the lens shows", () => {
    const { lantern } = h();
    lantern.setSources(SOURCES, AREA);
    expect(lantern.listensToWheel).toBe(false);
    lantern.setMode("sticky");
    expect(lantern.listensToWheel).toBe(true);
    lantern.setMode("off");
    expect(lantern.listensToWheel).toBe(false);
  });

  it("steps through time with the wheel over the map, and the camera never sees it", () => {
    const { lantern, canvas } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    const zoom = vi.fn();
    canvas.addEventListener("wheel", zoom);
    const wheel = new WheelEvent("wheel", {
      deltaY: 100,
      deltaMode: 0,
      bubbles: true,
      cancelable: true,
    });
    canvas.dispatchEvent(wheel);
    expect(wheel.defaultPrevented).toBe(true);
    expect(zoom).not.toHaveBeenCalled();
    expect(lantern.currentIndex).toBe(1);
    // Older is down, newer is up.
    vi.spyOn(performance, "now").mockReturnValue(performance.now() + 1000);
    canvas.dispatchEvent(
      new WheelEvent("wheel", { deltaY: -3, deltaMode: 1, bubbles: true, cancelable: true }),
    );
    expect(lantern.currentIndex).toBe(0);
    vi.restoreAllMocks();
  });

  it("is closed by Esc, which takes the window and the lens away", () => {
    const { lantern, clipping, worldShader, frames } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    window.dispatchEvent(escape);
    expect(escape.defaultPrevented).toBe(true);
    expect(lantern.currentMode).toBe("off");
    expect(clipping.setLanternWindow).toHaveBeenLastCalledWith(null);
    expect(worldShader.setLantern).toHaveBeenLastCalledWith(null);
    expect(frames.at(-1)?.mode).toBe("off");
  });

  it("leaves Escape to a sheet or popover open over the map", () => {
    const { lantern } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    const sheet = document.createElement("div");
    sheet.setAttribute("role", "dialog");
    document.body.appendChild(sheet);
    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    window.dispatchEvent(escape);
    // Not prevented: Radix closes the sheet on an Escape nobody else handled.
    expect(escape.defaultPrevented).toBe(false);
    expect(lantern.currentMode).toBe("sticky");
    sheet.remove();
    const popover = document.createElement("div");
    popover.setAttribute("data-hud-popover", "");
    document.body.appendChild(popover);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }));
    expect(lantern.currentMode).toBe("sticky");
    popover.remove();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }));
    expect(lantern.currentMode).toBe("off");
  });

  it("leaves Alt to a scan object's candidates while they are cycled", () => {
    const { lantern, canvas } = h();
    lantern.setSources(SOURCES, AREA);
    canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: 400, clientY: 300 }));
    useSceneSelect.setState({ index: 0, candidates: [4, 7] } as never);
    const alt = new KeyboardEvent("keydown", { key: "Alt", cancelable: true });
    window.dispatchEvent(alt);
    expect(alt.defaultPrevented).toBe(false);
    expect(lantern.currentMode).toBe("off");
    useSceneSelect.setState({ index: -1, candidates: [] } as never);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Alt", cancelable: true }));
    expect(lantern.currentMode).toBe("peek");
  });

  it("steps with [ and ] over the map while the lens shows", () => {
    const { lantern } = h();
    lantern.setSources(SOURCES, AREA);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "]", cancelable: true }));
    expect(lantern.currentIndex).toBe(0);
    lantern.setMode("sticky");
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "]", cancelable: true }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "]", cancelable: true }));
    expect(lantern.currentIndex).toBe(2);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "[", cancelable: true }));
    expect(lantern.currentIndex).toBe(1);
  });

  it("peeks while Alt is held over the map", () => {
    const { lantern, canvas } = h();
    lantern.setSources(SOURCES, AREA);
    canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: 400, clientY: 300 }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Alt", cancelable: true }));
    expect(lantern.currentMode).toBe("peek");
    window.dispatchEvent(new KeyboardEvent("keyup", { key: "Alt", cancelable: true }));
    expect(lantern.currentMode).toBe("off");
  });
});

describe("the lens", () => {
  it("waits for the terrain under the cursor before it cuts the world", () => {
    const { lantern, scene, frame, tilesLoaded, worldShader, clipping } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    frame();
    // No ground loaded under the cursor: nothing is cut, nothing is windowed.
    expect(worldShader.setLantern).not.toHaveBeenCalledWith(expect.objectContaining({ x: 500 }));
    expect(clipping.setLanternWindow).not.toHaveBeenCalledWith(expect.any(Array));
    scene.globe.getHeight = () => 300;
    tilesLoaded();
    frame();
    expect(worldShader.setLantern).toHaveBeenLastCalledWith(
      expect.objectContaining({ x: 500, y: 400 }),
    );
    expect(clipping.setLanternWindow).toHaveBeenLastCalledWith(expect.any(Array));
    // Nothing moved: the window is not swapped again.
    const swaps = clipping.setLanternWindow.mock.calls.length;
    tilesLoaded();
    frame();
    expect(clipping.setLanternWindow.mock.calls.length).toBe(swaps);
  });
});

describe("away from the area listed", () => {
  it("draws no lens, and says the cursor is outside the area", () => {
    const { lantern, scene, frame, worldShader, frames } = h();
    scene.globe.getHeight = () => 300;
    lantern.setSources(SOURCES, { west: -121.06, south: 34.68, east: -121.02, north: 34.72 });
    lantern.setMode("sticky");
    frame();
    expect(worldShader.setLantern).not.toHaveBeenCalledWith(expect.objectContaining({ x: 500 }));
    expect(frames.at(-1)).toMatchObject({ lens: null, outside: true });
    expect(frames.at(-1)?.ground?.longitude).toBeCloseTo(-120.04, 3);
    lantern.setSources(SOURCES, AREA);
    frame();
    expect(frames.at(-1)).toMatchObject({ outside: false });
    expect(frames.at(-1)?.lens).not.toBeNull();
  });
});

describe("what closes it", () => {
  it("is forced off by explore mode, and not offered while exploring", () => {
    const { lantern, events, frames } = h();
    lantern.setSources(SOURCES, AREA);
    lantern.setMode("sticky");
    events.emit("explore", true);
    expect(lantern.currentMode).toBe("off");
    expect(frames.at(-1)?.availability).toEqual({ ok: false, reason: "explore" });
    lantern.setMode("sticky");
    expect(lantern.currentMode).toBe("off");
  });

  it("is not offered from high up, or while the object brush paints", () => {
    const { lantern, events } = h();
    events.emit("camera", { altitude: 30_000 } as SceneEvents["camera"]);
    expect(lantern.available).toEqual({ ok: false, reason: "altitude" });
    events.emit("camera", { altitude: 900 } as SceneEvents["camera"]);
    useSceneSelect.setState({ mode: "paint" } as never);
    expect(lantern.available).toEqual({ ok: false, reason: "paint" });
  });
});
