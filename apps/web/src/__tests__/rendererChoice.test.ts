/**
 * Choosing the splat renderer: CesiumJS unless chosen otherwise, and settings saved under the
 * PlayCanvas default moved to it once (docs/DECISIONS/0020-cesium-default-renderer.md). For
 * the WebGPU trial (docs/WEBGPU_TRIAL.md): `?renderer=` picks one for a visit over the saved
 * setting and never outlives it, choosing in the app ends it; the readouts name the engine and
 * API, say when the trial fell back and why, and give the latest gesture's frame rate; and the
 * reason PlayCanvas fell back to WebGL2 by itself.
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { whyNoWebgpu } from "@/cesium/scanView/playcanvasBackend";
import { RENDERER_NAMES, rendererReadout } from "@/lib/rendererReadout";
import {
  DEFAULT_SPLAT_RENDERER,
  migrateSettings,
  rendererFromQuery,
  SPLAT_RENDERERS,
  useRendererOverride,
  useSettings,
  useSplatRenderer,
} from "@/state/settings";

const STORE = "twin.settings.v1";

/** Settings as an older build stored them, then read back as on a page load. */
async function loadStored(state: Record<string, unknown>, version: number): Promise<void> {
  window.localStorage.setItem(STORE, JSON.stringify({ state, version }));
  await useSettings.persist.rehydrate();
}

describe("the default splat renderer", () => {
  beforeEach(() => window.localStorage.removeItem(STORE));
  afterEach(() => {
    useRendererOverride.setState({ renderer: null });
    useSettings.getState().reset();
    window.localStorage.removeItem(STORE);
  });

  it("is CesiumJS, on a fresh device and after a reset", async () => {
    expect(DEFAULT_SPLAT_RENDERER).toBe("cesium");
    useSettings.getState().set({ splatRenderer: "spark" });
    useSettings.getState().reset();
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    // Nothing stored: the default, and the store now writes version 4.
    window.localStorage.removeItem(STORE);
    await useSettings.persist.rehydrate();
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    useSettings.getState().set({ devReadouts: true });
    const stored = JSON.parse(window.localStorage.getItem(STORE) ?? "{}") as {
      state?: { splatRenderer?: string };
      version?: number;
    };
    expect(stored.version).toBe(4);
    expect(stored.state?.splatRenderer).toBe("cesium");
  });

  it("moves a v3 device on PlayCanvas, the old default, to CesiumJS", async () => {
    await loadStored({ splatRenderer: "playcanvas", quality: "ultra" }, 3);
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    // Nothing else is touched.
    expect(useSettings.getState().quality).toBe("ultra");
  });

  it("keeps a renderer that was only ever chosen: Spark, PlayCanvas on WebGPU, CesiumJS", async () => {
    for (const chosen of ["spark", "playcanvas-webgpu", "cesium"] as const) {
      await loadStored({ splatRenderer: chosen }, 3);
      expect(useSettings.getState().splatRenderer).toBe(chosen);
    }
  });

  it("keeps PlayCanvas chosen after the move: a v4 store is not migrated", async () => {
    await loadStored({ splatRenderer: "playcanvas" }, 4);
    expect(useSettings.getState().splatRenderer).toBe("playcanvas");
  });

  it("moves settings from before v3 (as the e2e fixtures seed them) to CesiumJS", async () => {
    await loadStored({ onboardingDismissed: false, quality: "performance" }, 1);
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    expect(useSettings.getState().world).toBe("photorealistic");
    // v3 overwrote any renderer stored before it, Spark included; that is unchanged.
    expect(migrateSettings({ splatRenderer: "spark" }, 2).splatRenderer).toBe("cesium");
  });

  it("leaves the page address's renderer winning over a migrated setting", async () => {
    useRendererOverride.setState({ renderer: "playcanvas" });
    await loadStored({ splatRenderer: "playcanvas" }, 3);
    const { result } = renderHook(() => useSplatRenderer());
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    expect(result.current).toBe("playcanvas");
    // And ?renderer= still reads the same: the default does not change what it names.
    expect(rendererFromQuery("?renderer=playcanvas")).toBe("playcanvas");
    expect(rendererFromQuery("?renderer=webgl")).toBe("playcanvas");
  });
});

describe("?renderer=", () => {
  afterEach(() => {
    useRendererOverride.setState({ renderer: null });
    useSettings.getState().reset();
  });

  it("names a renderer, or one of its short forms", () => {
    expect(rendererFromQuery("?renderer=playcanvas-webgpu")).toBe("playcanvas-webgpu");
    expect(rendererFromQuery("?renderer=WebGPU")).toBe("playcanvas-webgpu");
    expect(rendererFromQuery("?renderer=webgl")).toBe("playcanvas");
    expect(rendererFromQuery("?x=1&renderer=spark")).toBe("spark");
    expect(rendererFromQuery("?renderer=cesium")).toBe("cesium");
    for (const kind of SPLAT_RENDERERS) expect(rendererFromQuery(`?renderer=${kind}`)).toBe(kind);
  });

  it("is nothing when absent or not a renderer", () => {
    expect(rendererFromQuery("")).toBeNull();
    expect(rendererFromQuery("?renderer=")).toBeNull();
    expect(rendererFromQuery("?renderer=vulkan")).toBeNull();
    expect(rendererFromQuery("?stats")).toBeNull();
  });

  it("wins over the saved setting for the visit, which it leaves alone", () => {
    const { result } = renderHook(() => useSplatRenderer());
    expect(result.current).toBe(DEFAULT_SPLAT_RENDERER);
    act(() => useRendererOverride.setState({ renderer: "playcanvas-webgpu" }));
    expect(result.current).toBe("playcanvas-webgpu");
    expect(useSettings.getState().splatRenderer).toBe(DEFAULT_SPLAT_RENDERER);
  });

  it("ends when a renderer is chosen in the app, even the saved one", () => {
    const { result } = renderHook(() => useSplatRenderer());
    act(() => useRendererOverride.setState({ renderer: "playcanvas-webgpu" }));
    act(() => useSettings.getState().set({ splatRenderer: DEFAULT_SPLAT_RENDERER }));
    expect(useRendererOverride.getState().renderer).toBeNull();
    expect(result.current).toBe(DEFAULT_SPLAT_RENDERER);
    act(() => useRendererOverride.setState({ renderer: "spark" }));
    // Any other setting leaves it.
    act(() => useSettings.getState().set({ devReadouts: true }));
    expect(result.current).toBe("spark");
    act(() => useSettings.getState().reset());
    expect(result.current).toBe(DEFAULT_SPLAT_RENDERER);
  });
});

describe("the renderer readout", () => {
  const status = {
    kind: "playcanvas-webgpu" as const,
    active: true,
    api: "webgpu" as const,
    notice: null,
    webgl2ForObjects: false,
    meter: null,
  };

  it("names the choice until a renderer draws it", () => {
    expect(rendererReadout("playcanvas-webgpu", null)).toEqual({
      label: RENDERER_NAMES["playcanvas-webgpu"],
      meter: null,
      notice: null,
    });
    // Still the last renderer's status: the new one has not started.
    expect(rendererReadout("spark", status).label).toBe("Spark");
    expect(rendererReadout("cesium", null).label).toBe("CesiumJS");
  });

  it("names the engine and the API it draws with", () => {
    expect(rendererReadout("playcanvas-webgpu", status).label).toBe("PlayCanvas · WebGPU");
    expect(
      rendererReadout("playcanvas", { ...status, kind: "playcanvas", api: "webgl2" }).label,
    ).toBe("PlayCanvas · WebGL2");
    expect(rendererReadout("spark", { ...status, kind: "spark", api: "webgl2" }).label).toBe(
      "Spark · WebGL2",
    );
  });

  it("says when the trial fell back to WebGL2, and why", () => {
    const fellBack = rendererReadout("playcanvas-webgpu", {
      ...status,
      api: "webgl2",
      notice: "WebGPU device lost: GPU process restarted",
    });
    expect(fellBack.label).toBe("PlayCanvas · WebGL2 (WebGPU unavailable)");
    expect(fellBack.notice).toBe("WebGPU device lost: GPU process restarted");
  });

  it("says when the trial draws a scan with WebGL2 for its objects or motion", () => {
    const forObjects = rendererReadout("playcanvas-webgpu", {
      ...status,
      api: "webgl2",
      notice: "WebGL2 for scans with objects or motion",
      webgl2ForObjects: true,
    });
    // Not "WebGPU unavailable": WebGPU may well be there, the scan's modifiers are GLSL only.
    expect(forObjects.label).toBe("PlayCanvas · WebGL2 (objects or motion)");
    expect(forObjects.notice).toBe("WebGL2 for scans with objects or motion");
  });

  it("gives the latest gesture's rate, p95 and draw time, marked once it is over", () => {
    const meter = { fps: 58.4, p95Ms: 21.2, cpuMs: 2.34, frames: 60, live: true };
    expect(rendererReadout("playcanvas-webgpu", { ...status, meter }).meter).toBe(
      "58 fps · p95 21 ms · draw 2.3 ms",
    );
    expect(
      rendererReadout("playcanvas-webgpu", { ...status, meter: { ...meter, live: false } }).meter,
    ).toBe("last move 58 fps · p95 21 ms · draw 2.3 ms");
  });
});

describe("why PlayCanvas fell back to WebGL2 by itself", () => {
  it("tells no WebGPU from an insecure page from no adapter", () => {
    expect(whyNoWebgpu(undefined, true)).toBe("this browser has no WebGPU");
    expect(whyNoWebgpu(undefined, false)).toBe("WebGPU needs a secure (https) page");
    expect(whyNoWebgpu({}, true)).toBe("no WebGPU adapter or device");
  });
});
