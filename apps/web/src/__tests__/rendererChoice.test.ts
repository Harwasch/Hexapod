/**
 * Choosing the splat renderer for the WebGPU trial (docs/WEBGPU_TRIAL.md): `?renderer=` picks
 * one for a visit over the saved setting and never outlives it, choosing in the app ends it;
 * the readouts name the engine and API, say when the trial fell back and why, and give the
 * latest gesture's frame rate; and the reason PlayCanvas fell back to WebGL2 by itself.
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { whyNoWebgpu } from "@/cesium/scanView/playcanvasBackend";
import { RENDERER_NAMES, rendererReadout } from "@/lib/rendererReadout";
import {
  DEFAULT_SPLAT_RENDERER,
  rendererFromQuery,
  SPLAT_RENDERERS,
  useRendererOverride,
  useSettings,
  useSplatRenderer,
} from "@/state/settings";

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
