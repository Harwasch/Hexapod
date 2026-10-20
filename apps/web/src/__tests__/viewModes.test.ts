/**
 * The splat views and the renderers that can draw them (lib/viewModes.ts, state/viewMode.ts):
 * Photo everywhere, the analytic views on CesiumJS only, and a switch that really switches.
 */
import { afterEach, describe, expect, it } from "vitest";

import {
  drawWithCesium,
  NEEDS_CESIUM_NOTICE,
  needsCesium,
  VIEW_MODES,
  viewModeDrawable,
} from "@/lib/viewModes";
import { SPLAT_RENDERERS, useRendererOverride, useSettings } from "@/state/settings";
import { useViewMode } from "@/state/viewMode";

afterEach(() => {
  useViewMode.setState({ mode: "photo", cells: {} });
  useSettings.getState().reset();
});

describe("which renderer draws a view", () => {
  it("is any for Photo and CesiumJS alone for Confidence and Data age", () => {
    expect(VIEW_MODES.map((m) => m.value)).toEqual(["photo", "confidence", "data-age"]);
    expect(needsCesium("photo")).toBe(false);
    expect(needsCesium("confidence")).toBe(true);
    expect(needsCesium("data-age")).toBe(true);
    for (const renderer of SPLAT_RENDERERS) {
      expect(viewModeDrawable("photo", renderer)).toBe(true);
      expect(viewModeDrawable("confidence", renderer)).toBe(renderer === "cesium");
      expect(viewModeDrawable("data-age", renderer)).toBe(renderer === "cesium");
    }
  });

  it("names where the renderer is chosen", () => {
    expect(NEEDS_CESIUM_NOTICE).toBe(
      "Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
    );
  });

  it("switches to CesiumJS in one click, ending a ?renderer= choice for the visit", () => {
    useSettings.getState().set({ splatRenderer: "spark" });
    useRendererOverride.setState({ renderer: "playcanvas" });
    drawWithCesium();
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    expect(useRendererOverride.getState().renderer).toBeNull();
  });
});

describe("the view store", () => {
  it("starts on Photo, and keeps what the scene reports per model until it is cleared", () => {
    expect(useViewMode.getState().mode).toBe("photo");
    useViewMode.getState().set("confidence");
    expect(useViewMode.getState().mode).toBe("confidence");
    useViewMode.getState().report("a1", { kind: "loading" });
    useViewMode.getState().report("a2", { kind: "not-measured" });
    useViewMode.getState().report("a1", { kind: "ready", mode: "confidence" });
    expect(useViewMode.getState().cells).toEqual({
      a1: { kind: "ready", mode: "confidence" },
      a2: { kind: "not-measured" },
    });
    useViewMode.getState().report("a2", null);
    expect(Object.keys(useViewMode.getState().cells)).toEqual(["a1"]);
  });
});
