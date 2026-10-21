/**
 * The splat views and the renderers that can draw them (lib/viewModes.ts, state/viewMode.ts):
 * Photo everywhere, the analytic views everywhere but the WebGPU trial, what still needs
 * CesiumJS named, and a switch that really switches.
 */
import { afterEach, describe, expect, it } from "vitest";

import {
  cellLevelNote,
  drawWithCesium,
  NEEDS_CESIUM_NOTICE,
  needsCesiumFor,
  perCellDrawable,
  sunlightDrawable,
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
  it("is any for Photo, and any but the WebGPU trial for Confidence and Data age", () => {
    expect(VIEW_MODES.map((m) => m.value)).toEqual(["photo", "confidence", "data-age"]);
    for (const renderer of SPLAT_RENDERERS) {
      expect(viewModeDrawable("photo", renderer)).toBe(true);
      expect(viewModeDrawable("confidence", renderer)).toBe(renderer !== "playcanvas-webgpu");
      expect(viewModeDrawable("data-age", renderer)).toBe(renderer !== "playcanvas-webgpu");
    }
  });

  it("is CesiumJS or PlayCanvas for Sunlight: not Spark, not the WebGPU trial", () => {
    expect(SPLAT_RENDERERS.filter((r) => sunlightDrawable(r))).toEqual(["playcanvas", "cesium"]);
  });

  it("is CesiumJS or PlayCanvas for the per-cell rules: not Spark, not the WebGPU trial", () => {
    expect(SPLAT_RENDERERS.filter((r) => perCellDrawable(r))).toEqual(["playcanvas", "cesium"]);
  });

  it("names what still needs CesiumJS under each renderer, and nothing under CesiumJS", () => {
    expect(needsCesiumFor("cesium", true)).toBeNull();
    // Sunlight is not CesiumJS's alone: the Sun control says what draws it (SunControl). Nor
    // is Per cell: PlayCanvas draws the rules too (scanView/scanRule.ts).
    expect(needsCesiumFor("playcanvas", true)).toBeNull();
    expect(needsCesiumFor("spark", true)).toBe("Per cell");
    expect(needsCesiumFor("spark", false)).toBeNull();
    expect(needsCesiumFor("playcanvas", false)).toBeNull();
    expect(needsCesiumFor("playcanvas-webgpu", true)).toBe("Confidence, Data age and Per cell");
    expect(needsCesiumFor("playcanvas-webgpu", false)).toBe("Confidence and Data age");
  });

  it("says when cells are sized per tile rather than per splat", () => {
    expect(cellLevelNote("cesium")).toBeNull();
    expect(cellLevelNote("playcanvas")).toBe(
      "Cell size picked per tile by PlayCanvas (per splat by CesiumJS)",
    );
    expect(cellLevelNote("spark")).toBe(
      "Cell size picked per tile by Spark (per splat by CesiumJS)",
    );
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
