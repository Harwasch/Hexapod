/**
 * The splat views (state/viewMode.ts) and which renderer can draw them (design M4-6 C5).
 *
 * Photo is every renderer's. Confidence and Data age tint each splat by its cell in the
 * vertex shader of the CesiumJS splat primitive (cesium/splatCellView.ts), which PlayCanvas
 * and Spark do not have: under them the control says so and offers the switch, rather than
 * leaving a button that silently does nothing.
 */
import { useSettings, type SplatRenderer } from "@/state/settings";
import type { ViewMode } from "@/state/viewMode";

export const VIEW_MODES: readonly { value: ViewMode; label: string }[] = [
  { value: "photo", label: "Photo" },
  { value: "confidence", label: "Confidence" },
  { value: "data-age", label: "Data age" },
];

/** Whether `mode` is drawn by the CesiumJS splat primitive only. */
export function needsCesium(mode: ViewMode): boolean {
  return mode !== "photo";
}

/** Whether `mode` can be drawn now, under `renderer`. */
export function viewModeDrawable(mode: ViewMode, renderer: SplatRenderer): boolean {
  return !needsCesium(mode) || renderer === "cesium";
}

/** Where the renderer is chosen, as Settings names it (SettingsSheet `AdvancedSection`). */
export const NEEDS_CESIUM_NOTICE =
  "Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)";
export const DRAW_WITH_CESIUM = "Draw with CesiumJS";

/**
 * The notice's one-click switch: CesiumJS draws splats from now on. Choosing a renderer in the
 * app also ends a `?renderer=` choice for this visit (`useSettings.set`).
 */
export function drawWithCesium(): void {
  useSettings.getState().set({ splatRenderer: "cesium" });
}
