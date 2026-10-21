/**
 * The splat views (state/viewMode.ts) and which renderer can draw them (design M4-6 C5).
 *
 * Photo is every renderer's. Confidence and Data age tint each splat by its cell with one
 * colour code (lib/cellLook.ts) in CesiumJS's splat primitive, PlayCanvas's work-buffer
 * modifier and Spark's world modifier -- CesiumJS picking the cell level per splat, the other
 * two per tile, which the key says. The WebGPU trial does not draw them yet (the modifier is
 * GLSL only): under it the control says so and offers the switch, rather than leaving a button
 * that silently does nothing. The per-cell view rules and Sunlight are CesiumJS's alone, and
 * the notice names them.
 */
import { useSettings, type SplatRenderer } from "@/state/settings";
import type { ViewMode } from "@/state/viewMode";

export const VIEW_MODES: readonly { value: ViewMode; label: string }[] = [
  { value: "photo", label: "Photo" },
  { value: "confidence", label: "Confidence" },
  { value: "data-age", label: "Data age" },
];

/** Whether `mode` can be drawn now, under `renderer`. */
export function viewModeDrawable(mode: ViewMode, renderer: SplatRenderer): boolean {
  return mode === "photo" || renderer !== "playcanvas-webgpu";
}

/**
 * Whether `renderer` draws Sunlight (ADR 0019): CesiumJS in its splat primitive, PlayCanvas in
 * its per-frame splat hook (cesium/scanView/sunHook.ts). Not Spark, whose per-frame modifier
 * would generate every splat again each moving frame, nor the WebGPU trial (GLSL only).
 */
export function sunlightDrawable(renderer: SplatRenderer): boolean {
  return renderer === "cesium" || renderer === "playcanvas";
}

/**
 * What the splat view control offers that `renderer` cannot draw, in words for its notice
 * (`perCell`: the Per cell rules are offered, the site having two dates or more); null under
 * CesiumJS, which draws all of it.
 */
export function needsCesiumFor(renderer: SplatRenderer, perCell: boolean): string | null {
  if (renderer === "cesium") return null;
  const parts = [
    ...(viewModeDrawable("confidence", renderer) ? [] : ["Confidence", "Data age"]),
    ...(perCell ? ["Per cell"] : []),
    "Sunlight",
  ];
  const last = parts.pop() ?? "";
  return parts.length > 0 ? `${parts.join(", ")} and ${last}` : last;
}

const RENDERER_NAMES: Partial<Record<SplatRenderer, string>> = {
  playcanvas: "PlayCanvas",
  spark: "Spark",
};

/**
 * What the key says of the cell size under `renderer`: PlayCanvas and Spark read one level of
 * the grid per tile (cesium/scanView/cellModifier.ts); null under CesiumJS, per splat.
 */
export function cellLevelNote(renderer: SplatRenderer): string | null {
  const name = RENDERER_NAMES[renderer];
  return name ? `Cell size picked per tile by ${name} (per splat by CesiumJS)` : null;
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
