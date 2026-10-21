/**
 * The splat views (state/viewMode.ts) and which renderer can draw them (design M4-6 C5).
 *
 * Photo is every renderer's. Confidence and Data age tint each splat by its cell with one
 * colour code (lib/cellLook.ts) in CesiumJS's splat primitive, PlayCanvas's work-buffer
 * modifier and Spark's world modifier -- CesiumJS picking the cell level per splat, the other
 * two per tile, which the key says. The WebGPU trial does not draw them yet (the modifier is
 * GLSL only): under it the control says so and offers the switch, rather than leaving a button
 * that silently does nothing. The per-cell view rules are CesiumJS's and PlayCanvas's
 * (`perCellDrawable`): under Spark and the WebGPU trial the notice names them. Sunlight is CesiumJS's and PlayCanvas's (`sunlightDrawable`): under Spark and the WebGPU
 * trial the Sun control says so itself, with the switch to PlayCanvas, the default renderer.
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
 * Whether `renderer` draws the per-cell view rules (ADR 0017): CesiumJS on its frame anchor's
 * splat primitive, PlayCanvas in the copy pass of its work buffer (cesium/scanView/scanRule.ts).
 * Not Spark, whose world modifier has no port of it, nor the WebGPU trial (GLSL only).
 */
export function perCellDrawable(renderer: SplatRenderer): boolean {
  return renderer === "cesium" || renderer === "playcanvas";
}

/**
 * What the splat view control offers that `renderer` cannot draw and CesiumJS can, in words for
 * its notice (`perCell`: the Per cell rules are offered, the site having two dates or more);
 * null when there is none -- under CesiumJS, which draws all of it, or nothing of it missing.
 * Sunlight is not named: PlayCanvas draws it too, and the Sun control says so where it is not
 * drawn (`NEEDS_SUNLIGHT_RENDERER`). Per cell is named only where it is not drawn
 * (`perCellDrawable`).
 */
export function needsCesiumFor(renderer: SplatRenderer, perCell: boolean): string | null {
  if (renderer === "cesium") return null;
  const parts = [
    ...(viewModeDrawable("confidence", renderer) ? [] : ["Confidence", "Data age"]),
    ...(perCell && !perCellDrawable(renderer) ? ["Per cell"] : []),
  ];
  const last = parts.pop();
  if (last === undefined) return null;
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
/** The Sun control's notice under a renderer that does not draw Sunlight. */
export const NEEDS_SUNLIGHT_RENDERER =
  "Needs the CesiumJS or PlayCanvas renderer (Settings › Advanced › Splat renderer)";
export const DRAW_WITH_PLAYCANVAS = "Draw with PlayCanvas";

/**
 * The notice's one-click switch: CesiumJS draws splats from now on. Choosing a renderer in the
 * app also ends a `?renderer=` choice for this visit (`useSettings.set`).
 */
export function drawWithCesium(): void {
  useSettings.getState().set({ splatRenderer: "cesium" });
}

/**
 * The Sun control's one-click switch: PlayCanvas on WebGL2 -- the default renderer, which draws
 * Sunlight -- from now on, ending a `?renderer=` choice for this visit as the other does.
 */
export function drawWithPlaycanvas(): void {
  useSettings.getState().set({ splatRenderer: "playcanvas" });
}
