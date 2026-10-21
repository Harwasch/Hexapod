/**
 * Where a splat model's Confidence / Data age view hooks into the scene (SiteManager
 * `attachTileset`): a few lines on the globe's first load, the view itself (`cellView.ts`, its
 * shader part, the grid decoder) loaded on demand the first time a view other than Photo is
 * chosen. On Photo nothing is fetched and nothing joins the shader.
 */

import type { Cesium3DTileset, Scene } from "cesium";

import type { DataAgeContext } from "@/lib/dataAge";
import { useRendererOverride, useSettings } from "@/state/settings";
import { useViewMode, type ViewMode } from "@/state/viewMode";

import type { CellViewHandle, startCellView } from "./cellView";

export function attachCellView(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "requestRender">,
  assetId: string,
  context: () => DataAgeContext,
  options: {
    /**
     * Calls back when the site's dates change while the model stays loaded (a date published
     * or run again: `SiteManager.applyDatesRecord`); returns the unsubscriber.
     */
    onDatesChange?: (listener: () => void) => () => void;
    load?: () => Promise<{ startCellView: typeof startCellView }>;
  } = {},
): () => void {
  const load = options.load ?? (() => import("./cellView"));
  let handle: CellViewHandle | undefined;
  let loading = false;
  let disposed = false;
  /**
   * The view drawn: the one chosen while CesiumJS is the renderer, else Photo, as the control
   * says (ViewsControl). A rigged model is drawn by CesiumJS under any renderer, so the
   * renderer, not the model's tileset being shown, decides.
   */
  const drawn = (): ViewMode =>
    (useRendererOverride.getState().renderer ?? useSettings.getState().splatRenderer) === "cesium"
      ? useViewMode.getState().mode
      : "photo";
  let last = drawn();
  const apply = (mode: ViewMode): void => {
    if (handle) {
      handle.sync(mode);
      return;
    }
    if (mode === "photo" || loading) return;
    loading = true;
    load()
      .then(({ startCellView }) => {
        if (disposed) return;
        handle = startCellView(tileset, scene, assetId, context);
        handle.sync(drawn());
      })
      .catch((error: unknown) => {
        loading = false;
        const message = error instanceof Error ? error.message : String(error);
        if (!disposed) useViewMode.getState().report(assetId, { kind: "failed", message });
      });
  };
  const changed = (): void => {
    const next = drawn();
    if (next !== last) apply((last = next));
  };
  const unsubscribe = [
    useViewMode.subscribe(changed),
    useSettings.subscribe(changed),
    useRendererOverride.subscribe(changed),
  ];
  // Data age compares with the later dates: a new one counts once it is published.
  const unsubscribeDates = options.onDatesChange?.(() => handle?.sync(drawn()));
  apply(last);
  return () => {
    disposed = true;
    for (const off of unsubscribe) off();
    unsubscribeDates?.();
    handle?.dispose();
    handle = undefined;
    // A load that failed before there was a handle reported too; the model is gone either way.
    useViewMode.getState().report(assetId, null);
  };
}
