/**
 * Where a splat model's Confidence / Data age view hooks into the scene (SiteManager
 * `attachTileset`): a few lines on the globe's first load, the view itself (`cellView.ts`, its
 * shader part, the grid decoder) loaded on demand the first time a view other than Photo is
 * chosen. On Photo nothing is fetched and nothing joins the shader.
 */

import type { Cesium3DTileset, Scene } from "cesium";

import type { DataAgeContext } from "@/lib/dataAge";
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
        handle.sync(useViewMode.getState().mode);
      })
      .catch((error: unknown) => {
        loading = false;
        const message = error instanceof Error ? error.message : String(error);
        if (!disposed) useViewMode.getState().report(assetId, { kind: "failed", message });
      });
  };
  const unsubscribe = useViewMode.subscribe((s, previous) => {
    if (s.mode !== previous.mode) apply(s.mode);
  });
  // Data age compares with the later dates: a new one counts once it is published.
  const unsubscribeDates = options.onDatesChange?.(() => handle?.sync(useViewMode.getState().mode));
  apply(useViewMode.getState().mode);
  return () => {
    disposed = true;
    unsubscribe();
    unsubscribeDates?.();
    handle?.dispose();
    handle = undefined;
    // A load that failed before there was a handle reported too; the model is gone either way.
    useViewMode.getState().report(assetId, null);
  };
}
