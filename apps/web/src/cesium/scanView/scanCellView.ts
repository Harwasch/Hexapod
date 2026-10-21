/**
 * Where a scan's Confidence / Data age view (cesium/cellView.ts) meets the dedicated renderer
 * that draws the scan (PlayCanvas or Spark, scanView/cellModifier.ts): per asset id, the view
 * to draw now -- its mode, texture and grid (`CellLookView`) -- or none, which is the photo.
 * The cell view puts it here as it changes; a renderer drawing that asset watches it.
 *
 * Both sides are loaded on demand (the cell view on the first analytic view, a renderer when
 * chosen), and so is this: nothing of it is on the globe's first load.
 *
 * Each renderer also says here what it drew -- the tiles tinted, the level steps that had a
 * tile copied again, and how many sampled splats read the level CesiumJS would have given
 * them (`ScanCellViewReading`) -- for the developer readouts and the tests.
 */

import type { CellLookView } from "@/lib/cellLook";

type Listener = (view: CellLookView | null) => void;

const views = new Map<string, CellLookView>();
const listeners = new Map<string, Set<Listener>>();

/** Puts asset `assetId`'s view (null: the photo) where its renderer reads it. */
export function publishScanCellView(assetId: string, view: CellLookView | null): void {
  const before = views.get(assetId) ?? null;
  if (
    before === view ||
    (before &&
      view &&
      before.mode === view.mode &&
      before.rows === view.rows &&
      before.meta === view.meta)
  ) {
    return;
  }
  if (view) views.set(assetId, view);
  else views.delete(assetId);
  for (const listener of listeners.get(assetId) ?? []) listener(view);
}

/** Calls `listener` with asset `assetId`'s view now and on every change; returns the remover. */
export function watchScanCellView(assetId: string, listener: Listener): () => void {
  let set = listeners.get(assetId);
  if (!set) listeners.set(assetId, (set = new Set()));
  set.add(listener);
  listener(views.get(assetId) ?? null);
  return () => {
    set.delete(listener);
    if (set.size === 0) listeners.delete(assetId);
  };
}

/** What a dedicated renderer drew of a cell view, last time it was asked. */
export interface ScanCellViewReading {
  renderer: string;
  mode: CellLookView["mode"] | null;
  /** Tiles drawn with the view now. */
  tiles: number;
  /** Times a tile's level stepped and the tile was copied or generated again. */
  steps: number;
  /** Splats sampled from the tiles drawn, and those reading the level CesiumJS would. */
  sampled: number;
  agree: number;
}

const readings = new Map<string, () => ScanCellViewReading>();

/** Registers how asset `assetId`'s renderer reads out its cell view; returns the remover. */
export function reportScanCellView(assetId: string, read: () => ScanCellViewReading): () => void {
  readings.set(assetId, read);
  return () => {
    if (readings.get(assetId) === read) readings.delete(assetId);
  };
}

/** What the renderer drawing asset `assetId` says of its cell view, or null when none does. */
export function scanCellViewReading(assetId: string): ScanCellViewReading | null {
  return readings.get(assetId)?.() ?? null;
}
