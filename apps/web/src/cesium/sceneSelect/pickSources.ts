/**
 * Where scene selection (SceneSelectController) finds the splats of a scan: per asset id, the
 * renderer drawing it hands over the tiles it draws now (`PickTile`, lib/splatPick.ts) and
 * the scan's frame. CesiumJS's own splats register at a low priority (cesiumPickSource.ts) and
 * a dedicated renderer (scanView/ScanRendererHost.ts) above it, so whichever draws the scan
 * is the one picked from.
 */

import type { Matrix4 } from "cesium";

import type { PickTile } from "@/lib/splatPick";

export interface PickSource {
  /** Which renderer: for diagnostics. */
  readonly renderer: string;
  /** The tiles drawn now; empty while the renderer draws nothing of the scan. */
  tiles(): readonly PickTile[];
  /** The scan's frame (tileset local east/north/up metres) to Earth-fixed, when known. */
  toWorld(): Matrix4 | undefined;
}

/** Dedicated renderers outrank CesiumJS's own splats. */
export const CESIUM_PRIORITY = 0;
export const DEDICATED_PRIORITY = 1;

const SOURCES = new Map<string, { source: PickSource; priority: number }[]>();

/** Registers `source` for `assetId`'s scan. Returns the disposer. */
export function registerPickSource(
  assetId: string,
  source: PickSource,
  priority: number,
): () => void {
  const entry = { source, priority };
  const list = SOURCES.get(assetId) ?? [];
  list.push(entry);
  list.sort((a, b) => b.priority - a.priority);
  SOURCES.set(assetId, list);
  return () => {
    const now = SOURCES.get(assetId);
    if (!now) return;
    const at = now.indexOf(entry);
    if (at >= 0) now.splice(at, 1);
    if (now.length === 0) SOURCES.delete(assetId);
  };
}

/** The source of the renderer drawing `assetId`'s scan: the highest priority one with tiles. */
export function pickSourceOf(assetId: string): PickSource | undefined {
  const list = SOURCES.get(assetId);
  if (!list) return undefined;
  for (const { source } of list) if (source.tiles().length > 0) return source;
  return list[0]?.source;
}

/** Every scan with a source. */
export function pickAssets(): string[] {
  return [...SOURCES.keys()];
}
