/**
 * The globe's memory budget for gaussian-splat tilesets, counted in splats rather than bytes.
 *
 * CesiumJS 1.145 cannot budget splat tilesets by bytes: `GaussianSplat3DTileContent` reports
 * `geometryByteLength` 0, and its `texturesByteLength` is a share of the one aggregate
 * texture spread over the *selected* tiles (Source/Scene/GaussianSplat3DTileContent.js). What
 * a loaded splat tile really holds -- its decoded attributes and the root-frame copies of
 * positions, rotations and scales it keeps for rebuilds (`_positions`, `_rotations`,
 * `_scales`) -- is not counted at all, so `cacheBytes` never trims a splat tileset and
 * `totalMemoryUsageInBytes` never shows the pressure. Every tile the packer writes says how
 * many gaussians it holds (`extras.gaussians`, tools/captures/splat_tiles.py), and a splat
 * tile of any origin says it again in `content.pointsLength`, so the count is exact.
 *
 * The budget is the phone's Detail choice (lib/detail.ts): the gaussians a view may hold at
 * once. SiteManager reports the count as a memory source, so the PerformanceManager holds
 * idle refinement at 70% of it and coarsens past 125% (its REFINE_MEMORY_RATIO and
 * MEMORY_PRESSURE_RATIO); and past 125% SiteManager first trims the tiles the last frame did
 * not select -- the trim Cesium's own cache would have made had it seen the bytes.
 */

/**
 * Bytes one loaded splat costs, for reporting the count in the HUD's memory terms. From the
 * Cesium source: the glTF attributes (position 12, rotation 16, scale 12, colour 4) plus the
 * root-frame copies of position, rotation and scale (40), plus its share of the aggregate
 * texture and sort buffers when selected (about 16 more). The budget never depends on it.
 */
export const SPLAT_BYTES_ESTIMATE = 100;

/** The parts of a `Cesium3DTile` this reads. */
export interface SplatTileLike {
  extras?: unknown;
  content?: { pointsLength?: number } | undefined;
}

/** How many gaussians a tile's content holds: the packer's count, or Cesium's own. */
export function tileGaussians(tile: SplatTileLike): number {
  const counted = (tile.extras as { gaussians?: unknown } | undefined)?.gaussians;
  if (typeof counted === "number" && Number.isFinite(counted) && counted >= 0) return counted;
  const points = tile.content?.pointsLength;
  return typeof points === "number" && Number.isFinite(points) ? points : 0;
}

/**
 * The gaussians loaded in one splat tileset, kept from its `tileLoad` and `tileUnload`
 * events. Each tile is counted once however often it is reported, and what is subtracted on
 * unload is what was added on load, so a tile whose content is already gone by then cannot
 * make the count drift.
 */
export class SplatCount {
  private readonly counted = new WeakMap<object, number>();
  total = 0;

  load(tile: SplatTileLike): void {
    if (this.counted.has(tile)) return;
    const gaussians = tileGaussians(tile);
    this.counted.set(tile, gaussians);
    this.total += gaussians;
  }

  unload(tile: SplatTileLike): void {
    const gaussians = this.counted.get(tile);
    if (gaussians === undefined) return;
    this.counted.delete(tile);
    this.total -= gaussians;
  }
}

/** The splats loaded against the Detail budget, in the memory source's terms. */
export function splatMemory(loaded: number, budget: number): { bytes: number; budget: number } {
  return { bytes: loaded * SPLAT_BYTES_ESTIMATE, budget: budget * SPLAT_BYTES_ESTIMATE };
}
