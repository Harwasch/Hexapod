import {
  Cartographic,
  EllipsoidTerrainProvider,
  sampleTerrainMostDetailed,
  type Cesium3DTileset,
  type TerrainProvider,
} from "cesium";
import { placedMatrix, placementFrame, lowestHeight } from "@/cesium/tilesetScale";
import { plausibleGround } from "@/cesium/placement";

/** Without catalog ground cells, use the main globe's root-box ground fallback. */
export async function seatScan(
  tiles: Cesium3DTileset,
  terrain: TerrainProvider,
  sample = sampleTerrainMostDetailed,
): Promise<void> {
  const bottom = lowestHeight(tiles);
  const frame = placementFrame(tiles, bottom);
  const point = Cartographic.fromCartesian(frame.origin);
  // Offline the visible ground really is the ellipsoid. A failed online sample is not zero.
  const ground =
    terrain instanceof EllipsoidTerrainProvider && sample === sampleTerrainMostDetailed
      ? 0
      : plausibleGround((await sample(terrain, [point]))[0]?.height);
  if (ground === undefined) throw Error("Terrain height unavailable");
  tiles.modelMatrix = placedMatrix(frame, 1, ground - (bottom ?? frame.ground.height));
}

export function scanRange(radius: number): number {
  return Math.max(0.5, radius * 2.5);
}

/** Resolution is in CSS pixels because the embed explicitly enables browser recommended resolution. */
export class ScanResolution {
  scale: number;
  private slow = 0;
  constructor(dpr: number) {
    this.scale = Math.max(1, Math.min(3, dpr || 1));
  }
  frame(milliseconds: number): boolean {
    // requestRenderMode can leave arbitrarily long gaps between rendered frames.
    if (milliseconds >= 250) {
      this.slow = 0;
      return false;
    }
    this.slow = milliseconds > 40 ? this.slow + 1 : 0;
    if (this.slow < 60 || this.scale <= 1) return false;
    this.scale = Math.max(1, this.scale - 0.5);
    this.slow = 0;
    return true;
  }
}
