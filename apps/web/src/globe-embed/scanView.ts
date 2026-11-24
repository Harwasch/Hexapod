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

/** A terrain outage must not make an otherwise valid scan unavailable. */
export async function seatScanWithRetry(
  tiles: Cesium3DTileset,
  terrain: TerrainProvider,
  sample = sampleTerrainMostDetailed,
): Promise<boolean> {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      await seatScan(tiles, terrain, sample);
      return true;
    } catch {
      // seatScan publishes its matrix only after a valid height is available.
    }
  }
  return false;
}

export function scanRange(radius: number): number {
  return Math.max(0.5, radius * 2.5);
}

/** Resolution is in CSS pixels because the embed explicitly enables browser recommended resolution. */
export class ScanResolution {
  scale: number;
  private readonly maximum: number;
  private slow = 0;
  private good = 0;
  private started: number | undefined;
  constructor(dpr: number) {
    this.maximum = Math.max(1, Math.min(3, dpr || 1));
    this.scale = this.maximum;
  }
  beginFrame(now: number): void {
    this.started = now;
  }
  endFrame(now: number): boolean {
    const started = this.started;
    this.started = undefined;
    return started !== undefined && this.frame(now - started);
  }
  /** Render duration only: never the idle interval between request-render frames. */
  frame(milliseconds: number): boolean {
    if (!Number.isFinite(milliseconds) || milliseconds < 0) return false;
    this.slow = milliseconds > 40 ? this.slow + 1 : 0;
    this.good = milliseconds < 20 ? this.good + 1 : 0;
    if (this.slow >= 60 && this.scale > 1) {
      this.scale = Math.max(1, this.scale - 0.5);
    } else if (this.good >= 180 && this.scale < this.maximum) {
      this.scale = Math.min(this.maximum, this.scale + 0.5);
    } else {
      return false;
    }
    this.slow = 0;
    this.good = 0;
    return true;
  }
}
