import { SCAN_DETAIL_CONFIG } from "@/lib/detail";

const POLICY = SCAN_DETAIL_CONFIG.embed;
const STORAGE_KEY = "twin.embed.memoryReduced";
export interface HeapUsage {
  usedJSHeapSize: number;
  jsHeapSizeLimit: number;
}
export interface ScanCache {
  cacheBytes: number;
  maximumCacheOverflowBytes: number;
  maximumScreenSpaceError: number;
  trimLoadedTiles(): void;
}
/** Two sticky cache tiers. Repeated pressure is idempotent and cannot create a reload loop.
 * This is a Cesium cache target, not a bound on GPU/decoder/process memory.
 */
export class EmbedMemoryTier {
  constructor(private reduced = false) {}
  get options(): Pick<
    ScanCache,
    "cacheBytes" | "maximumCacheOverflowBytes" | "maximumScreenSpaceError"
  > {
    const factor = this.reduced ? POLICY.pressureCacheFactor : 1;
    return {
      cacheBytes: POLICY.cacheBytes * factor,
      maximumCacheOverflowBytes: POLICY.maximumCacheOverflowBytes * factor,
      maximumScreenSpaceError:
        POLICY.maximumScreenSpaceError * (this.reduced ? POLICY.pressureErrorFactor : 1),
    };
  }
  stepDown(): boolean {
    if (this.reduced) return false;
    this.reduced = true;
    return true;
  }
  apply(tiles: ScanCache): void {
    Object.assign(tiles, this.options);
    if (this.reduced) tiles.trimLoadedTiles();
  }
}
export function heapNearLimit(memory?: HeapUsage): boolean {
  return (
    !!memory &&
    memory.jsHeapSizeLimit > 0 &&
    memory.usedJSHeapSize / memory.jsHeapSizeLimit >= POLICY.heapLimitShare
  );
}
/** Persist context-loss recovery for this page session. WK process death is additionally
 * remembered natively, since the WebContent process may lose its sessionStorage.
 */
export function restoreMemoryTier(): EmbedMemoryTier {
  try {
    return new EmbedMemoryTier(sessionStorage.getItem(STORAGE_KEY) === "true");
  } catch {
    return new EmbedMemoryTier();
  }
}
export function rememberMemoryPressure(): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, "true");
  } catch {
    /* Native replay still works. */
  }
}
