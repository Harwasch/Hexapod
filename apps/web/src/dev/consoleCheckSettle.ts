/** A stable draw, not just a momentarily idle sort while the next LOD mapping is on its way. */
export interface CheckProgress {
  stage: "loading-source" | "loading-tiles" | "settling";
  tiles: string[];
  tileGaussians: number;
  drawnSplats: number;
  activeSplats: number;
  sortSettled: boolean;
  lodSettled: boolean;
  version: number;
  mappingVersion: number;
}

export class ConsoleCheckSettle {
  private key = "";
  private since = 0;
  private samples = 0;
  private wasReady = false;
  constructor(private readonly stableMs = 1000) {}

  observe(now: number, progress: CheckProgress): boolean {
    const ready =
      progress.stage === "settling" &&
      progress.drawnSplats > 0 &&
      progress.sortSettled &&
      progress.lodSettled;
    const key = JSON.stringify([
      progress.tiles,
      progress.tileGaussians,
      progress.drawnSplats,
      progress.activeSplats,
      progress.version,
      progress.mappingVersion,
    ]);
    if (!ready || !this.wasReady || key !== this.key) {
      this.key = key;
      this.since = now;
      this.samples = 0;
    }
    this.wasReady = ready;
    if (!ready) return false;
    this.samples++;
    return this.samples >= 2 && now - this.since >= this.stableMs;
  }
}
