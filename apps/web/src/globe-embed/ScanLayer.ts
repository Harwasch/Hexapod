import type { Cesium3DTileset } from "cesium";

/** Own one visible scan and retire late loads after selection changes or disposal. */
export class ScanLayer {
  private generation = 0;
  private current: Cesium3DTileset | null = null;
  get visible(): Cesium3DTileset | null {
    return this.current;
  }
  constructor(
    private readonly load: (url: string) => Promise<Cesium3DTileset>,
    private readonly add: (tiles: Cesium3DTileset) => void,
    private readonly remove: (tiles: Cesium3DTileset) => void,
    private readonly changed: () => void,
    private readonly failed: () => void,
  ) {}
  async select(url: string | null): Promise<void> {
    const generation = ++this.generation;
    if (this.current) this.remove(this.current);
    this.current = null;
    this.changed();
    if (!url) return;
    try {
      const tiles = await this.load(url);
      if (generation !== this.generation) {
        tiles.destroy();
        return;
      }
      this.current = tiles;
      this.add(tiles);
      this.changed();
    } catch {
      if (generation === this.generation) this.failed();
    }
  }
  dispose(): void {
    ++this.generation;
    if (this.current) this.remove(this.current);
    this.current = null;
  }
}
