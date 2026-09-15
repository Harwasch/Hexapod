import {
  Cartesian3,
  ClippingPolygon,
  ClippingPolygonCollection,
  type Cesium3DTileset,
  type Scene,
} from "cesium";

import type { Footprint } from "@twin/contracts";
import { flattenRing, polygonsOf } from "@twin/geo";

import { createLogger } from "@/lib/log";

const log = createLogger("clipping");

/**
 * Cuts holes in the coarse world (globe/terrain and the global 3D tileset)
 * where a high-resolution reality model takes over, so both never fight.
 *
 * Polygons are immutable in CesiumJS ≥ 1.145, so each footprint is tracked by
 * key and swapped, never mutated.
 */
export class ClippingManager {
  readonly supported: boolean;
  private readonly globeCollection: ClippingPolygonCollection | null = null;
  private worldCollection: ClippingPolygonCollection | null = null;
  private worldTileset: Cesium3DTileset | null = null;
  private readonly footprints = new Map<string, Footprint>();
  private readonly globePolygons = new Map<string, ClippingPolygon[]>();
  private readonly worldPolygons = new Map<string, ClippingPolygon[]>();
  private enabled = true;

  constructor(private readonly scene: Scene) {
    this.supported = ClippingPolygonCollection.isSupported(scene);
    if (this.supported) {
      this.globeCollection = new ClippingPolygonCollection({ polygons: [], enabled: false });
      scene.globe.clippingPolygons = this.globeCollection;
    } else {
      log.warn("clipping polygons unsupported (WebGL 1); coarse geometry will not be cut");
    }
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  /** Master switch (developer panel). */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.syncEnabled();
  }

  /** Registers (or replaces) the footprint that should clip the world for a key (site id). */
  setFootprint(key: string, footprint: Footprint | null): void {
    if (!this.supported) return;
    this.removeFrom(this.globeCollection, this.globePolygons, key);
    this.removeFrom(this.worldCollection, this.worldPolygons, key);
    this.footprints.delete(key);
    if (!footprint) {
      this.syncEnabled();
      return;
    }
    this.footprints.set(key, footprint);
    this.addTo(this.globeCollection, this.globePolygons, key, footprint);
    this.addTo(this.worldCollection, this.worldPolygons, key, footprint);
    this.syncEnabled();
  }

  /** Attaches the global photorealistic tileset so it receives the same holes. */
  setWorldTileset(tileset: Cesium3DTileset | null): void {
    if (!this.supported) return;
    if (this.worldTileset === tileset) return;
    this.worldPolygons.clear();
    this.worldTileset = tileset;
    this.worldCollection = tileset
      ? new ClippingPolygonCollection({ polygons: [], enabled: false })
      : null;
    if (tileset && this.worldCollection) {
      tileset.clippingPolygons = this.worldCollection;
      for (const [key, footprint] of this.footprints)
        this.addTo(this.worldCollection, this.worldPolygons, key, footprint);
    }
    this.syncEnabled();
  }

  get activeKeys(): string[] {
    return Array.from(this.footprints.keys());
  }

  private addTo(
    collection: ClippingPolygonCollection | null,
    registry: Map<string, ClippingPolygon[]>,
    key: string,
    footprint: Footprint,
  ): void {
    if (!collection) return;
    const created: ClippingPolygon[] = [];
    for (const polygon of polygonsOf(footprint)) {
      const [outer, ...holes] = polygon;
      if (!outer || outer.length < 4) continue;
      try {
        created.push(
          collection.add(
            new ClippingPolygon({
              positions: Cartesian3.fromDegreesArray(flattenRing(outer)),
              holes: holes
                .filter((h) => h.length >= 4)
                .map((h) => Cartesian3.fromDegreesArray(flattenRing(h))),
            }),
          ),
        );
      } catch (error) {
        log.error("failed to build clipping polygon", { key, error: String(error) });
      }
    }
    registry.set(key, created);
  }

  private removeFrom(
    collection: ClippingPolygonCollection | null,
    registry: Map<string, ClippingPolygon[]>,
    key: string,
  ): void {
    const existing = registry.get(key);
    if (!existing || !collection) return;
    for (const polygon of existing) collection.remove(polygon);
    registry.delete(key);
  }

  private syncEnabled(): void {
    const any = this.footprints.size > 0 && this.enabled;
    if (this.globeCollection) this.globeCollection.enabled = any && this.globeCollection.length > 0;
    if (this.worldCollection) this.worldCollection.enabled = any && this.worldCollection.length > 0;
    this.scene.requestRender();
  }

  destroy(): void {
    this.footprints.clear();
    this.globePolygons.clear();
    this.worldPolygons.clear();
  }
}
