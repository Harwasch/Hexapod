import {
  Cartesian3,
  Rectangle,
  ClippingPolygon,
  ClippingPolygonCollection,
  type Cesium3DTileset,
  type Scene,
} from "cesium";

import type { Footprint } from "@twin/contracts";
import { flattenRing, polygonsOf } from "@twin/geo";

import { createLogger } from "@/lib/log";

const log = createLogger("clipping");

export interface ClipTargets {
  globe: boolean;
  world: boolean;
}

const ALL: ClipTargets = { globe: true, world: true };
/** About 200 m around the engaged outlines, so the terrain under an edge is loaded before it shows. */
const GLOBE_LIMIT_MARGIN_DEG = 0.002;
/** A few metres of Antarctica nobody looks at, so an inverse clip always has a polygon. */
const SENTINEL_KEY = "__sentinel";
const SENTINEL: Footprint = {
  type: "Polygon",
  coordinates: [
    [
      [0, -89.99],
      [0.001, -89.99],
      [0.0005, -89.989],
      [0, -89.99],
    ],
  ],
};

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
  private readonly footprints = new Map<string, { footprint: Footprint; targets: ClipTargets }>();
  private readonly globePolygons = new Map<string, ClippingPolygon[]>();
  private readonly worldPolygons = new Map<string, ClippingPolygon[]>();
  private enabled = true;
  private photorealistic = false;
  private floorless: string | null = null;

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

  /**
   * Open world: the globe carries terrain and imagery everywhere and gets holes under mesh
   * sites. Photorealistic world: the Google mesh is the ground, so the globe is hidden,
   * except that it is kept (inverse clip) inside every site footprint: splats and point
   * clouds need an opaque floor under their sparse patches, and a mesh never fills its
   * footprint exactly, so without terrain the gap between mesh and footprint shows sky.
   */
  setWorldMode(photorealistic: boolean): void {
    if (this.photorealistic === photorealistic) return;
    this.photorealistic = photorealistic;
    if (!this.supported || !this.globeCollection) {
      this.scene.globe.show = !photorealistic;
      return;
    }
    for (const key of Array.from(this.globePolygons.keys()))
      this.removeFrom(this.globeCollection, this.globePolygons, key);
    this.globeCollection.inverse = photorealistic;
    for (const [key, entry] of this.footprints)
      if (this.globeCarries(key, entry.targets))
        this.addTo(this.globeCollection, this.globePolygons, key, entry.footprint);
    // In the photorealistic world the globe is never hidden, only clipped away: a hidden
    // globe stops loading, and the terrain and imagery inside a site's outline would then
    // start from level 0 the moment the site engages, showing a dark band for seconds while
    // they compete with the world's requests. A sentinel polygon at the pole keeps the
    // inverse clip active (and every tile culled) when no site is engaged.
    if (photorealistic)
      this.addTo(this.globeCollection, this.globePolygons, SENTINEL_KEY, SENTINEL);
    this.syncEnabled();
  }

  /** Bounding rectangle of every footprint the globe carries, with a margin; the sentinel alone otherwise. */
  private globeLimit(): Rectangle {
    let west = Number.POSITIVE_INFINITY;
    let south = Number.POSITIVE_INFINITY;
    let east = Number.NEGATIVE_INFINITY;
    let north = Number.NEGATIVE_INFINITY;
    for (const [key, entry] of this.footprints) {
      if (!this.globeCarries(key, entry.targets)) continue;
      for (const polygon of polygonsOf(entry.footprint))
        for (const [lon, lat] of polygon[0] ?? []) {
          if (lon === undefined || lat === undefined) continue;
          west = Math.min(west, lon);
          east = Math.max(east, lon);
          south = Math.min(south, lat);
          north = Math.max(north, lat);
        }
    }
    if (!Number.isFinite(west)) return Rectangle.fromDegrees(0, -89.99, 0.001, -89.989);
    const margin = GLOBE_LIMIT_MARGIN_DEG;
    return Rectangle.fromDegrees(
      Math.max(-180, west - margin),
      Math.max(-90, south - margin),
      Math.min(180, east + margin),
      Math.min(90, north + margin),
    );
  }

  /**
   * Whether a footprint belongs in the globe collection for the current world. A floorless
   * site (`setFloorless`) never keeps terrain under it: in the photorealistic world it is
   * left out of the inverse clip, in the open world it is cut like a mesh site.
   */
  private globeCarries(key: string, targets: ClipTargets): boolean {
    if (key === this.floorless) return !this.photorealistic;
    return this.photorealistic ? targets.world : targets.globe;
  }

  /**
   * Seen from inside it, a splat scan is its own ground: the terrain kept under it as an
   * opaque floor (for sparse patches seen from above) sits wherever the elevation model
   * says, often a metre above the scan's ground, and from eye height it buries that ground
   * and everything low beyond it. `key` (a site id), or null, has no terrain under it.
   */
  setFloorless(key: string | null): void {
    if (!this.supported || this.floorless === key) return;
    const previous = this.floorless;
    this.floorless = key;
    for (const changed of [previous, key]) {
      if (changed === null) continue;
      this.removeFrom(this.globeCollection, this.globePolygons, changed);
      const entry = this.footprints.get(changed);
      if (entry && this.globeCarries(changed, entry.targets))
        this.addTo(this.globeCollection, this.globePolygons, changed, entry.footprint);
    }
    this.syncEnabled();
  }

  /**
   * Registers (or replaces) the footprint that should clip the world for a key (site id).
   * `targets` chooses what gets cut: the globe (terrain + imagery) and/or the global 3D
   * tileset. A Gaussian splat blends over terrain without z-fighting, so it usually clips
   * only the world tileset and keeps the ground under it.
   */
  setFootprint(key: string, footprint: Footprint | null, targets: ClipTargets = ALL): void {
    if (!this.supported) return;
    this.removeFrom(this.globeCollection, this.globePolygons, key);
    this.removeFrom(this.worldCollection, this.worldPolygons, key);
    this.footprints.delete(key);
    if (!footprint) {
      this.syncEnabled();
      return;
    }
    this.footprints.set(key, { footprint, targets });
    if (this.globeCarries(key, targets))
      this.addTo(this.globeCollection, this.globePolygons, key, footprint);
    if (targets.world) this.addTo(this.worldCollection, this.worldPolygons, key, footprint);
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
      for (const [key, entry] of this.footprints)
        if (entry.targets.world)
          this.addTo(this.worldCollection, this.worldPolygons, key, entry.footprint);
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
    // In the photorealistic world the globe only exists inside the inverse clip, which the
    // sentinel keeps active; the globe itself stays shown so it keeps loading. Its work is
    // limited to the engaged outlines (plus a margin): loading and uploading imagery for
    // terrain that is clipped away everywhere else cost a third of a slow drag's main thread.
    if (this.globeCollection && this.photorealistic) this.globeCollection.enabled = this.enabled;
    this.scene.globe.show = !this.photorealistic || (this.globeCollection?.enabled ?? false);
    this.scene.globe.cartographicLimitRectangle = this.photorealistic
      ? this.globeLimit()
      : Rectangle.MAX_VALUE;
    this.scene.requestRender();
  }

  destroy(): void {
    this.footprints.clear();
    this.globePolygons.clear();
    this.worldPolygons.clear();
  }
}
