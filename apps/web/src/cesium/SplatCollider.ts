/**
 * The splats on the globe as something the camera can touch: an occupancy grid per splat
 * tileset (lib/occupancy.ts), kept in step with the tiles drawn, answering the two questions
 * navigation asks -- what is under the cursor, and how far may the camera move.
 *
 * Gaussian splats write no depth, so `scene.pickPosition` sees through them to the ground,
 * and CesiumJS's own collision (`enableCollision`) only knows triangles. Without this a zoom
 * aimed at a bench flies through it, an orbit pivots on the ground behind, and nothing stops
 * the camera inside a wall.
 *
 * Each grid is in its primitive's east/north/up frame (`_rootTransform`), where the snapshot's
 * positions are baked; world points are converted at the edges. It follows the committed
 * snapshot tile by tile (`snapshotTiles`): a tile swapped out takes back its cells, a tile
 * swapped in adds its own, so the grid is fine where the view is fine and merged where it is
 * coarse. The work is done a few milliseconds a frame, and never while the camera moves
 * (camera first: the splat motion gate's `holding`).
 */

import {
  BoundingSphere,
  Cartesian3,
  Matrix4,
  PrimitiveCollection,
  type Ray,
  type Scene,
} from "cesium";

import { SplatOccupancy, type Vec3 } from "@/lib/occupancy";

import type { SplatPrimitive } from "./splatInternals";
import { snapshotTiles } from "./splatTiles";

/** Main-thread milliseconds a frame spent adding tiles to grids, at rest. */
const BUDGET_MS = 4;
/** A move longer than this in one frame is a jump (a bookmark, a search), not a motion. */
const JUMP_M = 50;

interface Tracked {
  grid: SplatOccupancy;
  /** Local east/north/up to world, and back. */
  toWorld: Matrix4;
  toLocal: Matrix4;
  /** World bounding sphere of the tileset, for a quick "nowhere near" test. */
  bounds: BoundingSphere;
  generation: number;
  /** Tiles still to add: their content key, splat range. */
  queue: { key: unknown; start: number; count: number }[];
  positions: Float32Array | undefined;
  colors: Uint8Array | undefined;
}

interface SplatTilesetShape {
  gaussianSplatPrimitive?: SplatPrimitive;
  boundingSphere: BoundingSphere;
  show: boolean;
}

const scratchA = new Cartesian3();
const scratchB = new Cartesian3();

function toVec(c: Cartesian3): Vec3 {
  return [c.x, c.y, c.z];
}

export class SplatCollider {
  private readonly tracked = new Map<SplatTilesetShape, Tracked>();
  private readonly off: () => void;

  /** `holding` says the camera is moving: no grid work then. */
  constructor(
    private readonly scene: Scene,
    private readonly holding: () => boolean,
  ) {
    this.off = scene.preUpdate.addEventListener(() => this.tick());
  }

  /** Whether any splat has a grid with something in it. */
  get active(): boolean {
    for (const entry of this.tracked.values()) if (entry.grid.sourceKeys.length > 0) return true;
    return false;
  }

  /** Per tileset: the cell sizes in use, tiles and solid cells, and tiles still queued. */
  describe(): { cells: number[]; tiles: number; solidCells: number; queued: number }[] {
    return [...this.tracked.values()].map((entry) => ({
      cells: entry.grid.cells,
      tiles: entry.grid.sourceKeys.length,
      solidCells: entry.grid.solidCells,
      queued: entry.queue.length,
    }));
  }

  /** Metres from `world` to the nearest splat surface within `radius`, or null. */
  distanceToSurface(world: Cartesian3, radius: number): number | null {
    let best: number | null = null;
    for (const entry of this.near(world, radius)) {
      const local = toVec(Matrix4.multiplyByPoint(entry.toLocal, world, scratchA));
      const d = entry.grid.distance(local, radius);
      if (d !== null && (best === null || d < best)) best = d;
    }
    return best;
  }

  /** How close the camera may come to a splat near `world`, or 0 when no splat is near. */
  clearance(world: Cartesian3): number {
    let clearance = 0;
    for (const entry of this.near(world, 0)) {
      clearance = Math.max(clearance, entry.grid.clearance);
    }
    return clearance;
  }

  /** The nearest splat surface along a world ray, within `maxDistance` metres. */
  raycast(ray: Ray, maxDistance = 5_000): { point: Cartesian3; distance: number } | null {
    let best: { point: Cartesian3; distance: number } | null = null;
    for (const [tileset, entry] of this.tracked) {
      if (!tileset.show || entry.grid.sourceKeys.length === 0) continue;
      const origin = Matrix4.multiplyByPoint(entry.toLocal, ray.origin, scratchA);
      const direction = Matrix4.multiplyByPointAsVector(entry.toLocal, ray.direction, scratchB);
      const t = entry.grid.raycast(toVec(origin), toVec(direction), maxDistance);
      if (t === null || (best && t >= best.distance)) continue;
      const point = Cartesian3.add(
        ray.origin,
        Cartesian3.multiplyByScalar(ray.direction, t, new Cartesian3()),
        new Cartesian3(),
      );
      best = { point, distance: t };
    }
    return best;
  }

  /**
   * Where the camera may go from `from` towards `to`, in world coordinates: stopped short of
   * any splat surface and sliding along it (OccupancyGrid.sweep). A jump -- farther than
   * `JUMP_M` in one step -- is let through: it is a bookmark or a search, not a motion.
   */
  resolve(from: Cartesian3, to: Cartesian3): { position: Cartesian3; blocked: boolean } {
    if (Cartesian3.distance(from, to) > JUMP_M) return { position: to, blocked: false };
    let position = Cartesian3.clone(to);
    let blocked = false;
    for (const entry of this.near(to, JUMP_M)) {
      const a = toVec(Matrix4.multiplyByPoint(entry.toLocal, from, scratchA));
      const b = toVec(Matrix4.multiplyByPoint(entry.toLocal, position, scratchB));
      const swept = entry.grid.sweep(a, b);
      if (!swept.blocked) continue;
      blocked = true;
      position = Matrix4.multiplyByPoint(
        entry.toWorld,
        Cartesian3.fromArray(swept.position),
        new Cartesian3(),
      );
    }
    return { position, blocked };
  }

  private near(world: Cartesian3, margin: number): Tracked[] {
    const found: Tracked[] = [];
    for (const [tileset, entry] of this.tracked) {
      if (!tileset.show || entry.grid.sourceKeys.length === 0) continue;
      const reach = entry.bounds.radius + margin;
      if (Cartesian3.distanceSquared(world, entry.bounds.center) <= reach * reach) {
        found.push(entry);
      }
    }
    return found;
  }

  private tick(): void {
    const seen = new Set<SplatTilesetShape>();
    for (const tileset of splatTilesets(this.scene.primitives)) {
      seen.add(tileset);
      this.follow(tileset);
    }
    for (const tileset of [...this.tracked.keys()]) {
      if (!seen.has(tileset)) this.tracked.delete(tileset);
    }
    if (this.holding()) return;
    const until = performance.now() + BUDGET_MS;
    for (const entry of this.tracked.values()) {
      while (entry.queue.length > 0 && performance.now() < until) {
        const next = entry.queue.shift();
        const { positions, colors } = entry;
        if (!next || !positions || !colors) break;
        entry.grid.add(
          next.key,
          positions,
          next.start,
          next.count,
          (i) => (colors[i * 4 + 3] ?? 0) / 255,
        );
      }
    }
  }

  /** Brings a tileset's grid in step with its committed snapshot's tiles. */
  private follow(tileset: SplatTilesetShape): void {
    const primitive = tileset.gaussianSplatPrimitive;
    const root = primitive?._rootTransform;
    const generation = primitive?._snapshot?.generation;
    const positions = primitive?._positions;
    const colors = primitive?._colors;
    const numSplats = primitive?._numSplats;
    if (!primitive || !root || generation === undefined || !positions || !colors || !numSplats) {
      return;
    }
    let entry = this.tracked.get(tileset);
    const toWorld = Matrix4.fromArray(Array.from(root));
    if (!entry || !Matrix4.equalsEpsilon(entry.toWorld, toWorld, 1e-9)) {
      // A new tileset, or one moved (a placement edit re-bakes every position): start over.
      entry = {
        grid: new SplatOccupancy(),
        toWorld,
        toLocal: Matrix4.inverseTransformation(toWorld, new Matrix4()),
        bounds: BoundingSphere.clone(tileset.boundingSphere),
        generation: -1,
        queue: [],
        positions: undefined,
        colors: undefined,
      };
      this.tracked.set(tileset, entry);
    }
    if (entry.generation === generation) return;
    const tiles = snapshotTiles(tileset, primitive, positions, numSplats);
    if (tiles.kind !== "tiles") return;
    entry.generation = generation;
    entry.positions = positions;
    entry.colors = colors;
    BoundingSphere.clone(tileset.boundingSphere, entry.bounds);
    const now = new Set(tiles.tiles.map((tile) => tile.content));
    for (const key of entry.grid.sourceKeys) {
      if (!now.has(key as (typeof tiles.tiles)[number]["content"])) entry.grid.remove(key);
    }
    // New tiles, and every tile whose range moved (the queue reads the new arrays).
    entry.queue = tiles.tiles
      .filter((tile) => !entry.grid.has(tile.content))
      .map((tile) => ({ key: tile.content, start: tile.start, count: tile.count }));
  }

  destroy(): void {
    this.off();
    this.tracked.clear();
  }
}

/** Every splat tileset under a collection, nested collections included. */
function splatTilesets(collection: PrimitiveCollection): SplatTilesetShape[] {
  const found: SplatTilesetShape[] = [];
  const visit = (items: PrimitiveCollection): void => {
    for (let i = 0; i < items.length; i++) {
      const item: unknown = items.get(i);
      if (item instanceof PrimitiveCollection) visit(item);
      else if ((item as SplatTilesetShape | undefined)?.gaussianSplatPrimitive) {
        found.push(item as SplatTilesetShape);
      }
    }
  };
  visit(collection);
  return found;
}
