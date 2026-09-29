/**
 * A sparse occupancy grid built from gaussian splat centres: what the camera collides with,
 * and what a ray from the cursor hits, where the renderer offers neither.
 *
 * A splat writes no depth (CesiumJS draws it after the opaque pass; Spark sorts it back to
 * front), so neither `scene.pickPosition` nor a depth read sees it: a zoom aimed at a
 * bench goes through it to the ground behind, and nothing stops the camera inside a wall.
 * The splats themselves are the geometry. Each one adds its opacity to the cell its centre
 * falls in; a cell holding at least `SOLID` opacity is solid. One opaque splat is enough,
 * a lone faint floater is not, and a surface of many translucent ones adds up.
 *
 * Centres only, not extents: a surface a scan sees is covered by splats far closer together
 * than a cell (a 22.6M-gaussian site of 150 m puts one every few centimetres), and the camera
 * is kept `radius` from any solid cell, which closes what gaps remain. Large gaussians far
 * from anything (sky, distant background) are rare and sit where nobody walks.
 *
 * Contributions are kept per source (a tile), so a level-of-detail swap removes exactly what a
 * tile added: the grid always describes the tiles drawn, fine close up and merged far away.
 *
 * Pure and frame-agnostic: callers give points in one local metric frame (the splat
 * primitive's east/north/up origin, or a scan's own) and convert at the edges.
 */

export type Vec3 = [number, number, number];

/** Opacity a cell must gather to be solid. */
export const SOLID = 0.6;
/** Splats fainter than this add nothing: they are haze, not surface. */
export const MIN_OPACITY = 0.2;

const OFFSET = 1 << 16;
const SPAN = 1 << 17;

/** One source's cells, and the opacity it put in each. */
interface Contribution {
  keys: number[];
  weights: number[];
}

/**
 * Solid voxels and the three questions navigation asks of them -- the first surface along a
 * ray, the nearest surface to a point, and how far a sphere may move -- whatever holds the
 * voxels: a grid built from splats at run time (OccupancyGrid) or one precomputed when the
 * scan was packaged (BrickGrid). Cell `(ix, iy, iz)` spans `origin + [i, i + 1) * cell`.
 */
export abstract class VoxelSolids {
  abstract readonly cell: number;
  readonly origin: Vec3 = [0, 0, 0];

  /** Whether the cell with these indices is solid. */
  abstract solidAt(ix: number, iy: number, iz: number): boolean;

  /** An inclusive box of cell indices holding every solid cell, or null when none is. */
  abstract solidBounds(): { lo: Vec3; hi: Vec3 } | null;

  /**
   * The part of a ray (unit `d`) inside the solid cells' box, as [skip, until] distances,
   * or null when it misses the box within `maxDistance`. Starts a cell short of the box,
   * so the walk's first cell -- which never counts -- is outside it.
   */
  protected clipToSolids(start: Vec3, d: Vec3, maxDistance: number): [number, number] | null {
    const bounds = this.solidBounds();
    if (bounds === null) return null;
    let enter = 0;
    let exit = maxDistance;
    for (let k = 0; k < 3; k++) {
      const lo = (bounds.lo[k] ?? 0) * this.cell + (this.origin[k] ?? 0);
      const hi = ((bounds.hi[k] ?? 0) + 1) * this.cell + (this.origin[k] ?? 0);
      const o = start[k] ?? 0;
      const dk = d[k] ?? 0;
      if (Math.abs(dk) < 1e-12) {
        if (o < lo || o > hi) return null;
        continue;
      }
      let t0 = (lo - o) / dk;
      let t1 = (hi - o) / dk;
      if (t0 > t1) [t0, t1] = [t1, t0];
      enter = Math.max(enter, t0);
      exit = Math.min(exit, t1);
      if (enter > exit) return null;
    }
    return [Math.max(0, enter - this.cell), Math.min(maxDistance, exit + this.cell)];
  }

  /**
   * The distance along a ray to the first solid cell it enters, or null within `maxDistance`
   * (Amanatides & Woo's voxel walk: every cell the ray crosses, in order, none skipped).
   * `direction` need not be unit length; the answer is in its units times its length.
   */
  raycast(start: Vec3, direction: Vec3, maxDistance: number): number | null {
    const length = Math.hypot(direction[0], direction[1], direction[2]);
    if (!(length > 0)) return null;
    const d: Vec3 = [direction[0] / length, direction[1] / length, direction[2] / length];
    const cell = this.cell;
    // Only the stretch of the ray inside the box that holds every solid cell is walked: a ray
    // from far away, or one that misses the scan, costs a few divisions instead of thousands
    // of empty cells.
    const clip = this.clipToSolids(start, d, maxDistance);
    if (clip === null) return null;
    const [skip, until] = clip;
    const origin: Vec3 = [
      start[0] - this.origin[0] + d[0] * skip,
      start[1] - this.origin[1] + d[1] * skip,
      start[2] - this.origin[2] + d[2] * skip,
    ];
    const reach = until - skip;
    const index: Vec3 = [
      Math.floor(origin[0] / cell),
      Math.floor(origin[1] / cell),
      Math.floor(origin[2] / cell),
    ];
    const step: Vec3 = [0, 0, 0];
    const next: Vec3 = [0, 0, 0];
    const delta: Vec3 = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const dk = d[k] ?? 0;
      const ik = index[k] ?? 0;
      const ok = origin[k] ?? 0;
      if (dk > 0) {
        step[k] = 1;
        next[k] = ((ik + 1) * cell - ok) / dk;
        delta[k] = cell / dk;
      } else if (dk < 0) {
        step[k] = -1;
        next[k] = (ik * cell - ok) / dk;
        delta[k] = -cell / dk;
      } else {
        next[k] = Number.POSITIVE_INFINITY;
        delta[k] = Number.POSITIVE_INFINITY;
      }
    }
    // The cell the ray starts in does not count: from inside a solid cell, the way out is
    // not a hit.
    for (let guard = 0; guard < 1_000_000; guard++) {
      const axis = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : next[1] < next[2] ? 1 : 2;
      const travelled = next[axis] ?? Number.POSITIVE_INFINITY;
      if (travelled > reach) return null;
      index[axis] = (index[axis] ?? 0) + (step[axis] ?? 0);
      next[axis] = travelled + (delta[axis] ?? 0);
      if (this.solidAt(index[0], index[1], index[2])) return travelled + skip;
    }
    return null;
  }

  /**
   * The nearest solid cell within `radius` of `point`: how far its box is, and the unit
   * direction from it to the point (the way out). Null when nothing is that close.
   */
  nearest(world: Vec3, radius: number): { distance: number; away: Vec3 } | null {
    const cell = this.cell;
    const point: Vec3 = [
      world[0] - this.origin[0],
      world[1] - this.origin[1],
      world[2] - this.origin[2],
    ];
    const bounds = this.solidBounds();
    if (bounds === null) return null;
    const lo = point.map((v, k) =>
      Math.max(Math.floor((v - radius) / cell), bounds.lo[k] ?? 0),
    ) as Vec3;
    const hi = point.map((v, k) =>
      Math.min(Math.floor((v + radius) / cell), bounds.hi[k] ?? 0),
    ) as Vec3;
    let best = Number.POSITIVE_INFINITY;
    let away: Vec3 = [0, 0, 1];
    for (let ix = lo[0]; ix <= hi[0]; ix++) {
      for (let iy = lo[1]; iy <= hi[1]; iy++) {
        for (let iz = lo[2]; iz <= hi[2]; iz++) {
          if (!this.solidAt(ix, iy, iz)) continue;
          // Closest point of the cell's box to the point.
          const cx = Math.min(Math.max(point[0], ix * cell), (ix + 1) * cell);
          const cy = Math.min(Math.max(point[1], iy * cell), (iy + 1) * cell);
          const cz = Math.min(Math.max(point[2], iz * cell), (iz + 1) * cell);
          let dx = point[0] - cx;
          let dy = point[1] - cy;
          let dz = point[2] - cz;
          let distance = Math.hypot(dx, dy, dz);
          if (distance === 0) {
            // Inside the cell: out through its centre's far side.
            dx = point[0] - (ix + 0.5) * cell;
            dy = point[1] - (iy + 0.5) * cell;
            dz = point[2] - (iz + 0.5) * cell;
            distance = 0;
          }
          if (distance < best) {
            best = distance;
            const norm = Math.hypot(dx, dy, dz) || 1;
            away = [dx / norm, dy / norm, dz / norm];
          }
        }
      }
    }
    return best <= radius ? { distance: best, away } : null;
  }

  /**
   * Where a sphere of `radius` moving from `from` towards `to` may go: the whole way when
   * nothing is in the way; otherwise sliding along what it meets, the way a game camera or
   * character moves along a wall instead of stopping dead. Each step moves, then pushes the
   * sphere back out of any surface it entered (along the way out of the nearest solid cell,
   * a few times over for a crease), which keeps the part of the move along the surface and
   * drops the part into it.
   *
   * Steps no longer than half a cell, so a fast move cannot tunnel through a thin surface.
   * A sphere that starts already too close (the grid changed under it, or it passed through
   * on purpose) is not pushed about: any step that does not bring it closer is allowed, so it
   * can always get out, and none that goes deeper.
   */
  sweep(from: Vec3, to: Vec3, radius: number): { position: Vec3; blocked: boolean } {
    const move: Vec3 = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    const length = Math.hypot(move[0], move[1], move[2]);
    if (!(length > 0)) return { position: [...to], blocked: false };
    const steps = Math.min(4096, Math.max(1, Math.ceil(length / (this.cell * 0.5))));
    const part: Vec3 = [move[0] / steps, move[1] / steps, move[2] / steps];
    let position: Vec3 = [...from];
    let blocked = false;
    for (let s = 0; s < steps; s++) {
      const here = this.nearest(position, radius);
      const candidate: Vec3 = [position[0] + part[0], position[1] + part[1], position[2] + part[2]];
      if (here && here.distance < radius * 0.9) {
        // Already inside the margin: out, or along, but never deeper.
        const there = this.nearest(candidate, radius);
        if (there && there.distance < here.distance) {
          blocked = true;
          break;
        }
        position = candidate;
        continue;
      }
      let pushed = false;
      for (let k = 0; k < 4; k++) {
        const hit = this.nearest(candidate, radius);
        if (!hit || hit.distance >= radius - 1e-9) break;
        const out = radius - hit.distance + this.cell * 1e-3;
        candidate[0] += hit.away[0] * out;
        candidate[1] += hit.away[1] * out;
        candidate[2] += hit.away[2] * out;
        pushed = true;
      }
      if (pushed) {
        blocked = true;
        const left = this.nearest(candidate, radius);
        // A crease it could not get out of: stay put rather than end up inside.
        if (left && left.distance < radius * 0.9) break;
      }
      position = candidate;
    }
    // Unobstructed, arrive exactly: steps summed in floating point drift by an ulp or two.
    return blocked ? { position, blocked } : { position: [...to], blocked };
  }
}

export class OccupancyGrid extends VoxelSolids {
  private readonly weight = new Map<number, number>();
  private readonly sources = new Map<unknown, Contribution>();
  /** A box around every cell ever added; it only grows (removal is rare and a loose box is
   *  still correct, just less of a shortcut). */
  private bounds: { lo: Vec3; hi: Vec3 } | null = null;

  /** `cell` is the edge in metres; the grid spans ±65 536 cells around the frame origin. */
  constructor(readonly cell: number) {
    super();
  }

  /** Sources added and not removed. */
  get size(): number {
    return this.sources.size;
  }

  solidBounds(): { lo: Vec3; hi: Vec3 } | null {
    return this.sources.size === 0 ? null : this.bounds;
  }

  /** Number of solid cells (for diagnostics and tests). */
  get solidCells(): number {
    let solid = 0;
    this.weight.forEach((value) => (solid += value >= SOLID ? 1 : 0));
    return solid;
  }

  has(source: unknown): boolean {
    return this.sources.has(source);
  }

  /** Every source added and not removed. */
  get sourceKeys(): unknown[] {
    return [...this.sources.keys()];
  }

  /**
   * Adds `count` splats: centres at `positions[(start + i) * 3 ..]`, opacity from
   * `opacity(start + i)` in [0, 1]. Replaces what `source` added before, if anything.
   */
  add(
    source: unknown,
    positions: ArrayLike<number>,
    start: number,
    count: number,
    opacity: (index: number) => number,
  ): void {
    this.remove(source);
    const own = new Map<number, number>();
    const inverse = 1 / this.cell;
    for (let i = start; i < start + count; i++) {
      const alpha = opacity(i);
      if (!(alpha >= MIN_OPACITY)) continue;
      const x = positions[i * 3] ?? Number.NaN;
      const y = positions[i * 3 + 1] ?? Number.NaN;
      const z = positions[i * 3 + 2] ?? Number.NaN;
      const ix = Math.floor(x * inverse);
      const iy = Math.floor(y * inverse);
      const iz = Math.floor(z * inverse);
      const key = this.keyOf(ix, iy, iz);
      if (key === null) continue;
      if (this.bounds === null) this.bounds = { lo: [ix, iy, iz], hi: [ix, iy, iz] };
      else {
        const { lo, hi } = this.bounds;
        if (ix < lo[0]) lo[0] = ix;
        if (iy < lo[1]) lo[1] = iy;
        if (iz < lo[2]) lo[2] = iz;
        if (ix > hi[0]) hi[0] = ix;
        if (iy > hi[1]) hi[1] = iy;
        if (iz > hi[2]) hi[2] = iz;
      }
      own.set(key, (own.get(key) ?? 0) + alpha);
    }
    const contribution: Contribution = { keys: [], weights: [] };
    own.forEach((value, key) => {
      contribution.keys.push(key);
      contribution.weights.push(value);
      this.weight.set(key, (this.weight.get(key) ?? 0) + value);
    });
    this.sources.set(source, contribution);
  }

  /** Takes back what `source` added. */
  remove(source: unknown): void {
    const contribution = this.sources.get(source);
    if (!contribution) return;
    contribution.keys.forEach((key, i) => {
      const left = (this.weight.get(key) ?? 0) - (contribution.weights[i] ?? 0);
      if (left > 1e-6) this.weight.set(key, left);
      else this.weight.delete(key);
    });
    this.sources.delete(source);
  }

  clear(): void {
    this.weight.clear();
    this.sources.clear();
    this.bounds = null;
  }

  private keyOf(ix: number, iy: number, iz: number): number | null {
    const x = ix + OFFSET;
    const y = iy + OFFSET;
    const z = iz + OFFSET;
    if (!(x >= 0 && x < SPAN && y >= 0 && y < SPAN && z >= 0 && z < SPAN)) return null;
    return (x * SPAN + y) * SPAN + z;
  }

  solidAt(ix: number, iy: number, iz: number): boolean {
    const key = this.keyOf(ix, iy, iz);
    return key !== null && (this.weight.get(key) ?? 0) >= SOLID;
  }
}

/** Splats per occupied cell a tile's cell is sized for: enough that neighbouring cells on a
 *  surface are all occupied (on a surface sampled every s, a cell c holds about (c/s)^2). */
export const SPLATS_PER_CELL = 3;
/** Cell edges tried, as powers of two metres: 4 mm to 32 m. */
const MIN_CELL_EXP = -8;
const MAX_CELL_EXP = 5;
/** Splats sampled per tile to size its cell. */
const SIZING_SAMPLE = 20_000;
/** The camera keeps this many of a level's cells from a solid one, within these bounds (m). */
export const CLEARANCE_CELLS = 1.5;
const MIN_CLEARANCE_M = 0.005;
const MAX_CLEARANCE_M = 0.5;

/**
 * The cell a tile's splats are dense enough to fill: the smallest power of two metres at
 * which its opaque splats average `SPLATS_PER_CELL` per occupied cell. Measured from the
 * data, not from the scan's size: a merged parent 1/8 as dense gets cells about twice as big,
 * a leaf of a close-range capture millimetres. A sparse cloud read at too fine a cell is
 * isolated specks a ray passes between; too coarse a cell is a blob the camera cannot get
 * near. Powers of two, so tiles of one level share a grid.
 */
export function cellFor(
  positions: ArrayLike<number>,
  start: number,
  count: number,
  opacity: (index: number) => number,
): number {
  const stride = Math.max(1, Math.floor(count / SIZING_SAMPLE));
  const sample: number[] = [];
  for (let i = start; i < start + count; i += stride) {
    if (!(opacity(i) >= MIN_OPACITY)) continue;
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x === undefined || y === undefined || z === undefined) continue;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    sample.push(x, y, z);
  }
  const n = sample.length / 3;
  if (n === 0) return 2 ** MAX_CELL_EXP;
  // A stride thins the sample: each sampled splat stands for `stride`, so the per-cell count
  // at full density is the sampled one times it.
  for (let exp = MIN_CELL_EXP; exp < MAX_CELL_EXP; exp++) {
    const inverse = 2 ** -exp;
    const cells = new Set<string>();
    for (let i = 0; i < n; i++) {
      cells.add(
        `${String(Math.floor((sample[i * 3] ?? 0) * inverse))},${String(
          Math.floor((sample[i * 3 + 1] ?? 0) * inverse),
        )},${String(Math.floor((sample[i * 3 + 2] ?? 0) * inverse))}`,
      );
    }
    if ((n * stride) / cells.size >= SPLATS_PER_CELL) return 2 ** exp;
  }
  return 2 ** MAX_CELL_EXP;
}

/**
 * Occupancy at the resolution each part of a scan is drawn at: one OccupancyGrid per cell
 * size, each tile added to the level its own density calls for (`cellFor`). With REPLACE
 * level of detail each place is covered by one tile at a time, so each place is described
 * by one level: fine where the view is fine, coarse where it is merged.
 *
 * Queries ask every level: a ray stops at the nearest hit of any; a move is swept against
 * each in turn, keeping `CLEARANCE_CELLS` of that level's cells from its surfaces (so the
 * camera comes centimetres from a finely drawn bench, and stays clear of a coarse blob until
 * it refines).
 */
export class SplatOccupancy {
  private readonly levels = new Map<number, OccupancyGrid>();
  private readonly levelOf = new Map<unknown, OccupancyGrid>();

  /** Adds a tile's splats at the level its density calls for, replacing what it added before. */
  add(
    source: unknown,
    positions: ArrayLike<number>,
    start: number,
    count: number,
    opacity: (index: number) => number,
  ): void {
    this.remove(source);
    const cell = cellFor(positions, start, count, opacity);
    let level = this.levels.get(cell);
    if (!level) {
      level = new OccupancyGrid(cell);
      this.levels.set(cell, level);
    }
    level.add(source, positions, start, count, opacity);
    this.levelOf.set(source, level);
  }

  remove(source: unknown): void {
    this.levelOf.get(source)?.remove(source);
    this.levelOf.delete(source);
  }

  has(source: unknown): boolean {
    return this.levelOf.has(source);
  }

  get sourceKeys(): unknown[] {
    return [...this.levelOf.keys()];
  }

  get empty(): boolean {
    return this.levelOf.size === 0;
  }

  get solidCells(): number {
    let total = 0;
    this.levels.forEach((level) => (total += level.solidCells));
    return total;
  }

  /** The cell sizes in use, finest first. */
  get cells(): number[] {
    return [...this.levels.entries()]
      .filter(([, level]) => level.size > 0)
      .map(([cell]) => cell)
      .sort((a, b) => a - b);
  }

  /** How close the camera may come to the finest surfaces drawn (metres). */
  get clearance(): number {
    const finest = this.cells[0];
    return finest === undefined ? MIN_CLEARANCE_M : clearanceOf(finest);
  }

  raycast(origin: Vec3, direction: Vec3, maxDistance: number): number | null {
    let best: number | null = null;
    for (const level of this.levels.values()) {
      if (level.size === 0) continue;
      const t = level.raycast(origin, direction, best ?? maxDistance);
      if (t !== null && (best === null || t < best)) best = t;
    }
    return best;
  }

  /** Distance from `point` to the nearest solid cell of any level within `radius`, or null. */
  distance(point: Vec3, radius: number): number | null {
    let best: number | null = null;
    for (const level of this.levels.values()) {
      if (level.size === 0) continue;
      const hit = level.nearest(point, radius);
      if (hit && (best === null || hit.distance < best)) best = hit.distance;
    }
    return best;
  }

  sweep(from: Vec3, to: Vec3): { position: Vec3; blocked: boolean } {
    let position: Vec3 = [...to];
    let blocked = false;
    for (const [cell, level] of this.levels) {
      if (level.size === 0) continue;
      const swept = level.sweep(from, position, clearanceOf(cell));
      if (swept.blocked) {
        blocked = true;
        position = swept.position;
      }
    }
    return { position, blocked };
  }

  clear(): void {
    this.levels.clear();
    this.levelOf.clear();
  }
}

function clearanceOf(cell: number): number {
  return Math.min(MAX_CLEARANCE_M, Math.max(MIN_CLEARANCE_M, cell * CLEARANCE_CELLS));
}
