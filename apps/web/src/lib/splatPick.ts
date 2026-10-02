/**
 * Which splats are under a point of the screen, on the CPU, the same for every splat renderer
 * (CesiumJS, PlayCanvas, Spark): each renderer hands over the tiles it draws now as
 * `PickTile`s -- the tile's own positions in the scan's frame (the order and frame
 * `instances.json` keys its ids by, through the tile's checksum), each splat's largest axis
 * and its opacity -- and a ray from the camera through the cursor is tested against them.
 *
 * A splat is a soft ellipsoid; here it is a sphere of its largest axis (never thinner than a
 * couple of pixels at its distance, so a far splat can still be hit), and its opacity at the
 * ray is `opacity · exp(-d² / 2r²)` for a ray passing `d` from its centre. The hits are
 * composited front to back as the renderer blends them: each gets `T · α` of the pixel, where
 * `T` is what the splats in front let through, so what is behind a solid surface counts for
 * nothing and a thin veil of fragments in front counts for little.
 *
 * Every tile gets a spatial index the first time it is picked: its splats in Morton order cut
 * into blocks of `BLOCK` with a box each, so a ray only visits the splats of boxes it crosses.
 */

export type Vec3 = readonly [number, number, number];

/** One tile as a renderer draws it, in the tile's own splat order. */
export interface PickTile {
  /** `checksumPositions` of the tile's own positions: its key in `instances.json`. */
  readonly checksum: string;
  readonly count: number;
  /** xyz per splat, in the scan's frame (tileset-local east/north/up metres). */
  readonly positions: Float32Array;
  /** Each splat's largest axis (metres, linear). */
  readonly radii: Float32Array;
  /** Each splat's opacity, 0..1. */
  readonly opacity: Float32Array;
}

/** A ray in the scan's frame; `direction` need not be unit. */
export interface Ray {
  origin: Vec3;
  direction: Vec3;
}

export interface PickOptions {
  /** Radians a pixel spans at the centre of the view: the least radius a splat is hit by. */
  pixelAngle?: number;
  /** Pixels the least radius is. */
  minPixels?: number;
  /** A splat is hit within this many radii of the ray. */
  reach?: number;
  /** Radii are capped here, so a huge background splat does not swallow the ray. */
  maxRadius?: number;
  /** Nothing nearer than this (metres along the ray). */
  near?: number;
  /** Compositing stops once less than this is let through. */
  transmittanceFloor?: number;
  /** Whether a splat is drawn (a hidden object's splats are not, and let the ray through). */
  include?: (tile: number, index: number) => boolean;
}

/** One splat a ray hit, and the share of the pixel it is (front-to-back compositing). */
export interface RayHit {
  /** Index into the tiles given. */
  tile: number;
  /** The splat, in the tile's own order. */
  index: number;
  /** Distance along the (unit) ray. */
  t: number;
  /** Its opacity at the ray. */
  alpha: number;
  /** `T · α`: how much of the pixel it is. */
  weight: number;
}

/** Splats a block of the index holds. */
export const BLOCK = 64;

/** One tile's spatial index. */
export interface TileIndex {
  /** Splat indices in Morton order. */
  readonly order: Uint32Array;
  /** Per block: min xyz, max xyz (each splat's reach included). */
  readonly boxes: Float32Array;
  /** The radius cap the boxes were built with. */
  readonly maxRadius: number;
}

const DEFAULTS = {
  pixelAngle: 0.001,
  minPixels: 3,
  reach: 2.5,
  maxRadius: 0.6,
  near: 0.05,
  transmittanceFloor: 0.004,
};

/** Spreads the low 10 bits of `v` to every third bit. */
function spread(v: number): number {
  let x = v & 0x3ff;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x >>> 0;
}

/** Builds `tile`'s index (blocks of `BLOCK` splats along a Morton curve). */
export function buildTileIndex(tile: PickTile, maxRadius = DEFAULTS.maxRadius): TileIndex {
  const { count, positions, radii } = tile;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3] ?? 0;
    const y = positions[i * 3 + 1] ?? 0;
    const z = positions[i * 3 + 2] ?? 0;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  const q = 1023 / span;
  // Morton code (30 bits) above the index (up to 22 bits): one numeric sort orders both.
  const keys = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const code =
      spread(((positions[i * 3] ?? 0) - minX) * q) |
      (spread(((positions[i * 3 + 1] ?? 0) - minY) * q) << 1) |
      (spread(((positions[i * 3 + 2] ?? 0) - minZ) * q) << 2);
    keys[i] = (code >>> 0) * 4194304 + i;
  }
  keys.sort();
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = (keys[i] ?? 0) % 4194304;
  const blocks = Math.ceil(count / BLOCK);
  const boxes = new Float32Array(blocks * 6);
  for (let b = 0; b < blocks; b++) {
    let x0 = Infinity;
    let y0 = Infinity;
    let z0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    let z1 = -Infinity;
    const end = Math.min(count, (b + 1) * BLOCK);
    for (let k = b * BLOCK; k < end; k++) {
      const i = order[k] ?? 0;
      const r = DEFAULTS.reach * Math.min(maxRadius, radii[i] ?? 0);
      const x = positions[i * 3] ?? 0;
      const y = positions[i * 3 + 1] ?? 0;
      const z = positions[i * 3 + 2] ?? 0;
      x0 = Math.min(x0, x - r);
      y0 = Math.min(y0, y - r);
      z0 = Math.min(z0, z - r);
      x1 = Math.max(x1, x + r);
      y1 = Math.max(y1, y + r);
      z1 = Math.max(z1, z + r);
    }
    boxes.set([x0, y0, z0, x1, y1, z1], b * 6);
  }
  return { order, boxes, maxRadius };
}

const INDEXES = new WeakMap<PickTile, TileIndex>();

/** `tile`'s index, built once. */
export function tileIndexOf(tile: PickTile): TileIndex {
  let index = INDEXES.get(tile);
  if (!index) {
    index = buildTileIndex(tile);
    INDEXES.set(tile, index);
  }
  return index;
}

/** The entry and exit distances of a ray through a box, or null when it misses (slab test). */
function slab(
  o: Vec3,
  inv: Vec3,
  boxes: Float32Array,
  at: number,
  pad: number,
): [number, number] | null {
  let t0 = -Infinity;
  let t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    const lo = (boxes[at + a] ?? 0) - pad;
    const hi = (boxes[at + 3 + a] ?? 0) + pad;
    const ia = inv[a] ?? 0;
    if (!Number.isFinite(ia)) {
      if ((o[a] ?? 0) < lo || (o[a] ?? 0) > hi) return null;
      continue;
    }
    let ta = (lo - (o[a] ?? 0)) * ia;
    let tb = (hi - (o[a] ?? 0)) * ia;
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta);
    t1 = Math.min(t1, tb);
    if (t0 > t1) return null;
  }
  return [t0, t1];
}

function unit(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/**
 * Every splat the ray passes through, front to back, with its share of the pixel. Hits that
 * end up with almost no share (behind a solid surface) are dropped.
 */
export function castRay(tiles: readonly PickTile[], ray: Ray, options: PickOptions = {}): RayHit[] {
  const o = { ...DEFAULTS, ...options };
  const d = unit(ray.direction);
  const origin = ray.origin;
  const inv: Vec3 = [1 / d[0], 1 / d[1], 1 / d[2]];
  const raw: { tile: number; index: number; t: number; alpha: number }[] = [];
  for (let ti = 0; ti < tiles.length; ti++) {
    const tile = tiles[ti];
    if (!tile || tile.count === 0) continue;
    const index = tileIndexOf(tile);
    const { positions, radii, opacity } = tile;
    const blocks = index.boxes.length / 6;
    for (let b = 0; b < blocks; b++) {
      // The least radius grows with distance: pad the box by it at the box's far side.
      const at = b * 6;
      const bx = index.boxes;
      const far = Math.hypot(
        Math.max(Math.abs((bx[at] ?? 0) - origin[0]), Math.abs((bx[at + 3] ?? 0) - origin[0])),
        Math.max(Math.abs((bx[at + 1] ?? 0) - origin[1]), Math.abs((bx[at + 4] ?? 0) - origin[1])),
        Math.max(Math.abs((bx[at + 2] ?? 0) - origin[2]), Math.abs((bx[at + 5] ?? 0) - origin[2])),
      );
      const pad = o.reach * far * o.pixelAngle * o.minPixels * 0.5;
      const hit = slab(origin, inv, bx, at, pad);
      if (!hit || hit[1] < o.near) continue;
      const end = Math.min(tile.count, (b + 1) * BLOCK);
      for (let k = b * BLOCK; k < end; k++) {
        const i = index.order[k] ?? 0;
        const vx = (positions[i * 3] ?? 0) - origin[0];
        const vy = (positions[i * 3 + 1] ?? 0) - origin[1];
        const vz = (positions[i * 3 + 2] ?? 0) - origin[2];
        const t = vx * d[0] + vy * d[1] + vz * d[2];
        if (t < o.near) continue;
        const d2 = Math.max(0, vx * vx + vy * vy + vz * vz - t * t);
        const r = Math.max(
          Math.min(o.maxRadius, radii[i] ?? 0),
          t * o.pixelAngle * o.minPixels * 0.5,
          1e-4,
        );
        if (d2 > o.reach * o.reach * r * r) continue;
        const alpha = Math.min(0.99, (opacity[i] ?? 1) * Math.exp((-0.5 * d2) / (r * r)));
        if (alpha < 0.01) continue;
        if (o.include && !o.include(ti, i)) continue;
        raw.push({ tile: ti, index: i, t, alpha });
      }
    }
  }
  raw.sort((a, b) => a.t - b.t);
  const out: RayHit[] = [];
  let transmittance = 1;
  for (const h of raw) {
    if (transmittance < o.transmittanceFloor) break;
    const weight = transmittance * h.alpha;
    transmittance *= 1 - h.alpha;
    if (weight >= 1e-3) out.push({ ...h, weight });
  }
  return out;
}

/** Per instance id (0: none), the share of the pixel its splats are and the nearest hit. */
export function hitWeights(
  hits: readonly RayHit[],
  idOf: (hit: RayHit) => number,
): Map<number, { weight: number; t: number }> {
  const out = new Map<number, { weight: number; t: number }>();
  for (const hit of hits) {
    const id = idOf(hit);
    const current = out.get(id);
    if (current) current.weight += hit.weight;
    else out.set(id, { weight: hit.weight, t: hit.t });
  }
  return out;
}
