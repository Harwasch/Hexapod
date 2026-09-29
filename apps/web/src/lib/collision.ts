/**
 * A scan's collision voxels as packaged (hexapod.collision v1, written by
 * `tools/captures/splat_tiles.py` next to `tileset.json` and declared on its root tile as
 * `extras.collision`), and the one interface navigation uses for solids, whether precomputed
 * here or built from splats at run time (SplatOccupancy).
 *
 * Why precomputed: building a grid from the splats on screen cost the page's main thread a
 * fifth of its time in a trace of the globe, and only ever described the tiles loaded. The
 * packager sees every gaussian once, at full resolution, so its grid is exact everywhere and
 * costs the page a download and a parse.
 *
 * `collision.bin` is gzip; inside, little-endian, `bricks` records of 76 bytes: int32 bx, by,
 * bz, then a 64-byte mask of the brick's 8 x 8 x 8 cells, bit `lx + 8 ly + 64 lz`. Cell
 * `(ix, iy, iz)` spans `origin + [i, i + 1) * cell`; its brick is `i >> 3`.
 */

import { CLEARANCE_CELLS, VoxelSolids, type Vec3 } from "./occupancy";

export interface CollisionMeta {
  format: "hexapod.collision";
  version: 1;
  uri: string;
  cell: number;
  origin: Vec3;
  brick: 8;
  bricks: number;
  solidCells?: number;
}

/** Reads `extras.collision` off a tileset's root tile; null when absent or not this format. */
export function collisionMetaOf(extras: unknown): CollisionMeta | null {
  const meta = (extras as { collision?: Record<string, unknown> } | null | undefined)?.collision;
  if (meta?.format !== "hexapod.collision" || meta.version !== 1) return null;
  const origin = meta.origin;
  if (
    typeof meta.uri !== "string" ||
    typeof meta.cell !== "number" ||
    !(meta.cell > 0) ||
    meta.brick !== 8 ||
    typeof meta.bricks !== "number" ||
    !Array.isArray(origin) ||
    origin.length !== 3 ||
    !origin.every((v) => typeof v === "number" && Number.isFinite(v))
  ) {
    return null;
  }
  return meta as unknown as CollisionMeta;
}

const RECORD_BYTES = 76;
const SPAN = 1 << 17;

/** Packaged solid voxels: bricks of 8 x 8 x 8 cells, one bit each. */
export class BrickGrid extends VoxelSolids {
  override readonly origin: Vec3;
  private readonly bounds: { lo: Vec3; hi: Vec3 } | null;

  constructor(
    readonly cell: number,
    origin: Vec3,
    private readonly bricks: Map<number, Uint8Array>,
    brickBounds: { lo: Vec3; hi: Vec3 } | null,
    readonly solidCells: number,
  ) {
    super();
    this.origin = origin;
    this.bounds = brickBounds && {
      lo: [brickBounds.lo[0] * 8, brickBounds.lo[1] * 8, brickBounds.lo[2] * 8],
      hi: [brickBounds.hi[0] * 8 + 7, brickBounds.hi[1] * 8 + 7, brickBounds.hi[2] * 8 + 7],
    };
  }

  get empty(): boolean {
    return this.bricks.size === 0;
  }

  solidBounds(): { lo: Vec3; hi: Vec3 } | null {
    return this.bounds;
  }

  solidAt(ix: number, iy: number, iz: number): boolean {
    const bx = ix >> 3;
    const by = iy >> 3;
    const bz = iz >> 3;
    if (bx < 0 || by < 0 || bz < 0 || bx >= SPAN || by >= SPAN || bz >= SPAN) return false;
    const mask = this.bricks.get((bx * SPAN + by) * SPAN + bz);
    if (mask === undefined) return false;
    const n = (ix & 7) + 8 * (iy & 7) + 64 * (iz & 7);
    return (((mask[n >> 3] ?? 0) >> (n & 7)) & 1) === 1;
  }
}

/** The decompressed payload as a grid. */
export function parseCollision(raw: Uint8Array, meta: CollisionMeta): BrickGrid {
  if (raw.byteLength !== meta.bricks * RECORD_BYTES) {
    throw new Error(`collision: ${String(raw.byteLength)} bytes for ${String(meta.bricks)} bricks`);
  }
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const bricks = new Map<number, Uint8Array>();
  let lo: Vec3 | null = null;
  let hi: Vec3 | null = null;
  let solid = 0;
  for (let r = 0; r < meta.bricks; r++) {
    const at = r * RECORD_BYTES;
    const bx = view.getInt32(at, true);
    const by = view.getInt32(at + 4, true);
    const bz = view.getInt32(at + 8, true);
    if (bx < 0 || by < 0 || bz < 0 || bx >= SPAN || by >= SPAN || bz >= SPAN) continue;
    const mask = raw.subarray(at + 12, at + RECORD_BYTES);
    for (const byte of mask) {
      let b = byte;
      while (b) {
        solid += b & 1;
        b >>= 1;
      }
    }
    bricks.set((bx * SPAN + by) * SPAN + bz, mask);
    if (lo === null || hi === null) {
      lo = [bx, by, bz];
      hi = [bx, by, bz];
    } else {
      lo = [Math.min(lo[0], bx), Math.min(lo[1], by), Math.min(lo[2], bz)];
      hi = [Math.max(hi[0], bx), Math.max(hi[1], by), Math.max(hi[2], bz)];
    }
  }
  return new BrickGrid(
    meta.cell,
    [meta.origin[0], meta.origin[1], meta.origin[2]],
    bricks,
    lo && hi ? { lo, hi } : null,
    solid,
  );
}

async function gunzip(bytes: ArrayBuffer): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Fetches and parses a scan's collision file, `meta.uri` relative to its tileset. */
export async function loadCollision(tilesetUrl: string, meta: CollisionMeta): Promise<BrickGrid> {
  const response = await fetch(new URL(meta.uri, tilesetUrl).toString());
  if (!response.ok) throw new Error(`collision answered ${String(response.status)}`);
  return parseCollision(await gunzip(await response.arrayBuffer()), meta);
}

/** What navigation asks of the solids, precomputed or built from splats. */
export interface Solids {
  raycast(origin: Vec3, direction: Vec3, maxDistance: number): number | null;
  sweep(from: Vec3, to: Vec3): { position: Vec3; blocked: boolean };
  distance(point: Vec3, radius: number): number | null;
  /** How close the camera may come to a surface (metres). */
  readonly clearance: number;
  readonly empty: boolean;
}

const MIN_CLEARANCE_M = 0.005;
const MAX_CLEARANCE_M = 0.5;

/** A packaged grid as navigation's solids: one level, its clearance from its cell. */
export class PrecomputedSolids implements Solids {
  readonly clearance: number;

  constructor(readonly grid: BrickGrid) {
    this.clearance = Math.min(
      MAX_CLEARANCE_M,
      Math.max(MIN_CLEARANCE_M, grid.cell * CLEARANCE_CELLS),
    );
  }

  get empty(): boolean {
    return this.grid.empty;
  }

  raycast(origin: Vec3, direction: Vec3, maxDistance: number): number | null {
    return this.grid.raycast(origin, direction, maxDistance);
  }

  sweep(from: Vec3, to: Vec3): { position: Vec3; blocked: boolean } {
    return this.grid.sweep(from, to, this.clearance);
  }

  distance(point: Vec3, radius: number): number | null {
    return this.grid.nearest(point, radius)?.distance ?? null;
  }
}
