/**
 * Per-cell site data on one lattice (hexapod.cellgrid v1): what the confidence map, and later
 * the surfaces, say about each 2.5D column of a site. Cesium-free, so view.html and the tests
 * read it without the engine.
 *
 * The lattice (decision C1): axis-aligned columns in the tileset frame -- the site frame when
 * the run had one -- with x east and y north, cell size `baseM * 2^level` and cell index
 * `i = floor(x / cell)`, `j = floor(y / cell)` from the frame's origin. Every epoch's cells
 * therefore nest, and two epochs compare at the coarser of their two levels.
 *
 * The container (C2): the root tile's `extras.<kind>` is the descriptor; `uri` (relative to
 * the tileset) is the gzip of the level planes concatenated. A plane is `nj * ni` cells,
 * row-major (j outer, i inner), the channels interleaved per cell, little-endian. Each level
 * says where its plane starts (`offset`) and how long it is (`bytes`) in the uncompressed
 * stream; `sha256` is of that whole stream.
 */

export type CellChannelType = "u1" | "i1" | "u2" | "i2";

export interface CellGridChannel {
  name: string;
  type: CellChannelType;
  /** What one step of the stored integer is, in metres, when the channel is a length. */
  unitM?: number;
}

export interface CellGridLevel {
  level: number;
  cellM: number;
  i0: number;
  j0: number;
  ni: number;
  nj: number;
  /** Bytes into the uncompressed planes. */
  offset: number;
  bytes: number;
}

export interface CellGridSummary {
  cells?: number;
  scored?: number;
  weak?: number;
  unseen?: number;
  weakShare?: number;
  reasons?: Record<string, number>;
  medianGsdMm?: number;
  metric?: boolean;
}

export interface CellGridMeta {
  format: "hexapod.cellgrid";
  version: 1;
  kind: string;
  uri: string;
  frame: "tileset";
  lattice: { baseM: number; level: number };
  cellM: number;
  channels: CellGridChannel[];
  /** Finest first, each one lattice level coarser than the one before. */
  levels: CellGridLevel[];
  reasonBits?: Record<string, number>;
  policy?: string;
  weakBelow?: number;
  alignmentCap?: Record<string, unknown>;
  summary?: CellGridSummary;
  sha256?: string;
}

/** A decoded grid: the descriptor and its uncompressed planes. */
export interface CellGrid {
  meta: CellGridMeta;
  planes: Uint8Array;
  /** Bytes a cell holds (every channel). */
  stride: number;
}

/** The lattice's finest cell (C1). */
export const LATTICE_BASE_M = 0.25;
/** The finest level plus up to eight coarser (C3's pyramid): what the shader can address. */
export const MAX_CELL_LEVELS = 9;
/** `weak-v1`'s threshold when a file does not state one (C3). */
export const DEFAULT_WEAK_BELOW = 128;
/** `reasonBits.unseen` when a file does not state it (C2). */
export const DEFAULT_UNSEEN_BIT = 16;

const CHANNEL_BYTES: Record<CellChannelType, number> = { u1: 1, i1: 1, u2: 2, i2: 2 };

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const int = (v: unknown): v is number => Number.isInteger(v);

/** Bytes per cell for `channels`. */
export function cellStride(channels: readonly CellGridChannel[]): number {
  return channels.reduce((sum, c) => sum + CHANNEL_BYTES[c.type], 0);
}

/**
 * Reads `extras[key]` (a root tile's extras) as a cell grid; null when absent, another format
 * or version, or inconsistent in a way the decoder could only guess around.
 */
export function cellGridMeta(extras: unknown, key = "confidence"): CellGridMeta | null {
  const raw = (extras as Record<string, unknown> | null | undefined)?.[key] as
    Record<string, unknown> | undefined;
  if (raw?.format !== "hexapod.cellgrid" || raw.version !== 1) return null;
  if (typeof raw.uri !== "string" || raw.frame !== "tileset") return null;
  const lattice = raw.lattice as { baseM?: unknown; level?: unknown } | undefined;
  if (!finite(lattice?.baseM) || !(lattice.baseM > 0) || !int(lattice.level)) return null;
  const channels = raw.channels as CellGridChannel[] | undefined;
  if (!Array.isArray(channels) || channels.length === 0) return null;
  if (!channels.every((c) => typeof c?.name === "string" && c.type in CHANNEL_BYTES)) return null;
  const stride = cellStride(channels);
  const levels = raw.levels as CellGridLevel[] | undefined;
  if (!Array.isArray(levels) || levels.length === 0 || levels.length > MAX_CELL_LEVELS) return null;
  const sorted = [...levels].sort((a, b) => a.level - b.level);
  for (const [k, l] of sorted.entries()) {
    if (![l.level, l.i0, l.j0, l.ni, l.nj, l.offset, l.bytes].every(int)) return null;
    if (l.ni <= 0 || l.nj <= 0 || l.offset < 0 || l.bytes !== l.ni * l.nj * stride) return null;
    const expected = lattice.baseM * 2 ** l.level;
    if (!finite(l.cellM) || Math.abs(l.cellM - expected) > expected * 1e-9) return null;
    // The shader steps one lattice level at a time from the finest.
    if (k > 0 && l.level !== (sorted[k - 1]?.level ?? Number.NaN) + 1) return null;
  }
  if (sorted[0]?.level !== lattice.level) return null;
  return { ...(raw as unknown as CellGridMeta), channels, levels: sorted };
}

async function gunzip(bytes: ArrayBuffer): Promise<Uint8Array> {
  const body = new Response(bytes).body;
  if (!body) throw new Error("cell grid: empty response body");
  const stream = body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Lower-case hex SHA-256 of `bytes`. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Checks decoded planes against their descriptor and wraps them. */
export async function decodeCellGrid(meta: CellGridMeta, planes: Uint8Array): Promise<CellGrid> {
  const end = Math.max(...meta.levels.map((l) => l.offset + l.bytes));
  if (planes.length < end) {
    throw new Error(`cell grid is ${String(planes.length)} bytes; its levels need ${String(end)}`);
  }
  if (meta.sha256 !== undefined && (await sha256Hex(planes)) !== meta.sha256.toLowerCase()) {
    throw new Error("cell grid does not match its sha256");
  }
  return { meta, planes, stride: cellStride(meta.channels) };
}

/** Fetches a grid, `meta.uri` relative to its tileset, and checks its size and checksum. */
export async function loadCellGrid(tilesetUrl: string, meta: CellGridMeta): Promise<CellGrid> {
  const base = new URL(tilesetUrl, globalThis.location?.href);
  const response = await fetch(new URL(meta.uri, base).toString());
  if (!response.ok) throw new Error(`cell grid answered ${String(response.status)}`);
  return decodeCellGrid(meta, await gunzip(await response.arrayBuffer()));
}

/** Byte position of channel `name` within a cell, and its type; undefined when absent. */
export function channelAt(
  meta: Pick<CellGridMeta, "channels">,
  name: string,
): { at: number; type: CellChannelType } | undefined {
  let at = 0;
  for (const c of meta.channels) {
    if (c.name === name) return { at, type: c.type };
    at += CHANNEL_BYTES[c.type];
  }
  return undefined;
}

function readChannel(planes: Uint8Array, at: number, type: CellChannelType): number {
  const lo = planes[at] ?? 0;
  if (type === "u1") return lo;
  if (type === "i1") return lo >= 128 ? lo - 256 : lo;
  const v = lo | ((planes[at + 1] ?? 0) << 8);
  return type === "u2" ? v : v >= 32768 ? v - 65536 : v;
}

/** The level entry for lattice level `level`, if the grid has it. */
export function levelOf(grid: CellGrid, level: number): CellGridLevel | undefined {
  return grid.meta.levels.find((l) => l.level === level);
}

/** The value of each channel of the cell at lattice index `(i, j)` of `level`, or null. */
export function cellAtIndex(
  grid: CellGrid,
  level: number,
  i: number,
  j: number,
): Record<string, number> | null {
  const l = levelOf(grid, level);
  if (!l) return null;
  const ci = i - l.i0;
  const cj = j - l.j0;
  if (ci < 0 || cj < 0 || ci >= l.ni || cj >= l.nj) return null;
  const at = l.offset + (cj * l.ni + ci) * grid.stride;
  const out: Record<string, number> = {};
  let k = 0;
  for (const c of grid.meta.channels) {
    out[c.name] = readChannel(grid.planes, at + k, c.type);
    k += CHANNEL_BYTES[c.type];
  }
  return out;
}

/** The cell of `level` holding tileset-frame point `(x, y)` (metres), or null outside it. */
export function cellAt(
  grid: CellGrid,
  x: number,
  y: number,
  level: number,
): Record<string, number> | null {
  const cell = grid.meta.lattice.baseM * 2 ** level;
  return cellAtIndex(grid, level, Math.floor(x / cell), Math.floor(y / cell));
}

/**
 * Lattice levels above `cellM` a cell must climb to cover at least `minPx` pixels at
 * `distanceM`, seen at `pxPerRadian`: the smallest `k >= 0` with
 * `cellM * 2^k * pxPerRadian / distanceM >= minPx`. The shader's rule (`splatCellView.ts`):
 * near, the finest cells; far, coarser ones, as the tiles' level of detail coarsens.
 */
export function cellLevelFor(
  distanceM: number,
  pxPerRadian: number,
  cellM: number,
  minPx: number,
): number {
  const need = (minPx * Math.max(distanceM, 1e-3)) / (pxPerRadian * cellM);
  return need <= 1 ? 0 : Math.ceil(Math.log2(need));
}

/** What a confidence cell says under `weak-v1` (C3). */
export type CellVerdict = "none" | "weak" | "fair" | "strong";

/**
 * `weak-v1`: weak below the threshold, or unseen; a score of 0 without the unseen bit is no
 * evidence either way. Above the threshold, the upper half of the range is "strong".
 */
export function cellVerdict(score: number, reasons: number, meta: CellGridMeta): CellVerdict {
  const unseen = meta.reasonBits?.unseen ?? DEFAULT_UNSEEN_BIT;
  const weakBelow = meta.weakBelow ?? DEFAULT_WEAK_BELOW;
  if ((reasons & unseen) !== 0) return "weak";
  if (score === 0) return "none";
  if (score < weakBelow) return "weak";
  return score >= weakBelow + (256 - weakBelow) / 2 ? "strong" : "fair";
}

/**
 * Whether `grid` has evidence (a scored cell) under lattice cell `(i, j)` of `level`. Grids of
 * one frame nest (C1): a finer request reads its grid's containing cell, a coarser one any of
 * the cells it covers.
 */
export function evidenceAt(grid: CellGrid, level: number, i: number, j: number): boolean {
  const score = channelAt(grid.meta, "score");
  if (!score) return false;
  const levels = grid.meta.levels;
  const finest = levels[0];
  const coarsest = levels[levels.length - 1];
  if (!finest || !coarsest) return false;
  const scored = (l: CellGridLevel, ci: number, cj: number): boolean => {
    const a = ci - l.i0;
    const b = cj - l.j0;
    if (a < 0 || b < 0 || a >= l.ni || b >= l.nj) return false;
    return readChannel(grid.planes, l.offset + (b * l.ni + a) * grid.stride + score.at, "u1") > 0;
  };
  const at = levelOf(grid, level);
  if (at) return scored(at, i, j);
  if (level < finest.level) {
    const f = 2 ** (finest.level - level);
    return scored(finest, Math.floor(i / f), Math.floor(j / f));
  }
  const f = 2 ** (level - coarsest.level);
  for (let dj = 0; dj < f; dj += 1) {
    for (let di = 0; di < f; di += 1) {
      if (scored(coarsest, i * f + di, j * f + dj)) return true;
    }
  }
  return false;
}
