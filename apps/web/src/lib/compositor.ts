/**
 * Per-cell view rules (design M4-6 W5b; ADR 0017 "Per-cell view rules"): which one of a site's
 * dates is drawn in each cell of the site lattice, so a view can show "the newest thing we
 * know, everywhere" instead of one capture at a time. Pure and Cesium-free: the scene
 * (`cesium/compositorSession.ts`) feeds it the dates and draws what it decides.
 *
 * **The plane.** One 2D grid on the site lattice (decision C1: cells of `0.25 m * 2^level`,
 * `i = floor(x / cell)` east, `j = floor(y / cell)` north, in the tileset frame, which every
 * date drawn with the site's one placement shares). Its level is the coarsest of the drawn
 * dates' finest levels -- the finest at which every date can say whether it saw a cell -- made
 * coarser while it would exceed `MAX_PLANE_CELLS`. Each cell holds one owner: a source index
 * (`cesium/splatSources.ts`), or `NO_OWNER`. One byte a cell is the invariant: a cell can never
 * have two owners, so no pixel blends two dates of one cell.
 *
 * **What a date covers.** A date with a confidence grid (ADR 0016, `extras.confidence`) covers a
 * cell where it scored it (score > 0: evidence, good or weak). A date without one -- a splat
 * upload, an older run -- covers the cells under its leaf tiles' boxes, which overstate what it
 * saw (a box is empty in places); the legend flags it `tile-bounds`. An inferred layer has no
 * evidence of its own, and covers its tiles' boxes too.
 *
 * **The rules.** Only dates observed at or before `asOf` (the date chosen in the capture-date
 * control, which reads "As of" while a rule is on) take part.
 * - `latest`: the newest observed date that covers the cell; else an older one; else, where
 *   inferred layers are drawn at all, the newest inferred layer that covers it.
 * - `ground-truth`: as `latest`, never an inferred layer.
 * - `highlight-invented`: as `latest`; the scene tints the inferred owners' splats.
 * - `data-age`: as `latest`; the scene colours each cell by its owner's age (`ownerAgeGrid`).
 */

import {
  evidenceAt,
  LATTICE_BASE_M,
  type CellGrid,
  type CellGridLevel,
  type CellGridMeta,
} from "./cellGrid";
import { ageBucket } from "./dataAge";

export type ViewRule = "latest" | "ground-truth" | "highlight-invented" | "data-age";

/** The rules as the control offers them. */
export const VIEW_RULES: readonly { value: ViewRule; label: string; hint: string }[] = [
  { value: "latest", label: "Latest", hint: "Each cell from the newest date that saw it" },
  { value: "ground-truth", label: "Ground truth", hint: "Latest, never anything inferred" },
  {
    value: "highlight-invented",
    label: "Highlight invented",
    hint: "Latest, inferred cells tinted",
  },
  { value: "data-age", label: "Data age", hint: "Latest, coloured by each cell's date" },
];

/** A cell no date owns: nothing is drawn there under a rule. */
export const NO_OWNER = 255;
/** The most cells the owner plane holds (its texture, `cesium/splatCompositor.ts`). */
export const MAX_PLANE_CELLS = 2 ** 22;
/** The plane's lattice level when no drawn date has a grid: 1 m cells. */
export const DEFAULT_PLANE_LEVEL = 2;

/**
 * Seam feather width, metres: a UX constant, not a measurement. A splat within it of an edge
 * where the next cell has another owner is drawn at up to half opacity, so a seam reads as a
 * soft line rather than a cut.
 */
export function featherWidth(cellM: number): number {
  return Math.min(cellM / 4, 0.5);
}

/** An axis-aligned box in the tileset frame (metres, x east, y north). */
export interface TileBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** One drawn tileset, as the rules see it. */
export interface CompositorSource {
  /** Its index in the primitive's source list (`cesium/splatSources.ts`). */
  source: number;
  epochId: string;
  /** When its date was observed (ISO 8601). */
  observedAt: string;
  /** An inferred layer: what an image model filled in, not observed. */
  inferred: boolean;
  /** Its confidence grid, when its root carries one. */
  grid?: CellGrid | null;
  /** Its leaf tiles' boxes: what it covers without a grid. */
  boxes?: readonly TileBox[] | null;
}

export type Coverage = "grid" | "tile-bounds" | "none";

/** How a source's coverage is read: from its grid, its tiles' bounds, or not at all (yet). */
export function coverageOf(source: CompositorSource): Coverage {
  if (source.grid && !source.inferred) return "grid";
  return source.boxes && source.boxes.length > 0 ? "tile-bounds" : "none";
}

/** The owner of every cell of the plane, and what each source took. */
export interface OwnerPlane {
  rule: ViewRule;
  lattice: { baseM: number; level: number };
  cellM: number;
  i0: number;
  j0: number;
  ni: number;
  nj: number;
  /** `nj * ni` bytes, row-major (j outer): the owner's source index, or `NO_OWNER`. */
  owners: Uint8Array;
  /** Per source taking part: how its coverage was read, and the cells it owns. */
  sources: { source: number; epochId: string; coverage: Coverage; cells: number }[];
}

function time(iso: string | null | undefined): number {
  const at = iso ? Date.parse(iso) : Number.NaN;
  return Number.isFinite(at) ? at : Number.NEGATIVE_INFINITY;
}

/** The plane's lattice level for these sources (see the file comment). */
export function planeLevel(sources: readonly CompositorSource[]): number {
  let level: number | undefined;
  for (const s of sources) {
    const finest = coverageOf(s) === "grid" ? s.grid?.meta.levels[0]?.level : undefined;
    if (finest !== undefined) level = Math.max(level ?? finest, finest);
  }
  return level ?? DEFAULT_PLANE_LEVEL;
}

/** A source's extent in the tileset frame, metres; null when it covers nothing known. */
function extentOf(source: CompositorSource): TileBox | null {
  const boxes: TileBox[] = [];
  if (coverageOf(source) === "grid") {
    for (const l of source.grid?.meta.levels ?? []) {
      boxes.push({
        minX: l.i0 * l.cellM,
        minY: l.j0 * l.cellM,
        maxX: (l.i0 + l.ni) * l.cellM,
        maxY: (l.j0 + l.nj) * l.cellM,
      });
    }
  } else if (source.boxes) boxes.push(...source.boxes);
  if (boxes.length === 0) return null;
  return {
    minX: Math.min(...boxes.map((b) => b.minX)),
    minY: Math.min(...boxes.map((b) => b.minY)),
    maxX: Math.max(...boxes.map((b) => b.maxX)),
    maxY: Math.max(...boxes.map((b) => b.maxY)),
  };
}

/** The cells of `source` at the plane's level, as a mask over the plane. */
function coverageMask(
  source: CompositorSource,
  level: number,
  plane: { i0: number; j0: number; ni: number; nj: number },
): Uint8Array {
  const { i0, j0, ni, nj } = plane;
  const mask = new Uint8Array(ni * nj);
  const cell = LATTICE_BASE_M * 2 ** level;
  const kind = coverageOf(source);
  if (kind === "grid" && source.grid) {
    const grid = source.grid;
    for (let cj = 0; cj < nj; cj += 1)
      for (let ci = 0; ci < ni; ci += 1)
        if (evidenceAt(grid, level, i0 + ci, j0 + cj)) mask[cj * ni + ci] = 1;
  } else if (kind === "tile-bounds") {
    // A cell is under a box when they overlap by more than an edge.
    for (const b of source.boxes ?? []) {
      const a0 = Math.max(0, Math.floor(b.minX / cell) - i0);
      const a1 = Math.min(ni, Math.ceil(b.maxX / cell) - i0);
      const c0 = Math.max(0, Math.floor(b.minY / cell) - j0);
      const c1 = Math.min(nj, Math.ceil(b.maxY / cell) - j0);
      for (let cj = c0; cj < c1; cj += 1) mask.fill(1, cj * ni + a0, cj * ni + Math.max(a0, a1));
    }
  }
  return mask;
}

/**
 * The owner plane of `rule` over `sources`, as of `asOf` (ISO 8601; null: every date).
 * `allowInferred` says whether inferred layers are drawn at all (the viewer's Inferred style
 * is not Hide); `ground-truth` never takes one either way.
 */
export function ownerPlane(
  rule: ViewRule,
  sources: readonly CompositorSource[],
  asOf: string | null,
  options: { allowInferred?: boolean } = {},
): OwnerPlane {
  const until = asOf === null ? Number.POSITIVE_INFINITY : time(asOf);
  const inferredOk = rule !== "ground-truth" && options.allowInferred !== false;
  const taking = sources.filter(
    (s) => time(s.observedAt) <= until && (inferredOk || !s.inferred) && s.source !== NO_OWNER,
  );
  let level = planeLevel(taking);
  const extents = taking.map(extentOf).filter((e): e is TileBox => e !== null);
  let bounds = { i0: 0, j0: 0, ni: 0, nj: 0 };
  for (;;) {
    const cell = LATTICE_BASE_M * 2 ** level;
    if (extents.length === 0) break;
    const i0 = Math.floor(Math.min(...extents.map((e) => e.minX)) / cell);
    const j0 = Math.floor(Math.min(...extents.map((e) => e.minY)) / cell);
    const ni = Math.max(1, Math.ceil(Math.max(...extents.map((e) => e.maxX)) / cell) - i0);
    const nj = Math.max(1, Math.ceil(Math.max(...extents.map((e) => e.maxY)) / cell) - j0);
    bounds = { i0, j0, ni, nj };
    if (ni * nj <= MAX_PLANE_CELLS) break;
    level += 1;
  }
  // Newest first; an observed date before any inferred layer.
  const order = [...taking].sort(
    (a, b) =>
      Number(a.inferred) - Number(b.inferred) ||
      time(b.observedAt) - time(a.observedAt) ||
      b.source - a.source,
  );
  const masks = order.map((s) => coverageMask(s, level, bounds));
  const owners = new Uint8Array(bounds.ni * bounds.nj).fill(NO_OWNER);
  const cells = new Map<number, number>();
  for (let c = 0; c < owners.length; c += 1) {
    const k = masks.findIndex((m) => m[c] === 1);
    const owner = order[k]?.source;
    if (owner === undefined) continue;
    owners[c] = owner;
    cells.set(owner, (cells.get(owner) ?? 0) + 1);
  }
  return {
    rule,
    lattice: { baseM: LATTICE_BASE_M, level },
    cellM: LATTICE_BASE_M * 2 ** level,
    ...bounds,
    owners,
    sources: taking.map((s) => ({
      source: s.source,
      epochId: s.epochId,
      coverage: coverageOf(s),
      cells: cells.get(s.source) ?? 0,
    })),
  };
}

/** The owner of plane cell `(ci, cj)` (plane-relative), `NO_OWNER` outside the plane. */
export function ownerAtIndex(plane: OwnerPlane, ci: number, cj: number): number {
  if (ci < 0 || cj < 0 || ci >= plane.ni || cj >= plane.nj) return NO_OWNER;
  return plane.owners[cj * plane.ni + ci] ?? NO_OWNER;
}

/** The owner of the cell holding tileset-frame point `(x, y)`. */
export function ownerAt(plane: OwnerPlane, x: number, y: number): number {
  return ownerAtIndex(
    plane,
    Math.floor(x / plane.cellM) - plane.i0,
    Math.floor(y / plane.cellM) - plane.j0,
  );
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * The opacity weight a splat of `source` at `(x, y)` gets under `plane` -- what the
 * compositor's visibility part computes (`cesium/splatCompositor.ts`): 0 unless its source owns
 * the cell; else `mix(0.5, 1, smoothstep(0, w, d))`, with `d` the distance to the nearest edge
 * of the cell whose 4-neighbour across it has another owner, `w` the feather width.
 */
export function compositorWeight(
  plane: OwnerPlane,
  source: number,
  x: number,
  y: number,
  featherM: number = featherWidth(plane.cellM),
): number {
  if (source === NO_OWNER) return 0;
  const gx = x / plane.cellM;
  const gy = y / plane.cellM;
  const fi = Math.floor(gx);
  const fj = Math.floor(gy);
  const at = (di: number, dj: number) =>
    ownerAtIndex(plane, fi - plane.i0 + di, fj - plane.j0 + dj);
  const owner = at(0, 0);
  if (owner !== source) return 0;
  const tx = (gx - fi) * plane.cellM;
  const ty = (gy - fj) * plane.cellM;
  let d = Number.POSITIVE_INFINITY;
  if (at(-1, 0) !== owner) d = Math.min(d, tx);
  if (at(1, 0) !== owner) d = Math.min(d, plane.cellM - tx);
  if (at(0, -1) !== owner) d = Math.min(d, ty);
  if (at(0, 1) !== owner) d = Math.min(d, plane.cellM - ty);
  return 0.5 + 0.5 * smoothstep(0, featherM, d);
}

/** A node of a tileset.json, as far as its boxes go. */
export interface TilesetNode {
  boundingVolume?: { box?: readonly number[]; sphere?: readonly number[] };
  transform?: readonly number[];
  content?: unknown;
  contents?: unknown;
  children?: readonly TilesetNode[];
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function multiply(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1)
    for (let r = 0; r < 4; r += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
      out[c * 4 + r] = sum;
    }
  return out;
}

/**
 * The boxes of a tileset's leaf tiles with content, in its root's frame (each child's
 * `transform` composed down from the root, the root's own left out: it is the frame), as
 * axis-aligned boxes in x and y. What a date without a grid covers (`tile-bounds`).
 */
export function leafTileBoxes(root: TilesetNode | undefined): TileBox[] {
  const out: TileBox[] = [];
  const walk = (node: TilesetNode, m: readonly number[]): void => {
    const children = node.children ?? [];
    if (children.length > 0) {
      for (const child of children)
        walk(child, child.transform?.length === 16 ? multiply(m, child.transform) : m);
      return;
    }
    if (node.content === undefined && node.contents === undefined) return;
    const box = node.boundingVolume?.box;
    const sphere = node.boundingVolume?.sphere;
    const corners: [number, number, number][] = [];
    if (box?.length === 12) {
      const at = (a: number, sx: number, sy: number, sz: number) =>
        (box[a] ?? 0) + sx * (box[3 + a] ?? 0) + sy * (box[6 + a] ?? 0) + sz * (box[9 + a] ?? 0);
      for (const sx of [-1, 1])
        for (const sy of [-1, 1])
          for (const sz of [-1, 1])
            corners.push([at(0, sx, sy, sz), at(1, sx, sy, sz), at(2, sx, sy, sz)]);
    } else if (sphere?.length === 4) {
      const [x, y, z, r] = sphere as [number, number, number, number];
      for (const s of [-1, 1]) corners.push([x + s * r, y + s * r, z]);
    } else return;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const [x, y, z] of corners) {
      xs.push((m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z + (m[12] ?? 0));
      ys.push((m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z + (m[13] ?? 0));
    }
    out.push({
      minX: Math.min(...xs),
      minY: Math.min(...ys),
      maxX: Math.max(...xs),
      maxY: Math.max(...ys),
    });
  };
  if (root) walk(root, IDENTITY);
  return out;
}

/**
 * Data age under a rule: the owner plane as a one-level cell grid, for W4a's colour part
 * (`cesium/splatCellView.ts`) to draw in its Data age mode. R = 1 + the age bucket of the
 * cell's owner (`lib/dataAge.ts`), 0 where no observed date owns it; G = 255 where a date
 * observed after `asOf` -- one the rule leaves out -- has evidence (striped, "newer data
 * here").
 */
export function ownerAgeGrid(
  plane: OwnerPlane,
  sources: readonly CompositorSource[],
  asOf: string | null,
  now: number,
): { grid: CellGrid; rg: Uint8Array } {
  const { ni, nj, i0, j0, cellM } = plane;
  const level: CellGridLevel = {
    level: plane.lattice.level,
    cellM,
    i0,
    j0,
    ni,
    nj,
    offset: 0,
    bytes: ni * nj,
  };
  const meta: CellGridMeta = {
    format: "hexapod.cellgrid",
    version: 1,
    kind: "owners",
    uri: "",
    frame: "tileset",
    lattice: plane.lattice,
    cellM,
    channels: [{ name: "score", type: "u1" }],
    levels: [level],
  };
  const bySource = new Map(sources.map((s) => [s.source, s]));
  const until = asOf === null ? Number.POSITIVE_INFINITY : time(asOf);
  const newer = sources.filter((s) => !s.inferred && time(s.observedAt) > until);
  const masks = newer.map((s) => coverageMask(s, plane.lattice.level, plane));
  const planes = new Uint8Array(ni * nj);
  const rg = new Uint8Array(ni * nj * 2);
  for (let c = 0; c < ni * nj; c += 1) {
    const index = plane.owners[c] ?? NO_OWNER;
    // A cell nobody owns is nobody's age, whatever else the sources list (a later date).
    const owner = index === NO_OWNER ? undefined : bySource.get(index);
    const bucket = owner && !owner.inferred ? ageBucket(owner.observedAt, now) : null;
    planes[c] = owner ? 1 : 0;
    rg[c * 2] = bucket === null ? 0 : bucket + 1;
    rg[c * 2 + 1] = masks.some((m) => m[c] === 1) ? 255 : 0;
  }
  return { grid: { meta, planes, stride: 1 }, rg };
}
