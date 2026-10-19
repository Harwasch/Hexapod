/**
 * Show once, everywhere on the client: the similarity grid the API sends, the slider's rule,
 * the tally, the patch a tap protects and the drape's texture. Pure, so it is tested without
 * a scene.
 *
 * Wire format (`SignatureSimilarityRead.grid`, apps/api/app/schemas/land_similarity.py): a
 * u8 lon/lat grid, row 0 at the north; 0..250 = cosine x 250 (below 0 counts as 0), 255 =
 * outside the land or no full signature. The slider compares that number exactly as the
 * server's areas do (`raw_threshold` in apps/api/app/analysis/similarity.py), so the cells
 * lit, the tally and the areas a job gets agree.
 */

import type { LandArea, SignatureSimilarity } from "@twin/contracts";
import { boundsOf } from "@twin/geo";

import type { GridImage } from "@/cesium/GroundGridLayer";
import { rawValues } from "@/features/knowledge/grid";

export const NO_SCORE = 255;
export const SCORE_SCALE = 250;
/** The slider's range and its start: below 30% most of a land is alike, above 99% only the tap. */
export const ALIKE_MIN = 0.3;
export const ALIKE_MAX = 0.99;
export const ALIKE_DEFAULT = 0.8;
const ACRE_M2 = 4046.8564224;

export interface ScoreGrid {
  west: number;
  south: number;
  east: number;
  north: number;
  columns: number;
  rows: number;
  cellM: number;
  cellAreaM2: number;
  /** Quantized cosine per cell, row-major from the north-west; 255 for none. */
  scores: Uint8Array;
}

/**
 * A grid spacing as the card and the job's provenance write it: "10 m", or "11.9 m" where a
 * large land's grid was coarsened (`resolution_label` in apps/api/app/analysis/similarity.py).
 */
export function spacingLabel(metres: number): string {
  const rounded = Math.round(metres * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded} m` : `${rounded.toFixed(1)} m`;
}

/**
 * The spacing a signature run will have on a boundary before it is made: the request's, or
 * coarser where the land is more than `maxDimension - 6` cells across (`make_grid` in
 * apps/api/app/analysis/raster_grid.py). Estimated from the outline's extent; the server
 * measures it in a local projection, so a land right at the limit may differ by a hair.
 */
export function expectedSpacingM(
  boundary: LandArea["boundary"],
  request: { resolutionM: number; maxDimension: number },
): number {
  const { west, south, east, north } = boundsOf(boundary);
  const latitude = ((south + north) / 2) * (Math.PI / 180);
  const width = (east - west) * 111_320 * Math.cos(latitude);
  const height = (north - south) * 110_574;
  const span = Math.max(width, height);
  return Number.isFinite(span)
    ? Math.max(request.resolutionM, span / (request.maxDimension - 6))
    : request.resolutionM;
}

export function decodeScores(read: Pick<SignatureSimilarity, "grid" | "cellAreaM2">): ScoreGrid {
  const { grid } = read;
  return {
    west: grid.west,
    south: grid.south,
    east: grid.east,
    north: grid.north,
    columns: grid.columns,
    rows: grid.rows,
    cellM: grid.cellM,
    cellAreaM2: read.cellAreaM2,
    scores: Uint8Array.from(rawValues(grid)),
  };
}

/** The smallest quantized score the slider at `threshold` (0..1) lights. */
export function rawThreshold(threshold: number): number {
  return Math.max(0, Math.min(SCORE_SCALE, Math.ceil(threshold * SCORE_SCALE - 1e-9)));
}

/**
 * The drape's `threshold` uniform for the slider: half a step under the smallest lit score
 * (as a texel, raw / 255), so the GPU's comparison lights exactly the cells the tally counts.
 */
export function rampThreshold(threshold: number): number {
  return (rawThreshold(threshold) - 0.5) / 255;
}

export function isLit(score: number | undefined, raw: number): boolean {
  return score !== undefined && score !== NO_SCORE && score >= raw;
}

/**
 * The lit cells at a threshold, protected ones aside, and the protected ones on their own:
 * those lit at the threshold, the ground protection keeps out of this job. The server counts
 * the same cells (`land_similarity.zones`), so the card, the hatch and the job's note agree.
 */
export function litStats(
  grid: ScoreGrid,
  threshold: number,
  guarded: Uint8Array | null,
): { cells: number; acres: number; protectedCells: number; protectedAcres: number } {
  const raw = rawThreshold(threshold);
  let cells = 0;
  let held = 0;
  for (let i = 0; i < grid.scores.length; i++) {
    if (!isLit(grid.scores[i], raw)) continue;
    if (guarded?.[i]) held += 1;
    else cells += 1;
  }
  return {
    cells,
    acres: (cells * grid.cellAreaM2) / ACRE_M2,
    protectedCells: held,
    protectedAcres: (held * grid.cellAreaM2) / ACRE_M2,
  };
}

/** Cells connected to `start` (8-neighbours) that `member` accepts, `start` first. */
function flood(grid: ScoreGrid, start: number, member: (index: number) => boolean): number[] {
  if (start < 0 || start >= grid.scores.length || !member(start)) return [];
  const { columns, rows } = grid;
  const seen = new Uint8Array(grid.scores.length);
  const out: number[] = [];
  const stack = [start];
  seen[start] = 1;
  while (stack.length) {
    const index = stack.pop() ?? 0;
    out.push(index);
    const col = index % columns;
    const row = (index - col) / columns;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const r = row + dr;
        const c = col + dc;
        if (r < 0 || c < 0 || r >= rows || c >= columns) continue;
        const next = r * columns + c;
        if (seen[next] || !member(next)) continue;
        seen[next] = 1;
        stack.push(next);
      }
    }
  }
  return out;
}

/** The lit patch around a cell at a threshold (protected cells are not lit); [] if unlit. */
export function patchAt(
  grid: ScoreGrid,
  index: number,
  threshold: number,
  guarded: Uint8Array | null = null,
): number[] {
  const raw = rawThreshold(threshold);
  return flood(grid, index, (i) => !guarded?.[i] && isLit(grid.scores[i], raw));
}

/**
 * A protect tap: on protected ground still lit (hatched) it frees the protected patch it is
 * part of; on a lit patch it protects the whole patch; elsewhere nothing changes (null).
 */
export function toggleProtected(
  grid: ScoreGrid,
  guarded: Uint8Array | null,
  index: number,
  threshold: number,
): { mask: Uint8Array; change: "protected" | "freed"; cells: number } | null {
  const mask = guarded ? Uint8Array.from(guarded) : new Uint8Array(grid.scores.length);
  if (mask[index] && isLit(grid.scores[index], rawThreshold(threshold))) {
    const cells = flood(grid, index, (i) => mask[i] === 1);
    for (const i of cells) mask[i] = 0;
    return { mask, change: "freed", cells: cells.length };
  }
  const cells = patchAt(grid, index, threshold, mask);
  if (cells.length === 0) return null;
  for (const i of cells) mask[i] = 1;
  return { mask, change: "protected", cells: cells.length };
}

/** The protected cells as lon/lat rectangles (west, south, east, north), one per run along a row. */
export function protectedRectangles(grid: ScoreGrid, guarded: Uint8Array | null): number[][] {
  if (!guarded) return [];
  const dlon = (grid.east - grid.west) / grid.columns;
  const dlat = (grid.north - grid.south) / grid.rows;
  const out: number[][] = [];
  for (let row = 0; row < grid.rows; row++) {
    let start = -1;
    for (let col = 0; col <= grid.columns; col++) {
      const on = col < grid.columns && guarded[row * grid.columns + col] === 1;
      if (on && start < 0) start = col;
      if (!on && start >= 0) {
        const north = grid.north - row * dlat;
        out.push([grid.west + start * dlon, north - dlat, grid.west + col * dlon, north]);
        start = -1;
      }
    }
  }
  return out;
}

/**
 * The drape's texture (`GroundGridLayer` material "ramp"): R = the quantized score as is
 * (0..250, so a texel reads raw / 255, the scale `rampThreshold` uses), G = a scored cell,
 * B = protected, A = 255. Row 0 at the north, as the grid.
 */
export function rampImage(grid: ScoreGrid, guarded: Uint8Array | null): GridImage {
  const count = grid.columns * grid.rows;
  const data = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i++) {
    const score = grid.scores[i] ?? NO_SCORE;
    const scored = score !== NO_SCORE;
    data[i * 4] = scored ? score : 0;
    data[i * 4 + 1] = scored ? 255 : 0;
    data[i * 4 + 2] = scored && guarded?.[i] ? 255 : 0;
    data[i * 4 + 3] = 255;
  }
  return { width: grid.columns, height: grid.rows, data };
}

/** Acres for copy: one decimal under 10, whole above. */
export function formatAcres(acres: number): string {
  if (acres > 0 && acres < 0.1) return "<0.1";
  return acres < 10 ? acres.toFixed(1) : Math.round(acres).toLocaleString();
}
