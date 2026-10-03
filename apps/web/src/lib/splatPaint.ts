/**
 * Painting a selection on screen (`sceneSelect.ts`): which splats are under the brush and
 * front-most there. Every splat of the tiles drawn now (`splatPick.PickTile`) is projected
 * once (the view holds still while painting); the screen is cut into cells of `cellPx`, and
 * each cell keeps the nearest depth of its fairly solid splats -- the depth the renderer's
 * blend is decided by. A splat is visible when it is not much behind its cell's front, and
 * painted when it is visible and its cell is under a stroke.
 */

import type { PickTile } from "./splatPick";

/** Splats below this opacity do not set a cell's front (they still count as visible). */
export const SOLID_OPACITY = 0.35;
/** Splats below this opacity are ignored. */
export const MIN_OPACITY = 0.05;
/** A splat is front-most within this share of its cell's front depth, plus `DEPTH_SLACK_M`. */
export const DEPTH_TOLERANCE = 0.04;
export const DEPTH_SLACK_M = 0.12;

/** Every splat that lands on screen, projected. */
export interface ScreenSplats {
  cols: number;
  rows: number;
  cellPx: number;
  /** Per projected splat: which tile, which splat of it, its cell, depth and opacity. */
  tile: Uint32Array;
  index: Uint32Array;
  cell: Int32Array;
  depth: Float32Array;
  opacity: Float32Array;
  count: number;
  /** Per cell, the nearest solid depth (Infinity where none). */
  front: Float32Array;
}

/**
 * Projects `tiles` with `viewProj` (column-major 4x4, scan frame → clip space, w the view
 * depth) onto a `width` × `height` CSS-pixel screen.
 */
export function projectTiles(
  tiles: readonly PickTile[],
  viewProj: ArrayLike<number>,
  width: number,
  height: number,
  cellPx = 4,
  include?: (tile: number, index: number) => boolean,
): ScreenSplats {
  const cols = Math.max(1, Math.ceil(width / cellPx));
  const rows = Math.max(1, Math.ceil(height / cellPx));
  let total = 0;
  for (const tile of tiles) total += tile.count;
  const out: ScreenSplats = {
    cols,
    rows,
    cellPx,
    tile: new Uint32Array(total),
    index: new Uint32Array(total),
    cell: new Int32Array(total),
    depth: new Float32Array(total),
    opacity: new Float32Array(total),
    count: 0,
    front: new Float32Array(cols * rows).fill(Infinity),
  };
  const m = Array.from({ length: 16 }, (_, i) => viewProj[i] ?? 0);
  let n = 0;
  for (let ti = 0; ti < tiles.length; ti++) {
    const tile = tiles[ti];
    if (!tile) continue;
    const { positions, opacity } = tile;
    for (let i = 0; i < tile.count; i++) {
      const a = opacity[i] ?? 1;
      if (a < MIN_OPACITY) continue;
      if (include && !include(ti, i)) continue;
      const x = positions[i * 3] ?? 0;
      const y = positions[i * 3 + 1] ?? 0;
      const z = positions[i * 3 + 2] ?? 0;
      const w = (m[3] ?? 0) * x + (m[7] ?? 0) * y + (m[11] ?? 0) * z + (m[15] ?? 0);
      if (w <= 1e-4) continue;
      const cx = ((m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z + (m[12] ?? 0)) / w;
      const cy = ((m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z + (m[13] ?? 0)) / w;
      if (cx < -1 || cx > 1 || cy < -1 || cy > 1) continue;
      const px = ((cx + 1) / 2) * width;
      const py = ((1 - cy) / 2) * height;
      const c =
        Math.min(rows - 1, Math.floor(py / cellPx)) * cols +
        Math.min(cols - 1, Math.floor(px / cellPx));
      out.tile[n] = ti;
      out.index[n] = i;
      out.cell[n] = c;
      out.depth[n] = w;
      out.opacity[n] = a;
      if (a >= SOLID_OPACITY && w < (out.front[c] ?? Infinity)) out.front[c] = w;
      n++;
    }
  }
  out.count = n;
  return out;
}

/** Per projected splat, 1 when it is front-most in its cell. */
export function visibleSplats(
  screen: ScreenSplats,
  tolerance = DEPTH_TOLERANCE,
  slack = DEPTH_SLACK_M,
): Uint8Array {
  const out = new Uint8Array(screen.count);
  for (let k = 0; k < screen.count; k++) {
    const front = screen.front[screen.cell[k] ?? 0] ?? Infinity;
    const depth = screen.depth[k] ?? 0;
    out[k] = !Number.isFinite(front) || depth <= front * (1 + tolerance) + slack ? 1 : 0;
  }
  return out;
}

/** The painted cells: strokes of discs, added or taken away. */
export class BrushMask {
  readonly data: Uint8Array;
  constructor(
    readonly cols: number,
    readonly rows: number,
    readonly cellPx: number,
  ) {
    this.data = new Uint8Array(cols * rows);
  }

  clear(): void {
    this.data.fill(0);
  }

  /** A disc of `radius` CSS px at (`x`, `y`): painted (`value` 1) or erased (0). */
  stamp(x: number, y: number, radius: number, value: 0 | 1): void {
    const s = this.cellPx;
    const c0 = Math.max(0, Math.floor((x - radius) / s));
    const c1 = Math.min(this.cols - 1, Math.floor((x + radius) / s));
    const r0 = Math.max(0, Math.floor((y - radius) / s));
    const r1 = Math.min(this.rows - 1, Math.floor((y + radius) / s));
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const dx = (c + 0.5) * s - x;
        const dy = (r + 0.5) * s - y;
        if (dx * dx + dy * dy <= radius * radius) this.data[r * this.cols + c] = value;
      }
    }
  }

  /** Discs every `radius / 2` along the segment from (`x0`, `y0`) to (`x1`, `y1`). */
  line(x0: number, y0: number, x1: number, y1: number, radius: number, value: 0 | 1): void {
    const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / Math.max(1, radius / 2)));
    for (let i = 0; i <= steps; i++) {
      const f = i / steps;
      this.stamp(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, radius, value);
    }
  }

  get empty(): boolean {
    return !this.data.some((v) => v !== 0);
  }
}

/** Per projected splat, 1 when it is visible and under the mask. */
export function paintedSplats(
  screen: ScreenSplats,
  visible: Uint8Array,
  mask: BrushMask,
): Uint8Array {
  const out = new Uint8Array(screen.count);
  for (let k = 0; k < screen.count; k++) {
    out[k] = visible[k] && mask.data[screen.cell[k] ?? 0] ? 1 : 0;
  }
  return out;
}
