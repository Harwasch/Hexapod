/**
 * The fog's grid on the client: decoding the API's `LonLatGrid`s, finding the cell under a
 * point, and the texture the drape samples. Pure, so it is tested without a scene.
 *
 * Wire format (apps/api/app/schemas/grids.py): cells are equal steps of longitude and
 * latitude, row 0 is the northern row, column 0 the western one. Density is u8, 0..250 =
 * density x 250, 255 outside the land; best is u16 little-endian, the index of the record that
 * sets the cell, 65535 for none.
 */

import type { KnowledgeGrid, LonLatGrid } from "@twin/contracts";
import type { LonLat } from "@twin/geo";

import type { GridImage } from "@/cesium/GroundGridLayer";

export const OUTSIDE = 255;
export const NO_RECORD = 65535;

export interface DecodedGrid {
  west: number;
  south: number;
  east: number;
  north: number;
  columns: number;
  rows: number;
  cellM: number;
  /** Density x 250 per cell, row-major from the north-west; 255 outside. */
  density: Uint8Array;
  /** Index into the grid's records per cell; 65535 where no record covers it. */
  best: Uint16Array;
}

function bytesOf(base64: string): Uint8Array {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

/** The raw values of a grid: u8 as bytes, u16 read little-endian whatever the platform. */
export function rawValues(grid: LonLatGrid): Uint8Array | Uint16Array | Float32Array {
  const bytes = bytesOf(grid.data);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = grid.columns * grid.rows;
  if (grid.encoding === "u8") return bytes.subarray(0, count);
  if (grid.encoding === "u16") {
    const out = new Uint16Array(count);
    for (let i = 0; i < count; i++) out[i] = view.getUint16(i * 2, true);
    return out;
  }
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export function decodeGrid(read: Pick<KnowledgeGrid, "density" | "best">): DecodedGrid {
  const { density, best } = read;
  return {
    west: density.west,
    south: density.south,
    east: density.east,
    north: density.north,
    columns: density.columns,
    rows: density.rows,
    cellM: density.cellM,
    density: Uint8Array.from(rawValues(density)),
    best: Uint16Array.from(rawValues(best)),
  };
}

type GridFrame = Pick<DecodedGrid, "west" | "south" | "east" | "north" | "columns" | "rows">;

/** The cell holding a point, row 0 at the north; null outside the grid. */
export function cellAt(
  grid: GridFrame,
  longitude: number,
  latitude: number,
): { col: number; row: number } | null {
  const dlon = (grid.east - grid.west) / grid.columns;
  const dlat = (grid.north - grid.south) / grid.rows;
  const col = Math.floor((longitude - grid.west) / dlon);
  const row = Math.floor((grid.north - latitude) / dlat);
  if (col < 0 || row < 0 || col >= grid.columns || row >= grid.rows) return null;
  return { col, row };
}

/** A cell's outline, closed, counter-clockwise from its south-west corner. */
export function cellRing(grid: GridFrame, col: number, row: number): LonLat[] {
  const dlon = (grid.east - grid.west) / grid.columns;
  const dlat = (grid.north - grid.south) / grid.rows;
  const west = grid.west + col * dlon;
  const north = grid.north - row * dlat;
  const east = west + dlon;
  const south = north - dlat;
  return [
    { longitude: west, latitude: south },
    { longitude: east, latitude: south },
    { longitude: east, latitude: north },
    { longitude: west, latitude: north },
    { longitude: west, latitude: south },
  ];
}

/** Density 0..1 at a cell, or null outside the land. */
export function densityAt(grid: DecodedGrid, col: number, row: number): number | null {
  const value = grid.density[row * grid.columns + col];
  return value === undefined || value === OUTSIDE ? null : value / 250;
}

/**
 * The drape's texture: R = density (0..250 stretched to 0..255); G = inside the land (255) or
 * not (0), averaged with its four neighbours (an edge cell's own value beyond the grid), so
 * one bilinear read in the shader gives an edge softened over a cell and the 20 m staircase
 * does not show; B = 0, A = 255. Row 0 at the north, as the grid.
 */
export function densityImage(grid: DecodedGrid): GridImage {
  const { columns, rows } = grid;
  const count = columns * rows;
  const inside = (col: number, row: number) => {
    const c = Math.min(columns - 1, Math.max(0, col));
    const r = Math.min(rows - 1, Math.max(0, row));
    return (grid.density[r * columns + c] ?? OUTSIDE) !== OUTSIDE ? 255 : 0;
  };
  const data = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i++) {
    const value = grid.density[i] ?? OUTSIDE;
    const col = i % columns;
    const row = Math.floor(i / columns);
    data[i * 4] = value !== OUTSIDE ? Math.round((value * 255) / 250) : 0;
    data[i * 4 + 1] = Math.round(
      (inside(col, row) +
        inside(col - 1, row) +
        inside(col + 1, row) +
        inside(col, row - 1) +
        inside(col, row + 1)) /
        5,
    );
    data[i * 4 + 2] = 0;
    data[i * 4 + 3] = 255;
  }
  return { width: columns, height: rows, data };
}

/** True when the next grid is the same grid with fog gone somewhere: worth a dissolve. */
export function fogLifted(previous: DecodedGrid | null, next: DecodedGrid): boolean {
  if (
    previous?.columns !== next.columns ||
    previous.rows !== next.rows ||
    previous.west !== next.west ||
    previous.north !== next.north
  )
    return false;
  for (let i = 0; i < next.density.length; i++) {
    const before = previous.density[i] ?? OUTSIDE;
    const after = next.density[i] ?? OUTSIDE;
    if (before !== OUTSIDE && after !== OUTSIDE && after < before) return true;
  }
  return false;
}

/** A small stable number from a land id, so each land's fog has its own pattern. */
export function seedOf(id: string): number {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  return hash % 10_000;
}
