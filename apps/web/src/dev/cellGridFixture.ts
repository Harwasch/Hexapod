/**
 * A confidence grid (hexapod.cellgrid v1, `lib/cellGrid.ts`) built in TypeScript from the
 * container's specification (design M4-6 C2, C3), for the unit tests, the e2e routes and the
 * scene harness. The pipeline's own encoder (Python, `tools/captures`) lands separately; a
 * cross-language test then decodes its output with `lib/cellGrid.ts` against this one's rules.
 *
 * Deterministic: the same arguments give the same bytes and the same descriptor. The planes
 * are returned uncompressed; the caller gzips them (Node's zlib in tests, `CompressionStream`
 * in a page), since only the uncompressed stream is checksummed.
 *
 * Only tests and dev harnesses import this; nothing in the app does. Relative imports only:
 * the e2e specs load it in Node, outside Vite's `@/` alias.
 */

import {
  DEFAULT_UNSEEN_BIT,
  DEFAULT_WEAK_BELOW,
  sha256Hex,
  type CellGridLevel,
  type CellGridMeta,
} from "../lib/cellGrid";

/** One finest-level cell: score 0-255, reason bits, relief in quarter metres. */
export interface FixtureCell {
  score: number;
  reasons?: number;
  relief?: number;
}

export interface ConfidenceFixture {
  /** Lattice level of the finest cells (cell = 0.25 m x 2^level). */
  level: number;
  i0: number;
  j0: number;
  ni: number;
  nj: number;
  /** The cell at lattice index (i, j) of the finest level. */
  cell: (i: number, j: number) => FixtureCell;
  /** Coarser levels to build above the finest (C3: up to 8). */
  coarser?: number;
  uri?: string;
  alignmentCap?: Record<string, unknown>;
}

export const REASON_BITS = {
  views: 1,
  spread: 2,
  gsd: 4,
  accuracy: 8,
  unseen: DEFAULT_UNSEEN_BIT,
  alignment: 32,
} as const;

interface Plane {
  level: number;
  i0: number;
  j0: number;
  ni: number;
  nj: number;
  cells: Uint8Array;
}

/** C3's pyramid: score = min of scored children, reasons = OR, relief = max. */
function parentOf(child: Plane): Plane {
  const i0 = Math.floor(child.i0 / 2);
  const j0 = Math.floor(child.j0 / 2);
  const ni = Math.floor((child.i0 + child.ni - 1) / 2) - i0 + 1;
  const nj = Math.floor((child.j0 + child.nj - 1) / 2) - j0 + 1;
  const cells = new Uint8Array(ni * nj * 3);
  for (let cj = 0; cj < child.nj; cj += 1) {
    for (let ci = 0; ci < child.ni; ci += 1) {
      const from = (cj * child.ni + ci) * 3;
      const pi = Math.floor((child.i0 + ci) / 2) - i0;
      const pj = Math.floor((child.j0 + cj) / 2) - j0;
      const to = (pj * ni + pi) * 3;
      const score = child.cells[from] ?? 0;
      const current = cells[to] ?? 0;
      if (score > 0 && (current === 0 || score < current)) cells[to] = score;
      cells[to + 1] = (cells[to + 1] ?? 0) | (child.cells[from + 1] ?? 0);
      cells[to + 2] = Math.max(cells[to + 2] ?? 0, child.cells[from + 2] ?? 0);
    }
  }
  return { level: child.level + 1, i0, j0, ni, nj, cells };
}

/** The descriptor (root `extras.confidence`) and the uncompressed planes. */
export async function encodeConfidenceGrid(
  fixture: ConfidenceFixture,
): Promise<{ meta: CellGridMeta; planes: Uint8Array }> {
  const finest: Plane = {
    level: fixture.level,
    i0: fixture.i0,
    j0: fixture.j0,
    ni: fixture.ni,
    nj: fixture.nj,
    cells: new Uint8Array(fixture.ni * fixture.nj * 3),
  };
  const reasons: Record<string, number> = Object.fromEntries(
    Object.keys(REASON_BITS).map((name) => [name, 0]),
  );
  let scored = 0;
  let weak = 0;
  let unseen = 0;
  for (let cj = 0; cj < fixture.nj; cj += 1) {
    for (let ci = 0; ci < fixture.ni; ci += 1) {
      const c = fixture.cell(fixture.i0 + ci, fixture.j0 + cj);
      const at = (cj * fixture.ni + ci) * 3;
      const bits = c.reasons ?? 0;
      finest.cells[at] = c.score;
      finest.cells[at + 1] = bits;
      finest.cells[at + 2] = c.relief ?? 0;
      if (c.score > 0) scored += 1;
      const isUnseen = (bits & REASON_BITS.unseen) !== 0;
      if (isUnseen) unseen += 1;
      if (isUnseen || (c.score > 0 && c.score < DEFAULT_WEAK_BELOW)) weak += 1;
      for (const [name, bit] of Object.entries(REASON_BITS)) {
        if ((bits & bit) !== 0) reasons[name] = (reasons[name] ?? 0) + 1;
      }
    }
  }
  const planes: Plane[] = [finest];
  for (let k = 0; k < (fixture.coarser ?? 0); k += 1) {
    const last = planes[planes.length - 1];
    if (!last || (last.ni === 1 && last.nj === 1)) break;
    planes.push(parentOf(last));
  }
  const total = planes.reduce((sum, p) => sum + p.cells.length, 0);
  const bytes = new Uint8Array(total);
  const levels: CellGridLevel[] = [];
  let offset = 0;
  for (const p of planes) {
    bytes.set(p.cells, offset);
    levels.push({
      level: p.level,
      cellM: 0.25 * 2 ** p.level,
      i0: p.i0,
      j0: p.j0,
      ni: p.ni,
      nj: p.nj,
      offset,
      bytes: p.cells.length,
    });
    offset += p.cells.length;
  }
  const cells = fixture.ni * fixture.nj;
  const meta: CellGridMeta = {
    format: "hexapod.cellgrid",
    version: 1,
    kind: "confidence",
    uri: fixture.uri ?? "confidence.bin",
    frame: "tileset",
    lattice: { baseM: 0.25, level: fixture.level },
    cellM: 0.25 * 2 ** fixture.level,
    channels: [
      { name: "score", type: "u1" },
      { name: "reasons", type: "u1" },
      { name: "relief", type: "u1", unitM: 0.25 },
    ],
    levels,
    reasonBits: { ...REASON_BITS },
    policy: "weak-v1",
    weakBelow: DEFAULT_WEAK_BELOW,
    alignmentCap: fixture.alignmentCap ?? {
      uncertaintyM: 0,
      priorEpochs: 0,
      factor: 1,
      source: "site_context.json",
    },
    summary: {
      cells,
      scored,
      weak,
      unseen,
      weakShare: scored > 0 ? Math.round((weak / scored) * 1000) / 1000 : 0,
      reasons,
      metric: true,
    },
    sha256: await sha256Hex(bytes),
  };
  return { meta, planes: bytes };
}
