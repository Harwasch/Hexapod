/**
 * The web's `hexapod.cellgrid` reader (lib/cellGrid.ts) against the pipeline's bytes
 * (cross-language). `data/tiles/synthetic-yard-epochs` was written by tools/captures
 * (`synthetic_cell_grids.py`, through `cell_grid.pack`); `grids.json` holds each epoch's
 * descriptor, the file's sha256 and each level's sha256 of its plane as the pipeline lays it
 * out. The web fetches and gunzips the same files with its own decoder and must land on the
 * same planes, level by level, and call the same cells weak (apps/api's
 * `test_cell_grid_decode.py` is the API's half of this check).
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { cellAtIndex, cellGridMeta, cellVerdict, levelOf, loadCellGrid } from "@/lib/cellGrid";

const ROOT = resolve(process.cwd(), "../../data/tiles/synthetic-yard-epochs");

interface Fixture {
  descriptor: { summary: { weak: number; scored: number; unseen: number } };
  file: string;
  fileSha256: string;
  planes: { level: number; sha256: string }[];
}

const GRIDS = JSON.parse(readFileSync(resolve(ROOT, "grids.json"), "utf8")) as {
  epochs: Record<string, Fixture>;
};

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

afterEach(() => vi.unstubAllGlobals());

describe.each(["e1", "e2"])("the pipeline's %s grid", (epoch) => {
  const fixture = GRIDS.epochs[epoch];
  if (!fixture) throw new Error(`grids.json has no ${epoch}`);
  const file = readFileSync(resolve(ROOT, fixture.file));

  /** Served beside a tileset, as a root's `extras.confidence` names it. */
  async function load() {
    const meta = cellGridMeta({ confidence: fixture?.descriptor });
    if (!meta) throw new Error("the web refused the pipeline's descriptor");
    const fetched: string[] = [];
    vi.stubGlobal("fetch", (url: string) => {
      fetched.push(url);
      return Promise.resolve(new Response(new Uint8Array(file)));
    });
    const grid = await loadCellGrid(`/tiles/${epoch}/tileset.json`, meta);
    expect(fetched).toEqual([`${location.origin}/tiles/${epoch}/confidence.bin`]);
    return grid;
  }

  it("is the file grids.json describes", () => {
    expect(sha256(file)).toBe(fixture.fileSha256);
  });

  it("decodes to the pipeline's planes, level by level", async () => {
    // loadCellGrid also checks the descriptor's sha256 of the whole uncompressed stream.
    const grid = await load();
    expect(grid.meta.levels.map((l) => l.level)).toEqual(fixture.planes.map((p) => p.level));
    for (const { level, sha256: expected } of fixture.planes) {
      const l = levelOf(grid, level);
      if (!l) throw new Error(`no level ${String(level)}`);
      const plane = grid.planes.subarray(l.offset, l.offset + l.bytes);
      expect(sha256(plane), `level ${String(level)}`).toBe(expected);
    }
  });

  it("calls the summary's cells weak under weak-v1", async () => {
    const grid = await load();
    const finest = grid.meta.levels[0];
    if (!finest) throw new Error("no levels");
    let weak = 0;
    let covered = 0;
    for (let j = finest.j0; j < finest.j0 + finest.nj; j += 1) {
      for (let i = finest.i0; i < finest.i0 + finest.ni; i += 1) {
        const cell = cellAtIndex(grid, finest.level, i, j);
        const verdict = cellVerdict(cell?.score ?? 0, cell?.reasons ?? 0, grid.meta);
        if (verdict === "weak") weak += 1;
        if (verdict !== "none") covered += 1;
      }
    }
    const { summary } = fixture.descriptor;
    expect(weak).toBe(summary.weak);
    expect(covered).toBe(summary.scored + summary.unseen);
  });
});
