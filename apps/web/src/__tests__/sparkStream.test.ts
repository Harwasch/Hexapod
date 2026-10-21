/**
 * One tile into Spark (view/sparkStream.ts `loadSplatTile`), as the scan viewer and the globe's
 * Spark renderer both load it: Spark's `SplatMesh` is its constructor-and-`initialized`
 * contract here (2.2), nothing else, so the test reads what the mesh was asked for and where it
 * was put.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";

import { loadSplatTile } from "@/view/sparkStream";
import type { TileNode } from "@/view/tiles";

const made = vi.hoisted(() => [] as { options: Record<string, unknown> }[]);

vi.mock("@sparkjsdev/spark", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  class SplatMesh extends three.Object3D {
    readonly initialized: Promise<SplatMesh>;
    constructor(readonly options: Record<string, unknown>) {
      super();
      this.initialized = Promise.resolve(this);
      made.push(this);
    }
  }
  return { SplatMesh, SplatFileType: { SPZ: "spz" } };
});

const FAR = resolve(process.cwd(), "../../data/tiles/synthetic-far/splat");

function serve(name: string): void {
  const bytes = readFileSync(resolve(FAR, name));
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(new Uint8Array(bytes)))),
  );
}

const tile = (uri: string, offset?: [number, number, number]): TileNode => ({
  uri,
  gaussians: 950,
  geometricError: 0,
  children: [],
  ...(offset ? { offset } : {}),
});

describe("a tile into Spark", () => {
  afterEach(() => {
    made.length = 0;
    vi.unstubAllGlobals();
  });

  it("loadSplatTile places the mesh at the tile offset with 32-bit centres", async () => {
    serve("splat_4-4-2.glb");
    const { mesh, bytes } = await loadSplatTile(
      "https://tiles.example/far/tileset.json",
      tile("splat_4-4-2.glb", [2601, 1550, 0]),
    );
    expect(mesh.position.toArray()).toEqual([2601, 1550, 0]);
    expect(made[0]?.options).toMatchObject({ extSplats: true, lod: true, fileType: "spz" });
    // The SPZ handed over is the tile's own, gzip-framed, untouched.
    expect([bytes[0], bytes[1]]).toEqual([0x1f, 0x8b]);
  });

  it("leaves a tile with no frame of its own at the scan's origin, and asks for what it is told", async () => {
    serve("splat.glb");
    const { mesh } = await loadSplatTile(
      "https://tiles.example/far/tileset.json",
      tile("splat.glb"),
      {
        lod: false,
        extSplats: false,
      },
    );
    expect(mesh.position.toArray()).toEqual([0, 0, 0]);
    expect(made[0]?.options).toMatchObject({ extSplats: false, lod: false });
  });
});
