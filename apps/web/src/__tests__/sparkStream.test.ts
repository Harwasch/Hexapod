/**
 * One tile into Spark (view/sparkStream.ts `loadSplatTile`), as the scan viewer and the globe's
 * Spark renderer both load it: Spark's `SplatMesh` is its constructor-and-`initialized`
 * contract here (2.2), nothing else, so the test reads what the mesh was asked for and where it
 * was put.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import * as THREE from "three";
import type { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchSplatTileset,
  loadSplatTile,
  openScanStream,
  splatAntialiasing,
} from "@/view/sparkStream";
import { loadCollision } from "@/lib/collision";
import type { TileNode } from "@/view/tiles";

const made = vi.hoisted(() => [] as { options: Record<string, unknown> }[]);

vi.mock("@sparkjsdev/spark", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  class SplatMesh extends three.Object3D {
    readonly initialized: Promise<SplatMesh>;
    override dispose(): void {
      /* Test mesh owns no GPU resources. */
    }
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

describe("one scan's shared lifecycle", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });
  it.each(["ADD", "REPLACE"] as const)("removes and disposes a %s scan on stop", async (refine) => {
    const group = new THREE.Group();
    group.position.set(2, 3, 4);
    const root = tile("splat.glb", [10, 20, 30]);
    const mesh = new THREE.Object3D() as SplatMesh;
    mesh.position.set(...(root.offset ?? [0, 0, 0]));
    const dispose = vi.fn();
    mesh.dispose = dispose;
    const stream = openScanStream(
      { url: "/tileset.json", tree: { root, refine } },
      group,
      new THREE.PerspectiveCamera(),
      {} as THREE.WebGLRenderer,
      { lodSplatCount: 1000 } as SparkRenderer,
      { load: () => Promise.resolve(mesh) },
    );
    await stream.ready;
    expect(mesh.getWorldPosition(new THREE.Vector3()).toArray()).toEqual([12, 23, 34]);
    stream.stop();
    expect(group.children).toHaveLength(0);
    expect(dispose).toHaveBeenCalledOnce();
  });
  it("disposes a root finishing after the caller leaves", async () => {
    const mesh = new THREE.Object3D() as SplatMesh;
    const dispose = vi.fn();
    mesh.dispose = dispose;
    let finish!: (mesh: SplatMesh) => void;
    const group = new THREE.Group();
    const stream = openScanStream(
      { url: "/tileset.json", tree: { root: tile("root.glb"), refine: "REPLACE" } },
      group,
      new THREE.PerspectiveCamera(),
      {} as THREE.WebGLRenderer,
      { lodSplatCount: 1000 } as SparkRenderer,
      {
        load: () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      },
    );
    const rejected = expect(stream.ready).rejects.toMatchObject({ name: "AbortError" });
    stream.stop();
    finish(mesh);
    await rejected;
    expect(group.children).toHaveLength(0);
    expect(dispose).toHaveBeenCalledOnce();
  });
  it("uses opacity-compensated AA for trained AA splats and scan defaults", () => {
    expect(splatAntialiasing(true)).toEqual({ preBlurAmount: 0, blurAmount: 0.3 });
    expect(splatAntialiasing()).toEqual(splatAntialiasing(true));
    expect(splatAntialiasing(false)).toEqual({ preBlurAmount: 0.3, blurAmount: 0 });
  });
});

it("removes the lat=lon=height=0 globe placement while retaining tile-local offsets", async () => {
  // The bench packer's root maps local (x,y,z) to ECEF (6378137+z,x,y).
  // The reader removes precisely that placement: SPZ bytes remain solve-local.
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            root: {
              transform: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6378137, 0, 0, 1],
              content: { uri: "root.glb" },
              boundingVolume: { sphere: [1, 2, 3, 4] },
              children: [
                {
                  content: { uri: "child.glb" },
                  transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1],
                  boundingVolume: { sphere: [1, 2, 3, 4] },
                },
              ],
            },
          }),
        ),
      ),
    ),
  );
  try {
    const { tree } = await fetchSplatTileset("/tileset.json");
    expect(tree.root.bounds?.center).toEqual([1, 2, 3]);
    expect(tree.root.offset).toBeUndefined();
    expect(tree.root.children[0]?.offset).toEqual([10, 20, 30]);
    expect(tree.root.children[0]?.bounds?.center).toEqual([11, 22, 33]);
  } finally {
    vi.unstubAllGlobals();
  }
});

it("keeps relative GLB and collision requests on the API after a signed tileset redirect", async () => {
  const base = `${location.origin}/api/v1/captures/c/live/tiles/30/`;
  const manifest: unknown = JSON.parse(readFileSync(resolve(FAR, "tileset.json"), "utf8"));
  const fetched: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      fetched.push(url);
      const name = url.slice(base.length);
      const response =
        name === "tileset.json"
          ? new Response(JSON.stringify(manifest))
          : new Response(new Uint8Array(readFileSync(resolve(FAR, name))));
      // fetch has followed the API's 302 to a private bucket. Relative files must use
      // the original API URL so that each receives its own short-lived signature.
      Object.defineProperty(response, "url", { value: `https://private.example/${name}?signed=1` });
      Object.defineProperty(response, "redirected", { value: true });
      return Promise.resolve(response);
    }),
  );
  try {
    const { url, tree } = await fetchSplatTileset(`${base}tileset.json`);
    expect(url).toBe(`${base}tileset.json`);
    await loadSplatTile(url, tree.root);
    expect(tree.collision).toBeTruthy();
    if (tree.collision) await loadCollision(url, tree.collision);
    expect(fetched).toEqual([
      `${base}tileset.json`,
      new URL(tree.root.uri, base).href,
      `${base}collision.bin`,
    ]);
  } finally {
    vi.unstubAllGlobals();
  }
});

describe("full-resolution diagnostic stream", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });
  it("loads every leaf, even beyond the normal cap and below the screen-error target", async () => {
    vi.useFakeTimers();
    serve("splat.glb");
    const root = {
      ...tile("splat.glb"),
      children: [
        { ...tile("left.glb"), gaussians: 7_000_000 },
        { ...tile("right.glb"), gaussians: 7_000_000 },
      ],
    };
    const group = new THREE.Group();
    const stream = openScanStream(
      { url: "/tileset.json", tree: { root, refine: "REPLACE" } },
      group,
      new THREE.PerspectiveCamera(),
      { domElement: { clientHeight: 100 } } as THREE.WebGLRenderer,
      { lodSplatCount: 1 } as SparkRenderer,
      { fullResolution: true },
    );
    await stream.ready;
    expect(stream.settled).toBe(false);
    for (let i = 0; i < 8; i++) {
      await vi.advanceTimersByTimeAsync(160);
      stream.frame();
    }
    expect(stream.settled).toBe(true);
    expect(group.children).toHaveLength(2);
    expect(made.slice(-3).map((mesh) => mesh.options.lod)).toEqual([false, false, false]);
    stream.stop();
    expect(group.children).toHaveLength(0);
  });
});
