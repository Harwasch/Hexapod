/**
 * Picking objects in a scan PlayCanvas draws (playcanvasBackend.ts `pickTiles`), whose tile
 * worker hands each tile over in Morton order (playcanvasTile.worker.ts) with `order` beside it:
 * slot `k` holds the tile's own splat `order[k]`. `instances.json` lists a tile's ids in the
 * tile's own order, and scene selection reads the id of a hit as `ids[hit.index]`
 * (SceneSelectController's `idOf`, `labelsNear`, the castRay fallback), so the pick copy must
 * be put back in the tile's own order -- read in Morton order, every pick named the object of
 * some other splat of the tile.
 *
 * PlayCanvas is faked (no WebGL in jsdom); the worker's columns are made by the worker's own
 * functions (lib/splatLayout.ts).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BackendHooks } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import { centreColumns, mortonOrder, reorderColumns } from "@/lib/splatLayout";
import { castRay, labelsNear, type PickTile } from "@/lib/splatPick";
import type { TileNode } from "@/view/tiles";

vi.mock("playcanvas", () => {
  class Handlers {
    readonly handlers = new Map<string, ((...args: unknown[]) => void)[]>();
    on(name: string, handler: (...args: unknown[]) => void): void {
      this.handlers.set(name, [...(this.handlers.get(name) ?? []), handler]);
    }
  }
  class Vec3 {
    x = 0;
    y = 0;
    z = 0;
    set(x = 0, y = 0, z = 0): this {
      this.x = x;
      this.y = y;
      this.z = z;
      return this;
    }
  }
  /** Column-major, as PlayCanvas's. */
  class Mat4 {
    data = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    set(values: number[]): this {
      this.data = values.slice();
      return this;
    }
    copy(other: Mat4): this {
      this.data = other.data.slice();
      return this;
    }
    transformPoint(p: Vec3, out: Vec3): Vec3 {
      const m = this.data;
      const { x, y, z } = p;
      return out.set(
        (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z + (m[12] ?? 0),
        (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z + (m[13] ?? 0),
        (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z + (m[14] ?? 0),
      );
    }
  }
  class Quat {
    static IDENTITY = new Quat();
    setFromMat4(): this {
      return this;
    }
  }
  class Entity {
    parent: { removeChild(e: Entity): void } | null = null;
    camera?: Record<string, unknown>;
    gsplat?: Record<string, unknown>;
    position: number[] = [0, 0, 0];
    constructor(readonly name = "") {}
    addComponent(type: string, data: Record<string, unknown>): void {
      if (type === "camera") this.camera = { ...data };
      if (type === "gsplat") this.gsplat = { ...data, setParameter: () => undefined };
    }
    setPosition = (): void => undefined;
    setLocalPosition = (...p: unknown[]): void => {
      this.position = p.length === 1 ? [0, 0, 0] : (p as number[]);
    };
    setLocalRotation = (): void => undefined;
    lookAt = (): void => undefined;
    destroy(): void {
      this.parent?.removeChild(this);
    }
  }
  class Application extends Handlers {
    readonly graphicsDevice = { isWebGPU: false, maxPixelRatio: 1, gl: null };
    readonly scene = Object.assign(new Handlers(), { gsplat: {} });
    readonly systems = { gsplat: new Handlers() };
    readonly root = {
      addChild: (e: Entity) => {
        e.parent = this.root;
      },
      removeChild: (e: Entity) => {
        e.parent = null;
      },
    };
    autoRender = true;
    requestAnimationFrame = (): void => undefined;
    setCanvasFillMode = (): void => undefined;
    setCanvasResolution = (): void => undefined;
    resizeCanvas = (): void => undefined;
    start = (): void => undefined;
    render = (): void => undefined;
    destroy = (): void => undefined;
  }
  class GSplatResource {
    destroy = (): void => undefined;
  }
  class GSplatData {
    activated = false;
  }
  class Color {
    readonly rgba = [0, 0, 0, 0];
  }
  return {
    Application,
    Entity,
    GSplatResource,
    GSplatData,
    Vec3,
    Mat4,
    Quat,
    Color,
    FILLMODE_NONE: "NONE",
    RESOLUTION_AUTO: "AUTO",
    ASPECT_AUTO: 0,
  };
});

/** A tile of `count` splats on a jittered 3D grid, in the tile's own (file) order. */
function ownTile(count: number) {
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const z = new Float32Array(count);
  let seed = 7;
  const next = (): number => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  for (let i = 0; i < count; i++) {
    // Far from sorted: the file's order walks the grid backwards, with jitter.
    const cell = count - 1 - i;
    x[i] = 100 + (cell % 8) * 0.5 + next() * 0.05;
    y[i] = 40 + (Math.floor(cell / 8) % 8) * 0.5 + next() * 0.05;
    z[i] = 2 + Math.floor(cell / 64) * 0.5 + next() * 0.05;
  }
  const scale = new Float32Array(count).fill(0.05);
  const opacity = new Float32Array(count).map((_, i) => 0.5 + (i % 5) * 0.1);
  // Ids in the tile's own order: a different object every few splats, some with none.
  const ids = new Uint32Array(count).map((_, i) => (i % 11 === 0 ? 0 : 1 + (i % 7)));
  return { x, y, z, scale, opacity, ids };
}

const OWN = ownTile(256);
const CHECKSUM = "fnv1a32:256:0000abcd";

/** The worker's answer for the tile (playcanvasTile.worker.ts), Morton order and all. */
function workerAnswer(id: number) {
  const columns: Record<string, Float32Array> = {
    x: OWN.x.slice(),
    y: OWN.y.slice(),
    z: OWN.z.slice(),
    scale_0: OWN.scale.slice(),
    scale_1: OWN.scale.slice(),
    scale_2: OWN.scale.slice(),
    opacity: OWN.opacity.slice(),
  };
  const origin = centreColumns(columns);
  const empty = new Float32Array(0);
  const order = mortonOrder(columns.x ?? empty, columns.y ?? empty, columns.z ?? empty);
  return {
    id,
    count: OWN.x.length,
    properties: reorderColumns(columns, order),
    origin,
    checksum: CHECKSUM,
    order,
  };
}

class FakeWorker {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  postMessage(message: { id: number }): void {
    queueMicrotask(() => this.onmessage?.({ data: workerAnswer(message.id) }));
  }
  terminate = (): void => undefined;
}

const tile = (uri: string) => ({ uri }) as unknown as TileNode;

async function backend() {
  const { createBackend } = await import("@/cesium/scanView/playcanvasBackend");
  const hooks: BackendHooks = {
    frameWanted: () => undefined,
    work: { run: (task: () => unknown) => Promise.resolve(task()) } as unknown as TileWork,
    maxShDegree: 0,
  };
  return createBackend(document.createElement("canvas"), 1_000_000, hooks);
}

/** The splat a ray straight down through splat `k` meets first. */
function pickDownThrough(tiles: readonly PickTile[], k: number) {
  const [x, y, z] = [OWN.x[k] ?? 0, OWN.y[k] ?? 0, OWN.z[k] ?? 0];
  return castRay(tiles, { origin: [x, y, z + 10], direction: [0, 0, -1] }, { pixelAngle: 1e-4 });
}

describe("picking in a scan PlayCanvas draws in Morton order", () => {
  beforeEach(() => vi.stubGlobal("Worker", FakeWorker));
  afterEach(() => vi.unstubAllGlobals());

  it("hands the tile back in its own order, so a pick names the right object", async () => {
    const answer = workerAnswer(0);
    // The worker really did reorder: the test means something.
    expect([...answer.order].some((from, slot) => from !== slot)).toBe(true);

    const b = await backend();
    const entity = await b.load("https://scan.test/tileset.json", tile("0.glb"));
    b.add(entity);
    const tiles = b.pickTiles?.() ?? [];
    expect(tiles).toHaveLength(1);
    const pick = tiles[0];
    if (!pick) throw new Error("no pick tile");
    expect(pick.checksum).toBe(CHECKSUM);
    expect(pick.count).toBe(256);
    // Splat i of the pick copy is splat i of the file: position, radius, opacity.
    for (let i = 0; i < 256; i++) {
      expect(pick.positions[i * 3]).toBeCloseTo(OWN.x[i] ?? 0, 4);
      expect(pick.positions[i * 3 + 1]).toBeCloseTo(OWN.y[i] ?? 0, 4);
      expect(pick.positions[i * 3 + 2]).toBeCloseTo(OWN.z[i] ?? 0, 4);
      expect(pick.radii[i]).toBeCloseTo(0.05, 6);
      expect(pick.opacity[i]).toBeCloseTo(OWN.opacity[i] ?? 0, 6);
    }

    // A click on the top splats: the hit is that splat, and its id is the file's.
    const idOf = (_tile: number, index: number): number => OWN.ids[index] ?? 0;
    let named = 0;
    for (let k = 0; k < 64; k++) {
      const front = pickDownThrough(tiles, k)[0];
      expect(front?.index, `splat ${String(k)}`).toBe(k);
      expect(idOf(0, front?.index ?? -1)).toBe(OWN.ids[k]);
      // Read in the worker's Morton order instead, the same index is mostly another splat
      // with another object: what the pick copy used to be.
      if (OWN.ids[answer.order[k] ?? 0] !== OWN.ids[k]) named += 1;
    }
    expect(named).toBeGreaterThan(32);
    // The labelled splats around a point are found by their own ids too (labelsNear).
    const near = labelsNear(tiles, [OWN.x[5] ?? 0, OWN.y[5] ?? 0, OWN.z[5] ?? 0], 0.01, idOf);
    expect([...near.keys()]).toEqual([OWN.ids[5]]);
    b.destroy();
  });

  it("makes the copy once, when a pick asks, and moves a placed tile only then", async () => {
    const b = await backend();
    const entity = await b.load("https://scan.test/tileset.json", tile("object.glb"));
    b.add(entity);
    const first = b.pickTiles?.()[0];
    expect(b.pickTiles?.()[0]).toBe(first);
    // A split object placed 2 m east, twice: the pick reads the last placement.
    const east = (m: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, m, 0, 0, 1];
    b.place?.(entity, east(1));
    b.place?.(entity, east(2));
    const placed = b.pickTiles?.()[0];
    expect(placed).not.toBe(first);
    expect(placed?.positions[0]).toBeCloseTo((OWN.x[0] ?? 0) + 2, 4);
    expect(placed?.positions[1]).toBeCloseTo(OWN.y[0] ?? 0, 4);
    expect(b.pickTiles?.()[0]).toBe(placed);
    // Back where it was decoded: the tile's own copy again.
    b.place?.(entity, null);
    expect(b.pickTiles?.()[0]).toBe(first);
    // A tile taken off the screen is not picked.
    b.remove(entity);
    expect(b.pickTiles?.()).toEqual([]);
    b.destroy();
  });
});
