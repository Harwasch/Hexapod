import { describe, expect, it, vi } from "vitest";

vi.mock("@/cesium/scanView/playcanvasBackend", () => ({ createBackend: vi.fn() }));
import { cameraPath, fullDetail } from "@/dev/viewerPerf";
import { parseTileset } from "@/view/tiles";

const bounds = { center: [10, -20, 30] as [number, number, number], radius: 4 };

describe("offline viewer benchmark", () => {
  it("orbits fully, then dollies deterministically to 0.15 bounding radii", () => {
    const poses = cameraPath(bounds, 450, 150);
    expect(poses).toHaveLength(600);
    expect(poses).toEqual(cameraPath(bounds, 450, 150));
    const range = (i: number) => Math.hypot(...poses[i]!.eye.map((v, j) => v - bounds.center[j]!));
    expect(range(0)).toBeCloseTo(12);
    expect(poses[449]!.eye[0]).toBeCloseTo(poses[0]!.eye[0]);
    expect(poses[449]!.eye[1]).toBeCloseTo(poses[0]!.eye[1]);
    expect(poses[225]!.eye[0]).toBeLessThan(bounds.center[0]);
    expect(range(450)).toBeLessThan(range(449));
    expect(range(599)).toBeCloseTo(0.6);
    expect(poses.every((p) => p.pixelRatio === 1 && p.near < 0.6)).toBe(true);
  });

  it("rejects empty paths and degenerate bounds", () => {
    expect(() => cameraPath(bounds, 1, 150)).toThrow();
    expect(() => cameraPath(bounds, 450, 0)).toThrow();
    expect(() => cameraPath({ ...bounds, radius: 0 }, 450, 150)).toThrow();
  });

  it("loads leaves for REPLACE and every tile for ADD", () => {
    const root = {
      content: { uri: "parent.glb" },
      children: [{ content: { uri: "a.glb" } }, { content: { uri: "b.glb" } }],
    };
    expect(fullDetail(parseTileset({ root })).map((t) => t.uri)).toEqual(["a.glb", "b.glb"]);
    expect(
      fullDetail(parseTileset({ root: { ...root, refine: "ADD" } })).map((t) => t.uri),
    ).toEqual(["parent.glb", "a.glb", "b.glb"]);
  });
});

const engine = vi.hoisted(() => ({
  ready: null as
    null | ((camera: unknown, layer: unknown, ready: boolean, loading: number) => void),
}));
vi.mock("playcanvas", () => ({
  Application: {
    getApplication: () => ({
      systems: {
        gsplat: {
          on: (_event: string, fn: NonNullable<typeof engine.ready>) => {
            engine.ready = fn;
          },
        },
      },
    }),
  },
}));

it("waits for load, drawn confirmation and idle loading, then records every path frame", async () => {
  const { createBackend } = await import("@/cesium/scanView/playcanvasBackend");
  const { prepare } = await import("@/dev/viewerPerf");
  const mesh = { gsplat: { resource: { shBands: 3, numSplats: 10 } } };
  let draws = 0;
  let tick = 0;
  const backend = {
    load: vi.fn().mockResolvedValue(mesh),
    add: vi.fn(),
    destroy: vi.fn(),
    isDrawn: () => draws >= 2,
    render: vi.fn(() => {
      draws++;
      // A ready frame with outstanding loads must not release the wait.
      engine.ready?.(null, null, true, draws < 8 ? 1 : 0);
    }),
  };
  vi.mocked(createBackend).mockResolvedValue(
    backend as unknown as Awaited<ReturnType<typeof createBackend>>,
  );
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    queueMicrotask(() => callback(++tick * 2));
    return tick;
  });
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          root: {
            content: { uri: "leaf.glb" },
            extras: { gaussians: 10 },
            boundingVolume: { sphere: [0, 0, 0, 1] },
          },
        }),
    }),
  );
  const canvas = document.createElement("canvas");
  let timing = false;
  let disjoint = false;
  const gl = {
    QUERY_RESULT_AVAILABLE: 3,
    QUERY_RESULT: 4,
    getExtension: (name: string) =>
      name === "WEBGL_debug_renderer_info"
        ? { UNMASKED_RENDERER_WEBGL: 1 }
        : timing
          ? { TIME_ELAPSED_EXT: 2, GPU_DISJOINT_EXT: 5 }
          : null,
    getParameter: (key: number) => (key === 1 ? "NVIDIA test" : disjoint),
    isContextLost: () => false,
    createQuery: () => ({}),
    beginQuery: vi.fn(),
    endQuery: vi.fn(),
    deleteQuery: vi.fn(),
    getQueryParameter: (_query: unknown, key: number) => (key === 3 ? true : 2_000_000),
  };
  vi.spyOn(canvas, "getContext").mockReturnValue(gl as unknown as WebGL2RenderingContext);
  try {
    const harness = await prepare(canvas, {
      tilesetUrl: "http://localhost/perf/tileset.json",
      degree: 3,
      warmup: 2,
      orbit: 3,
      dolly: 1,
      timeoutMs: 10000,
    });
    expect(draws).toBe(14); // 7 loading, 5 consecutive idle, 2 warmup
    expect(createBackend).toHaveBeenCalledWith(
      canvas,
      10,
      expect.objectContaining({ maxShDegree: 3 }),
    );
    expect(harness.shDegree).toBe(3);
    const wall = await harness.measure();
    expect(wall.timestamps).toHaveLength(5);
    expect(wall.gpuMs).toBeNull();
    expect(draws).toBe(18);
    timing = true;
    expect((await harness.measure()).gpuMs).toEqual([2, 2, 2, 2]);
    expect(gl.beginQuery).toHaveBeenCalledTimes(4);
    expect(gl.endQuery).toHaveBeenCalledTimes(4);
    expect(gl.deleteQuery).toHaveBeenCalledTimes(4);
    disjoint = true;
    expect((await harness.measure()).gpuMs).toBeNull();
    expect(gl.deleteQuery).toHaveBeenCalledTimes(8);
    harness.destroy();
    expect(backend.destroy).toHaveBeenCalledOnce();
    mesh.gsplat.resource.shBands = 1;
    await expect(
      prepare(canvas, {
        tilesetUrl: "http://localhost/perf/tileset.json",
        degree: 3,
        warmup: 2,
        orbit: 3,
        dolly: 1,
        timeoutMs: 10000,
      }),
    ).rejects.toThrow("Decoded tile differs");
  } finally {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});
