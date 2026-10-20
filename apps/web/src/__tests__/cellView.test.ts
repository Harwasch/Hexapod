/**
 * A splat model's Confidence / Data age view in the scene (cesium/cellViewSlot.ts and
 * cesium/cellView.ts): nothing fetched or joined to the shader on Photo; the grid fetched once
 * and the part installed when a view is chosen; renders asked for on every change; what was
 * found reported; and nothing left behind.
 */
import { gzipSync } from "node:zlib";

import { Event, type Cesium3DTileset } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { attachCellView } from "@/cesium/cellViewSlot";
import type { SplatCellView } from "@/cesium/splatCellView";
import { colorChainOf, type SplatVertexColor } from "@/cesium/splatColor";
import { encodeConfidenceGrid, REASON_BITS } from "@/dev/cellGridFixture";
import type { CellGridMeta } from "@/lib/cellGrid";
import { useViewMode } from "@/state/viewMode";

import { A1, A2, E1, E2 } from "./epochFixtures";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

let served: { meta: CellGridMeta; planes: Uint8Array };
let fetched: string[];

beforeEach(async () => {
  served = await encodeConfidenceGrid({
    level: 2,
    i0: 0,
    j0: 0,
    ni: 8,
    nj: 8,
    cell: (i) => (i < 4 ? { score: 50, reasons: REASON_BITS.spread } : { score: 230 }),
    coarser: 3,
  });
  fetched = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      fetched.push(url);
      if (url.endsWith("tileset.json")) {
        // A2's tileset, the later date: same frame, scored everywhere.
        return Promise.resolve(
          Response.json({ root: { transform: IDENTITY, extras: { confidence: served.meta } } }),
        );
      }
      return Promise.resolve(new Response(gzipSync(served.planes, { level: 9 })));
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  useViewMode.setState({ mode: "photo", cells: {} });
});

function scan(extras: unknown) {
  const primitive = {
    vertexColor: undefined as SplatVertexColor | undefined,
    _rootTransform: IDENTITY,
    isDestroyed: () => false,
  };
  const tileset = {
    root: { extras, computedTransform: IDENTITY, transform: IDENTITY },
    resource: { url: "https://tiles.example/a1/tileset.json" },
    tileLoad: new Event(),
    gaussianSplatPrimitive: undefined as typeof primitive | undefined,
  };
  return { tileset, primitive, asCesium: tileset as unknown as Cesium3DTileset };
}

const context = () => ({ epochs: [E1, E2], assets: [A1, A2] });
const parts = (p: { vertexColor?: SplatVertexColor }) =>
  colorChainOf(p)?.parts.map((part) => part.colorFunction) ?? [];

describe("on Photo", () => {
  it("loads nothing and joins nothing", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const load = vi.fn(() => import("@/cesium/cellView"));
    const requestRender = vi.fn();
    const dispose = attachCellView(asCesium, { requestRender }, A1.id, context, load);
    tileset.tileLoad.raiseEvent({});
    await Promise.resolve();
    expect(load).not.toHaveBeenCalled();
    expect(fetched).toEqual([]);
    expect(primitive.vertexColor).toBeUndefined();
    dispose();
  });
});

describe("Confidence, then Data age, then Photo", () => {
  it("fetches the grid once, installs the part when the primitive is there, and says so", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    const load = vi.fn(() => import("@/cesium/cellView"));
    const requestRender = vi.fn();
    const dispose = attachCellView(asCesium, { requestRender }, A1.id, context, load);
    useViewMode.getState().set("confidence");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({ kind: "ready", mode: "confidence" }),
    );
    expect(fetched).toEqual(["https://tiles.example/a1/confidence.bin"]);
    // No primitive yet: nothing to join. Its first tile brings it.
    expect(primitive.vertexColor).toBeUndefined();
    tileset.gaussianSplatPrimitive = primitive;
    requestRender.mockClear();
    tileset.tileLoad.raiseEvent({});
    expect(parts(primitive)).toEqual(["splatCellViewColor"]);
    expect(requestRender).toHaveBeenCalled();

    // Data age: the later date's tileset and grid, compared in the same frame.
    useViewMode.getState().set("data-age");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toMatchObject({
        kind: "ready",
        mode: "data-age",
        newer: { dates: 1, compared: 1, cells: 64, skipped: [] },
      }),
    );
    expect(fetched).toEqual([
      "https://tiles.example/a1/confidence.bin",
      "https://tiles.example/a2/tileset.json",
      "https://tiles.example/a2/confidence.bin",
    ]);
    const part = colorChainOf(primitive)?.parts[0] as SplatCellView | undefined;
    expect(part?.mode).toBe("data-age");
    expect(part?.ready("data-age")).toBe(true);

    // Back to Confidence: uniforms only, nothing fetched again.
    useViewMode.getState().set("confidence");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({ kind: "ready", mode: "confidence" }),
    );
    expect(part?.mode).toBe("confidence");
    expect(fetched).toHaveLength(3);

    // Photo: out of the chain, a render asked for.
    requestRender.mockClear();
    useViewMode.getState().set("photo");
    expect(primitive.vertexColor).toBeUndefined();
    expect(requestRender).toHaveBeenCalled();
    expect(load).toHaveBeenCalledTimes(1);

    dispose();
    expect(useViewMode.getState().cells[A1.id]).toBeUndefined();
  });

  it("says a model without a grid was not measured, and installs nothing", async () => {
    const { asCesium, tileset, primitive } = scan({ viewCones: {} });
    tileset.gaussianSplatPrimitive = primitive;
    useViewMode.getState().set("confidence");
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({ kind: "not-measured" }),
    );
    expect(primitive.vertexColor).toBeUndefined();
    expect(fetched).toEqual([]);
    dispose();
  });

  it("says why a grid did not load", async () => {
    vi.stubGlobal("fetch", () => Promise.resolve(new Response("", { status: 503 })));
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    useViewMode.getState().set("confidence");
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({
        kind: "failed",
        message: "cell grid answered 503",
      }),
    );
    expect(primitive.vertexColor).toBeUndefined();
    dispose();
  });

  it("leaves nothing behind when the model goes while its grid loads", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    useViewMode.getState().set("confidence");
    dispose();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(primitive.vertexColor).toBeUndefined();
    expect(useViewMode.getState().cells[A1.id]).toBeUndefined();
  });
});
