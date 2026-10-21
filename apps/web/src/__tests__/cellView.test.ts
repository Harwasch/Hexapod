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
import type { CellLookView } from "@/lib/cellLook";
import { useRendererOverride, useSettings } from "@/state/settings";
import { useViewMode } from "@/state/viewMode";

import { A1, A2, E1, E2, epoch, splat } from "./epochFixtures";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

let served: { meta: CellGridMeta; planes: Uint8Array };
let fetched: string[];

beforeEach(async () => {
  // CesiumJS draws the scan here (the part in its primitive is what these tests read).
  useSettings.getState().set({ splatRenderer: "cesium" });
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
  useSettings.getState().reset();
  useRendererOverride.setState({ renderer: null });
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

describe("for the renderer drawing the scan, when CesiumJS does not", () => {
  it("hands over what is drawn: Confidence's texture, Data age's once built, nothing on Photo or once gone", async () => {
    const { watchScanCellView } = await import("@/cesium/scanView/scanCellView");
    let last: CellLookView | null = null;
    const seen: (string | null)[] = [];
    const unwatch = watchScanCellView(A1.id, (view) => {
      last = view;
      seen.push(view?.mode ?? null);
    });
    // No CesiumJS primitive at all: a dedicated renderer draws the scan, CesiumJS's is hidden.
    const { asCesium } = scan({ confidence: served.meta });
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    const view = (): CellLookView | null => last;
    useViewMode.getState().set("confidence");
    await vi.waitFor(() => expect(view()?.mode).toBe("confidence"));
    expect(view()?.meta.levels).toEqual(served.meta.levels);
    expect(view()?.meta.weakBelow ?? 128).toBe(128);
    // The grid's own bytes: (score, reasons) of the first cell.
    expect(Array.from(view()?.rows.data.subarray(0, 2) ?? [])).toEqual([50, REASON_BITS.spread]);
    useViewMode.getState().set("data-age");
    await vi.waitFor(() => expect(view()?.mode).toBe("data-age"));
    // The later date has evidence everywhere: "newer data here" on the first cell.
    expect(view()?.rows.data[1]).toBe(255);
    useViewMode.getState().set("photo");
    expect(view()).toBeNull();
    useViewMode.getState().set("confidence");
    await vi.waitFor(() => expect(view()?.mode).toBe("confidence"));
    dispose();
    expect(view()).toBeNull();
    unwatch();
    // Data age never drew with Confidence's bytes: the photo until its own were built.
    expect(seen).toEqual([null, "confidence", null, "data-age", null, "confidence", null]);
  });
});

describe("on Photo", () => {
  it("loads nothing and joins nothing", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const load = vi.fn(() => import("@/cesium/cellView"));
    const requestRender = vi.fn();
    const dispose = attachCellView(asCesium, { requestRender }, A1.id, context, { load });
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
    const dispose = attachCellView(asCesium, { requestRender }, A1.id, context, { load });
    useViewMode.getState().set("confidence");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({
        kind: "ready",
        mode: "confidence",
        weakBelow: 128,
      }),
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
      expect(useViewMode.getState().cells[A1.id]).toEqual({
        kind: "ready",
        mode: "confidence",
        weakBelow: 128,
      }),
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

  it("says when the view's code did not load, and forgets it with the model", async () => {
    const { asCesium } = scan({ confidence: served.meta });
    const load = () => Promise.reject(new Error("chunk failed"));
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context, { load });
    useViewMode.getState().set("confidence");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toEqual({
        kind: "failed",
        message: "chunk failed",
      }),
    );
    dispose();
    expect(useViewMode.getState().cells[A1.id]).toBeUndefined();
  });

  it("builds Data age again when the site's dates change, so a newly published date counts", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    // Loaded with one date; June is published while Data age is on.
    let dates = { epochs: [E1], assets: [A1] };
    const listeners = new Set<() => void>();
    const onDatesChange = (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, () => dates, {
      onDatesChange,
    });
    useViewMode.getState().set("data-age");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toMatchObject({
        kind: "ready",
        newer: { dates: 0 },
      }),
    );
    dates = { epochs: [E1, E2], assets: [A1, A2] };
    for (const listener of listeners) listener();
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toMatchObject({
        kind: "ready",
        newer: { dates: 1, compared: 1, cells: 64 },
      }),
    );
    dispose();
    expect(listeners.size).toBe(0);
  });

  it("is drawn under every renderer but the WebGPU trial, as the control says, rigged model or not", async () => {
    // A rigged model is drawn by CesiumJS whatever the renderer (SiteManager `cesiumDraws`);
    // PlayCanvas and Spark draw the view too, so it stays; under the WebGPU trial the control
    // says Photo, so Photo is what it must show.
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    useViewMode.getState().set("confidence");
    await vi.waitFor(() => expect(parts(primitive)).toEqual(["splatCellViewColor"]));
    useSettings.getState().set({ splatRenderer: "playcanvas" });
    expect(parts(primitive)).toEqual(["splatCellViewColor"]);
    useSettings.getState().set({ splatRenderer: "playcanvas-webgpu" });
    expect(primitive.vertexColor).toBeUndefined();
    // A `?renderer=` choice counts as the renderer too.
    useRendererOverride.setState({ renderer: "spark" });
    await vi.waitFor(() => expect(parts(primitive)).toEqual(["splatCellViewColor"]));
    useRendererOverride.setState({ renderer: null });
    expect(primitive.vertexColor).toBeUndefined();
    dispose();
  });

  it("fetches and joins nothing under the WebGPU trial, and draws once the renderer can", async () => {
    useSettings.getState().set({ splatRenderer: "playcanvas-webgpu" });
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, context);
    useViewMode.getState().set("confidence");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(fetched).toEqual([]);
    expect(primitive.vertexColor).toBeUndefined();
    expect(useViewMode.getState().cells[A1.id]).toBeUndefined();
    useSettings.getState().set({ splatRenderer: "playcanvas" });
    await vi.waitFor(() => expect(parts(primitive)).toEqual(["splatCellViewColor"]));
    dispose();
  });

  it("says when the shown date's age is counted from its upload", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    const uploaded = { ...E1, observedAtSource: "created" } as typeof E1;
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, () => ({
      epochs: [uploaded],
      assets: [A1],
    }));
    useViewMode.getState().set("data-age");
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toMatchObject({
        kind: "ready",
        ageFromUpload: true,
      }),
    );
    dispose();
  });

  it("a comparison overtaken by a newer one never says what it found", async () => {
    const { asCesium, tileset, primitive } = scan({ confidence: served.meta });
    tileset.gaussianSplatPrimitive = primitive;
    // The first comparison's read of June's tileset is slow; the one after it is not.
    let release = (): void => undefined;
    const stub = vi.mocked(fetch);
    const answer = stub.getMockImplementation();
    let first = true;
    stub.mockImplementation((url) => {
      const reply = answer?.(url) ?? Promise.reject(new Error("no fetch"));
      const href = typeof url === "string" ? url : url instanceof URL ? url.href : url.url;
      if (!href.endsWith("a2/tileset.json") || !first) return reply;
      first = false;
      return new Promise((resolve) => (release = () => resolve(reply)));
    });
    let dates = { epochs: [E1, E2], assets: [A1, A2] };
    const listeners = new Set<() => void>();
    const dispose = attachCellView(asCesium, { requestRender: vi.fn() }, A1.id, () => dates, {
      onDatesChange: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    });
    useViewMode.getState().set("data-age");
    await vi.waitFor(() => expect(first).toBe(false));
    // September is published: compared again, and done first.
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    dates = { epochs: [E1, E2, E3], assets: [A1, A2, splat("a3", { epochId: "e3" })] };
    for (const listener of listeners) listener();
    await vi.waitFor(() =>
      expect(useViewMode.getState().cells[A1.id]).toMatchObject({ newer: { dates: 2 } }),
    );
    release();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(useViewMode.getState().cells[A1.id]).toMatchObject({ newer: { dates: 2 } });
    dispose();
  });
});
