/**
 * A per-cell view rule on the scene (cesium/compositorSession.ts), with the splat primitives
 * faked down to what it reads: the frame anchor's primitive draws the other dates as
 * companions (their own primitives draw nothing), knows each slot's source, and keeps each
 * cell's owner; a date's inferred layers are sources of their own; the host's view cones are
 * off while the rule is on; Data age joins W4a's colour part; a Living Survey rig ends it,
 * saying why; and letting go puts every primitive back as it was.
 */
import { Event, Matrix4, type Cesium3DTileset } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { setCompanions } from "@/cesium/companions";
import {
  startCompositor,
  type CompositorContext,
  type CompositorSite,
} from "@/cesium/compositorSession";
import type { CellViewGpu } from "@/cesium/splatCellView";
import { colorChainOf } from "@/cesium/splatColor";
import type { CompositorGpu } from "@/cesium/splatCompositor";
import { addVisibilityPart, visibilityChainOf } from "@/cesium/splatVisibility";
import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { decodeCellGrid } from "@/lib/cellGrid";
import type { ViewRule } from "@/lib/compositor";
import { useSettings } from "@/state/settings";
import { useViewMode } from "@/state/viewMode";
import { useViewRule } from "@/state/viewRule";

import { A1, A2, E1, E2 } from "./epochFixtures";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const texture = () => ({
  copyFrom: () => undefined,
  isDestroyed: () => false,
  destroy: () => undefined,
});
const gpu: CompositorGpu = {
  createR8: texture,
  mat4: (values) => [...values],
  vec4: (x, y, z, w) => [x, y, z, w],
};
const cellGpu: CellViewGpu = {
  createRG: texture,
  mat4: (values) => [...values],
  vec4: (x, y, z, w) => [x, y, z, w],
};

function primitive() {
  return {
    incremental: true,
    companions: [] as Cesium3DTileset[],
    drawnBy: undefined as unknown,
    vertexVisibility: undefined as unknown,
    vertexColor: undefined as unknown,
    _rootTransform: IDENTITY,
    _tileSlots: undefined as
      Map<{ tileset: unknown }, { start: number; count: number }> | undefined,
    isDestroyed: () => false,
  };
}

const CONFIDENCE = { format: "hexapod.cellgrid", uri: "confidence.bin" };

function tileset(name: string, extras: Record<string, unknown> = {}) {
  return {
    name,
    root: { computedTransform: IDENTITY, transform: Matrix4.IDENTITY, extras },
    resource: { url: `https://tiles.example/${name}/tileset.json` },
    tileLoad: new Event(),
    splatIncremental: true,
    isDestroyed: () => false,
    gaussianSplatPrimitive: primitive() as ReturnType<typeof primitive> | undefined,
  };
}
type FakeTileset = ReturnType<typeof tileset>;

/** A 1 m grid over x in [0, 20), y in [0, 10), every cell scored. */
async function everywhere() {
  const { meta, planes } = await encodeConfidenceGrid({
    level: 2,
    i0: 0,
    j0: 0,
    ni: 20,
    nj: 10,
    cell: () => ({ score: 200 }),
  });
  return { meta, grid: await decodeCellGrid(meta, planes) };
}

const settle = () => new Promise((done) => setTimeout(done, 0));

async function harness(rule: ViewRule = "latest") {
  const { meta, grid } = await everywhere();
  // March has a grid; June (an upload) has none: its one leaf tile's box stands in.
  const march = tileset("march", { confidence: { ...meta, ...CONFIDENCE } });
  const june = tileset("june");
  const fetchJson = vi.fn(() =>
    Promise.resolve({
      root: {
        boundingVolume: { box: [5, 5, 0, 5, 0, 0, 0, 5, 0, 0, 0, 1] },
        children: [
          {
            boundingVolume: { box: [12, 5, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1] },
            content: { uri: "a.glb" },
          },
        ],
      },
    }),
  );
  const preUpdate = new Event();
  // SiteManager's record of the site, as far as the rule reads it.
  const entry: CompositorSite = {
    rule: { rule },
    representation: "gaussian-splat",
    epochs: [E1, E2],
    epochId: null,
    site: { assets: [A1, A2] },
    // June copies March's placement (SiteManager `followAnchor`).
    handles: new Map([
      [A1.id, { tileset: march as never }],
      [A2.id, { tileset: june as never, follows: A1.id }],
    ]),
  };
  const end = vi.fn();
  const context: CompositorContext = {
    scene: { preUpdate, requestRender: vi.fn() },
    entry,
    cesium: () => true,
    end,
  };
  const handle = startCompositor(context, {
    gpu,
    cellGpu,
    fetchJson,
    loadGrid: () => Promise.resolve(grid),
    now: () => Date.parse("2026-10-20T00:00:00Z"),
  });
  const frame = async () => {
    preUpdate.raiseEvent();
    await settle();
    preUpdate.raiseEvent();
  };
  return { march, june, handle, frame, end, entry, fetchJson };
}

const hostOf = (t: FakeTileset) => t.gaussianSplatPrimitive!;

afterEach(() => {
  useSettings.getState().set({ inferredStyle: "hide" });
  useViewRule.setState({ legend: null });
});

describe("a view rule's session", () => {
  it("draws every date through the frame anchor's primitive, and each cell from one date", async () => {
    useViewMode.getState().set("confidence");
    const { march, june, handle, frame } = await harness();
    // A date's own analytic view cannot read the other dates' cells: back to Photo.
    expect(useViewMode.getState().mode).toBe("photo");
    await frame();
    const host = hostOf(march);
    expect(host.companions).toEqual([june]);
    expect(hostOf(june).drawnBy).toBe(host);
    expect(visibilityChainOf(host as never)?.parts.map((p) => p.visibilityFunction)).toEqual([
      "splatCompositorVisibility",
    ]);
    // Each slot's source: the host's own 0, June's 1.
    host._tileSlots = new Map([
      [{ tileset: march }, { start: 0, count: 10 }],
      [{ tileset: june }, { start: 10, count: 5 }],
    ]);
    await frame();
    const parts = handle.compositor();
    expect(parts?.sources.sourceAt(3)).toBe(0);
    expect(parts?.sources.sourceAt(12)).toBe(1);
    // June's box (x 10-14, y 4-6) is June's; the rest of March's grid is March's.
    const plane = parts?.plane;
    const at = (x: number, y: number) =>
      plane?.owners[(Math.floor(y) - plane.j0) * plane.ni + (Math.floor(x) - plane.i0)];
    expect(at(12, 5)).toBe(1);
    expect(at(3, 3)).toBe(0);
    expect(useViewRule.getState().legend).toEqual({
      cellM: 1,
      dates: [
        { epochId: E1.id, coverage: "grid", cells: 192, ageBucket: 3 },
        { epochId: E2.id, coverage: "tile-bounds", cells: 8, ageBucket: 2 },
      ],
      inferredCells: 0,
      left: 0,
    });
    handle.dispose();
    expect(host.companions).toEqual([]);
    expect(hostOf(june).drawnBy).toBeUndefined();
    expect(host.vertexVisibility).toBeUndefined();
    expect(useViewRule.getState().legend).toBeNull();
  });

  it("as of the earlier date, draws only it", async () => {
    const { march, handle, frame, entry } = await harness();
    entry.epochId = E1.id;
    expect(handle.assets("gaussian-splat")?.map((a) => a.id)).toEqual([A1.id]);
    await frame();
    expect(hostOf(march).companions).toEqual([]);
    expect(handle.compositor()?.plane?.owners.includes(1)).toBe(false);
    handle.dispose();
  });

  it("takes a date's inferred layers as sources, and tints them under Highlight invented", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { march, june, handle, frame, entry } = await harness("highlight-invented");
    const layer = tileset("june-fill");
    await frame();
    const host = hostOf(march);
    // June's inferred layer follows its scan onto the host (inferredLayers.ts).
    setCompanions(host, Symbol("inferred"), [layer as never], june);
    await frame();
    expect(host.companions).toEqual([june, layer]);
    expect(handle.compositor()?.tinted).toBe(1 << 2);
    // The host's own layer carries its scan's highlight (inferredLayers.ts) under the viewer's
    // Highlight: the rule tints the other dates' layers then, never one twice.
    const own = tileset("march-fill");
    setCompanions(host, Symbol("inferred"), [own as never], march);
    useSettings.getState().set({ inferredStyle: "highlight" });
    await frame();
    expect(host.companions).toEqual([june, layer, own]);
    expect(handle.compositor()?.tinted).toBe(1 << 2);
    useSettings.getState().set({ inferredStyle: "show" });
    if (entry.rule) entry.rule.rule = "latest";
    await frame();
    expect(handle.compositor()?.tinted).toBe(0);
    if (entry.rule) entry.rule.rule = "highlight-invented";
    await frame();
    expect(handle.compositor()?.tinted).toBe((1 << 2) | (1 << 3));
    // Ground truth never draws them.
    if (entry.rule) entry.rule.rule = "ground-truth";
    await frame();
    expect(handle.compositor()?.plane?.owners.includes(2)).toBe(false);
    handle.dispose();
  });

  it("turns the host's view cones off while it is on, and back after", async () => {
    const { march, handle, frame } = await harness();
    const cones = {
      visibilityFunction: "splatViewConeVisibility",
      visibilityOrder: 10,
      enabled: true,
      addToShader: () => undefined,
    };
    addVisibilityPart(hostOf(march) as never, cones);
    await frame();
    expect(cones.enabled).toBe(false);
    handle.dispose();
    expect(cones.enabled).toBe(true);
  });

  it("colours by each cell's date under Data age, through the cell view's colour part", async () => {
    const { march, handle, frame, entry } = await harness("data-age");
    await frame();
    const host = hostOf(march);
    expect(colorChainOf(host as never)?.parts.map((p) => p.colorFunction)).toContain(
      "splatCellViewColor",
    );
    if (entry.rule) entry.rule.rule = "latest";
    await frame();
    expect(colorChainOf(host as never)?.parts.map((p) => p.colorFunction) ?? []).not.toContain(
      "splatCellViewColor",
    );
    handle.dispose();
  });

  it("ends, saying why, when a Living Survey rig takes a drawn date", async () => {
    const { june, handle, frame, end } = await harness();
    await frame();
    june.splatIncremental = false;
    await frame();
    expect(end).toHaveBeenCalledWith("living");
    handle.dispose();
  });

  it("waits for the host's primitive, and tells a date's primitive before it uploads", async () => {
    const { march, june, handle, frame } = await harness();
    const host = march.gaussianSplatPrimitive;
    march.gaussianSplatPrimitive = undefined;
    june.gaussianSplatPrimitive = undefined;
    await frame();
    expect(handle.compositor()).toBeUndefined();
    march.gaussianSplatPrimitive = host;
    await frame();
    // June's primitive is made with its first tile: the tile's load tells it at once.
    const own = primitive();
    june.gaussianSplatPrimitive = own;
    june.tileLoad.raiseEvent();
    expect(own.drawnBy).toBe(host);
    handle.dispose();
  });
});
