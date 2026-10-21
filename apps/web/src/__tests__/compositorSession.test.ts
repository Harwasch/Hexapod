/**
 * A per-cell view rule on the scene (cesium/compositorSession.ts), with the splat primitives
 * faked down to what it reads: the frame anchor's primitive draws the other dates as
 * companions (their own primitives draw nothing), knows each slot's source, and keeps each
 * cell's owner; a date's inferred layers are sources of their own; the host's view cones are
 * off while the rule is on; Data age joins W4a's colour part; a Living Survey rig ends it,
 * saying why; and letting go puts every primitive back as it was.
 */
import { Cartesian3, Event, Matrix4, type Cesium3DTileset } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { setCompanions } from "@/cesium/companions";
import {
  startCompositor,
  type CompositorContext,
  type CompositorSite,
} from "@/cesium/compositorSession";
import type { CellViewGpu } from "@/cesium/splatCellView";
import { colorChainOf } from "@/cesium/splatColor";
import { ruledObjects } from "@/cesium/companions";
import type { CompositorGpu } from "@/cesium/splatCompositor";
import { addVisibilityPart, visibilityChainOf } from "@/cesium/splatVisibility";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import { watchScanRule, type ScanRuleView } from "@/cesium/scanView/scanRule";
import type { SplatRendererKind } from "@/cesium/scanView/types";
import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { decodeCellGrid } from "@/lib/cellGrid";
import type { ViewRule } from "@/lib/compositor";
import { useAsCaptured } from "@/state/asCaptured";
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
const EVIDENCE = {
  kind: "inferred",
  filler: "fixture",
  views: 3,
  gaussians: 10,
  meanConfidence: 0.5,
};

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

async function harness(
  rule: ViewRule = "latest",
  options: { renderer?: SplatRendererKind; juneExtras?: Record<string, unknown> } = {},
) {
  const { meta, grid } = await everywhere();
  // March has a grid; June (an upload) has none: its one leaf tile's box stands in.
  const march = tileset("march", { confidence: { ...meta, ...CONFIDENCE } });
  const june = tileset("june", options.juneExtras);
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
  // Who draws splats. The scan PlayCanvas draws is the date chosen: June, the latest.
  const drawing: { renderer: SplatRendererKind } = { renderer: options.renderer ?? "cesium" };
  const context: CompositorContext = {
    scene: { preUpdate, requestRender: vi.fn() },
    entry,
    renderer: () => drawing.renderer,
    kit: RULE_KIT,
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
  return { march, june, handle, frame, end, entry, fetchJson, drawing };
}

/** The rule handed to the renderer drawing asset `assetId` now (scanView/scanRule.ts). */
function ruleFor(assetId: string): ScanRuleView | null {
  let view: ScanRuleView | null = null;
  watchScanRule(assetId, (v) => {
    view = v;
  })();
  return view;
}

const hostOf = (t: FakeTileset) => t.gaussianSplatPrimitive!;

afterEach(() => {
  useAsCaptured.getState().set(false);
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
      leftOut: [],
    });
    handle.dispose();
    expect(host.companions).toEqual([]);
    expect(hostOf(june).drawnBy).toBeUndefined();
    expect(host.vertexVisibility).toBeUndefined();
    expect(useViewRule.getState().legend).toBeNull();
  });

  it("draws each date's split objects whole, from the date owning the cell under each", async () => {
    const { march, june, handle, frame } = await harness();
    await frame();
    // Split objects are tilesets of their own beside their scan (splitObjects.ts): the host's
    // shader parts never see them, so the rule says which are drawn.
    // An object is where its declared origin is (extras.objects[].origin, at its pose): its
    // own frame's translation, as PlayCanvas reads it too (lib/compositor.ts `objectAnchor`)
    // -- not the middle of its bounds, which the package pads by three scales a splat.
    const object = (x: number, y: number, middle = x) =>
      ({
        root: { computedTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, 0, 1] },
        boundingSphere: { center: new Cartesian3(middle, y, 0.5) },
      }) as unknown as Cesium3DTileset;
    const inJunes = object(12, 5);
    const inMarchs = object(3, 3);
    expect(ruledObjects.get(march)?.(inMarchs)).toBe(true);
    expect(ruledObjects.get(march)?.(inJunes)).toBe(false);
    expect(ruledObjects.get(june)?.(inJunes)).toBe(true);
    expect(ruledObjects.get(june)?.(inMarchs)).toBe(false);
    // Near the seam at x = 10: its origin in March's cell, the middle of its bounds in June's.
    // March's copy is drawn, as under PlayCanvas (playcanvasRule.test.ts, the same case).
    const astride = object(9.9, 5, 10.3);
    expect(ruledObjects.get(march)?.(astride)).toBe(true);
    expect(ruledObjects.get(june)?.(astride)).toBe(false);
    // Outside the plane no date owns the cell: nothing is drawn there, as for the splats.
    expect(ruledObjects.get(march)?.(object(-40, 3))).toBe(false);
    handle.dispose();
    expect(ruledObjects.get(march)).toBeUndefined();
    expect(ruledObjects.get(june)).toBeUndefined();
  });

  it("an upload-dated date fills only where no dated one saw, and has no age in the key", async () => {
    const { march, handle, frame, entry } = await harness();
    // June's date is only its upload's: when it was captured is not known.
    entry.epochs = [E1, { ...E2, observedAtSource: "created" }];
    await frame();
    hostOf(march)._tileSlots = new Map([[{ tileset: march }, { start: 0, count: 10 }]]);
    await frame();
    const plane = handle.compositor()?.plane;
    const at = (x: number, y: number) =>
      plane?.owners[(Math.floor(y) - plane.j0) * plane.ni + (Math.floor(x) - plane.i0)];
    // March saw June's box too: March is drawn there, though June's upload is newer.
    expect(at(12, 5)).toBe(0);
    expect(useViewRule.getState().legend?.dates).toEqual([
      { epochId: E1.id, coverage: "grid", cells: 200, ageBucket: 3 },
      { epochId: E2.id, coverage: "tile-bounds", cells: 0, ageBucket: null },
    ]);
    handle.dispose();
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

  it("under Latest, gives the viewer's own Highlight to the other dates' inferred layers", async () => {
    // Their scans' own highlight parts are not on the host (inferredLayers.ts): without the
    // rule's tint, Layers › Inferred › Highlight would do nothing to them under a rule.
    useSettings.getState().set({ inferredStyle: "highlight" });
    const { march, june, handle, frame } = await harness("latest");
    await frame();
    const host = hostOf(march);
    const layer = tileset("june-fill");
    const own = tileset("march-fill");
    setCompanions(host, Symbol("inferred"), [layer as never], june);
    setCompanions(host, Symbol("inferred"), [own as never], march);
    await frame();
    expect(host.companions).toEqual([june, layer, own]);
    // June's layer (source 2) tinted; March's own (3) keeps its scan's highlight part.
    expect(handle.compositor()?.tinted).toBe(1 << 2);
    useSettings.getState().set({ inferredStyle: "show" });
    await frame();
    expect(handle.compositor()?.tinted).toBe(0);
    handle.dispose();
  });

  it("names each date it leaves out, and why, the As of date too", async () => {
    const { handle, frame, entry } = await harness();
    // June is placed on its own (another frame): SiteManager decided it does not follow.
    const june = entry.handles.get(A2.id);
    (entry.handles as Map<string, unknown>).set(A2.id, { ...june, follows: undefined });
    await frame();
    expect(useViewRule.getState().legend?.dates.map((d) => d.epochId)).toEqual([E1.id]);
    expect(useViewRule.getState().legend?.leftOut).toEqual([
      { epochId: E2.id, reason: "frame", asOf: true },
    ]);
    // Still deciding (loading, or its placement being compared): said as such.
    (entry.handles as Map<string, unknown>).set(A2.id, {
      ...june,
      follows: undefined,
      followPending: true,
    });
    await frame();
    expect(useViewRule.getState().legend?.leftOut).toEqual([
      { epochId: E2.id, reason: "loading", asOf: true },
    ]);
    handle.dispose();
  });

  it("draws the rule chosen as Ground truth while As captured is on, and the rule after", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { march, june, handle, frame, entry } = await harness("data-age");
    await frame();
    const host = hostOf(march);
    const colours = () => colorChainOf(host as never)?.parts.map((p) => p.colorFunction) ?? [];
    expect(colours()).toContain("splatCellViewColor");
    useAsCaptured.getState().set(true);
    await frame();
    // No Data age colour, and the viewer's rule is still theirs.
    expect(colours()).not.toContain("splatCellViewColor");
    expect(entry.rule?.rule).toBe("data-age");
    useAsCaptured.getState().set(false);
    await frame();
    expect(colours()).toContain("splatCellViewColor");
    // Highlight invented: no inferred layer tinted while it is on.
    if (entry.rule) entry.rule.rule = "highlight-invented";
    setCompanions(host, Symbol("inferred"), [tileset("june-fill") as never], june);
    await frame();
    expect(handle.compositor()?.tinted).toBe(1 << 2);
    useAsCaptured.getState().set(true);
    await frame();
    expect(handle.compositor()?.tinted).toBe(0);
    useAsCaptured.getState().set(false);
    await frame();
    expect(handle.compositor()?.tinted).toBe(1 << 2);
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

describe("a view rule's session under PlayCanvas", () => {
  const MARCH = "https://tiles.example/march/tileset.json";
  const JUNE = "https://tiles.example/june/tileset.json";

  it("hands the renderer the plane and each date's source, and touches no primitive", async () => {
    useViewMode.getState().set("confidence");
    const { march, june, handle, frame } = await harness("latest", { renderer: "playcanvas" });
    expect(useViewMode.getState().mode).toBe("photo");
    await frame();
    // Nothing on CesiumJS's primitives: PlayCanvas draws every date in its one sort.
    expect(hostOf(march).companions).toEqual([]);
    expect(hostOf(june).drawnBy).toBeUndefined();
    expect(visibilityChainOf(hostOf(march) as never)?.parts ?? []).toEqual([]);
    expect(handle.compositor()).toBeUndefined();
    const view = ruleFor(A2.id);
    expect(view).not.toBeNull();
    expect(view?.sources.get(MARCH)).toEqual({ source: 0 });
    expect(view?.sources.get(JUNE)).toEqual({ source: 1 });
    // The scan PlayCanvas draws is June's; March is streamed beside it, where it is.
    expect(view?.native).toBe(1);
    expect(view?.dates).toEqual([
      { url: MARCH, assetId: A1.id, extras: march.root.extras, placement: null, root: IDENTITY },
    ]);
    // The same plane CesiumJS draws: June's box is June's, the rest March's.
    const [i0, j0, ni] = view?.plane ?? [0, 0, 0];
    const at = (x: number, y: number) =>
      view?.owners.data[(Math.floor(y) - j0) * ni + (Math.floor(x) - i0)];
    expect(at(12, 5)).toBe(1);
    expect(at(3, 3)).toBe(0);
    expect(view).toMatchObject({ cellM: 1, reach: 2, enabled: true, tinted: 0, age: null });
    expect(useViewRule.getState().legend?.dates.map((d) => [d.epochId, d.cells])).toEqual([
      [E1.id, 192],
      [E2.id, 8],
    ]);
    handle.dispose();
    expect(ruleFor(A2.id)).toBeNull();
    expect(useViewRule.getState().legend).toBeNull();
  });

  it("names each date's inferred layers and split objects by their tilesets, tints invented", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { handle, frame, entry } = await harness("highlight-invented", {
      renderer: "playcanvas",
      juneExtras: {
        inferredLayers: [{ uri: "fill/tileset.json", evidence: EVIDENCE }],
        objects: [{ uri: "objects/7/tileset.json", instance: 7, origin: [12, 5, 0] }],
      },
    });
    await frame();
    await frame();
    const view = ruleFor(A2.id);
    const fill = "https://tiles.example/june/fill/tileset.json";
    expect(view?.sources.get(fill)).toEqual({ source: 2 });
    expect(view?.sources.get("https://tiles.example/june/objects/7/tileset.json")).toEqual({
      source: 1,
      whole: true,
    });
    // Highlight invented: the inferred sources, by the layers' own look.
    expect(view?.tinted).toBe(1 << 2);
    if (entry.rule) entry.rule.rule = "latest";
    await frame();
    expect(ruleFor(A2.id)?.tinted).toBe(0);
    handle.dispose();
  });

  it("draws Data age from the plane's owners' ages", async () => {
    const { handle, frame, entry } = await harness("data-age", { renderer: "playcanvas" });
    await frame();
    expect(ruleFor(A2.id)?.age?.mode).toBe("data-age");
    if (entry.rule) entry.rule.rule = "latest";
    await frame();
    expect(ruleFor(A2.id)?.age).toBeNull();
    handle.dispose();
  });

  it("follows the renderer: CesiumJS's companions under CesiumJS, PlayCanvas's rule under it", async () => {
    const { march, june, handle, frame, drawing } = await harness("latest", {
      renderer: "playcanvas",
    });
    await frame();
    expect(ruleFor(A2.id)).not.toBeNull();
    drawing.renderer = "cesium";
    await frame();
    expect(ruleFor(A2.id)).toBeNull();
    expect(hostOf(march).companions).toEqual([june]);
    drawing.renderer = "playcanvas";
    await frame();
    expect(hostOf(march).companions).toEqual([]);
    expect(hostOf(june).drawnBy).toBeUndefined();
    expect(ruleFor(A2.id)?.sources.get(JUNE)).toEqual({ source: 1 });
    // A renderer that draws no rule: the rule cannot be drawn (SiteManager ends it, "changed").
    drawing.renderer = "spark";
    expect(handle.assets("gaussian-splat")).toBeNull();
    handle.dispose();
  });

  it("places a date in the scan's frame by the two tilesets' transforms", async () => {
    const { march, handle, frame } = await harness("latest", { renderer: "playcanvas" });
    // March's tiles sit 3 m east of June's frame on the globe.
    march.root.computedTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1];
    await frame();
    const date = ruleFor(A2.id)?.dates[0];
    expect(date?.placement).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]);
    expect(date?.root).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -3, 0, 0, 1]);
    handle.dispose();
  });
});
