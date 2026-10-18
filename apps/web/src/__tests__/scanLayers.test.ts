/**
 * An inferred layer drawn by the dedicated renderer that draws its scan (scanView/scanLayers.ts):
 * its tiles are the renderer's meshes, placed in the scan's frame and sorted with its splats;
 * they follow the fill pick and the Inferred style; and they look as CesiumJS draws them
 * (scanView/layerLook.ts): Highlight's purple, and the fade of their own view cones.
 */

import { afterEach, describe, expect, it } from "vitest";

import { inferredHighlightColor } from "@/lib/inferred";
import { OMNI, visibility, type ViewConesMeta } from "@/lib/viewCones";
import {
  coneEye,
  evaluateLayerLook,
  eyeMoved,
  layerCones,
  layerUniforms,
  LAYER_LOOK_GLSL,
  type LayerLook,
} from "@/cesium/scanView/layerLook";
import {
  layerPlacement,
  layerView,
  ScanLayers,
  type LayerBackend,
} from "@/cesium/scanView/scanLayers";
import { PLAYCANVAS_LAYER_MODIFIER } from "@/cesium/scanView/playcanvasBackend";
import { useSettings } from "@/state/settings";
import { useVariants } from "@/state/variants";
import type { View } from "@/view/stream";
import type { TileNode } from "@/view/tiles";

const EVIDENCE = {
  kind: "inferred",
  filler: "fixture",
  views: 3,
  gaussians: 10,
  meanConfidence: 0.5,
};

/** A scan's root transform somewhere on the globe (a rotation and an Earth-fixed origin). */
const SCAN_ROOT = [
  0.9327, -0.3606, 0, 0, 0.254, 0.6572, 0.7096, 0, -0.2559, -0.6619, 0.7046, 0, -1634618.7,
  -4228608.5, 4471328.9, 1,
];
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A one-tile layer at `root`. */
function layerJson(root: number[] = SCAN_ROOT, extras: Record<string, unknown> = {}) {
  return {
    asset: { version: "1.1" },
    geometricError: 0,
    root: {
      transform: root,
      refine: "REPLACE",
      geometricError: 0,
      boundingVolume: { box: [0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1] },
      content: { uri: "splat.glb" },
      extras: { gaussians: 100, evidence: EVIDENCE, ...extras },
    },
  };
}

const VIEW: View = {
  eye: [0, -10, 3],
  projection: 500,
  visible: () => true,
};

interface Mesh {
  uri: string;
}

/** A renderer that records what it is asked. */
function fakeBackend() {
  const calls: string[] = [];
  const looks = new Map<Mesh, LayerLook>();
  const placed = new Map<Mesh, readonly number[] | null>();
  const onScreen = new Set<Mesh>();
  const backend: LayerBackend<Mesh> = {
    loadLayer: (url, tile: TileNode) => {
      calls.push(`load ${new URL(tile.uri, url).pathname}`);
      return Promise.resolve({ uri: tile.uri });
    },
    add: (mesh) => {
      calls.push(`add ${mesh.uri}`);
      onScreen.add(mesh);
    },
    remove: (mesh) => {
      calls.push(`remove ${mesh.uri}`);
      onScreen.delete(mesh);
    },
    dispose: (mesh) => {
      calls.push(`dispose ${mesh.uri}`);
    },
    place: (mesh, matrix) => {
      placed.set(mesh, matrix);
    },
    setLayerLook: (mesh, look) => {
      looks.set(mesh, look);
    },
  };
  return { backend, calls, looks, placed, onScreen };
}

async function flush(): Promise<void> {
  for (let i = 0; i < 6; i += 1) await new Promise((r) => setTimeout(r, 0));
}

const EXTRAS = {
  inferredLayers: [{ uri: "inferred/tileset.json", evidence: EVIDENCE }],
  variants: {
    fill: [
      {
        name: "top",
        label: "Top",
        about: "A rebuilt top.",
        inferredLayers: [{ uri: "variants/fill/top/tileset.json", evidence: EVIDENCE }],
      },
    ],
  },
};

describe("an inferred layer drawn by the overlay's renderer", () => {
  afterEach(() => {
    useSettings.getState().set({ inferredStyle: "hide" });
    useVariants.getState().pick("layers-1", "fill", null);
  });

  it("loads nothing while Inferred is Hide, and the pick's layer once it is shown", async () => {
    const { backend, calls, onScreen } = fakeBackend();
    const fetched: string[] = [];
    let arrivals = 0;
    useSettings.getState().set({ inferredStyle: "hide" });
    const layers = new ScanLayers(
      backend,
      "layers-1",
      "https://tiles.example/site/splat/tileset.json",
      EXTRAS,
      SCAN_ROOT,
      {
        arrived: () => {
          arrivals += 1;
        },
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: (url) => {
          fetched.push(new URL(url).pathname);
          return Promise.resolve(layerJson());
        },
      },
    );
    await flush();
    expect(fetched).toEqual([]);
    // Show: Today's layer, fetched, then its tile planned from the view and put on screen.
    useSettings.getState().set({ inferredStyle: "show" });
    await flush();
    expect(fetched).toEqual(["/site/splat/inferred/tileset.json"]);
    expect(arrivals).toBeGreaterThan(0);
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    expect(calls).toContain("load /site/splat/inferred/splat.glb");
    expect([...onScreen].map((m) => m.uri)).toEqual(["splat.glb"]);
    expect(layers.status()).toMatchObject({ wanted: 1, loaded: 1, drawn: 1, tiles: 1 });
    // Picking a method swaps the layer for its own.
    useVariants.getState().pick("layers-1", "fill", "top");
    await flush();
    expect(onScreen.size).toBe(0);
    expect(calls).toContain("dispose splat.glb");
    expect(fetched.at(-1)).toBe("/site/splat/variants/fill/top/tileset.json");
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    expect(onScreen.size).toBe(1);
    layers.stop();
    expect(onScreen.size).toBe(0);
  });

  it("Hide takes the tiles off screen and Show puts the same ones back; Highlight is the look", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { backend, calls, looks, onScreen } = fakeBackend();
    const layers = new ScanLayers(
      backend,
      "layers-2",
      "https://tiles.example/s/tileset.json",
      EXTRAS,
      SCAN_ROOT,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(layerJson()),
      },
    );
    await flush();
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    const [mesh] = [...onScreen];
    if (!mesh) throw new Error("no layer tile on screen");
    expect(looks.get(mesh)?.highlight).toBe(false);
    useSettings.getState().set({ inferredStyle: "highlight" });
    expect(looks.get(mesh)?.highlight).toBe(true);
    useSettings.getState().set({ inferredStyle: "hide" });
    expect(onScreen.size).toBe(0);
    expect(layers.status()).toMatchObject({ drawn: 0, tiles: 0, style: "hide" });
    const loads = calls.filter((c) => c.startsWith("load")).length;
    useSettings.getState().set({ inferredStyle: "show" });
    expect([...onScreen]).toEqual([mesh]);
    expect(looks.get(mesh)?.highlight).toBe(false);
    // Nothing fetched again.
    expect(calls.filter((c) => c.startsWith("load")).length).toBe(loads);
    layers.stop();
  });

  it("draws a layer whose root is the scan's where it was decoded, and another under S⁻¹·O", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    expect(layerPlacement(SCAN_ROOT, SCAN_ROOT)).toBeNull();
    const shifted = SCAN_ROOT.slice();
    // One metre east of the scan's origin, in the scan's own east.
    shifted[12] = (shifted[12] ?? 0) + (SCAN_ROOT[0] ?? 0);
    shifted[13] = (shifted[13] ?? 0) + (SCAN_ROOT[1] ?? 0);
    shifted[14] = (shifted[14] ?? 0) + (SCAN_ROOT[2] ?? 0);
    const placement = layerPlacement(SCAN_ROOT, shifted);
    expect(placement?.[12]).toBeCloseTo(1, 6);
    expect(placement?.[13]).toBeCloseTo(0, 6);
    expect(placement?.[0]).toBeCloseTo(1, 9);
    const { backend, placed, looks, onScreen } = fakeBackend();
    const layers = new ScanLayers(
      backend,
      "layers-3",
      "https://tiles.example/s/tileset.json",
      { inferredLayers: EXTRAS.inferredLayers },
      SCAN_ROOT,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(layerJson(shifted)),
      },
    );
    await flush();
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    const [mesh] = [...onScreen];
    if (!mesh) throw new Error("no layer tile on screen");
    expect(placed.get(mesh)?.[12]).toBeCloseTo(1, 6);
    // Its look takes the scan's frame back into the layer's.
    expect(looks.get(mesh)?.fromScan[12]).toBeCloseTo(-1, 6);
    layers.stop();
  });

  it("plans a placed layer from the view taken into its frame", () => {
    const placement = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1];
    const seen: number[][] = [];
    const view = layerView(
      {
        eye: [5, -10, 0],
        projection: 100,
        visible: (b) => {
          seen.push(b.center);
          return true;
        },
        centre: { forward: [0, 1, 0], halfDiagonal: 0.5 },
      },
      placement,
    );
    expect(view.eye).toEqual([0, -10, 0]);
    expect(view.centre?.forward).toEqual([0, 1, 0]);
    view.visible({ center: [0, 0, 0], radius: 1 });
    expect(seen).toEqual([[5, 0, 0]]);
    const same = { eye: [1, 2, 3], projection: 1, visible: () => true } as View;
    expect(layerView(same, null)).toBe(same);
  });

  it("a layer that does not load is reported, and the rest still draw", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { backend, onScreen } = fakeBackend();
    const layers = new ScanLayers(
      backend,
      "layers-4",
      "https://tiles.example/s/tileset.json",
      {
        inferredLayers: [
          { uri: "a/tileset.json", evidence: EVIDENCE },
          { uri: "b/tileset.json", evidence: EVIDENCE },
        ],
      },
      SCAN_ROOT,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: (url) =>
          url.includes("/b/")
            ? Promise.reject(new Error("answered 404"))
            : Promise.resolve(layerJson()),
      },
    );
    await flush();
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    expect(onScreen.size).toBe(1);
    expect(layers.status()).toMatchObject({ wanted: 2, loaded: 1, failed: 1, drawn: 1 });
    layers.stop();
  });

  it("hands its view cones to the look, loaded beside its tileset", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const meta: ViewConesMeta = {
      format: "hexapod.viewcones",
      version: 1,
      uri: "viewcones.bin",
      origin: [-1, -1, -1],
      cell: 1,
      dims: [2, 2, 2],
      fadeDeg: 20,
    };
    const { backend, looks, onScreen } = fakeBackend();
    const asked: string[] = [];
    const layers = new ScanLayers(
      backend,
      "layers-5",
      "https://tiles.example/s/tileset.json",
      { inferredLayers: EXTRAS.inferredLayers },
      SCAN_ROOT,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(layerJson(SCAN_ROOT, { viewCones: meta })),
        loadCones: (url, m) => {
          asked.push(`${new URL(url).pathname} ${m.uri}`);
          return Promise.resolve(new Uint8Array(8 * 4).fill(OMNI));
        },
      },
    );
    await flush();
    layers.update(VIEW);
    await flush();
    layers.update(VIEW);
    expect(asked).toEqual(["/s/inferred/tileset.json viewcones.bin"]);
    const [mesh] = [...onScreen];
    const look = mesh ? looks.get(mesh) : undefined;
    expect(look?.cones?.meta.dims).toEqual([2, 2, 2]);
    expect(look?.cones?.height).toBe(1);
    layers.stop();
  });
});

describe("a layer's look under the overlay's renderers", () => {
  const meta: ViewConesMeta = {
    format: "hexapod.viewcones",
    version: 1,
    uri: "viewcones.bin",
    origin: [0, 0, 0],
    cell: 1,
    dims: [1, 1, 1],
    fadeDeg: 20,
  };
  /** One cell seen from straight above only (axis down, a 10 degree half-angle). */
  const fromAbove = layerCones(meta, Uint8Array.of(128, 128, Math.round((10 * 254) / 180), 255));

  it("is as painted when shown, and Highlight's purple and hatching otherwise", () => {
    const look: LayerLook = { highlight: false, cones: null, fromScan: IDENTITY };
    const grey = [0.5, 0.5, 0.5, 1] as const;
    expect(evaluateLayerLook([0.2, 0.3, 0.4], grey, look, [0, 0, 10])).toEqual([0.5, 0.5, 0.5, 1]);
    const lit = evaluateLayerLook([0.2, 0.3, 0.4], grey, { ...look, highlight: true }, [0, 0, 10]);
    expect(lit).toEqual(inferredHighlightColor(grey, [0.2, 0.3, 0.4], true));
    expect(lit[2]).toBeGreaterThan(lit[1] + 0.2);
    // While one of the scan's objects is highlighted, the layer is dimmed with the rest (it
    // belongs to none), before Highlight's purple: CesiumJS's order.
    expect(evaluateLayerLook([0.2, 0.3, 0.4], grey, look, [0, 0, 10], [0.35, 0.5])).toEqual([
      0.175, 0.175, 0.175, 0.5,
    ]);
    expect(LAYER_LOOK_GLSL).toContain(
      "vec2 hexapodLayerDim(vec4 instanceParams, vec4 instanceDim)",
    );
  });

  it("fades by the layer's own cones, in the layer's frame", () => {
    // The octahedral axis (128, 128) is about +z: seen along it, i.e. from below looking up.
    const look: LayerLook = { highlight: false, cones: fromAbove, fromScan: IDENTITY };
    const below = evaluateLayerLook([0.5, 0.5, 0.5], [1, 1, 1, 1], look, [0.5, 0.5, -10]);
    const side = evaluateLayerLook([0.5, 0.5, 0.5], [1, 1, 1, 1], look, [10, 0.5, 0.5]);
    expect(below[3]).toBeCloseTo(1, 5);
    expect(side[3]).toBe(0);
    // The reference the shader follows.
    expect(side[3]).toBe(visibility([128, 128, 14, 255], [-1, 0, 0], 20));
    // A layer placed one metre east: the same splat and eye, moved with it, look the same.
    const moved: LayerLook = {
      ...look,
      fromScan: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1],
    };
    expect(evaluateLayerLook([1.5, 0.5, 0.5], [1, 1, 1, 1], moved, [11, 0.5, 0.5])).toEqual(side);
  });

  it("carries the eye into the layer's frame, and the fade only with cones", () => {
    const fromScan = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, -2, -3, 1];
    expect([...coneEye({ highlight: false, cones: fromAbove, fromScan }, [1, 2, 3])]).toEqual([
      0, 0, 0, 20,
    ]);
    expect(coneEye({ highlight: false, cones: null, fromScan }, [1, 2, 3])[3]).toBe(0);
    const uniforms = layerUniforms({ highlight: true, cones: fromAbove, fromScan });
    expect(uniforms.pattern[3]).toBe(1);
    expect([...uniforms.grid]).toEqual([0, 0, 0, 1]);
    expect(uniforms.dims[3]).toBe(4096);
  });

  it("works the fade out again only once the eye has moved enough to show", () => {
    expect(eyeMoved(null, [0, 0, 0], [0, 0, 0])).toBe(true);
    expect(eyeMoved([0, 0, 100], [0.5, 0, 100], [0, 0, 0])).toBe(false);
    expect(eyeMoved([0, 0, 100], [2, 0, 100], [0, 0, 0])).toBe(true);
    expect(eyeMoved([0, 0, 1], [0.03, 0, 1], [0, 0, 0])).toBe(true);
  });

  it("is one GLSL function both WebGL back-ends call", () => {
    expect(LAYER_LOOK_GLSL).toContain("vec4 hexapodLayerColor(vec3 scanPosition, vec4 color");
    expect(PLAYCANVAS_LAYER_MODIFIER.glsl).toContain("color = hexapodLayerColor(center, color");
    expect(PLAYCANVAS_LAYER_MODIFIER.glsl).toContain("void modifySplatCenter(inout vec3 center)");
    // GLSL only: the WebGPU trial draws a scan with inferred layers with WebGL2.
    expect(PLAYCANVAS_LAYER_MODIFIER.wgsl).toBeUndefined();
  });
});
