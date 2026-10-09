import { describe, expect, it } from "vitest";

import { Event, Matrix4 } from "cesium";
import type { Cesium3DTileset, Scene } from "cesium";

import {
  attachInferredLayers,
  COMPANION_COLOR_GLSL,
  CompanionHighlight,
  drawsCompanions,
  evaluateInferredColor,
  InferredHighlight,
} from "@/cesium/inferredLayers";
import {
  colorChainOf,
  composeColorGlsl,
  SplatColorChain,
  type SplatVertexColor,
} from "@/cesium/splatColor";
import {
  describeEvidence,
  evidenceOf,
  INFERRED_HIGHLIGHT,
  inferredLayersOf,
  resolveLayerUrl,
} from "@/lib/inferred";
import { useInferred } from "@/state/inferred";
import { useSettings } from "@/state/settings";
import { useVariants } from "@/state/variants";

const EVIDENCE = {
  kind: "inferred",
  filler: "nvidia-fixer",
  views: 8,
  gaussians: 64681,
  meanConfidence: 0.676,
};

describe("inferred layers", () => {
  it("reads only what says it is inferred", () => {
    expect(evidenceOf({ evidence: EVIDENCE })?.filler).toBe("nvidia-fixer");
    expect(evidenceOf({ evidence: { ...EVIDENCE, kind: "measured" } })).toBeNull();
    expect(evidenceOf(undefined)).toBeNull();
    expect(evidenceOf({ evidence: { kind: "inferred", meanConfidence: 7 } })).toMatchObject({
      filler: "unknown",
      meanConfidence: 1,
    });
  });

  it("lists declared layers and skips malformed ones", () => {
    const layers = inferredLayersOf({
      inferredLayers: [
        { uri: "inferred/tileset.json", evidence: EVIDENCE },
        { uri: "", evidence: EVIDENCE },
        { uri: "x/tileset.json", evidence: { kind: "guess" } },
        "nonsense",
      ],
    });
    expect(layers.map((l) => l.uri)).toEqual(["inferred/tileset.json"]);
    expect(inferredLayersOf({})).toEqual([]);
  });

  it("resolves beside the measured tileset, keeping a signed query", () => {
    expect(
      resolveLayerUrl(
        "https://tiles.example/site/splat/tileset.json?sig=abc",
        "../inferred/tileset.json",
      ),
    ).toBe("https://tiles.example/site/inferred/tileset.json?sig=abc");
    expect(resolveLayerUrl("https://a.example/s/tileset.json", "https://b.example/t.json")).toBe(
      "https://b.example/t.json",
    );
  });

  it("says what painted it and that it is not measured", () => {
    const text = describeEvidence(evidenceOf({ evidence: EVIDENCE })!);
    expect(text).toContain("nvidia-fixer");
    expect(text).toContain("8 views");
    expect(text).toContain("68%");
    expect(text).toContain("Not measured");
  });

  it("draws a layer only while its scan is drawn and inferred layers are wanted", async () => {
    const parent = {
      root: { extras: { inferredLayers: [{ uri: "inferred/tileset.json", evidence: EVIDENCE }] } },
      resource: { url: "https://tiles.example/site/splat/tileset.json" },
      show: true,
      modelMatrix: Matrix4.fromTranslation({ x: 1, y: 2, z: 3 } as never),
      maximumScreenSpaceError: 16,
    } as unknown as Cesium3DTileset;
    const child = {
      root: { extras: { evidence: { ...EVIDENCE, views: 6 } } },
      show: false,
      modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
      tileLoad: new Event(),
      destroy: () => undefined,
    } as unknown as Cesium3DTileset;
    const added: unknown[] = [];
    const preUpdate = new Event();
    const scene = {
      primitives: { add: (p: unknown) => added.push(p), remove: () => true },
      preUpdate,
      requestRender: () => undefined,
    } as unknown as Scene;
    const urls: string[] = [];
    const dispose = attachInferredLayers(parent, scene, "asset-1", (url) => {
      urls.push(url);
      return Promise.resolve(child);
    });
    await new Promise((r) => setTimeout(r, 0));
    expect(urls).toEqual(["https://tiles.example/site/splat/inferred/tileset.json"]);
    expect(added).toEqual([child]);
    expect(useInferred.getState().layers["asset-1"]?.[0]?.views).toBe(6);
    // Hidden until a person chooses.
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    expect(Matrix4.equals(child.modelMatrix, parent.modelMatrix)).toBe(true);
    useSettings.getState().set({ inferredStyle: "hide" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    useSettings.getState().set({ inferredStyle: "highlight" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    (parent as { show: boolean }).show = false;
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    dispose();
    expect(useInferred.getState().layers["asset-1"]).toBeUndefined();
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("leaves the layer to the renderer that draws the scan: CesiumJS's copy only with its scan", async () => {
    const { parent, child, scene, preUpdate } = fixture();
    // Another renderer draws the scan (its tileset hidden): it draws the layer too
    // (scanView/scanLayers.ts), so CesiumJS's copy stays hidden, Show or not.
    (parent as { show: boolean }).show = false;
    const dispose = attachInferredLayers(parent, scene, "asset-2", () => Promise.resolve(child));
    await settle();
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    expect(useInferred.getState().layers["asset-2"]).toHaveLength(1);
    (parent as { show: boolean }).show = true;
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    dispose();
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("highlights through its own primitive's colour chain, switched by a uniform", async () => {
    const { parent, child, scene, preUpdate, primitive } = fixture();
    const dispose = attachInferredLayers(parent, scene, "asset-3", () => Promise.resolve(child));
    await settle();
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    const hook = colorChainOf(primitive as never)?.parts[0] as InferredHighlight | undefined;
    expect(hook).toBeInstanceOf(InferredHighlight);
    expect(hook?.on).toBe(false);
    useSettings.getState().set({ inferredStyle: "highlight" });
    preUpdate.raiseEvent();
    expect(hook?.on).toBe(true);
    // The uniform carries it: no new shader for a switch.
    const lines: string[] = [];
    const uniforms: Record<string, () => unknown> = {};
    const builder = {
      addUniform: () => undefined,
      addVertexLines: (l: string | readonly string[]) => void lines.push(String(l)),
    };
    (primitive.vertexColor as SplatColorChain | undefined)?.addToShader(builder, uniforms, {});
    expect(lines.join("\n")).toContain("vec4 splatInferredColor(");
    expect(lines.join("\n")).toContain("color = splatInferredColor(splatIndex, position, color);");
    expect((uniforms.u_inferredPattern?.() as { w: number }).w).toBe(1);
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    expect((uniforms.u_inferredPattern?.() as { w: number }).w).toBe(0);
    dispose();
    expect(primitive.vertexColor).toBeUndefined();
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("is drawn by the scan's own primitive, in its sort, while CesiumJS draws the scan", async () => {
    const { parent, child, scene, preUpdate, primitive } = fixture();
    const host = hostPrimitive();
    (parent as unknown as { gaussianSplatPrimitive: unknown }).gaussianSplatPrimitive = host;
    const dispose = attachInferredLayers(parent, scene, "asset-6", () => Promise.resolve(child));
    await settle();
    preUpdate.raiseEvent();
    // Hidden: nothing drawn, by either.
    expect(host.companions).toEqual([]);
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    expect(host.companions).toEqual([child]);
    expect((primitive as { drawnBy?: unknown }).drawnBy).toBe(host);
    // Highlight acts on the layer's slots of the scan's primitive, not on a measured splat.
    const companion = colorChainOf(host)?.parts.find((p) => p instanceof CompanionHighlight);
    expect(companion).toBeDefined();
    host._tileSlots = new Map([
      [{ tileset: parent }, { start: 0, count: 100 }],
      [{ tileset: child }, { start: 100, count: 20 }],
      [{ tileset: child }, { start: 120, count: 5 }],
    ]);
    expect(companion?.slots().map((v) => [v.x, v.y, v.z, v.w])[0]).toEqual([100, 125, 0, 0]);
    useSettings.getState().set({ inferredStyle: "highlight" });
    preUpdate.raiseEvent();
    expect(companion?.on).toBe(true);
    // Hide: the scan's primitive lets the layer go; the layer's own draws nothing either.
    useSettings.getState().set({ inferredStyle: "hide" });
    preUpdate.raiseEvent();
    expect(host.companions).toEqual([]);
    expect(child.show).toBe(false);
    expect((primitive as { drawnBy?: unknown }).drawnBy).toBeUndefined();
    expect(colorChainOf(host)).toBeUndefined();
    dispose();
  });

  it("draws itself, as before, beside a scan whose primitive cannot sort it in", async () => {
    const { parent, child, scene, preUpdate, primitive } = fixture();
    const host = hostPrimitive();
    host.incremental = false;
    (parent as unknown as { gaussianSplatPrimitive: unknown }).gaussianSplatPrimitive = host;
    const dispose = attachInferredLayers(parent, scene, "asset-7", () => Promise.resolve(child));
    await settle();
    useSettings.getState().set({ inferredStyle: "show" });
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    expect(host.companions).toEqual([]);
    expect((primitive as { drawnBy?: unknown }).drawnBy).toBeUndefined();
    dispose();
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("composes Highlight after the objects' highlight in one colour slot", () => {
    const chain = new SplatColorChain();
    const objects = {
      colorFunction: "splatInstanceColor",
      colorOrder: 0,
      addToShader: () => undefined,
    };
    chain.add(new CompanionHighlight());
    chain.add(objects);
    expect(chain.parts.map((p) => p.colorFunction)).toEqual([
      "splatInstanceColor",
      "splatCompanionColor",
    ]);
    expect(composeColorGlsl(chain.parts.map((p) => p.colorFunction))).toContain(
      "    color = splatInstanceColor(splatIndex, position, color);\n    color = splatCompanionColor(splatIndex, position, color);",
    );
    expect(COMPANION_COLOR_GLSL).toContain("uniform ivec4 u_companionSlots[4];");
    expect(drawsCompanions(undefined)).toBe(false);
  });

  it("turns an inferred splat purple and hatched, and leaves it as painted otherwise", () => {
    const grey = [0.5, 0.5, 0.5, 1] as const;
    expect(evaluateInferredColor(grey, [0, 0, 0], false)).toEqual([0.5, 0.5, 0.5, 1]);
    const lit = evaluateInferredColor(grey, [0, 0, 0], true);
    // Bluer and redder than green: purple.
    expect(lit[2]).toBeGreaterThan(lit[1] + 0.2);
    expect(lit[0]).toBeGreaterThan(lit[1]);
    expect(lit[3]).toBeLessThan(1);
    // Half a band further along, darker.
    const step = INFERRED_HIGHLIGHT.stripeM / 2 / 0.57735027 / 3;
    const dark = evaluateInferredColor(grey, [step + 0.01, step + 0.01, step + 0.01], true);
    expect(dark[2]).toBeLessThan(lit[2]);
  });

  it("swaps to a fill variant's layers, and back to Today's", async () => {
    const { scene, preUpdate } = fixture();
    const parent = {
      root: {
        extras: {
          inferredLayers: [{ uri: "inferred/tileset.json", evidence: EVIDENCE }],
          variants: {
            fill: [
              {
                name: "vace",
                label: "VACE",
                about: "Video inpainting.",
                inferredLayers: [
                  { uri: "variants/fill/vace/a.json", evidence: EVIDENCE },
                  { uri: "variants/fill/vace/b.json", evidence: EVIDENCE },
                ],
              },
            ],
          },
        },
      },
      resource: { url: "https://tiles.example/site/splat/tileset.json" },
      show: true,
      modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
      maximumScreenSpaceError: 16,
    } as unknown as Cesium3DTileset;
    const loaded: string[] = [];
    const removed: unknown[] = [];
    (scene.primitives as unknown as { remove: (p: unknown) => boolean }).remove = (p) => {
      removed.push(p);
      return true;
    };
    const dispose = attachInferredLayers(parent, scene, "asset-4", (url) => {
      loaded.push(url.replace("https://tiles.example/site/splat/", ""));
      return Promise.resolve(layer(url));
    });
    await settle();
    expect(loaded).toEqual(["inferred/tileset.json"]);
    expect(useVariants.getState().status["asset-4"]?.fill).toEqual({ state: "ready" });
    useVariants.getState().pick("asset-4", "fill", "vace");
    await settle();
    expect(loaded.slice(1)).toEqual(["variants/fill/vace/a.json", "variants/fill/vace/b.json"]);
    expect(removed).toHaveLength(1);
    expect(useInferred.getState().layers["asset-4"]).toHaveLength(2);
    preUpdate.raiseEvent();
    useVariants.getState().pick("asset-4", "fill", null);
    await settle();
    expect(loaded.at(-1)).toBe("inferred/tileset.json");
    expect(removed).toHaveLength(3);
    expect(useInferred.getState().layers["asset-4"]).toHaveLength(1);
    dispose();
    expect(useVariants.getState().status["asset-4"]).toBeUndefined();
  });

  it("says when a variant's layer did not load", async () => {
    const { scene } = fixture();
    const parent = {
      root: {
        extras: {
          variants: {
            fill: [
              {
                name: "broken",
                inferredLayers: [{ uri: "variants/fill/broken/tileset.json", evidence: EVIDENCE }],
              },
            ],
          },
        },
      },
      resource: { url: "https://tiles.example/site/splat/tileset.json" },
      show: true,
      modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
    } as unknown as Cesium3DTileset;
    const dispose = attachInferredLayers(parent, scene, "asset-5", () =>
      Promise.reject(new Error("layer answered 404")),
    );
    await settle();
    // Today has none: nothing to load.
    expect(useVariants.getState().status["asset-5"]?.fill).toEqual({ state: "ready" });
    useVariants.getState().pick("asset-5", "fill", "broken");
    await settle();
    expect(useVariants.getState().status["asset-5"]?.fill).toMatchObject({
      state: "error",
      message: expect.stringContaining("404") as unknown,
    });
    useVariants.getState().pick("asset-5", "fill", null);
    dispose();
  });
});

/** Lets the loads' promises settle. */
async function settle(): Promise<void> {
  for (let i = 0; i < 4; i += 1) await new Promise((r) => setTimeout(r, 0));
}

/** A loaded layer: its own root, a patched splat primitive with the colour hook's slot. */
function layer(url = "x") {
  const primitive = Object.defineProperty({}, "vertexColor", {
    value: undefined,
    writable: true,
    enumerable: true,
  }) as { vertexColor?: unknown };
  return {
    url,
    root: { extras: { evidence: { ...EVIDENCE, views: 6 } } },
    show: false,
    modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
    gaussianSplatPrimitive: primitive,
    tileLoad: new Event(),
    destroy: () => undefined,
  } as unknown as Cesium3DTileset & { gaussianSplatPrimitive: { vertexColor?: unknown } };
}

/** The scan's own primitive on the patched engine, in incremental mode. */
function hostPrimitive() {
  return {
    incremental: true,
    companions: [] as Cesium3DTileset[],
    vertexColor: undefined as SplatVertexColor | undefined,
    vertexVisibility: undefined as unknown,
    _tileSlots: undefined as
      Map<{ tileset: unknown }, { start: number; count: number }> | undefined,
    isDestroyed: () => false,
  };
}

function fixture() {
  const parent = {
    root: { extras: { inferredLayers: [{ uri: "inferred/tileset.json", evidence: EVIDENCE }] } },
    resource: { url: "https://tiles.example/site/splat/tileset.json" },
    show: true,
    modelMatrix: Matrix4.fromTranslation({ x: 1, y: 2, z: 3 } as never),
    maximumScreenSpaceError: 16,
  } as unknown as Cesium3DTileset;
  const child = layer();
  const preUpdate = new Event();
  const scene = {
    primitives: { add: () => undefined, remove: () => true },
    preUpdate,
    requestRender: () => undefined,
  } as unknown as Scene;
  return { parent, child, scene, preUpdate, primitive: child.gaussianSplatPrimitive };
}
