import { describe, expect, it } from "vitest";

import { Event, Matrix4 } from "cesium";
import type { Cesium3DTileset, Scene } from "cesium";

import { attachInferredLayers } from "@/cesium/inferredLayers";
import { describeEvidence, evidenceOf, inferredLayersOf, resolveLayerUrl } from "@/lib/inferred";
import { useInferred } from "@/state/inferred";

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
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    expect(Matrix4.equals(child.modelMatrix, parent.modelMatrix)).toBe(true);
    useInferred.getState().setShow(false);
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    useInferred.getState().setShow(true);
    (parent as { show: boolean }).show = false;
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    dispose();
    expect(useInferred.getState().layers["asset-1"]).toBeUndefined();
  });
});
