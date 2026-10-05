import type { Cesium3DTileset, Scene } from "cesium";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { declaresInstances, declaresObjectsOrMotion } from "@/cesium/scanView/ScanRendererHost";
import { declaresMotion } from "@/cesium/scanView/scanMotion";
import { attachVariants } from "@/cesium/scanVariants";
import {
  attachInstances,
  instancesDocOf,
  instancesHookOf,
  type InstanceGpu,
} from "@/cesium/splatInstances";
import { attachSkin, onSkinsChanged, skinningOf } from "@/cesium/splatSkin";
import type { MotionTextureFactory } from "@/cesium/splatGpuMotion";
import { parseInstances, type InstancesDoc } from "@/lib/instances";
import type { SkinDoc } from "@/lib/skin";
import {
  findVariant,
  hasVariants,
  inferredLayersFor,
  instancesRefFor,
  NO_VARIANTS,
  resolveVariantUrl,
  skinRefFor,
  todayOf,
  variantsOf,
} from "@/lib/variants";
import { useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";
import { onPickChange, pickedVariant, PICKS_KEY, useVariants } from "@/state/variants";

const EVIDENCE = {
  kind: "inferred",
  filler: "wan2.1-vace-14b",
  views: 4,
  gaussians: 9,
  meanConfidence: 0.5,
};

/** The shape the bake-off conventions publish, one entry per system. */
const DECLARED = {
  objects: [
    {
      name: "ground-first",
      label: "A · Ground first",
      about: "Takes the ground out first, then splits what stands on it.",
      instances: "variants/objects/ground-first/instances.json",
    },
  ],
  fill: [
    {
      name: "vace-14b",
      label: "Wan2.1-VACE 14B",
      about: "Video inpainting of the views no camera took.",
      inferredLayers: [{ uri: "variants/fill/vace-14b/tileset.json", evidence: EVIDENCE }],
    },
  ],
  skins: [
    {
      name: "freeform",
      label: "FreeForm (eigenmodes)",
      about: "Each object's lowest elastic modes.",
      skin: "variants/skins/freeform/skin.json",
    },
  ],
};

describe("extras.variants", () => {
  it("reads the published shape", () => {
    const v = variantsOf({ variants: DECLARED });
    expect(v.objects).toEqual([DECLARED.objects[0]]);
    expect(v.fill[0]?.inferredLayers[0]?.evidence.filler).toBe("wan2.1-vace-14b");
    expect(v.skins[0]?.skin).toBe("variants/skins/freeform/skin.json");
    expect(hasVariants(v)).toBe(true);
  });

  it("is none when nothing is declared", () => {
    expect(variantsOf(undefined)).toBe(NO_VARIANTS);
    expect(variantsOf({})).toBe(NO_VARIANTS);
    expect(variantsOf({ variants: [] })).toBe(NO_VARIANTS);
    expect(variantsOf({ variants: { objects: "nope" } })).toBe(NO_VARIANTS);
    expect(hasVariants(NO_VARIANTS)).toBe(false);
  });

  it("skips malformed entries and keeps the rest", () => {
    const v = variantsOf({
      variants: {
        objects: [
          { label: "no name", instances: "a.json" },
          { name: "no-file" },
          { name: "  ", instances: "a.json" },
          "nonsense",
          null,
          { name: "kept", instances: "variants/objects/kept/instances.json" },
          { name: "kept", label: "A second of the same name", instances: "b.json" },
          { name: "uri-form", instances: { uri: "variants/objects/u/instances.json" } },
        ],
        fill: [
          // One layer in its list does not read: the method is not there whole.
          {
            name: "half",
            inferredLayers: [
              { uri: "a/tileset.json", evidence: EVIDENCE },
              { uri: "b/tileset.json", evidence: { kind: "measured" } },
            ],
          },
          { name: "no-list", inferredLayers: "a/tileset.json" },
          { name: "none", label: "No fill", inferredLayers: [] },
        ],
        skins: [
          { name: "no-skin", skin: "" },
          { name: 7, skin: "s.json" },
        ],
        future: [{ name: "x" }],
      },
    });
    expect(v.objects.map((o) => o.name)).toEqual(["kept", "uri-form"]);
    expect(v.objects[0]).toMatchObject({ label: "kept", about: "" });
    expect(v.objects[1]?.instances).toBe("variants/objects/u/instances.json");
    expect(v.fill).toEqual([{ name: "none", label: "No fill", about: "", inferredLayers: [] }]);
    expect(v.skins).toEqual([]);
  });

  it("resolves beside the measured tileset, keeping a signed query", () => {
    expect(
      resolveVariantUrl(
        "https://tiles.example/a/tileset.json?sig=1",
        "variants/objects/x/instances.json",
      ),
    ).toBe("https://tiles.example/a/variants/objects/x/instances.json?sig=1");
  });

  it("falls back to Today's files", () => {
    const extras = {
      instances: { uri: "instances.json", count: 3 },
      skin: { uri: "skin.json", count: 2 },
      inferredLayers: [{ uri: "inferred/tileset.json", evidence: EVIDENCE }],
      variants: DECLARED,
    };
    const v = variantsOf(extras);
    expect(instancesRefFor(extras, null)).toEqual({ uri: "instances.json", count: 3 });
    expect(instancesRefFor(extras, v.objects[0] ?? null)?.uri).toBe(
      "variants/objects/ground-first/instances.json",
    );
    expect(skinRefFor(extras, null)?.uri).toBe("skin.json");
    expect(skinRefFor(extras, v.skins[0] ?? null)?.uri).toBe("variants/skins/freeform/skin.json");
    expect(inferredLayersFor(extras, null).map((l) => l.uri)).toEqual(["inferred/tileset.json"]);
    expect(inferredLayersFor(extras, v.fill[0] ?? null).map((l) => l.uri)).toEqual([
      "variants/fill/vace-14b/tileset.json",
    ]);
    expect(todayOf(extras)).toEqual({ objects: true, fill: true, skins: true });
    expect(todayOf({ variants: DECLARED })).toEqual({ objects: false, fill: false, skins: false });
    expect(findVariant(v, "skins", "freeform")?.label).toBe("FreeForm (eigenmodes)");
    expect(findVariant(v, "skins", "gone")).toBeNull();
  });

  it("counts as objects and motion for the renderers that cannot draw them", () => {
    expect(declaresInstances({ variants: { objects: DECLARED.objects } })).toBe(true);
    expect(declaresMotion({ variants: { skins: DECLARED.skins } })).toBe(true);
    expect(declaresObjectsOrMotion({ variants: { fill: DECLARED.fill } })).toBe(false);
    expect(declaresObjectsOrMotion({})).toBe(false);
  });
});

describe("the variants store", () => {
  beforeEach(() => {
    useVariants.setState({ offered: {}, picks: {}, status: {} });
    sessionStorage.clear();
  });

  it("offers what a loaded scan declares, and forgets it when it unloads", () => {
    const tileset = { root: { extras: { variants: DECLARED } } } as unknown as Cesium3DTileset;
    const off = attachVariants(tileset, "scan");
    expect(useVariants.getState().offered.scan?.objects).toHaveLength(1);
    expect(useVariants.getState().offered.scan?.today).toEqual({
      objects: false,
      fill: false,
      skins: false,
    });
    off();
    expect(useVariants.getState().offered.scan).toBeUndefined();
    const plain = { root: { extras: {} } } as unknown as Cesium3DTileset;
    attachVariants(plain, "plain")();
    expect(useVariants.getState().offered).toEqual({});
  });

  it("keeps a pick per scan and per system, and Today is no pick", () => {
    const { pick } = useVariants.getState();
    pick("a", "objects", "ground-first");
    pick("a", "fill", "vace-14b");
    pick("b", "objects", "other");
    expect(useVariants.getState().picks).toEqual({
      a: { objects: "ground-first", fill: "vace-14b" },
      b: { objects: "other" },
    });
    pick("a", "objects", null);
    pick("b", "objects", null);
    expect(useVariants.getState().picks).toEqual({ a: { fill: "vace-14b" } });
  });

  it("keeps the picks for the session", () => {
    useVariants.getState().pick("a", "skins", "freeform");
    expect(JSON.parse(sessionStorage.getItem(PICKS_KEY) ?? "{}")).toEqual({
      a: { skins: "freeform" },
    });
  });

  it("draws Today for a name the scan no longer offers", () => {
    const variants = variantsOf({ variants: DECLARED });
    useVariants.getState().pick("a", "objects", "ground-first");
    expect(pickedVariant("a", "objects", variants)?.name).toBe("ground-first");
    useVariants.getState().pick("a", "objects", "withdrawn");
    expect(pickedVariant("a", "objects", variants)).toBeNull();
    expect(pickedVariant("a", "fill", variants)).toBeNull();
  });

  it("tells each drawer of its own system's pick only", () => {
    const heard: string[] = [];
    const off = onPickChange("a", "fill", () => heard.push("fill"));
    useVariants.getState().pick("a", "objects", "x");
    useVariants.getState().pick("b", "fill", "x");
    useVariants.getState().pick("a", "fill", "x");
    useVariants.getState().pick("a", "fill", "x");
    off();
    expect(heard).toEqual(["fill"]);
  });

  it("reports how a pick's files are doing, and nothing when unchanged", () => {
    const { setStatus } = useVariants.getState();
    setStatus("a", "objects", { state: "loading" });
    const before = useVariants.getState().status;
    setStatus("a", "objects", { state: "loading" });
    expect(useVariants.getState().status).toBe(before);
    setStatus("a", "objects", { state: "error", message: "404" });
    expect(useVariants.getState().status.a?.objects).toEqual({ state: "error", message: "404" });
    setStatus("a", "objects", null);
    expect(useVariants.getState().status.a).toBeUndefined();
  });
});

function fakeGpu(): InstanceGpu {
  const texture = () => ({
    width: 1,
    height: 1,
    isDestroyed: () => false,
    destroy: () => undefined,
    copyFrom: () => undefined,
  });
  return {
    createUintQuads: texture,
    createBytes: texture,
    vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
  };
}

function fakeScene(): Pick<Scene, "preUpdate" | "requestRender"> {
  return {
    preUpdate: { addEventListener: () => () => undefined },
    requestRender: () => undefined,
  } as unknown as Pick<Scene, "preUpdate" | "requestRender">;
}

function instancesDoc(labels: string[]): InstancesDoc {
  const doc = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: labels.map((label, i) => ({
      id: i + 1,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      splats: 10,
      tags: [{ label, score: 0.9 }],
    })),
    tiles: {},
  });
  if (!doc) throw new Error("fixture");
  return doc;
}

async function settle(): Promise<void> {
  for (let i = 0; i < 4; i += 1) await new Promise((r) => setTimeout(r, 0));
}

describe("switching a scan's objects", () => {
  beforeEach(() => {
    useVariants.setState({ offered: {}, picks: {}, status: {} });
    useInstances.setState({ assets: {}, dimOthers: true });
    useSceneSelect.getState().clear();
  });

  const tileset = {
    root: {
      extras: {
        instances: { uri: "instances.json", count: 2 },
        variants: {
          objects: [
            {
              name: "coarse",
              label: "Coarse",
              instances: "variants/objects/coarse/instances.json",
            },
            {
              name: "broken",
              label: "Broken",
              instances: "variants/objects/broken/instances.json",
            },
          ],
        },
      },
    },
    resource: { url: "https://x.test/scan/tileset.json?sig=1" },
  } as unknown as Cesium3DTileset;

  const files: Record<string, InstancesDoc> = {
    "instances.json": instancesDoc(["oak tree", "bench"]),
    "variants/objects/coarse/instances.json": instancesDoc(["forest"]),
  };
  const load = (_url: string, ref: { uri: string }): Promise<InstancesDoc> => {
    const doc = files[ref.uri];
    return doc ? Promise.resolve(doc) : Promise.reject(new Error("instances answered 404"));
  };

  it("loads the pick's file into the panel's table, clearing the selection", async () => {
    const dispose = attachInstances(tileset, fakeScene(), "scan", fakeGpu(), load);
    await settle();
    expect(useInstances.getState().assets.scan?.instances.map((i) => i.tags[0]?.label)).toEqual([
      "oak tree",
      "bench",
    ]);
    useInstances.getState().setHidden("scan", [2], true);
    useSceneSelect.getState().select("scan", [1], 1, 0, null);
    useVariants.getState().pick("scan", "objects", "coarse");
    // The selection was of the old file's ids: gone at once, before the new file is in.
    expect(useSceneSelect.getState().candidates).toEqual([]);
    expect(useVariants.getState().status.scan?.objects).toEqual({ state: "loading" });
    await settle();
    const entry = useInstances.getState().assets.scan;
    expect(entry?.instances.map((i) => i.tags[0]?.label)).toEqual(["forest"]);
    expect(entry?.hidden.size).toBe(0);
    expect(instancesDocOf("scan")).toBe(files["variants/objects/coarse/instances.json"]);
    expect(useVariants.getState().status.scan?.objects).toEqual({ state: "ready" });
    // Back to Today.
    useVariants.getState().pick("scan", "objects", null);
    await settle();
    expect(useInstances.getState().assets.scan?.instances).toHaveLength(2);
    dispose();
    expect(useInstances.getState().assets.scan).toBeUndefined();
    expect(useVariants.getState().status.scan).toBeUndefined();
  });

  it("keeps the scan's painted objects over the new file", async () => {
    const dispose = attachInstances(tileset, fakeScene(), "painted", fakeGpu(), load);
    await settle();
    useSceneSelect.getState().addCustom("painted", {
      key: "p1",
      name: "my wheelbarrow",
      tiles: {},
      splats: 12,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      created: 0,
    });
    await settle();
    expect(instancesHookOf("painted")?.doc.maxId).toBe(3);
    useVariants.getState().pick("painted", "objects", "coarse");
    await settle();
    // The coarse file's one object, and the painted one past it.
    const drawn = instancesHookOf("painted")?.doc;
    expect(drawn?.maxId).toBe(2);
    expect(drawn?.byId.get(2)?.tags[0]?.label).toBe("my wheelbarrow");
    useSceneSelect.getState().removeCustom("painted", "p1");
    dispose();
  });

  it("draws no objects for a pick that did not load, and says why", async () => {
    const dispose = attachInstances(tileset, fakeScene(), "scan", fakeGpu(), load);
    await settle();
    useVariants.getState().pick("scan", "objects", "broken");
    await settle();
    expect(useInstances.getState().assets.scan).toBeUndefined();
    expect(useVariants.getState().status.scan?.objects).toMatchObject({ state: "error" });
    dispose();
  });

  it("draws the session's pick from the start", async () => {
    useVariants.getState().pick("scan", "objects", "coarse");
    const asked: string[] = [];
    const dispose = attachInstances(tileset, fakeScene(), "scan", fakeGpu(), (url, ref) => {
      asked.push(url.replace(/\?.*$/, "") + " " + ref.uri);
      return load(url, ref);
    });
    await settle();
    expect(asked).toEqual([
      "https://x.test/scan/tileset.json variants/objects/coarse/instances.json",
    ]);
    dispose();
  });

  it("attaches for a scan whose only objects are variants'", async () => {
    const only = {
      root: {
        extras: {
          variants: {
            objects: [{ name: "coarse", instances: "variants/objects/coarse/instances.json" }],
          },
        },
      },
      resource: { url: "https://x.test/scan/tileset.json" },
    } as unknown as Cesium3DTileset;
    const dispose = attachInstances(only, fakeScene(), "only", fakeGpu(), load);
    await settle();
    expect(useInstances.getState().assets.only).toBeUndefined();
    useVariants.getState().pick("only", "objects", "coarse");
    await settle();
    expect(useInstances.getState().assets.only?.instances).toHaveLength(1);
    dispose();
  });
});

describe("switching a scan's skin", () => {
  beforeEach(() => useVariants.setState({ offered: {}, picks: {}, status: {} }));

  const skinDoc = (skins: number): SkinDoc => ({
    skins: [],
    byId: new Map(),
    byInstance: new Map(),
    maxId: skins,
    tiles: new Map(),
    words: new Uint32Array(0),
    rows: 0,
    scale: 1 / 127,
    issues: [],
  });

  it("replaces the part the drivers move, and they hear of it", async () => {
    const tileset = {
      root: {
        extras: {
          skin: { uri: "skin.json", count: 4 },
          variants: {
            skins: [{ name: "trees", label: "Trees only", skin: "variants/skins/trees/skin.json" }],
          },
        },
      },
      resource: { url: "https://x.test/scan/tileset.json" },
    } as unknown as Cesium3DTileset;
    const docs: Record<string, SkinDoc> = {
      "skin.json": skinDoc(4),
      "variants/skins/trees/skin.json": skinDoc(1),
    };
    const factory = {} as MotionTextureFactory;
    const changed = vi.fn();
    const off = onSkinsChanged(changed);
    const dispose = attachSkin(tileset, fakeScene(), "scan", factory, (_url, ref) => {
      const doc = docs[ref.uri];
      return doc ? Promise.resolve(doc) : Promise.reject(new Error("skin answered 404"));
    });
    await settle();
    const today = skinningOf("scan");
    expect(today?.doc).toBe(docs["skin.json"]);
    useVariants.getState().pick("scan", "skins", "trees");
    await settle();
    expect(skinningOf("scan")).not.toBe(today);
    expect(skinningOf("scan")?.doc).toBe(docs["variants/skins/trees/skin.json"]);
    expect(changed).toHaveBeenCalledTimes(2);
    dispose();
    expect(skinningOf("scan")).toBeUndefined();
    off();
  });
});
