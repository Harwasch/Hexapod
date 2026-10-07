/**
 * Swap, don't stack (lib/supersedes.ts, state/supersedes.ts): a fill variant's `supersedes`
 * file is read defensively, its splats drawn under one reserved id past every other, and that
 * id hidden only while the variant is picked and Inferred is Show or Highlight -- Hide, Today
 * and other methods draw the untouched scan, with nothing left over from the last one.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  instanceStyle,
  linkScanInstances,
  type InstanceStyle,
} from "@/cesium/scanView/scanInstances";
import type { ScanBackend } from "@/cesium/scanView/types";
import { tileInstanceIds, parseInstances, type InstancesDoc } from "@/lib/instances";
import {
  parseSupersedes,
  supersededIdOf,
  withSuperseded,
  withSupersedes,
  type SupersedesDoc,
} from "@/lib/supersedes";
import { variantsOf } from "@/lib/variants";
import { useInstances } from "@/state/instances";
import { effectiveDoc, onCustomSetsChange, useSceneSelect } from "@/state/sceneSelect";
import { useSettings } from "@/state/settings";
import { activeSupersedes, followSupersedes, useSupersedes } from "@/state/supersedes";
import { useVariants } from "@/state/variants";

const A = "fnv1a32:6:0000abcd";
const B = "fnv1a32:4:0000abce";
const C = "fnv1a32:3:0000abcf";
const box = { min: [0, 0, 0], max: [1, 1, 1] };

/** Two instances; tile A `[1,1,2,2,2,0]`, tile B all 1; tile C not listed. */
function doc(): InstancesDoc {
  const parsed = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      { id: 1, bounds: box, tags: [{ label: "spool", score: 0.9 }] },
      { id: 2, bounds: box, tags: [{ label: "ground", score: 0.9 }] },
    ],
    tiles: { [A]: [1, 2, 2, 3, 0, 1], [B]: [1, 4] },
  });
  if (!parsed) throw new Error("fixture did not parse");
  return parsed;
}

const ids = (d: InstancesDoc, checksum: string): number[] => [
  ...(tileInstanceIds(d, checksum) ?? []),
];

function sup(tiles: Record<string, number[]>): SupersedesDoc {
  const parsed = parseSupersedes({ superseded: 0, tiles });
  if (!parsed) throw new Error("fixture did not parse");
  return parsed;
}

describe("reading a supersedes file", () => {
  it("keeps tiles with a superseded splat, and skips what is malformed", () => {
    const read = parseSupersedes({
      encoding: "rle per tile",
      superseded: 3,
      tiles: {
        [A]: [0, 1, 1, 2, 0, 3],
        [B]: [0, 4], // nothing superseded: nothing to do there
        [C]: [2, 3], // a flag must be 0 or 1
        "not-a-checksum": [1, 1],
        "fnv1a32:5:0000abcd": [1, 2], // runs short of the tile
      },
    });
    expect(read?.superseded).toBe(3);
    expect([...(read?.tiles.keys() ?? [])]).toEqual([A]);
    expect(read?.issues).toHaveLength(3);
    expect(parseSupersedes(null)).toBeNull();
    expect(parseSupersedes({ tiles: [] })).toBeNull();
    expect(parseSupersedes({ superseded: 2 })).toBeNull();
  });

  it("is a fill variant's optional field; a variant without it reads as before", () => {
    const layers = [{ uri: "variants/fill/a/tileset.json", evidence: { kind: "inferred" } }];
    const v = variantsOf({
      variants: {
        fill: [
          { name: "a", inferredLayers: layers, supersedes: "variants/fill/a/supersedes.json" },
          { name: "b", inferredLayers: layers, supersedes: "" },
          { name: "c", inferredLayers: layers, somethingNew: { x: 1 } },
        ],
      },
    });
    expect(v.fill.map((f) => f.supersedes)).toEqual([
      "variants/fill/a/supersedes.json",
      undefined,
      undefined,
    ]);
    expect(v.fill.map((f) => "supersedes" in f)).toEqual([true, false, false]);
  });
});

describe("drawing superseded splats under a reserved id", () => {
  it("re-labels exactly the flagged splats, past the largest id", () => {
    const d = doc();
    const swapped = withSupersedes(d, sup({ [A]: [0, 1, 1, 3, 0, 2], [C]: [1, 1, 0, 2] }));
    expect(swapped.maxId).toBe(3);
    expect(supersededIdOf(swapped)).toBe(3);
    expect(ids(swapped, A)).toEqual([1, 3, 3, 3, 2, 0]);
    expect(ids(swapped, B)).toEqual([1, 1, 1, 1]);
    // A tile the objects file does not list gets runs of its own.
    expect(ids(swapped, C)).toEqual([3, 0, 0]);
    // The file's document is untouched; no instance is added for the reserved id.
    expect(ids(d, A)).toEqual([1, 1, 2, 2, 2, 0]);
    expect(supersededIdOf(d)).toBeNull();
    expect(swapped.instances).toBe(d.instances);
    expect(swapped.byId.has(3)).toBe(false);
  });

  it("is the same document for the same pair, and the original for none", () => {
    const d = doc();
    const s = sup({ [A]: [1, 6] });
    expect(withSupersedes(d, s)).toBe(withSupersedes(d, s));
    expect(withSupersedes(d, null)).toBe(d);
    expect(withSupersedes(d, sup({ [B]: [0, 4] }))).toBe(d);
  });

  it("leaves a tile whose splat count disagrees with the objects file", () => {
    const d = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [{ id: 1, bounds: box }],
      tiles: { [A]: [1, 6] },
    });
    if (!d) throw new Error("fixture");
    // The same key with other runs cannot happen from one package; guard it all the same.
    const odd = { superseded: 1, tiles: new Map([[A, Int32Array.of(1, 5)]]), issues: [] };
    expect(ids(withSupersedes(d, odd), A)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("adds the reserved id to what is hidden, and leaves the store's set alone", () => {
    const d = doc();
    const swapped = withSupersedes(d, sup({ [A]: [1, 6] }));
    const stored = new Set([2]);
    expect([...withSuperseded(swapped, stored)]).toEqual([2, 3]);
    expect([...stored]).toEqual([2]);
    expect(withSuperseded(d, stored)).toBe(stored);
    // A dedicated renderer's state texels: 3 hidden though the store hides 2 alone.
    const style = instanceStyle(swapped, stored, new Set(), false);
    expect([style.state[2 * 4], style.state[3 * 4], style.state[1 * 4]]).toEqual([255, 255, 0]);
    expect(style.params[0]).toBe(1);
    expect(style.params[1]).toBe(3);
    // With nothing hidden by the person, the swap alone turns the rule on.
    expect(instanceStyle(swapped, new Set(), new Set(), false).params[0]).toBe(1);
  });
});

/** Settles the promises a follow chains. */
async function settle(): Promise<void> {
  for (let i = 0; i < 4; i += 1) await new Promise((r) => setTimeout(r, 0));
}

const LAYERS = [{ uri: "variants/fill/x/tileset.json", evidence: { kind: "inferred" } }];
const VARIANTS = variantsOf({
  variants: {
    fill: [
      { name: "rebuilt", inferredLayers: LAYERS, supersedes: "variants/fill/rebuilt/s.json" },
      { name: "other", inferredLayers: LAYERS, supersedes: "variants/fill/other/s.json" },
      { name: "stacked", inferredLayers: LAYERS },
    ],
  },
});

describe("when superseded splats are hidden", () => {
  const files: Record<string, SupersedesDoc> = {
    "variants/fill/rebuilt/s.json": sup({ [A]: [1, 2, 0, 4] }),
    "variants/fill/other/s.json": sup({ [B]: [1, 4] }),
  };
  let asked: string[] = [];
  /** Resolves each file on `release(uri)`, so the tests choose the order. */
  let pending: Map<string, () => void>;
  const load = (_url: string, uri: string): Promise<SupersedesDoc> => {
    asked.push(uri);
    return new Promise((resolve, reject) => {
      pending.set(uri, () => {
        const file = files[uri];
        if (file) resolve(file);
        else reject(new Error("supersedes answered 404"));
      });
    });
  };
  const release = async (uri: string): Promise<void> => {
    pending.get(uri)?.();
    pending.delete(uri);
    await settle();
  };

  beforeEach(() => {
    asked = [];
    pending = new Map();
    useVariants.setState({ offered: {}, picks: {}, status: {} });
    useSupersedes.setState({ active: {} });
    useSettings.getState().set({ inferredStyle: "show" });
  });

  it("only while the method is picked and Inferred is Show or Highlight", async () => {
    const off = followSupersedes("scan", VARIANTS, "https://x.test/scan/tileset.json", load);
    // Today: nothing.
    expect(activeSupersedes("scan")).toBeNull();
    useVariants.getState().pick("scan", "fill", "rebuilt");
    // Not until its file is in.
    expect(activeSupersedes("scan")).toBeNull();
    await release("variants/fill/rebuilt/s.json");
    const doc = files["variants/fill/rebuilt/s.json"];
    expect(activeSupersedes("scan")).toBe(doc);
    // Highlight: still swapped, and nothing reloaded or replaced.
    useSettings.getState().set({ inferredStyle: "highlight" });
    expect(activeSupersedes("scan")).toBe(doc);
    // Hide: the untouched scan.
    useSettings.getState().set({ inferredStyle: "hide" });
    expect(activeSupersedes("scan")).toBeNull();
    useSettings.getState().set({ inferredStyle: "show" });
    await settle();
    expect(activeSupersedes("scan")).toBe(doc);
    expect(asked).toEqual(["variants/fill/rebuilt/s.json"]); // loaded once
    // A method without the file, and Today: nothing hidden.
    useVariants.getState().pick("scan", "fill", "stacked");
    expect(activeSupersedes("scan")).toBeNull();
    useVariants.getState().pick("scan", "fill", "rebuilt");
    await settle();
    expect(activeSupersedes("scan")).toBe(doc);
    useVariants.getState().pick("scan", "fill", null);
    expect(activeSupersedes("scan")).toBeNull();
    off();
  });

  it("clears at once on a switch, and never applies a file the pick has left", async () => {
    const off = followSupersedes("scan", VARIANTS, "https://x.test/scan/tileset.json", load);
    useVariants.getState().pick("scan", "fill", "rebuilt");
    await release("variants/fill/rebuilt/s.json");
    useVariants.getState().pick("scan", "fill", "other");
    // The last method's splats come back at once, before the next file is in.
    expect(activeSupersedes("scan")).toBeNull();
    useVariants.getState().pick("scan", "fill", "stacked");
    await release("variants/fill/other/s.json");
    // The file arrived after the pick moved on: not applied.
    expect(activeSupersedes("scan")).toBeNull();
    off();
  });

  it("draws the whole scan when the file does not load, and tries again later", async () => {
    const off = followSupersedes("scan", VARIANTS, "https://x.test/scan/tileset.json", load);
    delete files["variants/fill/other/s.json"];
    useVariants.getState().pick("scan", "fill", "other");
    await release("variants/fill/other/s.json");
    expect(activeSupersedes("scan")).toBeNull();
    files["variants/fill/other/s.json"] = sup({ [B]: [1, 4] });
    useSettings.getState().set({ inferredStyle: "hide" });
    useSettings.getState().set({ inferredStyle: "show" });
    await release("variants/fill/other/s.json");
    expect(activeSupersedes("scan")).toBe(files["variants/fill/other/s.json"]);
    expect(asked).toEqual(["variants/fill/other/s.json", "variants/fill/other/s.json"]);
    // Unloading the scan clears it.
    off();
    expect(activeSupersedes("scan")).toBeNull();
  });
});

describe("what the renderers draw from", () => {
  beforeEach(() => {
    useSupersedes.setState({ active: {} });
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    useSceneSelect.getState().clear();
  });

  it("swaps after the painted objects, past their ids, and says when it changes", () => {
    const base = doc();
    let heard = 0;
    const off = onCustomSetsChange("scan", () => (heard += 1));
    useSceneSelect.getState().addCustom("scan", {
      key: "p",
      name: "painted",
      tiles: { [A]: [4, 2] },
      splats: 2,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      created: 0,
    });
    expect(heard).toBe(1);
    const painted = effectiveDoc("scan", base);
    expect(painted.maxId).toBe(3);
    expect(ids(painted, A)).toEqual([1, 1, 2, 2, 3, 3]);
    useSupersedes.getState().setActive("scan", sup({ [A]: [1, 1, 0, 4, 1, 1] }));
    expect(heard).toBe(2);
    const drawn = effectiveDoc("scan", base);
    expect(drawn).toBe(effectiveDoc("scan", base));
    expect(supersededIdOf(drawn)).toBe(4);
    expect(ids(drawn, A)).toEqual([4, 1, 2, 2, 3, 4]);
    useSupersedes.getState().setActive("scan", null);
    expect(heard).toBe(3);
    expect(effectiveDoc("scan", base)).toBe(painted);
    useSceneSelect.getState().removeCustom("scan", "p");
    off();
  });

  it("hands a dedicated renderer the swap, and the untouched scan when it ends", () => {
    const base = doc();
    const styles: (InstanceStyle | null)[] = [];
    const backend = {
      name: "spark",
      setInstances: (style: InstanceStyle | null) => void styles.push(style),
    } as unknown as ScanBackend<unknown>;
    const unlink = linkScanInstances("scan", backend, false, (id) => effectiveDoc(id, base));
    useInstances.getState().setTable("scan", base);
    expect(styles.at(-1)?.params[0]).toBe(0);
    useSupersedes.getState().setActive("scan", sup({ [B]: [1, 4] }));
    const swapped = styles.at(-1);
    expect(swapped?.doc.maxId).toBe(3);
    expect(swapped?.state[3 * 4]).toBe(255);
    expect(useInstances.getState().assets.scan?.hidden.size).toBe(0);
    useSupersedes.getState().setActive("scan", null);
    const restored = styles.at(-1);
    expect(restored?.doc).toBe(base);
    expect(restored?.params[0]).toBe(0);
    unlink();
  });
});
