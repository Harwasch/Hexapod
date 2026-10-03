/**
 * Scene objects under the dedicated splat renderers (cesium/scanView/scanInstances.ts): the
 * style handed to a back-end follows the store and matches CesiumJS's rule, a reordered tile's
 * ids follow its splats, a renderer that cannot draw them is reported, every committed yard
 * tile is found by the digest Spark computes from its SPZ, and the native-package probe is
 * skipped when the tileset says, or once a scan is known to have none.
 */

import { readFileSync } from "node:fs";

import { checksumPositions } from "@twin/world";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  declaredNativeLod,
  findNativeLod,
  NATIVE_LOD_PATH,
} from "@/cesium/scanView/ScanRendererHost";
import {
  idsInResourceOrder,
  instanceGap,
  instanceStyle,
  linkScanInstances,
  SCAN_INSTANCE_RULE_GLSL,
  type InstanceStyle,
} from "@/cesium/scanView/scanInstances";
import type { ScanBackend } from "@/cesium/scanView/types";
import {
  evaluateInstanceShading,
  HIGHLIGHT_STYLE,
  INSTANCE_TEXTURE_WIDTH,
} from "@/cesium/splatInstances";
import { parseInstances, tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { spzPositions } from "@/lib/spzPositions";
import { useInstances } from "@/state/instances";
import { spzFromGlb } from "@/view/glb";

import { yardPath, yardTiles } from "./splatYardFixture";

const box = { min: [0, 0, 0], max: [1, 1, 1] };

function doc(): InstancesDoc {
  const parsed = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      { id: 1, bounds: box, tags: [{ label: "oak", score: 0.8 }], behaviour: "in-place" },
      { id: 2, parent: 1, level: 1, bounds: box, tags: [], behaviour: "in-place" },
      { id: 3, bounds: box, tags: [{ label: "cabin", score: 0.5 }], behaviour: "static" },
    ],
    tiles: {},
  });
  if (!parsed) throw new Error("fixture did not parse");
  return parsed;
}

/** A back-end that only records what it is handed. */
function fakeBackend(
  name: ScanBackend<unknown>["name"] = "playcanvas",
  withInstances = true,
): ScanBackend<unknown> & { styles: (InstanceStyle | null)[] } {
  const styles: (InstanceStyle | null)[] = [];
  const backend = {
    name,
    loadFactor: 1,
    styles,
    load: () => Promise.resolve({}),
    add: () => undefined,
    remove: () => undefined,
    dispose: () => undefined,
    render: () => undefined,
    isDrawn: () => true,
    setBudget: () => undefined,
    destroy: () => undefined,
    ...(withInstances
      ? { setInstances: (style: InstanceStyle | null) => void styles.push(style) }
      : {}),
  };
  return backend;
}

describe("the style a dedicated renderer draws objects with", () => {
  it("applies the store's exact sets and the same numbers as CesiumJS's hooks", () => {
    const d = doc();
    const style = instanceStyle(d, new Set([1, 2]), new Set([3]), true);
    expect(style.rows).toBe(1);
    expect(style.state).toHaveLength(INSTANCE_TEXTURE_WIDTH * 4);
    // 1 and 2 (inside it, as the store expands it) hidden; 3 lit.
    expect([style.state[4], style.state[8], style.state[12 + 1]]).toEqual([255, 255, 255]);
    expect(style.state[12]).toBe(0);
    expect(style.params).toEqual([1, 3, 1, 0]);
    expect(style.tint).toEqual(HIGHLIGHT_STYLE.tint);
    expect(style.dim).toEqual([HIGHLIGHT_STYLE.dim[0], HIGHLIGHT_STYLE.dim[1], 0, 0]);
    // Per id, what the rule draws is CesiumJS's reference.
    const params = { active: true, highlightActive: true };
    expect(evaluateInstanceShading(2, style.state, d.maxId, params, [1, 1, 1, 1]).visibility).toBe(
      0,
    );
    expect(
      evaluateInstanceShading(3, style.state, d.maxId, params, [0, 0, 0, 1]).color[0],
    ).toBeCloseTo(HIGHLIGHT_STYLE.tint[0] * HIGHLIGHT_STYLE.tint[3] + 0.06);
    const nothing = instanceStyle(d, new Set(), new Set(), false);
    expect(nothing.params).toEqual([0, 3, 0, 0]);
    expect(nothing.dim).toEqual([1, 1, 0, 0]);
  });

  it("states the rule in GLSL: hide by opacity, tint the lit, dim the rest", () => {
    expect(SCAN_INSTANCE_RULE_GLSL).toContain("vec4(color.rgb, 0.0)");
    expect(SCAN_INSTANCE_RULE_GLSL).toContain(
      "mix(color.rgb, uInstanceTint.rgb, uInstanceTint.a) + 0.06",
    );
    expect(SCAN_INSTANCE_RULE_GLSL).toContain(
      "color.rgb * uInstanceDim.x, color.a * uInstanceDim.y",
    );
  });

  it("puts a reordered tile's ids where its splats went", () => {
    const ids = Uint32Array.from([7, 8, 9]);
    const out = new Uint32Array(4).fill(5);
    expect([...idsInResourceOrder(ids, Uint32Array.from([2, 0, 1]), out)]).toEqual([9, 7, 8, 0]);
    expect([...idsInResourceOrder(undefined, Uint32Array.from([2, 0, 1]), out)]).toEqual([
      0, 0, 0, 0,
    ]);
  });
});

describe("linking a renderer to the store", () => {
  beforeEach(() => useInstances.setState({ assets: {}, dimOthers: true, gaps: {} }));

  it("hands the back-end a style whenever what is hidden or lit changes", () => {
    const d = doc();
    const backend = fakeBackend();
    const unlink = linkScanInstances("scan", backend, false, () => d);
    // Nothing until the scan's instances load.
    expect(backend.styles).toEqual([]);
    useInstances.getState().setTable("scan", d);
    expect(backend.styles).toHaveLength(1);
    expect(backend.styles[0]?.params[0]).toBe(0);
    useInstances.getState().setHidden("scan", [3], true);
    expect(backend.styles.at(-1)?.state[12]).toBe(255);
    useInstances.getState().setQuery("scan", "oak");
    // A query alone changes nothing drawn.
    expect(backend.styles).toHaveLength(2);
    useInstances.getState().setDimOthers(false);
    expect(backend.styles.at(-1)?.dim).toEqual([1, 1, 0, 0]);
    expect(useInstances.getState().gaps.scan).toBeUndefined();
    unlink();
    expect(backend.styles.at(-1)).toBeNull();
    const count = backend.styles.length;
    useInstances.getState().setHidden("scan", [1], true);
    expect(backend.styles).toHaveLength(count);
  });

  it("asks the overlay for a frame after each style (it draws only on change)", () => {
    const d = doc();
    const backend = fakeBackend();
    let wakes = 0;
    const unlink = linkScanInstances(
      "scan",
      backend,
      false,
      () => d,
      () => (wakes += 1),
    );
    expect(wakes).toBe(0);
    useInstances.getState().setTable("scan", d);
    expect(wakes).toBe(1);
    useInstances.getState().setHidden("scan", [3], true);
    useInstances.getState().highlight("scan", [1]);
    expect(wakes).toBe(3);
    // Nothing drawn changed: no frame.
    useInstances.getState().setQuery("scan", "oak");
    expect(wakes).toBe(3);
    // The scan's instances going away is a change too.
    useInstances.setState({ assets: {} });
    expect(backend.styles.at(-1)).toBeNull();
    expect(wakes).toBe(4);
    unlink();
  });

  it("reports a renderer that cannot draw them, and clears it when it stops", () => {
    const off = linkScanInstances("scan", fakeBackend("spark", false), false, () => doc());
    expect(useInstances.getState().gaps.scan?.renderer).toBe("spark");
    off();
    expect(useInstances.getState().gaps.scan).toBeUndefined();
    const native = linkScanInstances("scan", fakeBackend(), true, () => doc());
    expect(useInstances.getState().gaps.scan?.reason).toMatch(/own format/);
    native();
    expect(instanceGap(fakeBackend(), false)).toBeNull();
  });
});

describe("the yard's tiles, as Spark digests them", () => {
  const instances = parseInstances(
    JSON.parse(readFileSync(yardPath("../instances/instances.json"), "utf8")),
  );

  it("reads each SPZ's centres bit for bit, so every tile is found in instances.json", async () => {
    if (!instances) throw new Error("no document");
    for (const tile of yardTiles.values()) {
      const glb = readFileSync(yardPath(tile.uri));
      const bytes = spzFromGlb(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength));
      const positions = await spzPositions(bytes);
      expect(positions, tile.uri).toBeDefined();
      if (!positions) continue;
      const checksum = checksumPositions(positions);
      expect(checksum, tile.uri).toBe(checksumPositions(tile.local));
      expect(tileInstanceIds(instances, checksum)?.length, tile.uri).toBe(positions.length / 3);
    }
  });
});

describe("the native level-of-detail probe", () => {
  const TILESET = "https://example.test/scan/tileset.json";
  beforeEach(() => localStorage.clear());

  it("reads what the tileset declares", () => {
    expect(declaredNativeLod(undefined)).toBeUndefined();
    expect(declaredNativeLod({ instances: {} })).toBeUndefined();
    expect(declaredNativeLod({ nativeLod: false })).toBe(false);
    expect(declaredNativeLod({ nativeLod: "sog/lod-meta.json" })).toBe("sog/lod-meta.json");
    expect(declaredNativeLod({ nativeLod: { uri: "x/lod-meta.json" } })).toBe("x/lod-meta.json");
    expect(declaredNativeLod({ nativeLod: 3 })).toBeUndefined();
  });

  it("asks nothing when declared, and probes an undeclared scan only until it knows", async () => {
    const probe = vi.fn((url: string) => Promise.resolve(url.includes("has-one")));
    expect(await findNativeLod(TILESET, { nativeLod: false }, probe)).toBeNull();
    expect(await findNativeLod(TILESET, { nativeLod: "sog/lod-meta.json" }, probe)).toBe(
      `https://example.test/scan/${NATIVE_LOD_PATH}`,
    );
    expect(probe).not.toHaveBeenCalled();
    // Undeclared and absent: probed once, then remembered.
    expect(await findNativeLod(TILESET, {}, probe)).toBeNull();
    expect(await findNativeLod(TILESET, {}, probe)).toBeNull();
    expect(probe).toHaveBeenCalledTimes(1);
    // Undeclared and present: found, as before, every time.
    const native = "https://example.test/has-one/tileset.json";
    expect(await findNativeLod(native, undefined, probe)).toBe(
      `https://example.test/has-one/${NATIVE_LOD_PATH}`,
    );
    expect(await findNativeLod(native, undefined, probe)).toBe(
      `https://example.test/has-one/${NATIVE_LOD_PATH}`,
    );
    expect(probe).toHaveBeenCalledTimes(3);
  });

  it("probes again once the remembered absence is old", async () => {
    const probe = vi.fn(() => Promise.resolve(false));
    const now = vi.spyOn(Date, "now").mockReturnValue(1_000);
    await findNativeLod(TILESET, {}, probe);
    now.mockReturnValue(1_000 + 4 * 24 * 3600 * 1000);
    await findNativeLod(TILESET, {}, probe);
    expect(probe).toHaveBeenCalledTimes(2);
    now.mockRestore();
  });
});
