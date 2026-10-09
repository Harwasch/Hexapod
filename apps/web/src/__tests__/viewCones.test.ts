import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

import { Event, type Cesium3DTileset } from "cesium";
import { describe, expect, it, vi } from "vitest";

import {
  attachLayerViewCones,
  COMPANION_SLOT_VECS,
  CompanionViewCones,
  companionSlots,
  inCompanionSlots,
  SplatViewCones,
  viewConesEnabled,
  type SplatVertexVisibility,
  type ViewConeGpu,
} from "@/cesium/splatViewCones";
import type { SplatShaderBuilder } from "@/cesium/splatInternals";
import { visibilityChainOf } from "@/cesium/splatVisibility";
import {
  cellCount,
  decodeAxis,
  gridFromModel,
  OMNI,
  textureRows,
  VIEW_CONES_TEXTURE_WIDTH,
  viewConesMetaOf,
  visibility,
  type ViewConesMeta,
} from "@/lib/viewCones";

const META: ViewConesMeta = {
  format: "hexapod.viewcones",
  version: 1,
  uri: "viewcones.bin",
  origin: [-5, -5, 0],
  cell: 0.5,
  dims: [20, 20, 10],
  fadeDeg: 20,
};

const deg = (rad: number): number => (rad * 180) / Math.PI;

/** `view_cones.encode_axis`, for a unit vector with z >= 0. */
function encodeUpper(v: readonly [number, number, number]): [number, number] {
  const n = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]);
  return [Math.round(((v[0] / n) * 0.5 + 0.5) * 255), Math.round(((v[1] / n) * 0.5 + 0.5) * 255)];
}

describe("extras.viewCones", () => {
  it("reads the packer's declaration and refuses anything else", () => {
    expect(viewConesMetaOf({ viewCones: META })).toEqual(META);
    expect(viewConesMetaOf({ collision: {} })).toBeNull();
    expect(viewConesMetaOf({ viewCones: { ...META, version: 2 } })).toBeNull();
    expect(viewConesMetaOf({ viewCones: { ...META, cell: 0 } })).toBeNull();
    expect(viewConesMetaOf({ viewCones: { ...META, dims: [20, 20] } })).toBeNull();
    expect(viewConesMetaOf({ viewCones: { ...META, dims: [20, 1.5, 3] } })).toBeNull();
    expect(viewConesMetaOf(undefined)).toBeNull();
  });

  it("is declared, and sized, on the committed yard", () => {
    const root = resolve(__dirname, "../../../../data/tiles/synthetic-yard/splat");
    const tileset = JSON.parse(readFileSync(resolve(root, "tileset.json"), "utf8")) as {
      root: { extras: unknown };
    };
    const meta = viewConesMetaOf(tileset.root.extras);
    expect(meta).not.toBeNull();
    const texels = gunzipSync(readFileSync(resolve(root, meta?.uri ?? "")));
    if (!meta) throw new Error("no extras.viewCones");
    expect(texels.length).toBe(cellCount(meta) * 4);
  });
});

describe("the grid's frame", () => {
  it("takes a baked position to cell coordinates through the inverse bake", () => {
    // B = translate(10, 0, 0): local (x, y, z) is baked (x + 10, y, z).
    const inverseBake = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1];
    const m = gridFromModel(META, inverseBake);
    const local: [number, number, number] = [-5 + 2.5 * 0.5, -5 + 4 * 0.5, 1.5];
    const baked: [number, number, number] = [local[0] + 10, local[1], local[2]];
    const at = (i: number): number => m[i] ?? Number.NaN;
    const g = [0, 1, 2].map(
      (r) => at(r) * baked[0] + at(4 + r) * baked[1] + at(8 + r) * baked[2] + at(12 + r),
    );
    expect(g[0]).toBeCloseTo(2.5, 9);
    expect(g[1]).toBeCloseTo(4, 9);
    expect(g[2]).toBeCloseTo(3, 9);
    expect(m.slice(12)).toEqual([expect.any(Number), expect.any(Number), expect.any(Number), 1]);
  });

  it("pads the texels to whole texture rows", () => {
    const texels = new Uint8Array(cellCount(META) * 4).fill(7);
    const { data, height } = textureRows(texels);
    expect(height).toBe(Math.ceil(cellCount(META) / VIEW_CONES_TEXTURE_WIDTH));
    expect(data.length).toBe(VIEW_CONES_TEXTURE_WIDTH * height * 4);
    expect(data.subarray(0, texels.length).every((v) => v === 7)).toBe(true);
  });
});

describe("the fade", () => {
  it("decodes the packer's octahedral axes", () => {
    for (const v of [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [Math.SQRT1_2, 0, Math.SQRT1_2],
    ] as const) {
      const [r, g] = encodeUpper(v);
      const back = decodeAxis(r, g);
      const angle = deg(Math.acos(v[0] * back[0] + v[1] * back[1] + v[2] * back[2]));
      expect(angle).toBeLessThan(1.5);
    }
  });

  it("is 1 inside the cone, half way through the margin, 0 past it, 1 seen from everywhere", () => {
    const [r, g] = encodeUpper([0, 0, 1]); // seen from below: the axis points up
    const texel = [r, g, Math.round((40 * 254) / 180), 255] as const;
    const at = (angleDeg: number): number => {
      const a = (angleDeg * Math.PI) / 180;
      return visibility(texel, [Math.sin(a), 0, Math.cos(a)], 20);
    };
    expect(at(0)).toBe(1); // looking up at it, as the capture did
    expect(at(39)).toBe(1);
    expect(at(50)).toBeCloseTo(0.5, 1);
    expect(at(61)).toBe(0);
    expect(at(180)).toBe(0); // looking down on it from above
    expect(visibility([128, 128, OMNI, 255], [0, 0, -1], 20)).toBe(1);
  });
});

describe("the vertexVisibility hook", () => {
  const created: number[][] = [];
  const gpu: ViewConeGpu = {
    createBytes: (_context, width, height) => {
      created.push([width, height]);
      let destroyed = false;
      return {
        copyFrom: () => undefined,
        isDestroyed: () => destroyed,
        destroy: () => {
          destroyed = true;
        },
      };
    },
    mat4: (values) => ({ mat4: [...values] }),
    vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
    vertexDestination: 1,
  };

  function build(hook: SplatViewCones): { lines: string; uniforms: Record<string, () => unknown> } {
    const out: string[] = [];
    const uniforms: Record<string, () => unknown> = {};
    const builder: SplatShaderBuilder = {
      addUniform: (type, name) => out.push(`uniform ${type} ${name};`),
      addVertexLines: (text) => out.push(...(typeof text === "string" ? [text] : text)),
    };
    hook.addToShader(builder, uniforms, { fake: "context" });
    return { lines: out.join("\n"), uniforms };
  }

  it("declares its uniforms and the function the patched shader calls", () => {
    const hook = new SplatViewCones(META, new Uint8Array(cellCount(META) * 4), gpu);
    const { lines, uniforms } = build(hook);
    expect(lines).toContain("float splatViewConeVisibility(uint splatIndex, vec3 position)");
    for (const name of ["u_viewCones", "u_viewConeFromModel", "u_viewConeDims", "u_viewConeFade"]) {
      expect(lines).toContain(name);
      expect(uniforms[name]).toBeTypeOf("function");
    }
    expect(uniforms.u_viewConeDims?.()).toEqual({ vec4: [20, 20, 10, VIEW_CONES_TEXTURE_WIDTH] });
    expect(created.at(-1)).toEqual([VIEW_CONES_TEXTURE_WIDTH, 1]);
  });

  it("draws everything until a tile's bake is known, then fades through its inverse", () => {
    const hook = new SplatViewCones(META, new Uint8Array(cellCount(META) * 4), gpu);
    const { uniforms } = build(hook);
    const tile = { content: {} as { _lastSplatTransform?: number[] } };
    const primitive = {
      vertexVisibility: undefined as SplatVertexVisibility | undefined,
      _selectedTileSet: new Set([tile]),
      isDestroyed: () => false,
    };
    expect(hook.install(primitive)).toBe(true);
    // In the primitive's visibility chain (splatVisibility.ts), not the slot itself.
    expect(visibilityChainOf(primitive)?.parts).toEqual([hook]);
    expect(hook.installed).toBe(true);
    expect(uniforms.u_viewConeActive?.()).toBe(0);
    tile.content._lastSplatTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1];
    expect(uniforms.u_viewConeActive?.()).toBe(1);
    const expected = gridFromModel(META, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1]);
    const value = uniforms.u_viewConeFromModel?.() as { mat4: number[] };
    value.mat4.forEach((v, i) => expect(v).toBeCloseTo(expected[i] ?? Number.NaN, 12));
    hook.enabled = false;
    expect(uniforms.u_viewConeActive?.()).toBe(0);
    hook.destroy();
    expect(primitive.vertexVisibility).toBeUndefined();
  });

  it("does nothing on an engine without the accessor", () => {
    const hook = new SplatViewCones(META, new Uint8Array(cellCount(META) * 4), gpu);
    expect(hook.install({ isDestroyed: () => false })).toBe(false);
  });

  it("fades a companion's splats on the scan's primitive, by its own grid, in its slots alone", () => {
    const layer = { name: "layer" };
    const scan = { name: "scan" };
    const hook = new CompanionViewCones(META, new Uint8Array(cellCount(META) * 4), gpu, layer);
    const other = new CompanionViewCones(META, new Uint8Array(cellCount(META) * 4), gpu, layer);
    const { lines, uniforms } = build(hook as unknown as SplatViewCones);
    // Named for itself: two layers' parts (and the scan's own cones) share one shader.
    expect(hook.visibilityFunction).not.toBe(other.visibilityFunction);
    expect(lines).toContain(`float ${hook.visibilityFunction}(uint splatIndex, vec3 position)`);
    expect(lines).not.toContain("float splatViewConeVisibility(");
    const bake = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1];
    interface Tile {
      tileset: unknown;
      content: { _lastSplatTransform?: number[] };
    }
    const scanBake = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
    const primitive = {
      vertexVisibility: undefined as SplatVertexVisibility | undefined,
      _tileSlots: new Map<Tile, { start: number; count: number }>([
        [
          { tileset: scan, content: { _lastSplatTransform: scanBake } },
          { start: 0, count: 50 },
        ],
        [
          { tileset: layer, content: { _lastSplatTransform: bake } },
          { start: 50, count: 10 },
        ],
      ]),
      isDestroyed: () => false,
    };
    expect(hook.install(primitive)).toBe(true);
    const key = (name: string): string =>
      Object.keys(uniforms).find((k) => k.startsWith(name)) ?? name;
    expect(uniforms[key("u_coneActive_")]?.()).toBe(1);
    expect(uniforms[key("u_coneSlots_")]?.()).toEqual([
      { vec4: [50, 60, 0, 0] },
      { vec4: [0, 0, 0, 0] },
      { vec4: [0, 0, 0, 0] },
      { vec4: [0, 0, 0, 0] },
    ]);
    // The layer's own bake, not the scan tile's that comes first.
    const expected = gridFromModel(META, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1]);
    const value = uniforms[key("u_coneFromModel_")]?.() as { mat4: number[] };
    value.mat4.forEach((v, i) => expect(v).toBeCloseTo(expected[i] ?? Number.NaN, 12));
    // Its slots gone (Hide): it acts on nothing.
    primitive._tileSlots = new Map([
      [
        { tileset: scan, content: {} },
        { start: 0, count: 50 },
      ],
    ]);
    expect(uniforms[key("u_coneActive_")]?.()).toBe(0);
    hook.destroy();
    expect(primitive.vertexVisibility).toBeUndefined();
  });

  it("fetches a layer's cones once CesiumJS shows it, and hooks them where its tiles are drawn", async () => {
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        asked.push(url);
        return Promise.resolve(new Response(gzipSync(new Uint8Array(cellCount(META) * 4))));
      }),
    );
    try {
      const own = {
        vertexVisibility: undefined as SplatVertexVisibility | undefined,
        isDestroyed: () => false,
      };
      const tileset = {
        root: { extras: { viewCones: META } },
        resource: { url: "https://tiles.example/fill/tileset.json" },
        show: false,
        gaussianSplatPrimitive: own,
        tileLoad: new Event(),
      } as unknown as Cesium3DTileset;
      const host = {
        vertexVisibility: undefined as SplatVertexVisibility | undefined,
        isDestroyed: () => false,
      };
      let drawnByHost = false;
      const cones = attachLayerViewCones(tileset, () => (drawnByHost ? host : undefined), gpu);
      cones.sync();
      // Hidden (another renderer draws the scan, and the layer): nothing fetched.
      expect(asked).toEqual([]);
      (tileset as { show: boolean }).show = true;
      cones.sync();
      expect(asked).toEqual(["https://tiles.example/fill/viewcones.bin"]);
      await vi.waitFor(() => expect(visibilityChainOf(own)?.parts).toHaveLength(1));
      expect(host.vertexVisibility).toBeUndefined();
      // Drawn by the scan's primitive: a part there, within the layer's slots.
      drawnByHost = true;
      cones.sync();
      expect(visibilityChainOf(host)?.parts[0]).toBeInstanceOf(CompanionViewCones);
      drawnByHost = false;
      cones.sync();
      expect(host.vertexVisibility).toBeUndefined();
      cones.sync();
      expect(asked).toHaveLength(1);
      cones.dispose();
      expect(own.vertexVisibility).toBeUndefined();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("packs a companion's slot ranges two to a vector, merged where they touch", () => {
    const layer = {};
    const slots = new Map([
      [{ tileset: layer }, { start: 30, count: 10 }],
      [{ tileset: {} }, { start: 0, count: 30 }],
      [{ tileset: layer }, { start: 40, count: 5 }],
      [{ tileset: layer }, { start: 100, count: 1 }],
    ]);
    const packed = companionSlots({ _tileSlots: slots }, new Set([layer]));
    expect(packed.vecs).toHaveLength(COMPANION_SLOT_VECS);
    expect(packed.vecs[0]).toEqual([30, 45, 100, 101]);
    expect(packed.count).toBe(2);
    expect(packed.overflow).toBe(0);
    expect(inCompanionSlots(packed.vecs, 44)).toBe(true);
    expect(inCompanionSlots(packed.vecs, 45)).toBe(false);
    expect(inCompanionSlots(packed.vecs, 100)).toBe(true);
    expect(inCompanionSlots(packed.vecs, 10)).toBe(false);
    expect(companionSlots(undefined, new Set([layer])).count).toBe(0);
  });

  it("is on unless the page says ?viewCones=off", () => {
    expect(viewConesEnabled("")).toBe(true);
    expect(viewConesEnabled("?viewCones=off")).toBe(false);
    expect(viewConesEnabled("?x=1&viewCones=on")).toBe(true);
  });
});
