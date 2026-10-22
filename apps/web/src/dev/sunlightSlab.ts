/**
 * A one-tile splat tileset for the sunlight scene spec (e2e/sunlightScene.spec.ts): a wall in
 * the north-up plane through the origin, facing east -- its south half flat discs (a normal
 * each, east or west), its north half round splats of the same grey (no normal: neutral). Built
 * here in TypeScript, as `tools/captures/splat_tiles.py` `pack_spz` and `build_glb` write a
 * tile (SPZ v2, 12 fractional bits, KHR_gaussian_splatting with the SPZ compression), so the
 * spec needs no committed fixture. Pure: the caller gzips (node:zlib in the spec).
 *
 * Loaded by the spec only; nothing in the app imports it.
 */

const SPZ_MAGIC = 0x5053474e;
const SPZ_VERSION = 2;
const FRACTIONAL_BITS = 12;

export interface SlabSplat {
  /** East, north, up (metres, the tile's frame). */
  position: readonly [number, number, number];
  /** Axis scales (metres). */
  scale: readonly [number, number, number];
}

/** The slab's place: the synthetic yard's frame origin (synthetic_yard.py). */
export const SLAB_ORIGIN = { lon: -123.88, lat: 46.1335, height: 0 } as const;
/** Where the round splats start (north of this, metres) and the discs end. */
export const SLAB_SPLIT_NORTH_M = 0.1;
/** The splats' grey (sRGB byte), the same for both halves. */
export const SLAB_GREY = 150;

/**
 * An inferred layer for the slab (lib/inferred.ts): east-facing discs like the wall's, but red
 * (`SLAB_LAYER_RGB`) and 40 cm in front of its north half -- flat, so a relight that did not
 * skip it would darken it as it darkens the wall's discs.
 */
export function slabLayerSplats(): SlabSplat[] {
  const out: SlabSplat[] = [];
  for (let i = 0; i < 10; i += 1) {
    for (let k = 0; k < 20; k += 1) {
      out.push({ position: [0.4, 0.5 + i * 0.1, 0.5 + k * 0.1], scale: [0.004, 0.09, 0.09] });
    }
  }
  return out;
}

/** The layer's colour (sRGB bytes): red, told apart from the grey wall by its hue alone. */
export const SLAB_LAYER_RGB = [220, 70, 60] as const;

/** The wall's splats: 30 x 30 discs south of the split, 30 x 30 round splats north of it. */
export function slabSplats(): SlabSplat[] {
  const out: SlabSplat[] = [];
  for (let i = 0; i < 30; i += 1) {
    for (let k = 0; k < 30; k += 1) {
      const up = 0.05 + k * 0.1;
      // Flat: 4 mm thick across the wall (east), 9 cm along it.
      out.push({ position: [0, -3 + 0.05 + i * 0.1, up], scale: [0.004, 0.09, 0.09] });
      // Round: 7 cm every way.
      out.push({
        position: [0, SLAB_SPLIT_NORTH_M + 0.05 + i * 0.1, up],
        scale: [0.07, 0.07, 0.07],
      });
    }
  }
  return out;
}

/** SPZ v2, uncompressed (the caller gzips): header, then positions, alphas, colours, scales, rotations. */
export function slabSpz(
  splats: readonly SlabSplat[],
  rgb: readonly [number, number, number] = [SLAB_GREY, SLAB_GREY, SLAB_GREY],
): Uint8Array {
  const n = splats.length;
  const bytes = new Uint8Array(16 + n * 19);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, SPZ_MAGIC, true);
  view.setUint32(4, SPZ_VERSION, true);
  view.setUint32(8, n, true);
  bytes[12] = 0; // SH degree
  bytes[13] = FRACTIONAL_BITS;
  let at = 16;
  for (const s of splats) {
    for (const v of s.position) {
      const fixed = Math.round(v * (1 << FRACTIONAL_BITS)) & 0xffffff;
      bytes[at] = fixed & 0xff;
      bytes[at + 1] = (fixed >> 8) & 0xff;
      bytes[at + 2] = (fixed >> 16) & 0xff;
      at += 3;
    }
  }
  for (let i = 0; i < n; i += 1) bytes[at++] = 250;
  // Colour bytes: sh0 * (0.15 * 255) + 127.5, with the engine reading 0.5 + C0 * sh0; the byte
  // for an sRGB channel g is 127.5 + (g / 255 - 0.5) / C0 * 38.25.
  const colour = rgb.map((g) =>
    Math.round(127.5 + ((g / 255 - 0.5) / 0.28209479177387814) * 38.25),
  );
  for (let i = 0; i < n; i += 1) for (const c of colour) bytes[at++] = c;
  for (const s of splats) {
    for (const v of s.scale) bytes[at++] = Math.round((Math.log(v) + 10) * 16);
  }
  // The identity rotation's x, y, z (w is implied, positive).
  for (let i = 0; i < n * 3; i += 1) bytes[at++] = 128;
  return bytes;
}

/** One point primitive whose attributes live in the SPZ block (splat_tiles.py `build_glb`). */
export function slabGlb(splats: readonly SlabSplat[], spz: Uint8Array): Uint8Array {
  const count = splats.length;
  const min = [0, 1, 2].map((a) => Math.min(...splats.map((s) => s.position[a] ?? 0)));
  const max = [0, 1, 2].map((a) => Math.max(...splats.map((s) => s.position[a] ?? 0)));
  const padded = new Uint8Array(Math.ceil(spz.length / 4) * 4);
  padded.set(spz);
  const gltf = {
    asset: { version: "2.0", generator: "twin sunlightSlab" },
    extensionsUsed: [
      "KHR_materials_unlit",
      "KHR_gaussian_splatting",
      "KHR_gaussian_splatting_compression_spz_2",
    ],
    extensionsRequired: ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"],
    scene: 0,
    scenes: [{ nodes: [0] }],
    // Data is z-up (east, north, up); glTF is y-up. Same swap as the tiler's.
    nodes: [{ mesh: 0, matrix: [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1] }],
    meshes: [
      {
        primitives: [
          {
            mode: 0,
            material: 0,
            attributes: {
              POSITION: 0,
              COLOR_0: 1,
              "KHR_gaussian_splatting:SCALE": 2,
              "KHR_gaussian_splatting:ROTATION": 3,
            },
            extensions: {
              KHR_gaussian_splatting: {
                extensions: { KHR_gaussian_splatting_compression_spz_2: { bufferView: 0 } },
              },
            },
          },
        ],
      },
    ],
    materials: [{ extensions: { KHR_materials_unlit: {} } }],
    accessors: [
      { componentType: 5126, count, type: "VEC3", min, max },
      { componentType: 5121, normalized: true, count, type: "VEC4" },
      { componentType: 5126, count, type: "VEC3" },
      { componentType: 5126, count, type: "VEC4" },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: spz.length }],
    buffers: [{ byteLength: padded.length }],
  };
  let json = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonPadded = new Uint8Array(Math.ceil(json.length / 4) * 4).fill(0x20);
  jsonPadded.set(json);
  json = jsonPadded;
  const total = 12 + 8 + json.length + 8 + padded.length;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, json.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.set(json, 20);
  view.setUint32(20 + json.length, padded.length, true);
  view.setUint32(24 + json.length, 0x004e4942, true);
  out.set(padded, 28 + json.length);
  return out;
}

/** East-north-up at `origin` to Earth-fixed, column-major (splat_tiles.py `enu_to_ecef`). */
export function enuToEcef(origin: { lon: number; lat: number; height: number }): number[] {
  const a = 6378137;
  const e2 = 6.69437999014e-3;
  const lat = (origin.lat * Math.PI) / 180;
  const lon = (origin.lon * Math.PI) / 180;
  const [sl, cl, sp, cp] = [Math.sin(lon), Math.cos(lon), Math.sin(lat), Math.cos(lat)];
  const n = a / Math.sqrt(1 - e2 * sp * sp);
  const x = (n + origin.height) * cp * cl;
  const y = (n + origin.height) * cp * sl;
  const z = (n * (1 - e2) + origin.height) * sp;
  return [-sl, cl, 0, 0, -sp * cl, -sp * sl, cp, 0, cp * cl, cp * sl, sp, 0, x, y, z, 1];
}

/**
 * The slab's tileset.json, its one tile at `content` (`slab.glb`); `extras` on its root (the
 * layer's `evidence`, or the measured slab's `inferredLayers`).
 */
export function slabTileset(
  splats: readonly SlabSplat[],
  options: { content?: string; extras?: Record<string, unknown> } = {},
): unknown {
  const min = [0, 1, 2].map((a) => Math.min(...splats.map((s) => s.position[a] ?? 0)) - 0.1);
  const max = [0, 1, 2].map((a) => Math.max(...splats.map((s) => s.position[a] ?? 0)) + 0.1);
  const centre = min.map((v, a) => (v + (max[a] ?? 0)) / 2);
  const half = min.map((v, a) => ((max[a] ?? 0) - v) / 2);
  const extensions = ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"];
  return {
    asset: { version: "1.1" },
    geometricError: 8,
    extensionsUsed: ["3DTILES_content_gltf"],
    extensions: {
      "3DTILES_content_gltf": { extensionsUsed: extensions, extensionsRequired: extensions },
    },
    root: {
      transform: enuToEcef(SLAB_ORIGIN),
      refine: "REPLACE",
      boundingVolume: {
        box: [...centre, half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]],
      },
      geometricError: 0,
      content: { uri: options.content ?? "slab.glb" },
      ...(options.extras ? { extras: options.extras } : {}),
    },
  };
}
