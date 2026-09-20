/**
 * The committed synthetic tree, and a fake splat primitive built from it.
 *
 * Shared by the frame and deformer tests. The fake reproduces the engine's shape closely enough
 * that everything except the GL call itself is exercised headlessly: real positions, a real bake
 * matrix of the shape `transformTile` composes, the real addressing parameters, and a texture
 * whose `copyFrom` records what would have been uploaded.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { vi } from "vitest";

import { parseRig, type MotionRig } from "@twin/world";

import {
  invertAffine,
  transformPoint,
  transformPositions,
  wgs84SurfaceNormal,
  type Mat4,
} from "@/cesium/splatFrames";
import type { SplatTexture, SplatTile, SplatTilesetLike } from "@/cesium/splatInternals";

/**
 * `data/tiles/synthetic-tree/`. Resolved from the working directory rather than
 * `import.meta.url`, which vitest's jsdom environment serves as an http URL.
 */
export function fixturePath(name: string): string {
  return resolve(process.cwd(), "../../data/tiles/synthetic-tree", name);
}

/** Node's `readFileSync` can return a view into a shared pool; copy before reading as float32. */
export const canonicalPositions = new Float32Array(
  Uint8Array.from(readFileSync(fixturePath("source/positions.f32"))).buffer,
);

export const fixtureRig: MotionRig = parseRig(readFileSync(fixturePath("source/rig.json"), "utf8"));

const tilesetJson = JSON.parse(readFileSync(fixturePath("splat/tileset.json"), "utf8")) as {
  root: { transform: number[]; boundingVolume: { box: number[] } };
};

/** Column-major 4×4 product, `a · b`. */
export function multiplyMat4(a: Mat4, b: Mat4): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

/**
 * `Transforms.eastNorthUpToFixedFrame`, transcribed. The engine builds
 * `primitive._rootTransform` with this from the tileset's bounding-sphere centre.
 */
export function eastNorthUpToFixedFrame(x: number, y: number, z: number): number[] {
  const up = wgs84SurfaceNormal(x, y, z);
  const eastLength = Math.hypot(-y, x);
  const east: [number, number, number] =
    eastLength > 0 ? [-y / eastLength, x / eastLength, 0] : [1, 0, 0];
  const north: [number, number, number] = [
    up[1] * east[2] - up[2] * east[1],
    up[2] * east[0] - up[0] * east[2],
    up[0] * east[1] - up[1] * east[0],
  ];
  return [...east, 0, ...north, 0, ...up, 0, x, y, z, 1];
}

/** The tileset's own ENU→ECEF placement, straight from the committed `tileset.json`. */
export const rootPlacement: readonly number[] = tilesetJson.root.transform;

const box = tilesetJson.root.boundingVolume.box;
const boundsCentre = transformPoint(
  rootPlacement,
  box[0] ?? 0,
  box[1] ?? 0,
  box[2] ?? 0,
  [0, 0, 0],
);

/** `primitive._rootTransform`: the ENU frame at the tileset's bounding-sphere centre. */
export const rootTransform: readonly number[] = eastNorthUpToFixedFrame(
  boundsCentre[0],
  boundsCentre[1],
  boundsCentre[2],
);

/** `B = inverse(rootTransform) · placement`, the shape `transformTile` composes and caches. */
export const bakeMatrix: readonly number[] = multiplyMat4(
  invertAffine(rootTransform) ?? [],
  rootPlacement,
);

/** What the engine would leave in `primitive._positions`, float32 rounding and all. */
export function bakeFixture(local: Float32Array, matrix: Mat4 = bakeMatrix): Float32Array {
  return transformPositions(local, matrix, new Float32Array(local.length));
}

/** One recorded upload, with the words copied out before the staging buffer moves on. */
export interface RecordedUpload {
  readonly width: number;
  readonly height: number;
  readonly xOffset: number;
  readonly yOffset: number;
  readonly words: Uint32Array;
}

/** A stand-in for `Renderer/Texture` that records rather than uploads. */
export class FakeSplatTexture implements SplatTexture {
  readonly uploads: RecordedUpload[] = [];
  destroyed = false;

  readonly copyFrom = vi.fn(
    (options: {
      source: { width: number; height: number; arrayBufferView: Uint32Array };
      xOffset?: number;
      yOffset?: number;
    }): void => {
      this.uploads.push({
        width: options.source.width,
        height: options.source.height,
        xOffset: options.xOffset ?? 0,
        yOffset: options.yOffset ?? 0,
        words: Uint32Array.from(options.source.arrayBufferView),
      });
    },
  );

  isDestroyed(): boolean {
    return this.destroyed;
  }
}

/** A mutable stand-in for the engine's primitive, shaped as `SplatPrimitive` declares it. */
export class FakeSplatPrimitive {
  _positions: Float32Array;
  _numSplats: number;
  _splatRowMask: number;
  _splatRowShift: number;
  _snapshot: { generation: number };
  _rootTransform: readonly number[];
  selectedTileLength: number;
  gaussianSplatTexture: SplatTexture | undefined;

  constructor(positions: Float32Array, options: { rowShift?: number } = {}) {
    const rowShift = options.rowShift ?? 12;
    this._positions = positions;
    this._numSplats = positions.length / 3;
    this._splatRowShift = rowShift;
    this._splatRowMask = (1 << rowShift) - 1;
    this._snapshot = { generation: 1 };
    this._rootTransform = rootTransform;
    this.selectedTileLength = 1;
    this.gaussianSplatTexture = new FakeSplatTexture();
  }

  get texture(): FakeSplatTexture {
    const texture = this.gaussianSplatTexture;
    if (!(texture instanceof FakeSplatTexture)) throw new Error("no fake texture attached");
    return texture;
  }
}

/** A mutable stand-in for `Cesium3DTileset`, viewed as the deformer sees it. */
export class FakeSplatTileset implements SplatTilesetLike {
  gaussianSplatPrimitive: FakeSplatPrimitive | undefined;
  root: { children: SplatTile[]; content: { _lastSplatTransform: readonly number[] | undefined } };

  constructor(primitive: FakeSplatPrimitive, matrix: readonly number[] = bakeMatrix) {
    this.gaussianSplatPrimitive = primitive;
    this.root = { children: [], content: { _lastSplatTransform: matrix } };
  }
}

/**
 * The packed RGBA32UI buffer the engine's WASM packer would produce: the baked positions in the
 * even columns, an arbitrary but distinctive pattern everywhere else.
 *
 * Deliberately shorter than the texture by one row, as the real packer's output is, so every
 * test that goes through here also exercises the zero-padding.
 */
export function packedBufferFor(baked: Float32Array): Uint32Array {
  const numSplats = baked.length / 3;
  // Eight words per splat, contiguous: the packer's layout is width-independent, which is what
  // lets the engine reinterpret it as a texture of whatever width the device allows.
  const words = new Uint32Array(numSplats * 8);
  const floats = new Float32Array(words.buffer);
  for (let i = 0; i < numSplats; i += 1) {
    floats[i * 8] = baked[i * 3] ?? 0;
    floats[i * 8 + 1] = baked[i * 3 + 1] ?? 0;
    floats[i * 8 + 2] = baked[i * 3 + 2] ?? 0;
    words[i * 8 + 3] = 0xdeadbeef;
    for (let c = 4; c < 8; c += 1) words[i * 8 + c] = (i * 2654435761 + c) >>> 0;
  }
  return words;
}
