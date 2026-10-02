/**
 * The real `MotionTextureFactory`: `Renderer/Texture` from the CesiumJS barrel.
 *
 * Split from `splatGpuMotion.ts` so that module — and `SplatDeformer` — stay importable, and
 * unit-testable, without loading CesiumJS. `Texture`, `Sampler`, `PixelFormat`,
 * `PixelDatatype` and `ShaderDestination` are all exported from the barrel at runtime
 * (`@cesium/engine/index.js`) but `Texture` and `ShaderDestination` are not declared in
 * `Cesium.d.ts`, so they are read the way `splatCapture.ts` reads the texture generator.
 */

import * as CesiumBarrel from "cesium";

import type { MotionTextureFactory, OwnedTexture } from "./splatGpuMotion";

type TextureConstructor = new (options: Record<string, unknown>) => OwnedTexture;

interface Barrel {
  Texture?: TextureConstructor;
  Sampler?: { NEAREST?: unknown };
  PixelFormat?: { RGBA?: number; RGBA_INTEGER?: number };
  PixelDatatype?: { FLOAT?: number; UNSIGNED_INT?: number };
  ShaderDestination?: { VERTEX?: number };
}

/** The factory, or `undefined` when this CesiumJS build does not export what it needs. */
export function cesiumMotionTextures(): MotionTextureFactory | undefined {
  const barrel = CesiumBarrel as unknown as Barrel;
  const Texture = barrel.Texture;
  const nearest = barrel.Sampler?.NEAREST;
  const rgba = barrel.PixelFormat?.RGBA;
  const rgbaInteger = barrel.PixelFormat?.RGBA_INTEGER;
  const float = barrel.PixelDatatype?.FLOAT;
  const unsignedInt = barrel.PixelDatatype?.UNSIGNED_INT;
  const vertex = barrel.ShaderDestination?.VERTEX;
  if (
    typeof Texture !== "function" ||
    nearest === undefined ||
    rgba === undefined ||
    rgbaInteger === undefined ||
    float === undefined ||
    unsignedInt === undefined ||
    vertex === undefined
  ) {
    return undefined;
  }
  const create = (
    context: unknown,
    width: number,
    height: number,
    data: Float32Array | Uint32Array,
    pixelFormat: number,
    pixelDatatype: number,
  ): OwnedTexture =>
    new Texture({
      context,
      source: { width, height, arrayBufferView: data },
      pixelFormat,
      pixelDatatype,
      preMultiplyAlpha: false,
      skipColorSpaceConversion: true,
      flipY: false,
      sampler: nearest,
    });
  return {
    createFloat: (context, width, height, data) =>
      create(context, width, height, data, rgba, float),
    createUintQuads: (context, width, height, data) =>
      create(context, width, height, data, rgbaInteger, unsignedInt),
    vertexDestination: vertex,
  };
}
