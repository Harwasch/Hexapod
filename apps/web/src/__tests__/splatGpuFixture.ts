/**
 * Stand-ins for the GPU motion path's collaborators: a texture factory that keeps what it was
 * given, primitives carrying the engine patch's `vertexMotion` accessor, and what the patched
 * `buildGSplatDrawCommand` does with the hook.
 */

import { expect } from "vitest";

import type { SplatGpuMotion, MotionTextureFactory, OwnedTexture } from "@/cesium/splatGpuMotion";
import type { SplatShaderBuilder } from "@/cesium/splatInternals";

import { FakeSplatPrimitive, FakeSplatTexture } from "./splatFixture";
import { FakeTiledPrimitive } from "./splatTilesFixture";

/** Textures that keep what they were given, for reading the GPU path's uploads back. */
export class FakeOwnedTexture extends FakeSplatTexture implements OwnedTexture {
  constructor(
    readonly width: number,
    readonly height: number,
    readonly initial: Float32Array | Uint32Array,
  ) {
    super();
  }
  destroy(): void {
    this.destroyed = true;
  }
}

export function fakeFactory(): MotionTextureFactory & { made: FakeOwnedTexture[] } {
  const made: FakeOwnedTexture[] = [];
  const make = (
    _context: unknown,
    width: number,
    height: number,
    data: Float32Array | Uint32Array,
  ): FakeOwnedTexture => {
    const texture = new FakeOwnedTexture(width, height, data.slice());
    made.push(texture);
    return texture;
  };
  return { createFloat: make, createUintPairs: make, vertexDestination: 0, made };
}

/** The tiled primitive, with the patch's accessor. */
export class FakeHookedPrimitive extends FakeTiledPrimitive {
  vertexMotion: SplatGpuMotion | undefined = undefined;
  isDestroyed(): boolean {
    return false;
  }
}

/** The single-tile primitive, with the patch's accessor. */
export class FakeHookedSinglePrimitive extends FakeSplatPrimitive {
  vertexMotion: SplatGpuMotion | undefined = undefined;
  isDestroyed(): boolean {
    return false;
  }
}

/** What the patched `buildGSplatDrawCommand` does with the hook. */
export function buildDrawCommand(primitive: {
  vertexMotion: SplatGpuMotion | undefined;
}): Record<string, () => unknown> {
  const uniformMap: Record<string, () => unknown> = {};
  const lines: string[] = [];
  const builder: SplatShaderBuilder = {
    addUniform: (type, name) => lines.push(`uniform ${type} ${name};`),
    addVertexLines: (text) => lines.push(...(typeof text === "string" ? [text] : text)),
  };
  primitive.vertexMotion?.addToShader(builder, uniformMap, { fake: "context" });
  expect(lines.join("\n")).toContain("vec3 splatVertexMotion(uint splatIndex, vec3 position)");
  return uniformMap;
}
