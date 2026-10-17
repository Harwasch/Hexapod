/**
 * The photorealistic world's one custom shader (cesium/worldShader.ts): attached on the first
 * lens, never before (it recompiles every tile once), kept attached after (a closed lens is a
 * uniform), carried over to a new world tileset, and a cut by discard, never by alpha.
 */
import type { Cartesian4, Cesium3DTileset, CustomShader } from "cesium";
import { describe, expect, it, vi } from "vitest";

import { WORLD_FRAGMENT_SHADER, WorldShader } from "@/cesium/worldShader";

function stubTileset(): Cesium3DTileset & { assigned: number } {
  const tileset = { assigned: 0, shader: undefined as CustomShader | undefined };
  Object.defineProperty(tileset, "customShader", {
    get: () => tileset.shader,
    set: (value: CustomShader) => {
      tileset.assigned += 1;
      tileset.shader = value;
    },
  });
  return tileset as unknown as Cesium3DTileset & { assigned: number };
}

const lens = { x: 400, y: 300, inner: 100, outer: 140 };
const uniform = (tileset: Cesium3DTileset, name: string) =>
  tileset.customShader!.uniforms[name]?.value as Cartesian4;

describe("the world shader", () => {
  it("is not attached until the first lens, then once", () => {
    const scene = { requestRender: vi.fn() };
    const shader = new WorldShader(scene);
    const tileset = stubTileset();
    shader.setTileset(tileset);
    expect(tileset.assigned).toBe(0);
    expect(shader.attached).toBe(false);
    shader.setLantern(lens);
    expect(tileset.assigned).toBe(1);
    expect(shader.attached).toBe(true);
    expect(uniform(tileset, "u_lantern")).toMatchObject({ x: 400, y: 300, z: 100, w: 140 });
    shader.setLantern({ ...lens, x: 410, limit: { x: 410, y: 300, r: 189 } });
    expect(tileset.assigned).toBe(1);
    expect(uniform(tileset, "u_lantern").x).toBe(410);
    expect(uniform(tileset, "u_lanternLimit")).toMatchObject({ x: 410, y: 300, z: 189 });
    expect(scene.requestRender).toHaveBeenCalled();
  });

  it("closes the lens with a uniform, and stays attached", () => {
    const shader = new WorldShader({ requestRender: vi.fn() });
    const tileset = stubTileset();
    shader.setTileset(tileset);
    shader.setLantern(lens);
    shader.setLantern(null);
    expect(uniform(tileset, "u_lantern").w).toBe(0);
    expect(tileset.customShader).toBeDefined();
    expect(tileset.assigned).toBe(1);
  });

  it("attaches to a new world tileset that replaces the old", () => {
    const shader = new WorldShader({ requestRender: vi.fn() });
    const first = stubTileset();
    shader.setTileset(first);
    shader.setLantern(lens);
    shader.setTileset(null);
    const second = stubTileset();
    shader.setTileset(second);
    expect(second.assigned).toBe(1);
    expect(second.customShader).toBe(first.customShader);
  });

  it("cuts by discard, never through the material's alpha", () => {
    expect(WORLD_FRAGMENT_SHADER).toContain("discard");
    expect(WORLD_FRAGMENT_SHADER).not.toContain("material.alpha");
    expect(WORLD_FRAGMENT_SHADER).toContain("gl_FragCoord");
  });
});
