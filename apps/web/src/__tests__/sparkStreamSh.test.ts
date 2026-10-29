import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";

import { loadSplatTile } from "@/view/sparkStream";

const made = vi.hoisted(() => [] as { maxSh: number; updateGenerator: ReturnType<typeof vi.fn> }[]);

vi.mock("@sparkjsdev/spark", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  class SplatMesh extends three.Object3D {
    readonly initialized: Promise<SplatMesh>;
    maxSh = 3;
    readonly updateGenerator = vi.fn();
    constructor(readonly options: Record<string, unknown>) {
      super();
      this.initialized = Promise.resolve(this);
      made.push(this);
    }
  }
  return { SplatMesh, SplatFileType: { SPZ: "spz" } };
});

describe("scan gallery SH cap", () => {
  afterEach(() => {
    made.length = 0;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it.each([
    { device: "phone", coarse: true, width: 390, height: 844, maxSh: 1 },
    { device: "tablet", coarse: true, width: 820, height: 1180, maxSh: 1 },
    { device: "desktop", coarse: false, width: 1920, height: 1080, maxSh: 3 },
    { device: "small fine-pointer screen", coarse: false, width: 390, height: 844, maxSh: 3 },
    { device: "large coarse-pointer screen", coarse: true, width: 1920, height: 1080, maxSh: 3 },
  ])("passes maxSh $maxSh to SplatMesh on a $device", async ({ coarse, width, height, maxSh }) => {
    const matchMedia = vi.fn(() => ({ matches: coarse }));
    vi.stubGlobal("matchMedia", matchMedia);
    vi.spyOn(window.screen, "width", "get").mockReturnValue(width);
    vi.spyOn(window.screen, "height", "get").mockReturnValue(height);
    const bytes = readFileSync(
      resolve(process.cwd(), "../../data/tiles/synthetic-far/splat/splat.glb"),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(new Uint8Array(bytes)))),
    );

    await loadSplatTile("https://tiles.example/tileset.json", {
      uri: "splat.glb",
      gaussians: 950,
      geometricError: 0,
      children: [],
    });

    expect(matchMedia).toHaveBeenCalledWith("(pointer: coarse)");
    expect(made).toHaveLength(1);
    expect(made[0]?.maxSh).toBe(maxSh);
    expect(made[0]?.updateGenerator).toHaveBeenCalledTimes(maxSh < 3 ? 1 : 0);
  });
});
