/**
 * Pins and leader ends sit at the height of the surface drawn under them. A sample read back
 * before that surface was drawn can come back far underground, which puts a pin between the
 * screen's centre and its spot; such readings are retried, never kept.
 */
import { Ellipsoid } from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { LandAnchorProjector } from "@/cesium/LandAnchorProjector";
import type { LandPoint } from "@/state/land";

function host(samples: (number | undefined)[], terrain?: number) {
  const sampleHeight = vi.fn(() => samples.shift());
  const manager = {
    isDestroyed: false,
    camera: { isMoving: false },
    scene: {
      sampleHeightSupported: true,
      sampleHeight,
      requestRender: () => undefined,
      postRender: { addEventListener: () => () => undefined },
      globe: { getHeight: () => terrain, ellipsoid: Ellipsoid.WGS84 },
    },
    viewer: { entities: { values: [] } },
  } as unknown as CesiumSceneManager;
  return { manager, sampleHeight };
}

/** Each call is one rendered frame's worth of sampling for this point. */
function heightAt(projector: LandAnchorProjector, point: LandPoint): number {
  (projector as unknown as { samplesLeft: number }).samplesLeft = 3;
  return (projector as unknown as { heightAt(point: LandPoint): number }).heightAt(point);
}

describe("LandAnchorProjector heights", () => {
  it("keeps a plausible surface sample and stops sampling that point", () => {
    const { manager, sampleHeight } = host([71]);
    const projector = new LandAnchorProjector(manager);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(71);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(71);
    expect(sampleHeight).toHaveBeenCalledTimes(1);
  });

  it("retries a sample from far underground instead of keeping it", () => {
    const { manager, sampleHeight } = host([-3_000, 71]);
    const projector = new LandAnchorProjector(manager);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(0);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(71);
    expect(sampleHeight).toHaveBeenCalledTimes(2);
  });

  it("rejects a sample far from the terrain drawn there", () => {
    const { manager } = host([2_400, 64], 60);
    const projector = new LandAnchorProjector(manager);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(60);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(64);
  });

  it("settles on the globe after repeated misses rather than sampling forever", () => {
    const { manager, sampleHeight } = host(
      Array.from({ length: 20 }, () => undefined),
      12,
    );
    const projector = new LandAnchorProjector(manager);
    for (let frame = 0; frame < 20; frame++) heightAt(projector, [-119.9, 36.6]);
    expect(sampleHeight).toHaveBeenCalledTimes(12);
    expect(heightAt(projector, [-119.9, 36.6])).toBe(12);
  });
});
