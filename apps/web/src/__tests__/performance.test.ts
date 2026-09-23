import type { Viewer } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  baseResolutionScale,
  buildLadder,
  decideScreenSpaceError,
  frameEvidence,
  frameWeight,
  PerformanceManager,
  shouldSharpenAtRest,
  splatMinimumScreenSpaceError,
  type QualitySample,
} from "@/cesium/PerformanceManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import { QUALITY_SSE } from "@/state/settings";
import type { PerformanceSnapshot } from "@/state/viewer";

const base: QualitySample = {
  bounds: QUALITY_SSE.balanced,
  current: 16,
  moving: false,
  loading: false,
  memoryRatio: 0.4,
};

describe("decideScreenSpaceError", () => {
  it("refines straight to the preset minimum at rest, at any height", () => {
    expect(decideScreenSpaceError(base)).toEqual({
      screenSpaceError: 2,
      reason: "idle refinement",
    });
    expect(decideScreenSpaceError({ ...base, current: 3 }).screenSpaceError).toBe(2);
    expect(decideScreenSpaceError({ ...base, current: 2 })).toEqual({
      screenSpaceError: 2,
      reason: "at finest",
    });
  });

  it("holds while tiles are loading and while memory headroom is gone", () => {
    expect(decideScreenSpaceError({ ...base, loading: true })).toEqual({
      screenSpaceError: 16,
      reason: "loading",
    });
    expect(decideScreenSpaceError({ ...base, memoryRatio: 0.9 }).screenSpaceError).toBe(16);
    expect(decideScreenSpaceError({ ...base, memoryRatio: 0.9 }).reason).toMatch(/holding/);
  });

  it("holds the tile selection while moving, however fine it got", () => {
    expect(decideScreenSpaceError({ ...base, moving: true, current: 6 })).toEqual({
      screenSpaceError: 6,
      reason: "moving (tiles held)",
    });
  });

  it("never returns to the base on its own: only memory pressure coarsens", () => {
    expect(
      decideScreenSpaceError({ ...base, current: 8, memoryRatio: 0.75 }).screenSpaceError,
    ).toBe(8);
    const d = decideScreenSpaceError({ ...base, current: 8, memoryRatio: 1.5 });
    expect(d.screenSpaceError).toBe(12);
    expect(d.reason).toMatch(/memory pressure/);
    expect(
      decideScreenSpaceError({ ...base, current: 31, memoryRatio: 1.5 }).screenSpaceError,
    ).toBe(32);
  });

  it("follows bounds shifted by a ladder penalty, coarsening at once when the floor rose", () => {
    const bounds = { base: 22, min: 12, max: 32 };
    expect(decideScreenSpaceError({ ...base, bounds, current: 13 }).screenSpaceError).toBe(12);
    expect(decideScreenSpaceError({ ...base, bounds, current: 12 }).reason).toBe("at finest");
    for (const sample of [
      { ...base, bounds, current: 4, moving: true },
      { ...base, bounds, current: 4, loading: true },
      { ...base, bounds, current: 4, memoryRatio: 0.9 },
    ])
      expect(decideScreenSpaceError(sample)).toEqual({
        screenSpaceError: 12,
        reason: "ladder floor",
      });
  });
});

describe("splatMinimumScreenSpaceError", () => {
  it("lets ultra refine splats further than balanced, never below 4", () => {
    expect(splatMinimumScreenSpaceError("performance")).toBe(12);
    expect(splatMinimumScreenSpaceError("balanced")).toBe(8);
    expect(splatMinimumScreenSpaceError("ultra")).toBe(4);
  });
});

describe("buildLadder", () => {
  it("cuts resolution to a half, then tiles, never past the preset maximum", () => {
    const steps = buildLadder("balanced");
    expect(steps.slice(0, 5).map((s) => s.label)).toEqual([
      "full",
      "resolution 0.8",
      "resolution 0.65",
      "resolution 0.5",
      "tiles +3 SSE",
    ]);
    expect(steps.every((s, i) => i < 4 || s.ssePenalty === (i - 3) * 3)).toBe(true);
    expect(steps[0]?.msaa).toBe(2);
    expect(buildLadder("ultra")[0]?.msaa).toBe(4);
    const last = steps[steps.length - 1];
    expect(QUALITY_SSE.balanced.base + (last?.ssePenalty ?? 0)).toBeLessThanOrEqual(
      QUALITY_SSE.balanced.max,
    );
  });

  it("has no anti-aliasing for the performance preset", () => {
    expect(buildLadder("performance")[0]?.msaa).toBe(1);
    expect(buildLadder("performance")[1]?.label).toBe("resolution 0.8");
  });

  it("caps balanced at 1.5 device pixels per CSS pixel; ultra keeps them all", () => {
    expect(baseResolutionScale("balanced", 1)).toBe(1);
    expect(baseResolutionScale("balanced", 2)).toBe(0.75);
    expect(baseResolutionScale("balanced", 3)).toBe(0.5);
    expect(baseResolutionScale("ultra", 2)).toBe(1);
    expect(baseResolutionScale("performance", 2)).toBe(1);
  });
});

describe("frameEvidence / frameWeight", () => {
  it("counts a frame nothing is driving as evidence of nothing", () => {
    expect(frameEvidence({ moving: false, animating: false })).toBe("none");
    expect(frameWeight("none")).toEqual({ ladder: false, recovery: false, judged: false });
  });

  it("treats a gesture during an animation as a gesture", () => {
    expect(frameEvidence({ moving: true, animating: true })).toBe("motion");
    expect(frameEvidence({ moving: true, animating: false })).toBe("motion");
    expect(frameWeight("motion")).toEqual({ ladder: true, recovery: true, judged: true });
  });

  it("lets an animated frame coarsen quality but never restore it", () => {
    expect(frameEvidence({ moving: false, animating: true })).toBe("animation");
    // The asymmetry the source documents: slow animated frames are evidence the machine
    // cannot keep up, smooth ones are not evidence that a gesture would be smooth.
    expect(frameWeight("animation")).toEqual({ ladder: true, recovery: false, judged: false });
  });

  it("keeps the benchmark and the step's before/after sample to camera motion only", () => {
    const judged = (["none", "motion", "animation"] as const).filter((e) => frameWeight(e).judged);
    expect(judged).toEqual(["motion"]);
  });
});

describe("shouldSharpenAtRest", () => {
  const rest = { moving: false, animating: false, restedMs: 600, loading: false };

  it("sharpens the still frame once the camera has rested", () => {
    expect(shouldSharpenAtRest(rest)).toBe(true);
    expect(shouldSharpenAtRest({ ...rest, restedMs: 400 })).toBe(false);
    expect(shouldSharpenAtRest({ ...rest, moving: true })).toBe(false);
  });

  it("waits for loading to settle, but not forever", () => {
    expect(shouldSharpenAtRest({ ...rest, loading: true })).toBe(false);
    expect(shouldSharpenAtRest({ ...rest, loading: true, restedMs: 3000 })).toBe(true);
  });

  it("never sharpens while animating, however long the camera has rested", () => {
    for (const restedMs of [600, 3000, 60_000])
      expect(shouldSharpenAtRest({ ...rest, animating: true, restedMs })).toBe(false);
  });
});

/** Enough of a viewer for the manager's constructor: events, the two settings it writes, a canvas. */
class FakeEvent {
  private readonly listeners = new Set<() => void>();

  readonly addEventListener = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  raise(): void {
    for (const listener of [...this.listeners]) listener();
  }
}

function createFakeViewer() {
  const scene = {
    preUpdate: new FakeEvent(),
    postUpdate: new FakeEvent(),
    preRender: new FakeEvent(),
    postRender: new FakeEvent(),
    globe: { maximumScreenSpaceError: 2 },
    msaaSamples: 4,
    postProcessStages: {
      add: () => undefined,
      remove: () => undefined,
      fxaa: { enabled: false },
    },
    requestRender: () => undefined,
    isDestroyed: () => false,
  };
  const camera = { changed: new FakeEvent(), moveEnd: new FakeEvent() };
  const viewer = {
    scene,
    camera,
    canvas: document.createElement("canvas"),
    useBrowserRecommendedResolution: false,
    resolutionScale: 1,
  };
  const events = new Emitter<SceneEvents>();
  const snapshots: Partial<PerformanceSnapshot>[] = [];
  events.on("performance", (snapshot) => snapshots.push(snapshot));
  const manager = new PerformanceManager(viewer as unknown as Viewer, events);
  manager.configure({ preset: "ultra", manualScreenSpaceError: null, adaptive: true });
  return { manager, scene, camera, snapshots };
}

describe("PerformanceManager.setAnimating", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("is off by default, and the still frame sharpens as it always did", () => {
    const { manager, scene, camera } = createFakeViewer();
    expect(scene.msaaSamples).toBe(1);
    camera.changed.raise();
    camera.moveEnd.raise();
    vi.advanceTimersByTime(1000);
    expect(scene.msaaSamples).toBe(4);
    manager.destroy();
  });

  it("suppresses the rest sharpen while animating and releases it afterwards", () => {
    const { manager, scene, camera } = createFakeViewer();
    manager.setAnimating(true);
    camera.changed.raise();
    camera.moveEnd.raise();
    // An animating scene renders every tick, so the camera resting means nothing: however
    // long it waits, the frame must stay at the ladder's step.
    vi.advanceTimersByTime(30_000);
    expect(scene.msaaSamples).toBe(1);
    manager.setAnimating(false);
    vi.advanceTimersByTime(1000);
    expect(scene.msaaSamples).toBe(4);
    manager.destroy();
  });

  it("drops a sharpened still frame back to the ladder's step when animation starts", () => {
    const { manager, scene, camera } = createFakeViewer();
    camera.changed.raise();
    camera.moveEnd.raise();
    vi.advanceTimersByTime(1000);
    expect(scene.msaaSamples).toBe(4);
    manager.setAnimating(true);
    expect(scene.msaaSamples).toBe(1);
    manager.destroy();
  });

  /** Renders `count` frames `frameMs` apart, letting the 500 ms evaluation tick run between them. */
  function renderFrames(scene: { postRender: FakeEvent }, count: number, frameMs: number): void {
    for (let i = 0; i < count; i += 1) {
      vi.advanceTimersByTime(frameMs);
      scene.postRender.raise();
    }
  }

  it("lets slow animated frames step the ladder down, which a still camera never could", () => {
    const still = createFakeViewer();
    renderFrames(still.scene, 20, 100);
    // Camera still and nothing animating: 10 fps here is tiles arriving into an idle scene.
    expect(still.manager.profile).toBe("full");
    still.manager.destroy();

    const animating = createFakeViewer();
    animating.manager.setAnimating(true);
    renderFrames(animating.scene, 20, 100);
    expect(animating.manager.profile).toBe("reduced");
    animating.manager.destroy();
  });

  it("never lets smooth animated frames undo a cut, but a smooth gesture does", () => {
    const { manager, scene, camera } = createFakeViewer();
    manager.setAnimating(true);
    // Just past SUSTAINED_LOW_MS of slow animated frames: one step down, not two.
    renderFrames(scene, 12, 100);
    expect(manager.profile).toBe("reduced");

    // 16 s of animation at 60 fps: the debt is paid off, but no recovery is earned.
    renderFrames(scene, 1000, 16);
    expect(manager.profile).toBe("reduced");

    // The same frames with the camera moving are evidence about responsiveness, and do.
    camera.changed.raise();
    renderFrames(scene, 600, 16);
    camera.moveEnd.raise();
    vi.advanceTimersByTime(2500);
    expect(manager.profile).toBe("full");
    manager.destroy();
  });

  it("reports `animating` in the performance snapshot, so `fps` says what it measures", () => {
    const { manager, snapshots } = createFakeViewer();
    vi.advanceTimersByTime(500);
    expect(snapshots[snapshots.length - 1]?.animating).toBe(false);
    manager.setAnimating(true);
    vi.advanceTimersByTime(500);
    expect(snapshots[snapshots.length - 1]?.animating).toBe(true);
    manager.destroy();
  });
});
