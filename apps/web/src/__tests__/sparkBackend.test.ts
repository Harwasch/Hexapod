/**
 * The Spark back-end (cesium/scanView/sparkBackend.ts) as the overlay drives it: when Spark's
 * `onDirty` asks for a frame. WebGL is not in jsdom, so three's renderer is one that tells the
 * scene's objects they are drawn, and Spark's `SparkRenderer` is its frame-and-sort contract
 * (2.2, `spark.module.js`), nothing else:
 *
 * - a frame regenerates the splats (`onBeforeRender` -> `updateInternal`) when the camera is
 *   away from where the latest sort was taken from, and says so (`setDirty` -> `onDirty`,
 *   unless already dirty); the frame clears `dirty` as it ends;
 * - one sort runs at a time, of the latest generation, and says so when it lands; a
 *   generation made while one runs is sorted once it has.
 */

import type { SplatMesh } from "@sparkjsdev/spark";
import type * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OverlayFrames, type FrameClock } from "@/cesium/scanView/overlayFrames";
import { createBackend } from "@/cesium/scanView/sparkBackend";
import { TileWork } from "@/cesium/scanView/tileWork";
import type { ScanBackend, ScanPose } from "@/cesium/scanView/types";

/** What a test reads and does of the fake `SparkRenderer`. */
interface FakeSpark {
  /** Regenerations: frames whose camera was away from the latest sort's. */
  generated: number;
  sorting: boolean;
  /** The sort in flight lands. */
  land(): void;
}

const sparks = vi.hoisted(() => [] as FakeSpark[]);

vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof THREE>();
  const nothing = (): void => undefined;
  class WebGLRenderer {
    readonly info = { render: { frame: 0 } };
    readonly setClearColor = nothing;
    readonly setPixelRatio = nothing;
    readonly setSize = nothing;
    readonly setAnimationLoop = nothing;
    readonly dispose = nothing;
    readonly forceContextLoss = nothing;
    /** As three's: the world matrices, then each object told it is about to be drawn. */
    render(scene: THREE.Scene, camera: THREE.Camera): void {
      this.info.render.frame += 1;
      scene.updateMatrixWorld();
      camera.updateMatrixWorld();
      scene.traverse((object) => {
        (object as { onBeforeRender: (...args: unknown[]) => void }).onBeforeRender(
          this,
          scene,
          camera,
        );
      });
    }
  }
  return { ...actual, WebGLRenderer };
});

vi.mock("@sparkjsdev/spark", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  class SparkRenderer extends three.Object3D implements FakeSpark {
    generated = 0;
    sorting = false;
    sortDirty = false;
    sortTimeoutId = -1;
    // Spark draws `display` and generates into `current`: the same here, one mapping.
    readonly display = {};
    readonly current = this.display;
    private dirty = true;
    private readonly sortedCenter = new three.Vector3().setScalar(Number.NEGATIVE_INFINITY);
    private readonly generatedAt = new three.Vector3();
    private readonly onDirty: (() => void) | undefined;
    private landing: (() => void) | null = null;

    constructor(options: { onDirty?: () => void }) {
      super();
      this.onDirty = options.onDirty;
      sparks.push(this);
    }

    override onBeforeRender(
      _renderer: THREE.WebGLRenderer,
      _scene: THREE.Scene,
      camera: THREE.Camera,
    ): void {
      const center = camera.getWorldPosition(new three.Vector3());
      if (center.distanceTo(this.sortedCenter) > 1e-3) {
        this.generated += 1;
        this.generatedAt.copy(center);
        this.sortDirty = true;
        this.setDirty();
      }
      this.driveSort();
      this.dirty = false;
    }

    land(): void {
      const landing = this.landing;
      this.landing = null;
      landing?.();
    }

    override dispose(): void {
      this.landing = null;
    }

    private setDirty(): void {
      if (!this.dirty) {
        this.dirty = true;
        this.onDirty?.();
      }
    }

    private driveSort(): void {
      if (this.sorting || !this.sortDirty) return;
      this.sorting = true;
      this.sortDirty = false;
      this.sortedCenter.copy(this.generatedAt);
      this.landing = () => {
        this.sorting = false;
        this.setDirty();
        this.driveSort();
      };
    }
  }
  const uniform = (value: unknown) => ({ value });
  return {
    SparkRenderer,
    dyno: { dynoSampler2D: uniform, dynoUsampler2D: uniform, dynoVec4: uniform },
  };
});

// Tiles are not loaded here; Spark's own tile loading stays out of the module graph.
vi.mock("@/view/sparkStream", () => ({ loadSplatTile: vi.fn() }));

function poseAt(eye: [number, number, number]): ScanPose {
  return {
    eye,
    direction: [0, 1, 0],
    up: [0, 0, 1],
    fovy: 1,
    near: 0.1,
    far: 1000,
    width: 960,
    height: 600,
    pixelRatio: 1,
  };
}

/**
 * The overlay's frames (overlayFrames.ts) drawing the Spark back-end, with `frameWanted` as
 * the host wires it; `display(n)` is n display frames.
 */
async function rig() {
  let now = 0;
  let callbacks: (() => void)[] = [];
  const clock: FrameClock = {
    now: () => now,
    requestFrame: (callback) => {
      callbacks.push(callback);
      return callbacks.length;
    },
    cancelFrame: () => {
      callbacks = [];
    },
    setTimer: () => 0,
    clearTimer: () => undefined,
  };
  const state = { pose: poseAt([0, -40, 10]), renders: 0 };
  let backend: ScanBackend<SplatMesh> | null = null;
  const driver = new OverlayFrames(
    () => () => undefined,
    {
      changed: () => false,
      draw: () => {
        backend?.render(state.pose);
        state.renders += 1;
        return { again: false, by: null };
      },
    },
    clock,
  );
  backend = await createBackend(document.createElement("canvas"), 3_000_000, {
    frameWanted: () => driver.wake("renderer"),
    work: new TileWork(),
    maxShDegree: 3,
  });
  const spark = sparks.at(-1);
  if (!spark) throw new Error("no SparkRenderer made");
  const display = (frames: number): void => {
    for (let i = 0; i < frames; i++) {
      now += 16;
      const due = callbacks;
      callbacks = [];
      for (const callback of due) callback();
    }
  };
  return { state, driver, backend, spark, display };
}

describe("the Spark back-end asking for frames", () => {
  afterEach(() => {
    sparks.length = 0;
  });

  it("a still view after a move draws nothing more while an earlier view's sort is in flight", async () => {
    const { state, driver, backend, spark, display } = await rig();
    // The first frame: generated, and its sort runs.
    driver.wake("start");
    display(1);
    expect(state.renders).toBe(1);
    expect(spark.sorting).toBe(true);

    // The camera moves on and stops (the globe's frame draws it) before that sort lands. The
    // frame regenerates -- the camera is away from where the sort was taken -- and Spark says
    // so from inside it. Asked for, the next frame did the same, and the next: a frame every
    // display frame of a still view, for as long as the sort took (seconds on a software GPU,
    // CI's `scanOverlayIdle` "spark draws nothing once a still scan has loaded").
    state.pose = poseAt([2, -40, 10]);
    driver.wake("camera");
    display(1);
    expect(state.renders).toBe(2);
    expect(spark.generated).toBe(2);
    display(120);
    expect(state.renders).toBe(2);
    expect(spark.generated).toBe(2);
    expect(backend.settled?.()).toBe(false);

    // The sort lands: one frame shows it, and the sort of the view that rests begins.
    spark.land();
    display(120);
    expect(state.renders).toBe(3);
    expect(spark.sorting).toBe(true);
    // That one lands: one frame more, and nothing after it.
    spark.land();
    display(120);
    expect(state.renders).toBe(4);
    expect(spark.generated).toBe(2);
    expect(backend.settled?.()).toBe(true);
    spark.land();
    display(600);
    expect(state.renders).toBe(4);
    backend.destroy();
  });

  it("a sort landing between frames asks for the frame that shows it", async () => {
    const { state, driver, backend, spark, display } = await rig();
    driver.wake("start");
    display(1);
    const before = state.renders;
    display(30);
    expect(state.renders).toBe(before);
    spark.land();
    display(1);
    expect(state.renders).toBe(before + 1);
    backend.destroy();
  });
});
