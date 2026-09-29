/**
 * A bare CesiumJS page that loads one splat tileset and deforms it — the driver for the Living
 * Survey's end-to-end checks.
 *
 * It exists because `SplatDeformer` is the one piece of this feature that cannot be proved
 * headlessly: the texel arithmetic, the frame round trip, the tile bookkeeping and the refusal
 * logic are all unit tested against fake primitives, but whether the engine's real packed
 * buffer, the real addressing parameters, a real `Texture.copyFrom` — or, on the GPU path, the
 * patched vertex shader — compose into a tree that moves can only be answered by a browser. This
 * is the smallest page that asks that question — no API, no app shell, no store.
 *
 * Loaded dynamically by `e2e/livingSurvey*.spec.ts`; nothing imports it, so it never reaches the
 * production bundle. It drives `apply()` by hand rather than from `scene.preUpdate` — the
 * animation hook belongs to `LivingSurveyManager`, and a test that installs its own is testing
 * the deformer rather than the manager.
 */

import {
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  type Scene,
} from "cesium";
import * as CesiumBarrel from "cesium";

import {
  deform,
  FLUTTER_STILL,
  flutterField,
  livingFrame,
  livingWindFromSettings,
  loadLivingMotion,
  parseRig,
  prepareLivingMotion,
  type FlutterField,
  type LivingMotion,
  type MotionRig,
  type NodeTransform,
  type WindSettings,
} from "@twin/world";

import { SplatDeformer, type DeformerStatus } from "@/cesium/SplatDeformer";
import { installSplatTextureInterception } from "@/cesium/splatCapture";
import { splatCaptureCount } from "@/cesium/splatCaptureRegistry";
import { cesiumMotionTextures } from "@/cesium/splatGpuTextures";
import { splatTilesetOf } from "@/cesium/splatInternals";

export interface LivingSurveyHarnessOptions {
  readonly container: HTMLElement;
  /** A `tileset.json` for a Gaussian splat capture: one tile, or a level-of-detail tree. */
  readonly tilesetUrl: string;
  /** The matching `rig.json`. */
  readonly rigUrl: string;
  /**
   * Evaluate motion in the vertex shader (engine patch) rather than rewriting the texture.
   * Default true, as in the app; `false` drives the CPU path.
   */
  readonly gpu?: boolean;
  /** Default 1, which keeps a single-tile capture fully refined; a LOD tree wants Cesium's 16. */
  readonly maximumScreenSpaceError?: number;
  /** Camera distance in bounding radii. Default 3.2. */
  readonly rangeRadii?: number;
  /**
   * Drive the rig with Living Mode (ADR 0008) from the motion sidecar it points at — modal sway
   * and advected leaf flutter — as `LivingSurveyManager` does for such a rig. Default: the
   * legacy model, whatever the rig carries.
   */
  readonly living?: boolean;
}

/** Which parts of a frame `step` applies: everything, or the sway with flutter held still. */
export type StepParts = "all" | "sway";

/** What the Playwright spec drives. Everything returns plain JSON so it crosses the bridge. */
export interface LivingSurveyHarness {
  /** Applies the rig at time `t` under `wind`, then renders one frame. */
  step(t: number, wind: WindSettings, parts?: StepParts): Promise<DeformerStatus>;
  /** Renders frames until the deformer attaches or `timeoutMs` elapses. */
  waitUntilReady(timeoutMs: number): Promise<DeformerStatus>;
  status(): HarnessStatus;
  /** Largest distance the rig's transforms move any splat, metres. Flutter is not included. */
  displacementM(t: number, wind: WindSettings): number;
  /** One `apply()` with no render, for timing the CPU and upload-submit cost alone. */
  applyOnly(t: number, wind: WindSettings): DeformerStatus;
  /**
   * Frames the tileset from `rangeM` metres and steps `(t, wind)` until the tile selection has
   * settled and the deformer is attached to it.
   */
  view(rangeM: number, t: number, wind: WindSettings, timeoutMs: number): Promise<HarnessStatus>;
  /** Times the engine's own radix sort of this snapshot from this camera, `repeats` times. */
  sortMs(repeats: number): Promise<{ numSplats: number; ms: number[] }>;
  /** One line per frame of the last `view`: snapshot key, load state and deformer phase. */
  lastViewTrace(): string[];
  /** Keeps a copy of the current frame's pixels under `name`. */
  capture(name: string): void;
  /** How two captured frames differ: mean absolute channel difference, and changed share. */
  diff(a: string, b: string): { meanAbs: number; changed: number };
  /**
   * Steps `frames` frames of wind from `t0` at 60 Hz and times them: `apply()` alone (the CPU
   * cost of the motion path, upload submit included) and the whole frame (apply + render).
   */
  measure(frames: number, t0: number, wind: WindSettings): Promise<FrameTiming>;
  /**
   * `count` consecutive `apply()` calls with no render between them: the motion path's CPU cost
   * per frame, upload submit included, for views too large to render many frames of here.
   */
  measureApply(count: number, t0: number, wind: WindSettings): { mean: number; max: number };
}

export interface HarnessStatus extends DeformerStatus {
  captures: number;
  numSplatsLoaded: number;
  selectedTiles: number;
  tilesLoaded: boolean;
  /** A snapshot rebuild is in flight. */
  pending: boolean;
  /** Camera distance from the tileset's bounding-sphere centre, metres. */
  cameraRangeM: number;
  /** The last traversal selected exactly the tiles the committed snapshot aggregates. */
  selectionSettled: boolean;
  /** Which motion model drives the rig. */
  model: "living" | "legacy";
  /** Side of the leaf-flutter texture the GPU path has uploaded; 0 for none. */
  gpuFlutterTextureSize: number;
}

export interface FrameTiming {
  readonly frames: number;
  readonly applyMsMean: number;
  readonly applyMsMax: number;
  readonly frameMsMean: number;
  readonly frameMsMedian: number;
  readonly numSplats: number;
  readonly motion: string;
  readonly lastUploadWords: number;
}

/** Resolves on the next completed frame. */
function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

export async function startLivingSurveyHarness(
  options: LivingSurveyHarnessOptions,
): Promise<LivingSurveyHarness> {
  // Before anything can load: the interception only sees calls made after it is installed.
  installSplatTextureInterception();

  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    // Continuous rendering. SwiftShader frames here run 740–1730 ms, and a request-render scene
    // that stops asking never fires `postRender` for the listener to resolve on.
    requestRenderMode: false,
    msaaSamples: 1,
    // So a frame can be read back after it is presented, for `capture`.
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");

  const rigResponse = await fetch(options.rigUrl);
  const rig: MotionRig = parseRig(await rigResponse.text());
  let living: LivingMotion | undefined;
  if (options.living === true) {
    if (rig.motionPath === undefined) throw new Error("living: the rig points at no sidecar");
    const url = new URL(rig.motionPath, new URL(options.rigUrl, window.location.href));
    const response = await fetch(url);
    if (!response.ok) throw new Error(`living: sidecar HTTP ${String(response.status)}`);
    living = loadLivingMotion(rig, await response.text());
    prepareLivingMotion(living);
  }
  /** One frame of the rig: Living Mode's when asked for, the legacy model's otherwise. */
  function frameAt(
    t: number,
    wind: WindSettings,
  ): { transforms: NodeTransform[]; flutter: FlutterField } {
    if (living !== undefined) {
      return livingFrame(living, t, livingWindFromSettings(wind, living.sidecar));
    }
    return { transforms: deform(rig, t, wind), flutter: flutterField(rig, t, wind) };
  }

  const tileset = await Cesium3DTileset.fromUrl(options.tilesetUrl, {
    maximumScreenSpaceError: options.maximumScreenSpaceError ?? 1,
    // The traversal site tilesets use for splats (providers/tiles.ts): REPLACE, no skipping.
    skipLevelOfDetail: false,
  });
  scene.primitives.add(tileset);

  const bounds = tileset.boundingSphere;
  const frame = (rangeM: number): void => {
    scene.camera.lookAt(
      bounds.center,
      new HeadingPitchRange(CesiumMath.toRadians(35), CesiumMath.toRadians(-12), rangeM),
    );
  };
  frame(bounds.radius * (options.rangeRadii ?? 3.2));

  const internals = splatTilesetOf(tileset);
  const wantGpu = options.gpu !== false;
  const gpu = wantGpu ? cesiumMotionTextures() : undefined;
  if (wantGpu && gpu === undefined) throw new Error("no GPU motion textures");
  const deformer = new SplatDeformer({ tileset: internals, rig, gpu });

  async function step(
    t: number,
    wind: WindSettings,
    parts: StepParts = "all",
  ): Promise<DeformerStatus> {
    const { transforms, flutter } = frameAt(t, wind);
    const status = deformer.apply(transforms, parts === "sway" ? FLUTTER_STILL : flutter);
    await nextFrame(scene);
    return status;
  }

  function status(): HarnessStatus {
    const primitive = internals.gaussianSplatPrimitive;
    // What the last traversal selected, against what the committed snapshot aggregated.
    const traversal = (tileset as unknown as { _selectedTiles?: readonly unknown[] })
      ._selectedTiles;
    const committed = primitive?._selectedTileSet;
    const selectionSettled =
      committed !== undefined &&
      traversal?.length === committed.size &&
      traversal.every((tile) => committed.has(tile as never));
    return {
      selectionSettled,
      ...deformer.status,
      captures: splatCaptureCount(),
      numSplatsLoaded: primitive?._numSplats ?? 0,
      selectedTiles: primitive?._selectedTileSet?.size ?? 0,
      tilesLoaded: tileset.tilesLoaded,
      pending: primitive?._pendingSnapshot !== undefined && primitive._pendingSnapshot !== null,
      cameraRangeM: Cartesian3.distance(scene.camera.positionWC, bounds.center),
      model: living === undefined ? "legacy" : "living",
      gpuFlutterTextureSize: deformer.gpuMotion?.flutterTextureSize ?? 0,
    };
  }

  const frames = new Map<string, Uint8ClampedArray>();
  /** One line per frame of the last `view`, for diagnosing a view that never settles. */
  const viewTrace: string[] = [];
  function pixels(): Uint8ClampedArray {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return new Uint8ClampedArray(0);
    context.drawImage(canvas, 0, 0);
    return context.getImageData(0, 0, copy.width, copy.height).data;
  }

  return {
    step,
    status,
    applyOnly(t: number, wind: WindSettings): DeformerStatus {
      const { transforms, flutter } = frameAt(t, wind);
      return deformer.apply(transforms, flutter);
    },
    async waitUntilReady(timeoutMs: number): Promise<DeformerStatus> {
      const deadline = Date.now() + timeoutMs;
      let current = deformer.status;
      while (Date.now() < deadline && current.phase !== "ready" && current.phase !== "refused") {
        current = await step(0, { strength: 0, bearingDeg: 0 });
      }
      return current;
    },
    async view(rangeM: number, t: number, wind: WindSettings, timeoutMs: number) {
      frame(rangeM);
      const deadline = Date.now() + timeoutMs;
      let key = "";
      let stable = 0;
      viewTrace.length = 0;
      // Settled: every selected tile loaded, the primitive rebuilt over exactly them, the
      // deformer attached to that snapshot, and all of it unchanged for a few frames.
      while (Date.now() < deadline && stable < 6) {
        await step(t, wind);
        const now = status();
        const next = `${String(now.generation)}:${String(now.selectedTiles)}:${String(now.numSplatsLoaded)}`;
        viewTrace.push(
          `${next} loaded=${String(now.tilesLoaded)} sel=${String(now.selectionSettled)} pending=${String(now.pending)} ${now.phase}/${now.reason ?? ""} tiles=${String(now.tiles)}`,
        );
        const settled =
          now.tilesLoaded &&
          now.selectionSettled &&
          !now.pending &&
          now.phase === "ready" &&
          now.tiles === now.selectedTiles &&
          now.numSplats === now.numSplatsLoaded;
        stable = settled && next === key ? stable + 1 : 0;
        key = next;
        if (now.phase === "refused") break;
      }
      return status();
    },
    async sortMs(repeats: number): Promise<{ numSplats: number; ms: number[] }> {
      // The engine's own radix sort (a WASM worker), over this snapshot's positions from this
      // camera: what one re-sort of displaced positions would cost, were it triggered.
      const sorter = (
        CesiumBarrel as unknown as {
          GaussianSplatSorter?: {
            radixSortIndexes(parameters: unknown): Promise<unknown> | undefined;
          };
        }
      ).GaussianSplatSorter;
      const primitive = internals.gaussianSplatPrimitive;
      const positions = primitive?._positions;
      const root = primitive?._rootTransform;
      const count = primitive?._numSplats ?? 0;
      const ms: number[] = [];
      if (sorter === undefined || positions === undefined || root === undefined) {
        return { numSplats: count, ms };
      }
      const modelView = Matrix4.multiply(
        scene.camera.viewMatrix,
        Matrix4.clone(root as Matrix4),
        new Matrix4(),
      );
      while (ms.length < repeats) {
        const start = performance.now();
        const promise = sorter.radixSortIndexes({
          primitive: {
            positions: new Float32Array(positions.subarray(0, count * 3)),
            modelView: Float32Array.from(modelView),
            count,
          },
          sortType: "Index",
        });
        if (promise === undefined) {
          await nextFrame(scene);
          continue;
        }
        await promise;
        ms.push(performance.now() - start);
      }
      return { numSplats: count, ms };
    },
    lastViewTrace(): string[] {
      return [...viewTrace];
    },
    capture(name: string): void {
      frames.set(name, pixels());
    },
    diff(a: string, b: string) {
      const first = frames.get(a);
      const second = frames.get(b);
      if (first === undefined || second?.length !== first.length) {
        return { meanAbs: Number.NaN, changed: Number.NaN };
      }
      let total = 0;
      let changed = 0;
      for (let i = 0; i < first.length; i += 4) {
        const d =
          Math.abs((first[i] ?? 0) - (second[i] ?? 0)) +
          Math.abs((first[i + 1] ?? 0) - (second[i + 1] ?? 0)) +
          Math.abs((first[i + 2] ?? 0) - (second[i + 2] ?? 0));
        total += d / 3;
        if (d > 0) changed += 1;
      }
      const count = first.length / 4;
      return { meanAbs: total / count, changed: changed / count };
    },
    measureApply(count: number, t0: number, wind: WindSettings) {
      const times: number[] = [];
      for (let i = 0; i < count; i += 1) {
        const { transforms, flutter } = frameAt(t0 + i / 60, wind);
        const start = performance.now();
        deformer.apply(transforms, flutter);
        times.push(performance.now() - start);
      }
      return {
        mean: times.reduce((sum, value) => sum + value, 0) / Math.max(1, times.length),
        max: Math.max(...times),
      };
    },
    async measure(count: number, t0: number, wind: WindSettings): Promise<FrameTiming> {
      const apply: number[] = [];
      const whole: number[] = [];
      for (let i = 0; i < count; i += 1) {
        const { transforms, flutter } = frameAt(t0 + i / 60, wind);
        const start = performance.now();
        deformer.apply(transforms, flutter);
        const applied = performance.now();
        await nextFrame(scene);
        apply.push(applied - start);
        whole.push(performance.now() - start);
      }
      const sorted = [...whole].sort((x, y) => x - y);
      const mean = (values: number[]): number =>
        values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
      const current = deformer.status;
      return {
        frames: count,
        applyMsMean: mean(apply),
        applyMsMax: Math.max(...apply),
        frameMsMean: mean(whole),
        frameMsMedian: sorted[Math.floor(sorted.length / 2)] ?? Number.NaN,
        numSplats: current.numSplats,
        motion: current.motion,
        lastUploadWords: current.lastUploadWords,
      };
    },
    displacementM(t: number, wind: WindSettings): number {
      const canonical = deformer.canonicalPositions;
      const assignment = deformer.assignment;
      if (canonical === undefined || assignment === undefined) return 0;
      const { transforms } = frameAt(t, wind);
      let worst = 0;
      const point = new Cartesian3();
      for (let i = 0; i < assignment.length; i += 1) {
        const transform = transforms[assignment[i] ?? 0];
        if (transform === undefined) continue;
        const x = canonical[i * 3] ?? 0;
        const y = canonical[i * 3 + 1] ?? 0;
        const z = canonical[i * 3 + 2] ?? 0;
        const [qx, qy, qz, qw] = transform.rotation;
        // q ⊗ v ⊗ q⁻¹, written out so this stays independent of @twin/world's own helpers.
        const ux = qy * z - qz * y;
        const uy = qz * x - qx * z;
        const uz = qx * y - qy * x;
        const tx = ux + qw * x;
        const ty = uy + qw * y;
        const tz = uz + qw * z;
        point.x = x + 2 * (qy * tz - qz * ty) + transform.translation[0];
        point.y = y + 2 * (qz * tx - qx * tz) + transform.translation[1];
        point.z = z + 2 * (qx * ty - qy * tx) + transform.translation[2];
        const moved = Math.hypot(point.x - x, point.y - y, point.z - z);
        if (moved > worst) worst = moved;
      }
      return worst;
    },
  };
}
