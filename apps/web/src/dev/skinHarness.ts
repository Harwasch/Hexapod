/**
 * A bare CesiumJS page that draws a skinned splat tileset and drives its handles
 * (`cesium/splatSkin.ts`) -- the driver for e2e/skin.spec.ts, and a dev page: `wobble` sways
 * one object with a sine on its first handles, and `drive` sets any handles at all.
 *
 * What it answers, in a real engine with the patch: does the skin GLSL compile in the motion
 * chain, do a driven object's pixels move while the others stay put, does the constant handle
 * move an object rigidly, and do covariances follow the skin (`covariance` on and off)? The
 * tileset is attached as `SiteManager` attaches one (`attachInstances`, `attachSkin`), so
 * hiding and highlighting compose with the motion exactly as in the app.
 *
 * `windOn` / `advance` / `windOff` drive every skin with the wind (`cesium/skinWind.ts`) on a
 * harness clock that moves only when told to, so a frame is a function of the steps taken: the
 * driver for e2e/wind.spec.ts.
 *
 * With `extras.telemetry` on the root (step C3), the scan's bindings drive their instances
 * (`cesium/telemetry.ts`) on a harness clock in milliseconds that moves only when told to
 * (`telemetryAt`): the driver for e2e/telemetry.spec.ts.
 *
 * With `renderer` set to `playcanvas` or `spark`, CesiumJS's splats are hidden and the scan is
 * drawn by that dedicated renderer over the globe (`cesium/scanView`), as the app does: the
 * same drivers move the same objects through `scanView/scanMotion.ts`, split objects declared
 * on the root are drawn at their poses (`objectPose`), and a frame is the globe and the
 * renderer's canvas composited: the driver for e2e/motionRenderers.spec.ts.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import {
  Cartesian2,
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  SceneTransforms,
  type Scene,
} from "cesium";

import { claimsOf } from "@/cesium/motionClaims";
import { ScanRendererHost, type ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import type { SplatRendererKind } from "@/cesium/scanView/types";
import { describeFromStore, SkinWindDriver } from "@/cesium/skinWind";
import { attachInstances, instanceSphere } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { motionChainOf } from "@/cesium/splatMotionChain";
import { attachSkin, skinningOf, type SplatSkinning } from "@/cesium/splatSkin";
import { installSplatSorter } from "@/cesium/splatSorter";
import { attachTelemetry, telemetryOf } from "@/cesium/telemetry";
import { withDescendants } from "@/lib/instances";
import { HANDLE_FLOATS, rigidHandle } from "@/lib/skin";
import { SyntheticSource } from "@/lib/telemetrySources";
import { SkinWindField, WIND_CALM, type WindSettings } from "@twin/world";
import type { ObjectPose } from "@/lib/sceneObjects";
import { useInstances, type RendererGap } from "@/state/instances";
import { useSceneObjects } from "@/state/sceneObjects";

const BACKGROUND = "#10141a";
const ASSET = "skin-harness";

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One handle's motion: `Z_j = [A | t]`, row-major, 12 numbers. */
export interface HandleMotion {
  handle: number;
  z: number[];
}

export interface SkinHarness {
  /** Points the camera and waits for the tiles, the instances and the skin. */
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  /** Every skin: its id, instance, handle count and size. */
  skins(): { id: number; instance: number; handles: number; scale: number }[];
  /** Sets one object's handles (the rest at rest) and waits for the frame. */
  drive(instance: number, motions: HandleMotion[]): Promise<void>;
  /** The constant handle as a rigid motion: a turn about the vertical, then a shift. */
  rigid(instance: number, yawDeg: number, shift: [number, number, number]): Promise<void>;
  /** Sways `instance`'s handles 1..k with sines (k handles, `amplitude` m) until `stop`. */
  wobble(instance: number, amplitude: number, handles?: number): void;
  stop(): void;
  /** Every object back at rest. */
  rest(): Promise<void>;
  /** Covariances follow the skin (default) or not. */
  covariance(on: boolean): Promise<void>;
  /** Hides instances through the store (the visibility chain), then waits. */
  hide(ids: number[]): Promise<void>;
  /** Hides every object but `id` (and what is below it). */
  isolate(id: number): Promise<void>;
  /** Keeps a copy of the frame; returns its index. */
  frame(): number;
  /** Share of pixels (in `rect`) that differ between frames `a` and `b` by more than `tol`. */
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  /** The largest channel difference (0-255) between frames `a` and `b` in `rect`. */
  maxDifference(a: number, b: number, rect?: Rect): number;
  /** Share of `rect` (or the canvas) that is not background in frame `a`. */
  coverage(a: number, rect?: Rect): number;
  /** Where instance `id` is on screen, or null. */
  rectOf(id: number, grow?: number): Rect | null;
  /** The motion chain's parts and whether a Jacobian is declared. */
  hooks(): { motion: string[]; jacobian: string[]; skin: boolean; active: boolean };
  /** Wind on every skin at `strength` towards `bearingDeg`, the harness clock set to `t0`. */
  windOn(strength: number, bearingDeg: number, t0?: number, seed?: number): Promise<void>;
  /** Moves the harness clock `seconds` on in steps of `1/fps`, then renders. */
  advance(seconds: number, fps?: number): Promise<void>;
  /** Calm: every skin handed `null`; waits for the frame. */
  windOff(): Promise<void>;
  /** What the wind driver made of each skin: its material, whether it sways. */
  windSkins(): {
    instance: number;
    wind: boolean;
    stiffness: number;
    evidence: string;
    claimed: boolean;
  }[];
  /** A square on screen around a point of skin `instance`'s rest frame (from its origin). */
  pointRect(instance: number, local: [number, number, number], radiusM: number): Rect | null;
  /** A square on screen around a point of the scan frame (tileset local ENU). */
  scanRect(point: [number, number, number], radiusM: number): Rect | null;
  /** Waits until the telemetry bindings are attached. */
  telemetryReady(): Promise<void>;
  /** Sets the telemetry clock (ms) and waits for the frame. */
  telemetryAt(ms: number): Promise<void>;
  /** Every binding: its state, age, motion path and pose shown (scan frame). */
  telemetryStatus(): TelemetryStatus[];
  /** Silences a synthetic source (a dropout), or lets it speak again. */
  mute(sourceId: string, muted: boolean): void;
  /** Every bound instance back at rest, tracks emptied, the clock at `ms`. */
  telemetryReset(ms: number): Promise<void>;
  /** The rigid part: instances driven, whether it acts, the slot of each id asked for. */
  rigidInfo(ids: number[]): { driven: number[]; active: boolean; slots: number[] };
  /** The handles a skinned instance's skin holds now, or null at rest. */
  skinHandles(instance: number): number[] | null;
  /** The dedicated renderer's state (null under CesiumJS). */
  rendererStatus(): ScanRendererStatus | null;
  /** Poses split object `instance` (null: its declared pose), then waits for the frame. */
  objectPose(instance: number, pose: ObjectPose | null): Promise<void>;
  /** What the store says the renderer cannot move, for the panels. */
  motionGap(): RendererGap | null;
}

export interface TelemetryStatus {
  instance: number;
  source: string;
  state: string;
  ageMs: number | null;
  via: string;
  position: number[] | null;
  orientation: number[] | null;
  playoutMs: number;
}

function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

export async function startSkinHarness(options: {
  container: HTMLElement;
  url: string;
  incremental?: boolean;
  maximumScreenSpaceError?: number;
  /** The app's distance sorter (`installSplatSorter`), which orders moving groups as drawn. */
  sorter?: boolean;
  /** Who draws the splats: CesiumJS (the default) or a dedicated renderer over the globe. */
  renderer?: SplatRendererKind;
}): Promise<SkinHarness> {
  if (options.sorter === true) installSplatSorter();
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  const tileset = await Cesium3DTileset.fromUrl(options.url, {
    maximumScreenSpaceError: options.maximumScreenSpaceError ?? 16,
    skipLevelOfDetail: false,
  });
  if (options.incremental ?? true) {
    keepOffscreenSplats(tileset);
    incrementalSplats(tileset, 0);
  }
  scene.primitives.add(tileset);
  attachInstances(tileset, scene, ASSET);
  attachSkin(tileset, scene, ASSET);
  let telemetryClock = 0;
  attachTelemetry(tileset, scene, ASSET, { clock: () => telemetryClock });
  // A dedicated renderer draws the scan instead; CesiumJS keeps the tileset, hidden, for its
  // frame, as SiteManager does.
  const kind = options.renderer ?? "cesium";
  let host: ScanRendererHost | undefined;
  if (kind !== "cesium") {
    tileset.show = false;
    tileset.preloadWhenHidden = false;
    host = new ScanRendererHost(widget, { preserveDrawingBuffer: true });
    host.setRenderer(kind);
    host.setTarget({ key: "harness", tileset, assetId: ASSET });
  }

  const settle = async (frames: number): Promise<void> => {
    for (let frame = 0; frame < frames; frame += 1) await nextFrame(scene);
    if (host) await converge();
  };
  /**
   * A dedicated renderer catches up over the frames after a change: drawn until it says it
   * shows what it was asked for (`settled`: Spark draws a new generation only once its
   * asynchronous sort lands, and frames repeat while it waits) and two frames running are the
   * same, so a frame read is the state, not the way to it.
   */
  const converge = async (): Promise<void> => {
    let last = pixels().data;
    for (let frame = 0; frame < 600; frame += 1) {
      await nextFrame(scene);
      const now = pixels().data;
      const same = now.length === last.length && now.every((v, i) => v === last[i]);
      if (same && (host?.status().settled ?? true)) return;
      last = now;
    }
    throw new Error("the splat renderer never settled");
  };
  const part = (): SplatSkinning => {
    const found = skinningOf(ASSET);
    if (!found) throw new Error("the skin never loaded");
    return found;
  };
  const frames: Uint8ClampedArray[] = [];
  let wobbleOff: (() => void) | undefined;
  let windClock = 0;
  let windSettings: WindSettings = WIND_CALM;
  let windDriver: SkinWindDriver | undefined;
  let windTickOff: (() => void) | undefined;
  const windDriverFor = (seed: number): SkinWindDriver => {
    windDriver ??= new SkinWindDriver(
      part(),
      describeFromStore(ASSET),
      new SkinWindField(seed),
      claimsOf(ASSET),
    );
    return windDriver;
  };

  function pixels(): { data: Uint8ClampedArray; width: number; height: number } {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) throw new Error("no 2d context");
    context.drawImage(canvas, 0, 0);
    const overlay = document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
    if (host && overlay) context.drawImage(overlay, 0, 0, copy.width, copy.height);
    return {
      data: context.getImageData(0, 0, copy.width, copy.height).data,
      width: copy.width,
      height: copy.height,
    };
  }

  function bounds(rect: Rect | undefined, width: number, height: number) {
    return {
      x0: Math.max(0, Math.floor(rect?.x ?? 0)),
      y0: Math.max(0, Math.floor(rect?.y ?? 0)),
      x1: Math.min(width, Math.ceil(rect ? rect.x + rect.width : width)),
      y1: Math.min(height, Math.ceil(rect ? rect.y + rect.height : height)),
    };
  }

  const handlesFor = (instance: number, motions: HandleMotion[]): Float64Array => {
    const skin = part().doc.byInstance.get(instance);
    if (!skin) throw new Error(`instance ${String(instance)} has no skin`);
    const out = new Float64Array(skin.handles * HANDLE_FLOATS);
    for (const { handle, z } of motions) {
      if (handle < 0 || handle >= skin.handles) continue;
      for (let i = 0; i < HANDLE_FLOATS; i += 1) out[handle * HANDLE_FLOATS + i] = z[i] ?? 0;
    }
    return out;
  };

  return {
    async view(headingDeg, pitchDeg, rangeM) {
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(
          CesiumMath.toRadians(headingDeg),
          CesiumMath.toRadians(pitchDeg),
          rangeM,
        ),
      );
      if (host) {
        // The renderer's tiles in and still, and nothing in flight, for 30 frames running.
        let still = 0;
        let last = -1;
        for (let frame = 0; frame < 1500 && still < 30; frame += 1) {
          await nextFrame(scene);
          const status = host.status();
          const ready =
            status.active &&
            (status.native || status.tiles > 0) &&
            status.loading === 0 &&
            useInstances.getState().assets[ASSET] !== undefined &&
            skinningOf(ASSET) !== undefined;
          still = ready && status.tiles === last ? still + 1 : 0;
          last = status.tiles;
        }
      } else {
        for (let frame = 0; frame < 600; frame += 1) {
          if (
            tileset.tilesLoaded &&
            splatTilesetOf(tileset).gaussianSplatPrimitive &&
            useInstances.getState().assets[ASSET] &&
            skinningOf(ASSET)
          )
            break;
          await nextFrame(scene);
        }
      }
      part();
      await settle(30);
    },
    skins() {
      return part().doc.skins.map(({ id, instance, handles, scale }) => ({
        id,
        instance,
        handles,
        scale,
      }));
    },
    async drive(instance, motions) {
      part().setInstanceHandles(instance, handlesFor(instance, motions));
      await settle(10);
    },
    async rigid(instance, yawDeg, shift) {
      const half = CesiumMath.toRadians(yawDeg) / 2;
      const z = Array.from(rigidHandle([0, 0, Math.sin(half), Math.cos(half)], shift));
      part().setInstanceHandles(instance, handlesFor(instance, [{ handle: 0, z }]));
      await settle(10);
    },
    wobble(instance, amplitude, handles = 3) {
      wobbleOff?.();
      const start = performance.now();
      wobbleOff = scene.preUpdate.addEventListener(() => {
        const t = (performance.now() - start) / 1000;
        const motions: HandleMotion[] = [];
        for (let j = 1; j <= handles; j += 1) {
          const s = amplitude * Math.sin(2 * Math.PI * (0.4 + 0.17 * j) * t + j);
          // A sideways sway of handle j: translation along east, a little north.
          motions.push({ handle: j, z: [0, 0, 0, s, 0, 0, 0, 0.4 * s, 0, 0, 0, 0] });
        }
        part().setInstanceHandles(instance, handlesFor(instance, motions));
      });
    },
    stop() {
      wobbleOff?.();
      wobbleOff = undefined;
    },
    async rest() {
      part().rest();
      await settle(10);
    },
    async covariance(on) {
      part().covariance = on;
      await settle(10);
    },
    async hide(ids) {
      const store = useInstances.getState();
      store.showAll(ASSET);
      store.setHidden(ASSET, ids, true);
      await settle(10);
    },
    async isolate(id) {
      const store = useInstances.getState();
      const entry = store.assets[ASSET];
      if (!entry) throw new Error("no instances");
      const keep = withDescendants(entry, [id]);
      const others = entry.instances.filter((i) => i.parent === null && !keep.has(i.id));
      store.showAll(ASSET);
      store.setHidden(
        ASSET,
        others.map((i) => i.id),
        true,
      );
      await settle(10);
    },
    frame() {
      frames.push(pixels().data);
      return frames.length - 1;
    },
    difference(a, b, rect, tol = 40) {
      const fa = frames[a];
      const fb = frames[b];
      if (!fa || !fb) throw new Error("no such frame");
      const width = scene.canvas.width;
      const height = scene.canvas.height;
      const { x0, y0, x1, y1 } = bounds(rect, width, height);
      let changed = 0;
      let total = 0;
      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const i = (y * width + x) * 4;
          const d =
            Math.abs((fa[i] ?? 0) - (fb[i] ?? 0)) +
            Math.abs((fa[i + 1] ?? 0) - (fb[i + 1] ?? 0)) +
            Math.abs((fa[i + 2] ?? 0) - (fb[i + 2] ?? 0));
          if (d > tol) changed += 1;
          total += 1;
        }
      }
      return total > 0 ? changed / total : 0;
    },
    maxDifference(a, b, rect) {
      const fa = frames[a];
      const fb = frames[b];
      if (!fa || !fb) throw new Error("no such frame");
      const width = scene.canvas.width;
      const { x0, y0, x1, y1 } = bounds(rect, width, scene.canvas.height);
      let worst = 0;
      for (let y = y0; y < y1; y += 1)
        for (let x = x0; x < x1; x += 1)
          for (let k = 0; k < 3; k += 1) {
            const i = (y * width + x) * 4 + k;
            worst = Math.max(worst, Math.abs((fa[i] ?? 0) - (fb[i] ?? 0)));
          }
      return worst;
    },
    coverage(a, rect) {
      const fa = frames[a];
      if (!fa) throw new Error("no such frame");
      const width = scene.canvas.width;
      const height = scene.canvas.height;
      const bg = Color.fromCssColorString(BACKGROUND);
      const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
      const { x0, y0, x1, y1 } = bounds(rect, width, height);
      let covered = 0;
      let total = 0;
      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const i = (y * width + x) * 4;
          const d =
            Math.abs((fa[i] ?? 0) - br) +
            Math.abs((fa[i + 1] ?? 0) - bgG) +
            Math.abs((fa[i + 2] ?? 0) - bb);
          if (d > 24) covered += 1;
          total += 1;
        }
      }
      return total > 0 ? covered / total : 0;
    },
    rectOf(id, grow = 1) {
      const sphere = instanceSphere(ASSET, id);
      if (!sphere) return null;
      const centre = SceneTransforms.worldToWindowCoordinates(
        scene,
        sphere.center,
        new Cartesian2(),
      );
      if (!centre) return null;
      const edge = Cartesian3.add(
        sphere.center,
        Cartesian3.multiplyByScalar(scene.camera.rightWC, sphere.radius, new Cartesian3()),
        new Cartesian3(),
      );
      const side = SceneTransforms.worldToWindowCoordinates(scene, edge, new Cartesian2());
      if (!side) return null;
      const radius = Math.hypot(side.x - centre.x, side.y - centre.y) * grow;
      const ratio = scene.canvas.width / Math.max(1, scene.canvas.clientWidth);
      return {
        x: (centre.x - radius) * ratio,
        y: (centre.y - radius) * ratio,
        width: 2 * radius * ratio,
        height: 2 * radius * ratio,
      };
    },
    async windOn(strength, bearingDeg, t0 = 0, seed = 1) {
      windDriver?.rest();
      windDriver = undefined;
      windTickOff?.();
      const driver = windDriverFor(seed);
      windClock = t0;
      windSettings = { strength, bearingDeg };
      windTickOff = scene.preUpdate.addEventListener(() => {
        driver.tick(windClock, windSettings);
      });
      await settle(2);
    },
    async advance(seconds, fps = 30) {
      const steps = Math.round(seconds * fps);
      const start = windClock;
      // The state lives on a fixed grid, so ticking the driver through the steps without
      // drawing each one lands where drawing them would (SwiftShader frames are slow).
      for (let k = 1; k <= steps; k += 1) {
        windClock = start + k / fps;
        windDriver?.tick(windClock, windSettings);
      }
      await settle(2);
    },
    async windOff() {
      windSettings = WIND_CALM;
      await settle(4);
      windTickOff?.();
      windTickOff = undefined;
    },
    windSkins() {
      const driver = windDriver;
      return part().doc.skins.map(({ instance }) => {
        const material = driver?.material(instance);
        return {
          instance,
          wind: material?.wind ?? false,
          stiffness: material?.stiffness ?? 0,
          evidence: material?.evidence ?? "none",
          claimed: driver?.claimed(instance) ?? false,
        };
      });
    },
    pointRect(instance, local, radiusM) {
      const skin = part().doc.byInstance.get(instance);
      const root = tileset.root as { computedTransform?: Matrix4 };
      if (!skin || !root.computedTransform) return null;
      const point = Matrix4.multiplyByPoint(
        root.computedTransform,
        new Cartesian3(
          skin.origin[0] + local[0],
          skin.origin[1] + local[1],
          skin.origin[2] + local[2],
        ),
        new Cartesian3(),
      );
      const centre = SceneTransforms.worldToWindowCoordinates(scene, point, new Cartesian2());
      const edge = Cartesian3.add(
        point,
        Cartesian3.multiplyByScalar(scene.camera.rightWC, radiusM, new Cartesian3()),
        new Cartesian3(),
      );
      const side = SceneTransforms.worldToWindowCoordinates(scene, edge, new Cartesian2());
      if (!centre || !side) return null;
      const radius = Math.hypot(side.x - centre.x, side.y - centre.y);
      const ratio = scene.canvas.width / Math.max(1, scene.canvas.clientWidth);
      return {
        x: (centre.x - radius) * ratio,
        y: (centre.y - radius) * ratio,
        width: 2 * radius * ratio,
        height: 2 * radius * ratio,
      };
    },
    scanRect(point, radiusM) {
      const root = tileset.root as { computedTransform?: Matrix4 };
      if (!root.computedTransform) return null;
      const world = Matrix4.multiplyByPoint(
        root.computedTransform,
        new Cartesian3(point[0], point[1], point[2]),
        new Cartesian3(),
      );
      const centre = SceneTransforms.worldToWindowCoordinates(scene, world, new Cartesian2());
      const edge = Cartesian3.add(
        world,
        Cartesian3.multiplyByScalar(scene.camera.rightWC, radiusM, new Cartesian3()),
        new Cartesian3(),
      );
      const side = SceneTransforms.worldToWindowCoordinates(scene, edge, new Cartesian2());
      if (!centre || !side) return null;
      const radius = Math.hypot(side.x - centre.x, side.y - centre.y);
      const ratio = scene.canvas.width / Math.max(1, scene.canvas.clientWidth);
      return {
        x: (centre.x - radius) * ratio,
        y: (centre.y - radius) * ratio,
        width: 2 * radius * ratio,
        height: 2 * radius * ratio,
      };
    },
    async telemetryReady() {
      for (let frame = 0; frame < 600 && !telemetryOf(ASSET); frame += 1) await nextFrame(scene);
      if (!telemetryOf(ASSET)) throw new Error("telemetry never attached");
      await settle(2);
    },
    async telemetryAt(ms) {
      telemetryClock = ms;
      await settle(4);
    },
    telemetryStatus() {
      return (telemetryOf(ASSET)?.driver.statuses ?? []).map((s) => ({
        instance: s.instance,
        source: s.source,
        state: s.state,
        ageMs: s.ageMs,
        via: s.via,
        position: s.pose ? [...s.pose.position] : null,
        orientation: s.pose ? [...s.pose.orientation] : null,
        playoutMs: s.playoutMs,
      }));
    },
    mute(sourceId, muted) {
      const source = telemetryOf(ASSET)?.sources.get(sourceId);
      if (!(source instanceof SyntheticSource)) throw new Error(`no synthetic ${sourceId}`);
      source.muted = muted;
    },
    async telemetryReset(ms) {
      const attached = telemetryOf(ASSET);
      if (!attached) throw new Error("no telemetry");
      attached.driver.reset(skinningOf(ASSET), () => telemetryOf(ASSET)?.rigid);
      telemetryClock = ms;
      await settle(4);
    },
    rigidInfo(ids) {
      const rigid = telemetryOf(ASSET)?.rigid;
      return {
        driven: rigid?.driven ?? [],
        active: rigid?.active ?? false,
        slots: ids.map((id) => rigid?.slotOf(id) ?? 0),
      };
    },
    rendererStatus() {
      return host?.status() ?? null;
    },
    async objectPose(instance, pose) {
      useSceneObjects.getState().setPose(ASSET, instance, pose);
      await settle(10);
    },
    motionGap() {
      return useInstances.getState().motionGaps[ASSET] ?? null;
    },
    skinHandles(instance) {
      const handles = skinningOf(ASSET)?.instanceHandles(instance);
      return handles ? Array.from(handles) : null;
    },
    hooks() {
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive;
      const chain = primitive ? motionChainOf(primitive) : undefined;
      const skin = skinningOf(ASSET);
      return {
        motion: chain?.parts.map((p) => p.motionFunction) ?? [],
        jacobian:
          chain?.parts.flatMap((p) => (p.jacobianFunction ? [p.jacobianFunction] : [])) ?? [],
        skin: skin?.installed ?? false,
        active: skin?.active ?? false,
      };
    },
  };
}
