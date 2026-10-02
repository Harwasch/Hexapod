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
  SceneTransforms,
  type Scene,
} from "cesium";

import { attachInstances, instanceSphere } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { motionChainOf } from "@/cesium/splatMotionChain";
import { attachSkin, skinningOf, type SplatSkinning } from "@/cesium/splatSkin";
import { HANDLE_FLOATS, rigidHandle } from "@/lib/skin";
import { useInstances } from "@/state/instances";

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
  /** Keeps a copy of the frame; returns its index. */
  frame(): number;
  /** Share of pixels (in `rect`) that differ between frames `a` and `b` by more than `tol`. */
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  /** Share of `rect` (or the canvas) that is not background in frame `a`. */
  coverage(a: number, rect?: Rect): number;
  /** Where instance `id` is on screen, or null. */
  rectOf(id: number, grow?: number): Rect | null;
  /** The motion chain's parts and whether a Jacobian is declared. */
  hooks(): { motion: string[]; jacobian: string[]; skin: boolean; active: boolean };
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
}): Promise<SkinHarness> {
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

  const settle = async (frames: number): Promise<void> => {
    for (let frame = 0; frame < frames; frame += 1) await nextFrame(scene);
  };
  const part = (): SplatSkinning => {
    const found = skinningOf(ASSET);
    if (!found) throw new Error("the skin never loaded");
    return found;
  };
  const frames: Uint8ClampedArray[] = [];
  let wobbleOff: (() => void) | undefined;

  function pixels(): { data: Uint8ClampedArray; width: number; height: number } {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) throw new Error("no 2d context");
    context.drawImage(canvas, 0, 0);
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
