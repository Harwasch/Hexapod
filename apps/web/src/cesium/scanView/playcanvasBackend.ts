/**
 * The PlayCanvas back-end of the scan renderer (ScanRendererHost): SuperSplat's own engine
 * (MIT). Each tile is decoded in a worker into the columns PlayCanvas's `GSplatData` takes
 * (playcanvasTile.worker.ts), already in PlayCanvas's Morton order, made a `GSplatResource`
 * within the main thread's frame budget (tileWork.ts), and drawn by PlayCanvas's unified
 * splat renderer, which sorts every tile's splats together. PlayCanvas has no level of detail
 * over resources given this way, so it draws what it is given: streamed at the budget itself.
 *
 * Drawn on demand. PlayCanvas runs its own update loop (streaming, the hand-off of sort
 * results) but only renders when the host draws a frame (`autoRender` off: the frame must be
 * drawn from the globe's camera, in step with it). The loop says when it has something new --
 * a sort result to apply, streamed detail -- with `frame:request`, PlayCanvas's own hook for
 * apps that render on demand, and the host draws a frame for it; `frame:ready` says whether a
 * rendered frame showed every change sorted, which is when a new tile counts as drawn
 * (handover.ts). Once nothing loads or sorts the loop itself pauses, and anything that could
 * give it work (a frame drawn, a sort landing) starts it again.
 *
 * The scan's objects (scanInstances.ts): each tile's resource carries one more stream, the
 * instance id of every splat (`splatInstance`, R32U, in the resource's own Morton order), and
 * a work-buffer modifier reads it with the shared state table when PlayCanvas copies the tile
 * into the buffer it sorts and draws from. A change of what is hidden or highlighted updates
 * the table and marks every tile for a new copy: no tile is decoded or uploaded again.
 *
 * Two entry points, one renderer. `createBackend` is the default: PlayCanvas's `Application`,
 * which always makes a WebGL2 device. `createWebgpuBackend` is the WebGPU trial
 * (docs/WEBGPU_TRIAL.md): the device is made first and asynchronously, WebGPU preferred,
 * PlayCanvas's own WebGL2 fallback after it, and the app is an `AppBase` with only what this
 * renderer uses (a camera, gsplats, the gsplat and texture asset handlers). On WebGPU three
 * things differ: the work-buffer modifier is WGSL (PlayCanvas picks the language by device);
 * PlayCanvas sorts on the GPU in the frame that draws, so no sort result arrives later to ask
 * for the frame that confirms a new tile is drawn -- the renderer asks for that one frame
 * itself (`frameWanted`); and a device can be lost for good (a driver reset, a GPU process
 * crash), which the host answers by drawing with WebGL2 instead (`hooks.deviceLost`).
 */

import * as pc from "playcanvas";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import type { TileNode } from "@/view/tiles";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import { splatMinPixelSize } from "./quality";
import {
  idsInResourceOrder,
  SCAN_INSTANCE_RULE_GLSL,
  SCAN_INSTANCE_RULE_WGSL,
  type InstanceStyle,
} from "./scanInstances";
import { countOverlayLoopTick } from "./stats";
import type { BackendHooks, GraphicsApi, ScanBackend, ScanPose, SplatRendererKind } from "./types";

interface Decoded {
  id: number;
  count?: number;
  /** In Morton order already (`order`): playcanvasTile.worker.ts. */
  properties?: Record<string, Float32Array>;
  /** The tile's centre: the positions are relative to it (playcanvasTile.worker.ts). */
  origin?: [number, number, number];
  /** `checksumPositions` of the tile's own positions, before centring. */
  checksum?: string;
  /** Slot `i` holds the tile's own splat `order[i]`. */
  order?: Uint32Array;
  error?: string;
}

/** The per-splat stream with each splat's instance id. */
const INSTANCE_STREAM = "splatInstance";

/**
 * The work-buffer modifier (PlayCanvas's `gsplatModifyVS`): the scene-object rule on the
 * colour, reading the tile's id stream (`loadSplatInstance`, declared by the resource's format)
 * at the splat being copied.
 */
export const PLAYCANVAS_INSTANCE_GLSL = `
uniform highp sampler2D uInstanceState;
uniform vec4 uInstanceParams;
uniform vec4 uInstanceTint;
uniform vec4 uInstanceDim;
${SCAN_INSTANCE_RULE_GLSL}
void modifySplatCenter(inout vec3 center) {
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
}
void modifySplatColor(vec3 center, inout vec4 color) {
    if (uInstanceParams.x < 0.5) {
        return;
    }
    color = hexapodInstanceColor(loadSplatInstance().r, color);
}
`;

/**
 * The same modifier in WGSL, for PlayCanvas on WebGPU: PlayCanvas's WGSL signatures for the
 * three functions (`gsplatModifyVS`), the colour passed by pointer, and the id stream's
 * `loadSplatInstance()` returning a `vec4u` (an R32U stream reads as `texture_2d<u32>`).
 */
export const PLAYCANVAS_INSTANCE_WGSL = `
uniform uInstanceParams: vec4f;
uniform uInstanceTint: vec4f;
uniform uInstanceDim: vec4f;
var uInstanceState: texture_2d<f32>;
${SCAN_INSTANCE_RULE_WGSL}
fn modifySplatCenter(center: ptr<function, vec3f>) {
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    if (uniform.uInstanceParams.x < 0.5) {
        return;
    }
    *color = hexapodInstanceColor(loadSplatInstance().r, *color);
}
`;

/** The modifier in both languages: PlayCanvas takes the one its device speaks
 *  (`GSplatComponent.setWorkBufferModifier`), so hide and highlight work on either API. */
export const PLAYCANVAS_INSTANCE_MODIFIER = {
  glsl: PLAYCANVAS_INSTANCE_GLSL,
  wgsl: PLAYCANVAS_INSTANCE_WGSL,
} as const;

/** What a tile needs to take its ids: its digest, and where each original splat went. */
interface TileBinding {
  checksum: string;
  /** Resource slot `i` holds the tile's splat `order[i]` (GSplatData's Morton reorder). */
  order: Uint32Array;
  resource: pc.GSplatResource;
  /** The doc the ids were written from, and whether the file lists the tile. */
  doc: InstancesDoc | null;
  matched: boolean;
}

/** Frames a disposed tile's GPU resource outlives its entity: PlayCanvas's unified renderer
 *  drops a removed entity from its placements in the next frame it renders, and destroying
 *  the resource first left a placement with none ("Cannot read properties of null (reading
 *  'hasCenters')"). Counted in frames rendered, not loop ticks: a removal is only processed
 *  by a render -- so while one is waiting, the renderer asks the host for those frames
 *  (`frameWanted`): a tile evicted at rest (an off-screen one dropped from the cache at the
 *  end of a plan, a load that landed after it was abandoned) had no frames coming, its
 *  resource was never freed, and PlayCanvas's loop, which pauses only once nothing is
 *  waiting, ticked every display frame for as long as the page stayed open. */
const DESTROY_AFTER_FRAMES = 3;

/**
 * When PlayCanvas cannot say (no `frame:ready`, which it fires for every rendered frame with
 * splats in it), a new entity counts as drawn this many rendered frames and milliseconds after
 * it was added: the unified renderer copies it into its work buffer and sorts in a worker
 * first.
 */
const SETTLE_FRAMES = 4;
const SETTLE_MS = 120;

/** Loop ticks in a row with nothing to load, sort or show before PlayCanvas's loop pauses. */
export const QUIET_TICKS_BEFORE_PAUSE = 10;

/** Tiles decoded at once: a quarter of the cores, one to three. */
function workerCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  return Math.min(3, Math.max(1, Math.floor(cores / 4)));
}

/** The `frame:ready` handler's arguments (gsplat/system.d.ts EVENT_FRAMEREADY). */
type FrameReady = (camera: unknown, layer: unknown, ready: boolean, loadingCount: number) => void;

/** The overlay's canvas: transparent, premultiplied, no multisampling (splats are smooth). */
const DEVICE_OPTIONS = {
  alpha: true,
  antialias: false,
  premultipliedAlpha: true,
  powerPreference: "high-performance",
} as const;

/** PlayCanvas on WebGL2, the default: its `Application`, which always makes a WebGL2 device. */
export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  hooks: BackendHooks,
): Promise<ScanBackend<pc.Entity>> {
  const app = new pc.Application(canvas, { graphicsDeviceOptions: { ...DEVICE_OPTIONS } });
  return Promise.resolve(assemble(app, budget, hooks, { name: "playcanvas", note: null }));
}

/**
 * Why WebGPU was not used when PlayCanvas fell back to WebGL2 by itself (it logs the reason
 * and moves on): the browser has none, the page is not a secure context, or no adapter or
 * device came (PlayCanvas also declines PowerVR GPUs' WebGPU).
 */
export function whyNoWebgpu(
  gpu: unknown = typeof navigator === "undefined"
    ? undefined
    : (navigator as { gpu?: unknown }).gpu,
  secure: boolean = typeof window === "undefined" || window.isSecureContext,
): string {
  if (!gpu) return secure ? "this browser has no WebGPU" : "WebGPU needs a secure (https) page";
  return "no WebGPU adapter or device";
}

/**
 * PlayCanvas on WebGPU, the trial (docs/WEBGPU_TRIAL.md). `createGraphicsDevice` tries WebGPU
 * and then WebGL2 on the same canvas -- WebGPU takes the canvas only once its adapter and
 * device exist, so a WebGL2 fallback still can -- and after that, silently, a Null device that
 * draws nothing. A Null device is refused here (the host then draws with `createBackend` on a
 * fresh canvas); a WebGL2 one is used, and says why (`apiNote`).
 */
export async function createWebgpuBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  hooks: BackendHooks,
): Promise<ScanBackend<pc.Entity>> {
  const device = (await pc.createGraphicsDevice(canvas, {
    ...DEVICE_OPTIONS,
    deviceTypes: [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2],
    // Not for a headset: an XR-compatible adapter can be another GPU than the display's.
    xrCompatible: false,
  })) as pc.GraphicsDevice;
  if (device.isNull || (!device.isWebGPU && !device.isWebGL2)) {
    device.destroy();
    throw new Error("PlayCanvas started neither WebGPU nor WebGL2 (its Null device draws nothing)");
  }
  // `Application` would make a WebGL2 device of its own: the app is assembled around this one,
  // with only what the renderer uses: a camera, gsplats, and the handlers a scan's streamed
  // package loads through (its octree and SOG chunks are gsplat assets; each chunk's WebP
  // planes are texture assets -- without that handler a native scan loads nothing).
  const app = new pc.AppBase(canvas);
  const options = new pc.AppOptions();
  options.graphicsDevice = device;
  options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
  options.resourceHandlers = [pc.GSplatHandler, pc.TextureHandler];
  app.init(options);
  return assemble(app, budget, hooks, {
    name: "playcanvas-webgpu",
    note: device.isWebGPU ? null : `WebGPU unavailable: ${whyNoWebgpu()}`,
  });
}

/** What a lost WebGPU device says (`GPUDeviceLostInfo`). */
interface LostInfo {
  reason: string;
  message: string;
}

/** The `lost` promise of a WebGPU device (`GPUDevice.lost`); null on WebGL2. */
function webgpuLost(device: pc.GraphicsDevice): Promise<LostInfo> | null {
  const wgpu = (device as unknown as { wgpu?: { lost?: Promise<LostInfo> } }).wgpu;
  return wgpu?.lost ?? null;
}

/** Everything but making the app and its device: the same renderer on either API. */
function assemble(
  app: pc.AppBase,
  budget: number,
  hooks: BackendHooks,
  identity: { name: SplatRendererKind; note: string | null },
): ScanBackend<pc.Entity> {
  const api: GraphicsApi = app.graphicsDevice.isWebGPU ? "webgpu" : "webgl2";
  // On WebGPU PlayCanvas sorts on the GPU, in the frame that draws (gsplat-params.js,
  // `_resolveRenderer`): no sort result comes back later to ask for the frame that confirms a
  // new tile is drawn, so this renderer asks for it itself (`render`).
  const sortsOnGpu = app.graphicsDevice.isWebGPU;
  /** The device is gone for good: nothing more is drawn, and the host replaces the renderer. */
  let lost = false;
  /** The renderer is letting go of its device itself: that loss is not news. */
  let destroying = false;
  // PlayCanvas answers a lost WebGPU device by making another on the same canvas. A loss on a
  // phone is mostly memory pressure or a GPU process restart, which a new WebGPU device meets
  // again; the trial's answer is the API every device here has drawn with, WebGL2, on a fresh
  // canvas (the host's), so a lost device costs the scan's tiles once, not the view.
  void webgpuLost(app.graphicsDevice)?.then((info) => {
    if (destroying) return;
    lost = true;
    hooks.deviceLost?.(`WebGPU device lost: ${info.message || info.reason}`);
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.scene.gsplat.splatBudget = budget;
  // Full-precision work buffer rather than the compact one (quantised transforms): the tiles
  // are already the detail the view asked for. The smallest splat kept is set per resolution
  // (`splatMinPixelSize`, in render).
  (app.scene.gsplat as unknown as { dataFormat: string }).dataFormat = "large";
  const camera = new pc.Entity("scan-camera");
  camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0, 0) });
  app.root.addChild(camera);
  // PlayCanvas updates in its own loop (streaming, sorting) but draws only from `render`, in
  // the same frame and from the same pose as the globe under it. Drawn in its own loop, it ran
  // before the host set the camera: a pose behind, so the scan slid on the map as the view
  // moved, and a resize cleared the canvas a frame before anything was drawn on it (a black
  // flash where the world is clipped away under the scan).
  app.autoRender = false;

  // The loop pauses when it has nothing to do (see the file comment): its next animation frame
  // is simply not asked for, and `resumeLoop` asks for it again.
  let looping = true;
  let quietTicks = 0;
  /** The last rendered frame showed every change sorted, with nothing left loading. */
  let settled = false;
  const tickAgain = app.requestAnimationFrame.bind(app);
  app.requestAnimationFrame = () => {
    if (looping) tickAgain();
  };
  const resumeLoop = (): void => {
    quietTicks = 0;
    if (looping) return;
    looping = true;
    tickAgain();
  };
  app.start();

  const workers = Array.from(
    { length: workerCount() },
    () => new Worker(new URL("./playcanvasTile.worker.ts", import.meta.url), { type: "module" }),
  );
  const waiting = new Map<number, (message: Decoded) => void>();
  let nextId = 1;
  let turn = 0;
  for (const worker of workers) {
    worker.onmessage = (event: MessageEvent<Decoded>) => {
      waiting.get(event.data.id)?.(event.data);
      waiting.delete(event.data.id);
    };
  }
  const decode = (url: string): Promise<Decoded> =>
    new Promise((resolve) => {
      const id = nextId++;
      waiting.set(id, resolve);
      const worker = workers[turn++ % workers.length];
      worker?.postMessage({ id, url, maxSh: hooks.maxShDegree });
    });

  const resources = new WeakMap<pc.Entity, pc.GSplatResource>();
  /** Tiles loaded and not yet disposed, with what binds them to the scan's objects. */
  const tiles = new Map<pc.Entity, TileBinding>();
  let style: InstanceStyle | null = null;
  let stateTexture: pc.Texture | null = null;
  const params = new Float32Array(4);
  const tint = new Float32Array(4);
  const dim = new Float32Array(4);
  /** Writes the tile's ids for the current doc (once a doc) and puts the modifier on it. */
  const bind = (entity: pc.Entity, tile: TileBinding): void => {
    const component = entity.gsplat;
    if (!component) return;
    if (!style) {
      if (tile.doc === null) return;
      params[0] = 0;
      component.setParameter("uInstanceParams", params);
      return;
    }
    if (tile.doc !== style.doc) {
      const ids = tileInstanceIds(style.doc, tile.checksum);
      const listed = ids?.length === tile.order.length ? ids : undefined;
      const resource = tile.resource as unknown as {
        format: { addExtraStreams(streams: { name: string; format: number }[]): void };
        getTexture(name: string): pc.Texture | null;
      };
      resource.format.addExtraStreams([{ name: INSTANCE_STREAM, format: pc.PIXELFORMAT_R32U }]);
      const texture = resource.getTexture(INSTANCE_STREAM);
      if (!texture) return;
      idsInResourceOrder(listed, tile.order, texture.lock() as Uint32Array);
      texture.unlock();
      tile.doc = style.doc;
      tile.matched = listed !== undefined;
      component.setWorkBufferModifier(PLAYCANVAS_INSTANCE_MODIFIER);
    }
    if (stateTexture) component.setParameter("uInstanceState", stateTexture);
    component.setParameter("uInstanceTint", tint);
    component.setParameter("uInstanceDim", dim);
    // Last: setting a parameter marks the tile for a new copy into the work buffer.
    component.setParameter("uInstanceParams", params);
  };
  const setInstances = (next: InstanceStyle | null): void => {
    style = next;
    if (next) {
      if (stateTexture?.height !== next.rows) {
        stateTexture?.destroy();
        stateTexture = new pc.Texture(app.graphicsDevice, {
          name: "instanceState",
          width: INSTANCE_TEXTURE_WIDTH,
          height: next.rows,
          format: pc.PIXELFORMAT_RGBA8,
          mipmaps: false,
          minFilter: pc.FILTER_NEAREST,
          magFilter: pc.FILTER_NEAREST,
          addressU: pc.ADDRESS_CLAMP_TO_EDGE,
          addressV: pc.ADDRESS_CLAMP_TO_EDGE,
        });
      }
      (stateTexture.lock() as Uint8Array).set(next.state);
      stateTexture.unlock();
      params.set(next.params);
      tint.set(next.tint);
      dim.set(next.dim);
    }
    for (const [entity, tile] of tiles) bind(entity, tile);
  };
  /** Frames rendered so far, the last one that showed every change sorted (`frame:ready`),
   *  and the frame count when each entity was added. */
  let framesDrawn = 0;
  let lastReadyFrame = -1;
  let frameReadySeen = false;
  const addedAt = new WeakMap<pc.Entity, number>();
  /** Added and not yet confirmed drawn by a ready frame two frames on (`isDrawn`). */
  const unconfirmed = new Set<pc.Entity>();
  const doomed: { resource: pc.GSplatResource; at: number }[] = [];
  const target = new pc.Vec3();
  const up = new pc.Vec3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  /** A decoded tile made into PlayCanvas's resource and entity: main-thread work. */
  const build = (tile: TileNode, decoded: Decoded): pc.Entity => {
    const { properties, count, order } = decoded;
    if (!properties || count === undefined || !order) {
      throw new Error(decoded.error ?? "The tile could not be decoded.");
    }
    const data = new pc.GSplatData([
      {
        name: "vertex",
        count,
        properties: Object.entries(properties).map(([name, storage]) => ({
          type: "float",
          name,
          storage,
          byteSize: 4,
        })),
      },
    ]);
    // glTF KHR_gaussian_splatting's convention: linear scale, opacity after the sigmoid.
    data.activated = true;
    // Already in Morton order (the worker's `mortonOrder`), as `reorderData` would leave it.
    const resource = new pc.GSplatResource(app.graphicsDevice, data);
    const entity = new pc.Entity(tile.uri);
    if (decoded.origin) entity.setLocalPosition(...decoded.origin);
    entity.addComponent("gsplat", { resource });
    resources.set(entity, resource);
    if (decoded.checksum !== undefined) {
      const binding: TileBinding = {
        checksum: decoded.checksum,
        order,
        resource,
        doc: null,
        matched: false,
      };
      tiles.set(entity, binding);
      if (style) bind(entity, binding);
    }
    return entity;
  };

  const backend: ScanBackend<pc.Entity> = {
    name: identity.name,
    api,
    apiNote: identity.note,
    loadFactor: 1,
    load: async (tilesetUrl: string, tile: TileNode, signal?: AbortSignal) => {
      const decoded = await decode(new URL(tile.uri, tilesetUrl).toString());
      signal?.throwIfAborted();
      // Building the resource packs every splat into textures on the main thread: within the
      // frame's budget, a little at a time while the camera moves.
      const entity = await hooks.work.run(() => {
        signal?.throwIfAborted();
        return build(tile, decoded);
      });
      return entity;
    },
    add: (entity) => {
      app.root.addChild(entity);
      addedAt.set(entity, framesDrawn);
      unconfirmed.add(entity);
    },
    // Drawn once a rendered frame after the one that took it in showed every change sorted;
    // without `frame:ready`, after a few rendered frames. The same on WebGPU, where the frame
    // that takes a tile in can report ready before PlayCanvas's update has made the tile part
    // of what it draws (its streaming update runs in its own loop, after the frame): one more
    // ready frame is the proof, and `render` asks for it.
    isDrawn: (entity, sinceMs) => {
      const added = addedAt.get(entity) ?? framesDrawn;
      if (frameReadySeen) return lastReadyFrame >= added + 2;
      return framesDrawn - added >= SETTLE_FRAMES && sinceMs >= SETTLE_MS;
    },
    remove: (entity) => {
      unconfirmed.delete(entity);
      if (entity.parent) entity.parent.removeChild(entity);
    },
    dispose: (entity) => {
      unconfirmed.delete(entity);
      tiles.delete(entity);
      const resource = resources.get(entity);
      entity.destroy();
      if (resource) {
        doomed.push({ resource, at: framesDrawn });
        // The renders that let it go (DESTROY_AFTER_FRAMES): asked for, since a dispose at
        // rest has none coming. Within a frame the host takes it as one more frame.
        hooks.frameWanted();
      }
    },
    render: (pose: ScanPose) => {
      // A lost device draws nothing; the host is already replacing this renderer.
      if (lost) return;
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        app.graphicsDevice.maxPixelRatio = pose.pixelRatio;
        app.resizeCanvas(pose.width, pose.height);
        app.scene.gsplat.minPixelSize = splatMinPixelSize(pose.pixelRatio);
      }
      const component = camera.camera;
      if (component) {
        component.fov = (pose.fovy * 180) / Math.PI;
        component.nearClip = pose.near;
        component.farClip = pose.far;
        component.aspectRatioMode = pc.ASPECT_AUTO;
      }
      camera.setPosition(...pose.eye);
      target.set(
        pose.eye[0] + pose.direction[0],
        pose.eye[1] + pose.direction[1],
        pose.eye[2] + pose.direction[2],
      );
      camera.lookAt(target, up.set(...pose.up));
      framesDrawn += 1;
      app.render();
      while (doomed[0] && framesDrawn - doomed[0].at >= DESTROY_AFTER_FRAMES) {
        doomed.shift()?.resource.destroy();
      }
      // A resource still waiting for its renders gets them: at most DESTROY_AFTER_FRAMES
      // frames after the last dispose, and then the loop can pause (`frameupdate`).
      if (doomed.length > 0) hooks.frameWanted();
      // A frame drawn may have started a sort or a load: the loop watches for it.
      resumeLoop();
      // Sorted on the GPU, this frame was ready and a tile added lately still needs a ready
      // frame two on to count as drawn: nothing else will ask for that frame (no sort result
      // comes back), so this one does -- one frame per batch of tiles, none at rest.
      for (const entity of unconfirmed) {
        if (lastReadyFrame >= (addedAt.get(entity) ?? 0) + 2) unconfirmed.delete(entity);
      }
      if (sortsOnGpu && lastReadyFrame === framesDrawn && unconfirmed.size > 0) {
        hooks.frameWanted();
      }
    },
    setBudget: (drawn) => {
      app.scene.gsplat.splatBudget = drawn;
    },
    setInstances,
    instanceTiles: () => {
      let matched = 0;
      for (const tile of tiles.values()) if (tile.matched) matched += 1;
      return { tiles: tiles.size, matched };
    },
    // PlayCanvas's own streamed level of detail -- what superspl.at runs: chunks of a few
    // hundred thousand splats per level as lossless WebP textures the browser decodes off the
    // main thread and uploads as they are, chosen against `splatBudget` from the camera.
    streamNative: async (url) => {
      const asset = new pc.Asset(url, "gsplat", { url });
      app.assets.add(asset);
      await new Promise<void>((resolve, reject) => {
        asset.once("load", () => resolve());
        asset.once("error", (error: unknown) => reject(new Error(String(error))));
        app.assets.load(asset);
      });
      const entity = new pc.Entity("scan");
      entity.addComponent("gsplat", { asset, unified: true });
      app.root.addChild(entity);
      resumeLoop();
      return {
        splats: () => app.scene.gsplat.splatBudget,
        stop: () => {
          entity.destroy();
          app.assets.remove(asset);
          asset.unload();
        },
      };
    },
    destroy: () => {
      for (const worker of workers) worker.terminate();
      waiting.clear();
      tiles.clear();
      stateTexture?.destroy();
      stateTexture = null;
      // PlayCanvas only lets go of its context (`gl = null`), and the browser frees a
      // context's memory whenever it collects the canvas -- with a new canvas and context per
      // session, a few sessions in a visit held several. Losing it frees it now, once
      // PlayCanvas has taken its own handlers off the canvas. A WebGPU device is destroyed by
      // PlayCanvas itself (`GPUDevice.destroy`), which frees its memory at once.
      destroying = true;
      const gl = (app.graphicsDevice as unknown as { gl?: WebGLRenderingContext | null }).gl;
      try {
        app.destroy();
      } catch (error) {
        // A lost WebGPU device can refuse what PlayCanvas does on the way out; nothing of it is
        // drawn again either way.
        if (!lost) throw error;
      }
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
  // A resource waiting to be destroyed keeps the loop going until the renders it asked for
  // (`render`, `dispose`) have let it go: a few frames, never for good.
  app.on("frameupdate", () => {
    countOverlayLoopTick();
    quietTicks += 1;
    if (quietTicks >= QUIET_TICKS_BEFORE_PAUSE && settled && doomed.length === 0) looping = false;
  });
  const gsplat = app.systems.gsplat as unknown as pc.EventHandler | undefined;
  // New streamed detail, or a sort result waiting to be applied: a frame shows it.
  gsplat?.on("frame:request", () => {
    quietTicks = 0;
    hooks.frameWanted();
  });
  gsplat?.on("frame:ready", ((_camera, _layer, ready, loadingCount) => {
    frameReadySeen = true;
    settled = ready && !loadingCount;
    if (ready) lastReadyFrame = framesDrawn;
  }) as FrameReady);
  // A sort that finished while the loop was paused: the loop hands it over (`frame:request`).
  // (Sorted on the CPU only: a GPU sort finishes in the frame that draws.)
  app.scene.on("gsplat:sorted", resumeLoop);
  return backend;
}
