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
 * Per-splat data rides as extra resource streams, in the resource's own Morton order, bound to
 * each tile by the checksum of its positions (the worker digests them as it decodes; `order`
 * says where each of the tile's own splats went):
 *
 * - `splatInstance` (R32U): the instance id (scanInstances.ts), for hide and highlight and for
 *   rigid motion;
 * - `splatSkin` (R32U) and `splatWeights` (RGBA32U): the skin id and the `skin.bin` row
 *   (scanMotion.ts), only on tiles `skin.json` lists with skinned splats -- written within the
 *   frame budget (`hooks.work`), as a tile's resource is built.
 *
 * A work-buffer modifier reads them when PlayCanvas copies a tile into the buffer it sorts and
 * draws from: the scene-object colour rule (`modifySplatColor`), and the motion -- skin
 * handles and rigid poses from shared tables -- on the centre (`modifySplatCenter`) and, through
 * the motion's linear part, on the rotation and scales (`modifySplatRotationScale`: the
 * covariance `J·Σ·Jᵀ`). A change re-copies only the tiles that hold what changed; nothing is
 * decoded or uploaded again.
 *
 * Sorting: PlayCanvas sorts on the CPU from each resource's `centers`, which the modifier does
 * not see. A tile holding a rigidly moved object has its centres moved the same way (and its
 * `centersVersion` bumped, which has PlayCanvas re-send them and sort again), at most every
 * `SORT_REFRESH_MS` while it moves -- the host is told when the held-back one is due
 * (`frameDueBy`), so the last pose is sorted even once nothing moves -- and at once when it
 * comes to rest; a skin's sway is sorted at rest, as CesiumJS sorts it. A split object is its
 * own entity, sorted where it is placed.
 *
 * Selecting in the scene (cesium/sceneSelect) reads the tiles drawn now in each tile's own
 * order (`pickTiles`): the Morton order put back, so a pick's index is the index the tile's
 * ids in `instances.json` are listed by. The copies are made only when a pick first asks for
 * them (a scan with objects), not for every tile of every scan.
 *
 * Two entry points, one renderer. `createBackend` is the default: PlayCanvas's `Application`,
 * which always makes a WebGL2 device. `createWebgpuBackend` is the WebGPU trial
 * (docs/WEBGPU_TRIAL.md): the device is made first and asynchronously, WebGPU preferred,
 * PlayCanvas's own WebGL2 fallback after it, and the app is an `AppBase` with only what this
 * renderer uses (a camera, gsplats, the gsplat and texture asset handlers). On WebGPU four
 * things differ: a modifier is applied only where it has a WGSL form (`WorkBufferModifier`:
 * hide and highlight have one, the motion not yet), so the renderer moves nothing there and
 * says so (no `setMotion`) -- the host draws a scan with objects or motion with WebGL2 instead;
 * PlayCanvas sorts on the GPU in the frame that draws, so no sort result arrives later to ask
 * for the frame that confirms a new tile is drawn -- the renderer asks for that one frame
 * itself (`frameWanted`); and a device can be lost for good (a driver reset, a GPU process
 * crash), which the host answers by drawing with WebGL2 instead (`hooks.deviceLost`).
 */

import * as pc from "playcanvas";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { tileSkin, type SkinDoc } from "@/lib/skin";
import type { PickTile } from "@/lib/splatPick";
import type { TileNode } from "@/view/tiles";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import { splatMinPixelSize } from "./quality";
import {
  idsInResourceOrder,
  SCAN_INSTANCE_RULE_GLSL,
  SCAN_INSTANCE_RULE_WGSL,
  type InstanceStyle,
} from "./scanInstances";
import {
  MOTION_TEXTURE_WIDTH,
  movedCenters,
  SCAN_MOTION_GLSL,
  type ScanMotion,
} from "./scanMotion";
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

/** The per-splat streams. */
const INSTANCE_STREAM = "splatInstance";
const SKIN_STREAM = "splatSkin";
const WEIGHTS_STREAM = "splatWeights";

/** Shortest time between two re-sorts of a moving object's tiles. */
export const SORT_REFRESH_MS = 100;

/**
 * The work-buffer modifier (PlayCanvas's `gsplatModifyVS`) for a tile with ids, skin weights,
 * or both: the motion on the centre and the covariance, the scene-object rule on the colour.
 */
export function playcanvasModifierGlsl(ids: boolean, skin: boolean): string {
  const id = ids ? "loadSplatInstance().r" : "0u";
  const skinStep = skin
    ? "hexapodSkinMotion(uSkinHandles, loadSplatSkin().r, loadSplatWeights(), uMotionParams, uMotionExtra.x, center, delta, hexapodLinear);"
    : "";
  const rigidStep = ids
    ? `hexapodRigidMotion(uRigidSlots, uRigidPoses, ${id}, uMotionParams, center, delta, hexapodLinear);`
    : "";
  return `
uniform highp sampler2D uInstanceState;
uniform vec4 uInstanceParams;
uniform vec4 uInstanceTint;
uniform vec4 uInstanceDim;
uniform highp sampler2D uSkinHandles;
uniform highp usampler2D uRigidSlots;
uniform highp sampler2D uRigidPoses;
uniform vec4 uMotionParams;
uniform vec4 uMotionExtra;
${SCAN_INSTANCE_RULE_GLSL}
${SCAN_MOTION_GLSL}
mat3 hexapodLinear = mat3(0.0);
void modifySplatCenter(inout vec3 center) {
    hexapodLinear = mat3(0.0);
    if (uMotionParams.x < 0.5 && uMotionParams.z < 0.5) {
        return;
    }
    vec3 delta = vec3(0.0);
    ${skinStep}
    ${rigidStep}
    center += delta;
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    if (uMotionExtra.y > 0.5) {
        hexapodCovariance(mat3(1.0) + hexapodLinear, rotation, scale);
    }
}
void modifySplatColor(vec3 center, inout vec4 color) {
    if (uInstanceParams.x < 0.5) {
        return;
    }
    color = hexapodInstanceColor(${id}, color);
}
`;
}

/**
 * Hide and highlight in WGSL, for PlayCanvas on WebGPU: PlayCanvas's WGSL signatures for the
 * three functions (`gsplatModifyVS`), the colour passed by pointer, and the id stream's
 * `loadSplatInstance()` returning a `vec4u` (an R32U stream reads as `texture_2d<u32>`). The
 * motion has no WGSL port yet: it is GLSL only (`playcanvasModifierGlsl`).
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

/**
 * A work-buffer modifier in the languages it has: PlayCanvas takes its device's
 * (`GSplatComponent.setWorkBufferModifier`), and one without WGSL is not applied on WebGPU --
 * the tile is drawn as it was decoded.
 */
export interface WorkBufferModifier {
  readonly glsl: string;
  readonly wgsl?: string;
}

const MODIFIERS = new Map<string, WorkBufferModifier>();

/**
 * The modifier for a tile with ids, skin weights, or both: always GLSL (the colour rule and
 * the motion); WGSL only for ids without a skin, and then hide and highlight alone -- a
 * WebGPU renderer moves nothing (see the file comment). One object per kind, so a tile's
 * modifier is compared by identity.
 */
export function playcanvasModifier(ids: boolean, skin: boolean): WorkBufferModifier {
  const key = `${String(ids)}|${String(skin)}`;
  let modifier = MODIFIERS.get(key);
  if (!modifier) {
    modifier = {
      glsl: playcanvasModifierGlsl(ids, skin),
      ...(ids && !skin ? { wgsl: PLAYCANVAS_INSTANCE_WGSL } : {}),
    };
    MODIFIERS.set(key, modifier);
  }
  return modifier;
}

/** The decoded columns a pick copy is made from (lib/splatLayout.ts names). */
type PickColumns = Readonly<Record<string, Float32Array>>;

/**
 * A tile's splats for picking (lib/splatPick.ts), in the tile's own order: the resource's
 * columns are in Morton order -- slot `k` holds the tile's splat `order[k]` -- so each is put
 * back at `order[k]`, and moved off the tile's centre (`origin`) into the scan's frame. A pick
 * then returns the index `instances.json` lists the tile's ids by. Null when a column is
 * missing.
 */
export function pickTileOf(
  columns: PickColumns,
  order: Uint32Array,
  origin: readonly [number, number, number],
  checksum: string,
): PickTile | null {
  const { x, y, z, scale_0: s0, scale_1: s1, scale_2: s2, opacity: alpha } = columns;
  if (!x || !y || !z || !s0 || !s1 || !s2 || !alpha) return null;
  const count = order.length;
  const [ox, oy, oz] = origin;
  const positions = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const opacity = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const i = order[k] ?? 0;
    positions[i * 3] = (x[k] ?? 0) + ox;
    positions[i * 3 + 1] = (y[k] ?? 0) + oy;
    positions[i * 3 + 2] = (z[k] ?? 0) + oz;
    radii[i] = Math.max(s0[k] ?? 0, s1[k] ?? 0, s2[k] ?? 0);
    opacity[i] = alpha[k] ?? 0;
  }
  return { checksum, count, positions, radii, opacity };
}

/** `pick` moved by `matrix` (a split object's placement in the scan frame). */
function placedPickTile(pick: PickTile, matrix: pc.Mat4): PickTile {
  const positions = new Float32Array(pick.positions.length);
  const point = new pc.Vec3();
  for (let i = 0; i < pick.count; i++) {
    point.set(
      pick.positions[i * 3] ?? 0,
      pick.positions[i * 3 + 1] ?? 0,
      pick.positions[i * 3 + 2] ?? 0,
    );
    matrix.transformPoint(point, point);
    positions[i * 3] = point.x;
    positions[i * 3 + 1] = point.y;
    positions[i * 3 + 2] = point.z;
  }
  return { ...pick, positions };
}

/** What binds a tile to the scan's objects: its digest, where each original splat went, and
 *  what was written from which document. */
interface TileBinding {
  checksum: string;
  /** Resource slot `i` holds the tile's splat `order[i]` (the worker's Morton reorder). */
  order: Uint32Array;
  resource: Resource;
  /** The decoded columns (Morton order; the resource's own data, not a copy). */
  columns: PickColumns;
  /** Where the entity sits (the tile's centre): resource centres are relative to it. */
  origin: [number, number, number];
  /** The doc the ids were written from, whether the file lists the tile, and the ids. */
  doc: InstancesDoc | null;
  matched: boolean;
  /** The tile's ids in resource order, when the file lists it. */
  ids: Uint32Array | null;
  idSet: ReadonlySet<number>;
  /** The skin doc the skin streams are (being) written from, and the skins the tile holds. */
  skinDoc: SkinDoc | null;
  skinned: boolean;
  skinSet: ReadonlySet<number>;
  /** The modifier on it now. */
  modifier: WorkBufferModifier | null;
  /** The tile's own centres, kept while a motion has moved them for the sorter. */
  restCenters: Float32Array | null;
  /** The tile's splats for picking, in its own order: made on first need (`pickTiles`). */
  pick: PickTile | null | undefined;
  /** Where `place` last put the tile (a split object at its pose), or null where decoded. */
  placement: pc.Mat4 | null;
  /** `pick` at `placement`: made on first need, so a moving object costs no copy a frame. */
  placedPick: PickTile | null;
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

/** The resource members used here that the typings do not show. */
type Resource = pc.GSplatResource & {
  format: { addExtraStreams(streams: { name: string; format: number }[]): void };
  getTexture(name: string): pc.Texture | null;
  centers: Float32Array | null;
  centersVersion: number;
};

/** The overlay's canvas: transparent, premultiplied, no multisampling (splats are smooth). */
const DEVICE_OPTIONS = {
  alpha: true,
  antialias: false,
  premultipliedAlpha: true,
  powerPreference: "high-performance",
} as const;

/** The device options for `hooks`: the drawn frame kept readable when a harness asks. */
function deviceOptions(hooks: BackendHooks) {
  return { ...DEVICE_OPTIONS, preserveDrawingBuffer: hooks.preserveDrawingBuffer === true };
}

/** PlayCanvas on WebGL2, the default: its `Application`, which always makes a WebGL2 device. */
export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  hooks: BackendHooks,
): Promise<ScanBackend<pc.Entity>> {
  const app = new pc.Application(canvas, { graphicsDeviceOptions: deviceOptions(hooks) });
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
    ...deviceOptions(hooks),
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
  const device = app.graphicsDevice;
  const api: GraphicsApi = device.isWebGPU ? "webgpu" : "webgl2";
  // On WebGPU PlayCanvas sorts on the GPU, in the frame that draws (gsplat-params.js,
  // `_resolveRenderer`): no sort result comes back later to ask for the frame that confirms a
  // new tile is drawn, so this renderer asks for it itself (`render`).
  const sortsOnGpu = device.isWebGPU;
  /** The motion modifier is GLSL only (`playcanvasModifier`): on WebGPU nothing moves. */
  const moves = api === "webgl2";
  /** The device is gone for good: nothing more is drawn, and the host replaces the renderer. */
  let lost = false;
  /** The renderer is letting go of its device itself: that loss is not news. */
  let destroying = false;
  // PlayCanvas answers a lost WebGPU device by making another on the same canvas. A loss on a
  // phone is mostly memory pressure or a GPU process restart, which a new WebGPU device meets
  // again; the trial's answer is the API every device here has drawn with, WebGL2, on a fresh
  // canvas (the host's), so a lost device costs the scan's tiles once, not the view.
  void webgpuLost(device)?.then((info) => {
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

  const dataTexture = (
    name: string,
    width: number,
    height: number,
    format: number,
    data?: Uint8Array | Uint32Array | Float32Array,
  ): pc.Texture => {
    const texture = new pc.Texture(device, {
      name,
      width,
      height,
      format,
      mipmaps: false,
      minFilter: pc.FILTER_NEAREST,
      magFilter: pc.FILTER_NEAREST,
      addressU: pc.ADDRESS_CLAMP_TO_EDGE,
      addressV: pc.ADDRESS_CLAMP_TO_EDGE,
    });
    if (data) {
      (texture.lock() as typeof data).set(data);
      texture.unlock();
    }
    return texture;
  };
  /** Stand-ins for tables not there yet: every sampler the GLSL modifier declares is bound. */
  let empty: { state: pc.Texture; float: pc.Texture; uint: pc.Texture } | null = null;
  const stand = (): { state: pc.Texture; float: pc.Texture; uint: pc.Texture } =>
    (empty ??= {
      state: dataTexture("hexapodEmptyState", 1, 1, pc.PIXELFORMAT_RGBA8, new Uint8Array(4)),
      float: dataTexture("hexapodEmptyFloat", 1, 1, pc.PIXELFORMAT_RGBA32F, new Float32Array(4)),
      uint: dataTexture("hexapodEmptyUint", 1, 1, pc.PIXELFORMAT_RGBA32U, new Uint32Array(4)),
    });

  const resources = new WeakMap<pc.Entity, pc.GSplatResource>();
  /** Tiles loaded and not yet disposed, with what binds them to the scan's objects. */
  const tiles = new Map<pc.Entity, TileBinding>();
  /** Where each entity was decoded to sit (its tile's centre), for `place`. */
  const origins = new WeakMap<pc.Entity, [number, number, number]>();
  /** Tiles on screen now (added, not removed): what a pick reads. */
  const shown = new Set<pc.Entity>();
  let style: InstanceStyle | null = null;
  let motion: ScanMotion | null = null;
  let stateTexture: pc.Texture | null = null;
  let handlesTexture: pc.Texture | null = null;
  let slotsTexture: pc.Texture | null = null;
  let posesTexture: pc.Texture | null = null;
  const params = new Float32Array(4);
  const tint = new Float32Array(4);
  const dim = new Float32Array(4);
  const motionParams = new Float32Array(4);
  const motionExtra = new Float32Array([0, 1, 0, 0]);
  let redrawn = 0;
  /** Tiles whose centres must follow a motion for the sorter, and when they last did. */
  const sortDirty = new Set<pc.Entity>();
  let lastSortAt = -Infinity;

  /** The instances whose ids tiles carry: hide and highlight's, or the motion's. */
  const instancesDoc = (): InstancesDoc | null => style?.doc ?? motion?.instances ?? null;

  const bindIds = (tile: TileBinding): void => {
    const doc = instancesDoc();
    if (!doc || tile.doc === doc) return;
    const ids = tileInstanceIds(doc, tile.checksum);
    const listed = ids?.length === tile.order.length ? ids : undefined;
    const resource = tile.resource;
    resource.format.addExtraStreams([{ name: INSTANCE_STREAM, format: pc.PIXELFORMAT_R32U }]);
    const texture = resource.getTexture(INSTANCE_STREAM);
    if (!texture) return;
    const ordered = idsInResourceOrder(listed, tile.order, texture.lock() as Uint32Array);
    // The texture is padded past the tile's splats: what the tile holds is the first `count`.
    const own = ordered.subarray(0, tile.order.length);
    tile.ids = listed ? Uint32Array.from(own) : null;
    tile.idSet = new Set(own);
    texture.unlock();
    tile.doc = doc;
    tile.matched = listed !== undefined;
  };

  /** Writes the tile's skin and weight streams from `doc`: main-thread work (`hooks.work`). */
  const writeSkin = (tile: TileBinding, doc: SkinDoc): void => {
    const found = tileSkin(doc, tile.checksum);
    const skinSet = new Set<number>();
    const fits = found?.skins.length === tile.order.length;
    if (fits) for (const id of found.skins) if (id !== 0) skinSet.add(id);
    tile.skinSet = skinSet;
    const resource = tile.resource;
    if (!fits || skinSet.size === 0) {
      // Listed before, not now: its skins are zeroed (a stream cannot be taken back).
      const skins = tile.skinned ? resource.getTexture(SKIN_STREAM) : null;
      if (skins) {
        (skins.lock() as Uint32Array).fill(0);
        skins.unlock();
      }
      return;
    }
    resource.format.addExtraStreams([
      { name: SKIN_STREAM, format: pc.PIXELFORMAT_R32U },
      { name: WEIGHTS_STREAM, format: pc.PIXELFORMAT_RGBA32U },
    ]);
    const skins = resource.getTexture(SKIN_STREAM);
    const weights = resource.getTexture(WEIGHTS_STREAM);
    if (!skins || !weights) return;
    const skinOut = skins.lock() as Uint32Array;
    const wordOut = weights.lock() as Uint32Array;
    skinOut.fill(0);
    wordOut.fill(0);
    for (let i = 0; i < tile.order.length; i += 1) {
      const from = tile.order[i] ?? 0;
      skinOut[i] = found.skins[from] ?? 0;
      for (let k = 0; k < 4; k += 1) wordOut[i * 4 + k] = found.words[from * 4 + k] ?? 0;
    }
    skins.unlock();
    weights.unlock();
    tile.skinned = true;
  };

  /** Sets every uniform the modifier reads; setting any marks the tile for a new copy. */
  const setUniforms = (component: pc.GSplatComponent): void => {
    component.setParameter("uInstanceState", stateTexture ?? stand().state);
    component.setParameter("uInstanceTint", tint);
    component.setParameter("uInstanceDim", dim);
    if (moves) {
      component.setParameter("uSkinHandles", handlesTexture ?? stand().float);
      component.setParameter("uRigidSlots", slotsTexture ?? stand().uint);
      component.setParameter("uRigidPoses", posesTexture ?? stand().float);
      component.setParameter("uMotionExtra", motionExtra);
      component.setParameter("uMotionParams", motionParams);
    }
    // Last: setting a parameter marks the tile for a new copy into the work buffer.
    component.setParameter("uInstanceParams", params);
  };

  /** Puts the modifier the tile's streams call for on it, with the current uniforms. */
  const applyModifier = (entity: pc.Entity, tile: TileBinding): void => {
    const component = entity.gsplat;
    if (!component) return;
    const hasIds = tile.doc !== null;
    if (!hasIds && !tile.skinned) return;
    const modifier = playcanvasModifier(hasIds, tile.skinned);
    if (modifier !== tile.modifier) {
      tile.modifier = modifier;
      component.setWorkBufferModifier(modifier);
    }
    setUniforms(component);
  };

  /**
   * Writes what the tile needs for the current style and motion and puts the modifier on. The
   * skin streams -- a pass over every splat of the tile -- are written within the frame's
   * budget (`hooks.work`): a new skin document (the wind switched on) rebinds every tile. A
   * tile being built is already a job of that budget (`inWork`), and writes them at once.
   */
  const bind = (entity: pc.Entity, tile: TileBinding, inWork = false): void => {
    if (!entity.gsplat) return;
    bindIds(tile);
    const skinDoc = motion?.skin ?? null;
    if (skinDoc && tile.skinDoc !== skinDoc) {
      // Claimed now, so a motion handed meanwhile does not queue it again.
      tile.skinDoc = skinDoc;
      if (inWork) {
        writeSkin(tile, skinDoc);
      } else {
        void hooks.work
          .run(() => {
            if (tiles.get(entity) !== tile || tile.skinDoc !== skinDoc) return;
            writeSkin(tile, skinDoc);
            applyModifier(entity, tile);
            hooks.frameWanted();
          })
          .catch(() => undefined);
      }
    }
    applyModifier(entity, tile);
  };

  const setInstances = (next: InstanceStyle | null): void => {
    style = next;
    if (next) {
      if (stateTexture?.height !== next.rows) {
        stateTexture?.destroy();
        stateTexture = dataTexture(
          "instanceState",
          INSTANCE_TEXTURE_WIDTH,
          next.rows,
          pc.PIXELFORMAT_RGBA8,
        );
      }
      (stateTexture.lock() as Uint8Array).set(next.state);
      stateTexture.unlock();
      params.set(next.params);
      tint.set(next.tint);
      dim.set(next.dim);
    } else {
      params[0] = 0;
    }
    for (const [entity, tile] of tiles) bind(entity, tile);
  };

  /** Uploads `data` into `texture`, or into a new texture when its size changed. */
  const table = (
    texture: pc.Texture | null,
    name: string,
    rows: number,
    format: number,
    data: Float32Array | Uint32Array,
  ): { texture: pc.Texture; replaced: boolean } => {
    if (texture?.height === rows) {
      (texture.lock() as typeof data).set(data);
      texture.unlock();
      return { texture, replaced: false };
    }
    texture?.destroy();
    return {
      texture: dataTexture(name, MOTION_TEXTURE_WIDTH, rows, format, data),
      replaced: true,
    };
  };

  const setMotion = (next: ScanMotion | null): void => {
    const before = motion;
    motion = next;
    let replaced = false;
    if (next) {
      const handles = table(
        handlesTexture,
        "hexapodSkinHandles",
        next.handleRows,
        pc.PIXELFORMAT_RGBA32F,
        next.handles,
      );
      const slots = table(
        slotsTexture,
        "hexapodRigidSlots",
        next.slotRows,
        pc.PIXELFORMAT_RGBA32U,
        next.slots,
      );
      const poses = table(
        posesTexture,
        "hexapodRigidPoses",
        next.poseRows,
        pc.PIXELFORMAT_RGBA32F,
        next.poses,
      );
      handlesTexture = handles.texture;
      slotsTexture = slots.texture;
      posesTexture = poses.texture;
      replaced = handles.replaced || slots.replaced || poses.replaced;
      motionParams.set(next.params);
      motionExtra.set(next.extra);
    } else {
      motionParams.fill(0);
    }
    const covarianceChanged = (before?.extra[1] ?? 1) !== (next?.extra[1] ?? 1);
    for (const [entity, tile] of tiles) {
      // A new table, a new document, or motion switched on or off: every tile binds again.
      const rebind =
        replaced ||
        next === null ||
        before === null ||
        (next.skin !== null && tile.skinDoc !== next.skin) ||
        (next.instances !== null && tile.doc === null);
      const touched =
        rebind ||
        covarianceChanged ||
        [...next.changedSkins].some((id) => tile.skinSet.has(id)) ||
        [...next.changedIds].some((id) => tile.idSet.has(id));
      if (!touched) continue;
      if (rebind) bind(entity, tile);
      else if (tile.modifier !== null) entity.gsplat?.setParameter("uMotionParams", motionParams);
      if (tile.modifier !== null) redrawn += 1;
      if (tile.ids) sortDirty.add(entity);
    }
  };

  /** Moves the centres of tiles holding rigidly moved objects, for the sorter. */
  const refreshSort = (now: number): void => {
    if (sortDirty.size === 0) return;
    const resting = motion === null || motion.rigidByLeaf.size === 0;
    // Held back while it moves: `frameDueBy` has the host draw the frame it is due in.
    if (!resting && now - lastSortAt < SORT_REFRESH_MS) return;
    lastSortAt = now;
    for (const entity of sortDirty) {
      const tile = tiles.get(entity);
      const resource = tile?.resource;
      if (!tile?.ids || !resource?.centers) continue;
      if (tile.restCenters === null) {
        if (resting) continue;
        tile.restCenters = Float32Array.from(resource.centers);
      }
      const moved = movedCenters(
        tile.restCenters,
        tile.origin,
        tile.ids,
        motion?.rigidByLeaf ?? new Map<number, Float64Array>(),
        resource.centers,
      );
      if (!moved) tile.restCenters = null;
      resource.centersVersion += 1;
    }
    sortDirty.clear();
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
  /** Scratch for `place` (split objects only). */
  let placing: { matrix: pc.Mat4; position: pc.Vec3; rotation: pc.Quat } | null = null;
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
    const resource = new pc.GSplatResource(device, data) as Resource;
    const entity = new pc.Entity(tile.uri);
    const origin = decoded.origin ?? [0, 0, 0];
    entity.setLocalPosition(...origin);
    origins.set(entity, origin);
    entity.addComponent("gsplat", { resource });
    resources.set(entity, resource);
    if (decoded.checksum !== undefined) {
      const binding: TileBinding = {
        checksum: decoded.checksum,
        order,
        resource,
        columns: properties,
        origin,
        doc: null,
        matched: false,
        ids: null,
        idSet: new Set(),
        skinDoc: null,
        skinned: false,
        skinSet: new Set(),
        modifier: null,
        restCenters: null,
        pick: undefined,
        placement: null,
        placedPick: null,
      };
      tiles.set(entity, binding);
      if (style || motion) {
        bind(entity, binding, true);
        if (binding.ids && motion && motion.rigidByLeaf.size > 0) sortDirty.add(entity);
      }
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
      shown.add(entity);
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
      shown.delete(entity);
      if (entity.parent) entity.parent.removeChild(entity);
    },
    dispose: (entity) => {
      unconfirmed.delete(entity);
      shown.delete(entity);
      tiles.delete(entity);
      sortDirty.delete(entity);
      const resource = resources.get(entity);
      entity.destroy();
      if (resource) {
        doomed.push({ resource, at: framesDrawn });
        // The renders that let it go (DESTROY_AFTER_FRAMES): asked for, since a dispose at
        // rest has none coming. Within a frame the host takes it as one more frame.
        hooks.frameWanted();
      }
    },
    place: (entity, matrix) => {
      const origin = origins.get(entity) ?? [0, 0, 0];
      const tile = tiles.get(entity);
      if (matrix === null) {
        if (tile) {
          tile.placement = null;
          tile.placedPick = null;
        }
        entity.setLocalPosition(...origin);
        entity.setLocalRotation(pc.Quat.IDENTITY);
        return;
      }
      placing ??= { matrix: new pc.Mat4(), position: new pc.Vec3(), rotation: new pc.Quat() };
      const { matrix: placement, position, rotation } = placing;
      placement.set(Array.from(matrix));
      // Only the matrix is kept: a pick moves the tile's splats there when it asks.
      if (tile) {
        (tile.placement ??= new pc.Mat4()).copy(placement);
        tile.placedPick = null;
      }
      placement.transformPoint(position.set(...origin), position);
      rotation.setFromMat4(placement);
      entity.setLocalPosition(position);
      entity.setLocalRotation(rotation);
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
        device.maxPixelRatio = pose.pixelRatio;
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
      refreshSort(performance.now());
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
    // A moving object's re-sort held back by SORT_REFRESH_MS: due then, whether or not
    // anything else changes by that time.
    frameDueBy: () => (sortDirty.size > 0 && !lost ? lastSortAt + SORT_REFRESH_MS : null),
    setBudget: (drawn) => {
      app.scene.gsplat.splatBudget = drawn;
    },
    setInstances,
    pickTiles: () => {
      const out: PickTile[] = [];
      for (const entity of shown) {
        const tile = tiles.get(entity);
        if (!tile) continue;
        tile.pick ??= pickTileOf(tile.columns, tile.order, tile.origin, tile.checksum);
        if (!tile.pick) continue;
        if (tile.placement) {
          tile.placedPick ??= placedPickTile(tile.pick, tile.placement);
          out.push(tile.placedPick);
        } else {
          out.push(tile.pick);
        }
      }
      return out;
    },
    instanceTiles: () => {
      let matched = 0;
      for (const tile of tiles.values()) if (tile.matched) matched += 1;
      return { tiles: tiles.size, matched };
    },
    ...(moves
      ? {
          setMotion,
          motionTiles: () => {
            let skinned = 0;
            for (const tile of tiles.values()) {
              if (tile.skinned && tile.skinSet.size > 0) skinned += 1;
            }
            return { skinned, redrawn };
          },
        }
      : {}),
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
      shown.clear();
      sortDirty.clear();
      for (const texture of [
        stateTexture,
        handlesTexture,
        slotsTexture,
        posesTexture,
        empty?.state,
        empty?.float,
        empty?.uint,
      ]) {
        texture?.destroy();
      }
      stateTexture = null;
      handlesTexture = null;
      slotsTexture = null;
      posesTexture = null;
      empty = null;
      // PlayCanvas only lets go of its context (`gl = null`), and the browser frees a
      // context's memory whenever it collects the canvas -- with a new canvas and context per
      // session, a few sessions in a visit held several. Losing it frees it now, once
      // PlayCanvas has taken its own handlers off the canvas. A WebGPU device is destroyed by
      // PlayCanvas itself (`GPUDevice.destroy`), which frees its memory at once.
      destroying = true;
      const gl = (device as unknown as { gl?: WebGLRenderingContext | null }).gl;
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
