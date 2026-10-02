/**
 * The PlayCanvas back-end of the scan renderer (ScanRendererHost): SuperSplat's own engine
 * (MIT). Each tile is decoded in a worker into the columns PlayCanvas's `GSplatData` takes
 * (playcanvasTile.worker.ts), made a `GSplatResource`, and drawn by PlayCanvas's unified
 * splat renderer, which sorts every tile's splats together. PlayCanvas has no level of detail
 * over resources given this way, so it draws what it is given: streamed at the budget itself.
 *
 * Per-splat data rides as extra resource streams, in the resource's own Morton order, bound to
 * each tile by the checksum of its positions (the worker digests them as it decodes):
 *
 * - `splatInstance` (R32U): the instance id (scanInstances.ts), for hide and highlight and for
 *   rigid motion;
 * - `splatSkin` (R32U) and `splatWeights` (RGBA32U): the skin id and the `skin.bin` row
 *   (scanMotion.ts), only on tiles `skin.json` lists with skinned splats.
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
 * `SORT_REFRESH_MS` while it moves and at once when it comes to rest; a skin's sway is sorted
 * at rest, as CesiumJS sorts it. A split object is its own entity, sorted where it is placed.
 */

import * as pc from "playcanvas";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { tileSkin, type SkinDoc } from "@/lib/skin";
import type { TileNode } from "@/view/tiles";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import { idsInResourceOrder, SCAN_INSTANCE_RULE_GLSL, type InstanceStyle } from "./scanInstances";
import {
  MOTION_TEXTURE_WIDTH,
  movedCenters,
  SCAN_MOTION_GLSL,
  type ScanMotion,
} from "./scanMotion";
import type { ScanBackend, ScanPose } from "./types";

interface Decoded {
  id: number;
  count?: number;
  properties?: Record<string, Float32Array>;
  /** The tile's centre: the positions are relative to it (playcanvasTile.worker.ts). */
  origin?: [number, number, number];
  /** `checksumPositions` of the tile's own positions, before centring. */
  checksum?: string;
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

/** What binds a tile to the scan's objects: its digest, where each original splat went, and
 *  what was written from which document. */
interface TileBinding {
  checksum: string;
  /** Resource slot `i` holds the tile's splat `order[i]` (GSplatData's Morton reorder). */
  order: Uint32Array;
  resource: pc.GSplatResource;
  /** Where the entity sits (the tile's centre): resource centres are relative to it. */
  origin: [number, number, number];
  /** The doc the ids were written from, whether the file lists the tile, and the ids. */
  doc: InstancesDoc | null;
  matched: boolean;
  ids: Uint32Array | null;
  idSet: ReadonlySet<number>;
  /** The skin doc the skin streams were written from, and the skins the tile holds. */
  skinDoc: SkinDoc | null;
  skinned: boolean;
  skinSet: ReadonlySet<number>;
  /** The modifier on it now. */
  glsl: string | null;
  /** The tile's own centres, kept while a motion has moved them for the sorter. */
  restCenters: Float32Array | null;
}

/** Frames a disposed tile's GPU resource outlives its entity: PlayCanvas's unified renderer
 *  drops a removed entity from its placements on its next update, and destroying the resource
 *  first left a placement with none ("Cannot read properties of null (reading 'hasCenters')"). */
const DESTROY_AFTER_FRAMES = 3;

/** A new entity counts as drawn this many frames and milliseconds after it was added: the
 *  unified renderer copies it into its work buffer and sorts in a worker first. */
const SETTLE_FRAMES = 4;
const SETTLE_MS = 120;

/** Tiles decoded at once: a quarter of the cores, one to three. */
function workerCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  return Math.min(3, Math.max(1, Math.floor(cores / 4)));
}

/** The resource members used here that the typings do not show. */
type Resource = pc.GSplatResource & {
  format: { addExtraStreams(streams: { name: string; format: number }[]): void };
  getTexture(name: string): pc.Texture | null;
  centers: Float32Array | null;
  centersVersion: number;
};

export interface BackendOptions {
  /** Keeps the drawn frame readable after it is shown (harnesses read pixels back). */
  preserveDrawingBuffer?: boolean;
}

export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  options: BackendOptions = {},
): Promise<ScanBackend<pc.Entity>> {
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: options.preserveDrawingBuffer === true,
    },
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.scene.gsplat.splatBudget = budget;
  // Full-precision work buffer rather than the compact one (quantised transforms), and splats
  // down to a pixel rather than two: the tiles are already the detail the view asked for.
  (app.scene.gsplat as unknown as { dataFormat: string }).dataFormat = "large";
  app.scene.gsplat.minPixelSize = 1;
  const camera = new pc.Entity("scan-camera");
  camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0, 0) });
  app.root.addChild(camera);
  // PlayCanvas updates in its own loop (streaming, sorting) but draws only from `render`, in
  // the same frame and from the same pose as the globe under it. Drawn in its own loop, it ran
  // before the host set the camera: a pose behind, so the scan slid on the map as the view
  // moved, and a resize cleared the canvas a frame before anything was drawn on it (a black
  // flash where the world is clipped away under the scan).
  app.autoRender = false;
  app.start();
  const device = app.graphicsDevice;

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
      worker?.postMessage({ id, url });
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
  /** Stand-ins for tables not there yet: every sampler the modifier declares is bound. */
  const empty = {
    state: dataTexture("hexapodEmptyState", 1, 1, pc.PIXELFORMAT_RGBA8, new Uint8Array(4)),
    float: dataTexture("hexapodEmptyFloat", 1, 1, pc.PIXELFORMAT_RGBA32F, new Float32Array(4)),
    uint: dataTexture("hexapodEmptyUint", 1, 1, pc.PIXELFORMAT_RGBA32U, new Uint32Array(4)),
  };

  const resources = new WeakMap<pc.Entity, pc.GSplatResource>();
  /** Tiles loaded and not yet disposed, with what binds them to the scan's objects. */
  const tiles = new Map<pc.Entity, TileBinding>();
  /** Where each entity was decoded to sit (its tile's centre), for `place`. */
  const origins = new WeakMap<pc.Entity, [number, number, number]>();
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
    const resource = tile.resource as Resource;
    resource.format.addExtraStreams([{ name: INSTANCE_STREAM, format: pc.PIXELFORMAT_R32U }]);
    const texture = resource.getTexture(INSTANCE_STREAM);
    if (!texture) return;
    const ordered = idsInResourceOrder(listed, tile.order, new Uint32Array(tile.order.length));
    (texture.lock() as Uint32Array).set(ordered);
    texture.unlock();
    tile.doc = doc;
    tile.matched = listed !== undefined;
    tile.ids = listed ? ordered : null;
    tile.idSet = new Set(ordered);
  };

  const bindSkin = (tile: TileBinding): void => {
    const doc = motion?.skin ?? null;
    if (!doc || tile.skinDoc === doc) return;
    tile.skinDoc = doc;
    const found = tileSkin(doc, tile.checksum);
    const skinSet = new Set<number>();
    const fits = found?.skins.length === tile.order.length;
    if (fits) for (const id of found.skins) if (id !== 0) skinSet.add(id);
    tile.skinSet = skinSet;
    const resource = tile.resource as Resource;
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
    component.setParameter("uInstanceState", stateTexture ?? empty.state);
    component.setParameter("uInstanceTint", tint);
    component.setParameter("uInstanceDim", dim);
    component.setParameter("uSkinHandles", handlesTexture ?? empty.float);
    component.setParameter("uRigidSlots", slotsTexture ?? empty.uint);
    component.setParameter("uRigidPoses", posesTexture ?? empty.float);
    component.setParameter("uMotionExtra", motionExtra);
    component.setParameter("uMotionParams", motionParams);
    component.setParameter("uInstanceParams", params);
  };

  /** Writes what the tile needs for the current style and motion and puts the modifier on. */
  const bind = (entity: pc.Entity, tile: TileBinding): void => {
    const component = entity.gsplat;
    if (!component) return;
    bindIds(tile);
    bindSkin(tile);
    const hasIds = tile.doc !== null;
    const glsl = hasIds || tile.skinned ? playcanvasModifierGlsl(hasIds, tile.skinned) : null;
    if (glsl === null) return;
    if (glsl !== tile.glsl) {
      tile.glsl = glsl;
      component.setWorkBufferModifier({ glsl });
    }
    setUniforms(component);
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
      else if (tile.glsl !== null) entity.gsplat?.setParameter("uMotionParams", motionParams);
      if (tile.glsl !== null) redrawn += 1;
      if (tile.ids) sortDirty.add(entity);
    }
  };

  /** Moves the centres of tiles holding rigidly moved objects, for the sorter. */
  const refreshSort = (now: number): void => {
    if (sortDirty.size === 0) return;
    const resting = motion === null || motion.rigidByLeaf.size === 0;
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

  /** Frames drawn so far, and the frame each entity was added at: PlayCanvas's unified
   *  renderer takes a new entity into its buffer and sorts it over the next frames. */
  let framesDrawn = 0;
  const addedAt = new WeakMap<pc.Entity, number>();
  const doomed: { resource: pc.GSplatResource; at: number }[] = [];
  const target = new pc.Vec3();
  const up = new pc.Vec3();
  const placement = new pc.Mat4();
  const placedPosition = new pc.Vec3();
  const placedRotation = new pc.Quat();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  const backend: ScanBackend<pc.Entity> = {
    name: "playcanvas",
    loadFactor: 1,
    load: async (tilesetUrl: string, tile: TileNode, signal?: AbortSignal) => {
      const decoded = await decode(new URL(tile.uri, tilesetUrl).toString());
      signal?.throwIfAborted();
      if (!decoded.properties || decoded.count === undefined) {
        throw new Error(decoded.error ?? "The tile could not be decoded.");
      }
      const data = new pc.GSplatData([
        {
          name: "vertex",
          count: decoded.count,
          properties: Object.entries(decoded.properties).map(([name, storage]) => ({
            type: "float",
            name,
            storage,
            byteSize: 4,
          })),
        },
      ]);
      // glTF KHR_gaussian_splatting's convention: linear scale, opacity after the sigmoid.
      data.activated = true;
      // As `reorderData`, keeping the order: the object ids are in the tile's own order.
      const order = data.calcMortonOrder();
      data.reorder(order);
      const resource = new pc.GSplatResource(device, data);
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
          origin,
          doc: null,
          matched: false,
          ids: null,
          idSet: new Set(),
          skinDoc: null,
          skinned: false,
          skinSet: new Set(),
          glsl: null,
          restCenters: null,
        };
        tiles.set(entity, binding);
        if (style || motion) {
          bind(entity, binding);
          if (binding.ids && motion && motion.rigidByLeaf.size > 0) sortDirty.add(entity);
        }
      }
      return entity;
    },
    add: (entity) => {
      app.root.addChild(entity);
      addedAt.set(entity, framesDrawn);
    },
    isDrawn: (entity, sinceMs) =>
      framesDrawn - (addedAt.get(entity) ?? framesDrawn) >= SETTLE_FRAMES && sinceMs >= SETTLE_MS,
    remove: (entity) => {
      if (entity.parent) entity.parent.removeChild(entity);
    },
    dispose: (entity) => {
      tiles.delete(entity);
      sortDirty.delete(entity);
      const resource = resources.get(entity);
      entity.destroy();
      if (resource) doomed.push({ resource, at: framesDrawn });
    },
    place: (entity, matrix) => {
      const origin = origins.get(entity) ?? [0, 0, 0];
      if (matrix === null) {
        entity.setLocalPosition(...origin);
        entity.setLocalRotation(pc.Quat.IDENTITY);
        return;
      }
      placement.set(Array.from(matrix));
      placement.transformPoint(placedPosition.set(...origin), placedPosition);
      placedRotation.setFromMat4(placement);
      entity.setLocalPosition(placedPosition);
      entity.setLocalRotation(placedRotation);
    },
    render: (pose: ScanPose) => {
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        device.maxPixelRatio = pose.pixelRatio;
        app.resizeCanvas(pose.width, pose.height);
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
      app.render();
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
    setMotion,
    motionTiles: () => {
      let skinned = 0;
      for (const tile of tiles.values()) if (tile.skinned && tile.skinSet.size > 0) skinned += 1;
      return { skinned, redrawn };
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
      for (const texture of [stateTexture, handlesTexture, slotsTexture, posesTexture]) {
        texture?.destroy();
      }
      stateTexture = null;
      handlesTexture = null;
      slotsTexture = null;
      posesTexture = null;
      app.destroy();
    },
  };
  app.on("frameend", () => {
    framesDrawn += 1;
    while (doomed[0] && framesDrawn - doomed[0].at >= DESTROY_AFTER_FRAMES) {
      doomed.shift()?.resource.destroy();
    }
  });
  return Promise.resolve(backend);
}
