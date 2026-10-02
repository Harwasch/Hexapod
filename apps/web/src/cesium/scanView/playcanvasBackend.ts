/**
 * The PlayCanvas back-end of the scan renderer (ScanRendererHost): SuperSplat's own engine
 * (MIT). Each tile is decoded in a worker into the columns PlayCanvas's `GSplatData` takes
 * (playcanvasTile.worker.ts), made a `GSplatResource`, and drawn by PlayCanvas's unified
 * splat renderer, which sorts every tile's splats together. PlayCanvas has no level of detail
 * over resources given this way, so it draws what it is given: streamed at the budget itself.
 *
 * The scan's objects (scanInstances.ts): each tile's resource carries one more stream, the
 * instance id of every splat (`splatInstance`, R32U, in the resource's own Morton order), and
 * a work-buffer modifier reads it with the shared state table when PlayCanvas copies the tile
 * into the buffer it sorts and draws from. A change of what is hidden or highlighted updates
 * the table and marks every tile for a new copy: no tile is decoded or uploaded again.
 */

import * as pc from "playcanvas";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import type { TileNode } from "@/view/tiles";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import { idsInResourceOrder, SCAN_INSTANCE_RULE_GLSL, type InstanceStyle } from "./scanInstances";
import { countOverlayLoopTick } from "./stats";
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

export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
): Promise<ScanBackend<pc.Entity>> {
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
      powerPreference: "high-performance",
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
      component.setWorkBufferModifier({ glsl: PLAYCANVAS_INSTANCE_GLSL });
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
  /** Frames drawn so far, and the frame each entity was added at: PlayCanvas's unified
   *  renderer takes a new entity into its buffer and sorts it over the next frames. */
  let framesDrawn = 0;
  const addedAt = new WeakMap<pc.Entity, number>();
  const doomed: { resource: pc.GSplatResource; at: number }[] = [];
  const target = new pc.Vec3();
  const up = new pc.Vec3();
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
      const resource = resources.get(entity);
      entity.destroy();
      if (resource) doomed.push({ resource, at: framesDrawn });
    },
    render: (pose: ScanPose) => {
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        app.graphicsDevice.maxPixelRatio = pose.pixelRatio;
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
      stateTexture?.destroy();
      stateTexture = null;
      app.destroy();
    },
  };
  app.on("frameupdate", countOverlayLoopTick);
  app.on("frameend", () => {
    framesDrawn += 1;
    while (doomed[0] && framesDrawn - doomed[0].at >= DESTROY_AFTER_FRAMES) {
      doomed.shift()?.resource.destroy();
    }
  });
  return Promise.resolve(backend);
}
