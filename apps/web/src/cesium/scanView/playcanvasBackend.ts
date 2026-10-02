/**
 * The PlayCanvas back-end of the scan renderer (ScanRendererHost): SuperSplat's own engine
 * (MIT). Each tile is decoded in a worker into the columns PlayCanvas's `GSplatData` takes
 * (playcanvasTile.worker.ts), made a `GSplatResource`, and drawn by PlayCanvas's unified
 * splat renderer, which sorts every tile's splats together. PlayCanvas has no level of detail
 * over resources given this way, so it draws what it is given: streamed at the budget itself.
 */

import * as pc from "playcanvas";

import type { TileNode } from "@/view/tiles";

import type { ScanBackend, ScanPose } from "./types";

interface Decoded {
  id: number;
  count?: number;
  properties?: Record<string, Float32Array>;
  /** The tile's centre: the positions are relative to it (playcanvasTile.worker.ts). */
  origin?: [number, number, number];
  error?: string;
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
      data.reorderData();
      const resource = new pc.GSplatResource(app.graphicsDevice, data);
      const entity = new pc.Entity(tile.uri);
      if (decoded.origin) entity.setLocalPosition(...decoded.origin);
      entity.addComponent("gsplat", { resource });
      resources.set(entity, resource);
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
