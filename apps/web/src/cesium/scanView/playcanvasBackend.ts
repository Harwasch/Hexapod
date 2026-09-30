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
  error?: string;
}

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
  const camera = new pc.Entity("scan-camera");
  camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0, 0) });
  app.root.addChild(camera);
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
  const target = new pc.Vec3();
  const up = new pc.Vec3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  return Promise.resolve({
    name: "playcanvas",
    loadFactor: 1,
    load: async (tilesetUrl: string, tile: TileNode) => {
      const decoded = await decode(new URL(tile.uri, tilesetUrl).toString());
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
      entity.addComponent("gsplat", { resource });
      resources.set(entity, resource);
      return entity;
    },
    add: (entity) => app.root.addChild(entity),
    remove: (entity) => {
      if (entity.parent) entity.parent.removeChild(entity);
    },
    dispose: (entity) => {
      const resource = resources.get(entity);
      entity.destroy();
      resource?.destroy();
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
      // PlayCanvas draws in its own loop (app.start); the camera is simply where it will look.
    },
    destroy: () => {
      for (const worker of workers) worker.terminate();
      waiting.clear();
      app.destroy();
    },
  });
}
