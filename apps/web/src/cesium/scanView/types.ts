import type { TileNode } from "@/view/tiles";

import type { InstanceStyle } from "./scanInstances";
import type { TileWork } from "./tileWork";

/**
 * What draws a splat scan on the globe: CesiumJS itself, or a dedicated splat renderer --
 * PlayCanvas on WebGL2, PlayCanvas on WebGPU where the device has it (a trial), or Spark.
 */
export type SplatRendererKind = "cesium" | "spark" | "playcanvas" | "playcanvas-webgpu";

/** The graphics API a dedicated renderer draws with. */
export type GraphicsApi = "webgl2" | "webgpu";

/** Cesium's camera in the scan's own east/north/up metres (the frame its splats are in). */
export interface ScanPose {
  eye: [number, number, number];
  direction: [number, number, number];
  up: [number, number, number];
  /** Vertical field of view (radians). */
  fovy: number;
  near: number;
  far: number;
  /** Canvas size in CSS pixels, and the device pixels a CSS pixel renders at. */
  width: number;
  height: number;
  pixelRatio: number;
}

/** What the host hands a back-end when it creates it. */
export interface BackendHooks {
  /**
   * The renderer has something new to show that the host did not cause -- a sort finished,
   * streamed detail arrived -- and wants a frame (overlayFrames.ts). Without it the overlay
   * would only draw when the camera or the tiles change.
   */
  frameWanted(): void;
  /** Main-thread tile work, within the frame's budget (tileWork.ts). */
  work: TileWork;
  /** Spherical-harmonic bands a tile keeps (quality.ts). */
  maxShDegree: number;
  /**
   * The renderer lost its GPU device for good and draws nothing more (a WebGPU device lost
   * to a driver reset, a GPU process crash, memory pressure): the host replaces it with the
   * WebGL2 renderer and draws again. `reason` is one line for the developer readouts.
   */
  deviceLost?(reason: string): void;
}

/** A scan the renderer streams by itself (ScanBackend.streamNative). */
export interface NativeStream {
  /** Splats it draws now, when it can say. */
  splats(): number;
  stop(): void;
}

/**
 * A dedicated splat renderer drawing onto its own transparent canvas above the globe. The
 * host (ScanRendererHost) owns streaming, the camera and the canvas; a back-end only turns
 * tiles into its meshes, puts them on screen and draws a frame.
 */
export interface ScanBackend<M> {
  readonly name: SplatRendererKind;
  /** What it draws with (the developer readouts); WebGL2 when it does not say. */
  readonly api?: GraphicsApi;
  /**
   * Why it does not draw with what was asked of it, in one line, or null: a WebGPU back-end
   * that came up on WebGL2 because the browser has no WebGPU or no adapter for it.
   */
  readonly apiNote?: string | null;
  /**
   * Gaussians streamed per gaussian drawn: a renderer with its own level of detail (Spark's
   * LoD trees) is given more than it draws and picks; one without draws all it is given.
   */
  readonly loadFactor: number;
  /** Fetches and decodes `tile`; `signal` aborts a fetch the view no longer wants. */
  load(tilesetUrl: string, tile: TileNode, signal?: AbortSignal): Promise<M>;
  add(mesh: M): void;
  remove(mesh: M): void;
  dispose(mesh: M): void;
  render(pose: ScanPose): void;
  /**
   * Streams a scan packaged in the renderer's own level-of-detail format instead of the 3D
   * Tiles: the renderer then chooses, fetches, caches and sorts by itself (PlayCanvas's
   * streamed SOG, `lod-meta.json`, which superspl.at serves). Absent when it has none.
   */
  streamNative?(url: string): Promise<NativeStream>;
  /** Whether `mesh`, added `sinceMs` ago, has been drawn (scanView/handover.ts). */
  isDrawn(mesh: M, sinceMs: number): boolean;
  /** How opaque `mesh` is drawn, 0 to 1, where the renderer can fade one (handover.ts). */
  fade?(mesh: M, alpha: number): void;
  /** The most gaussians it may draw a frame (the adaptive budget moved). */
  setBudget(drawn: number): void;
  /**
   * Draws the scan's objects hidden and highlighted as `style` says (scanInstances.ts), on
   * every tile loaded now or later; null draws every splat as it was. Absent when the
   * renderer cannot, and the objects panel then offers CesiumJS's renderer.
   */
  setInstances?(style: InstanceStyle | null): void;
  /** Tiles loaded now, and how many of them carry object ids: for tests and diagnostics. */
  instanceTiles?(): { tiles: number; matched: number };
  /**
   * Lets go of everything, the GPU context included: a lost context (`WEBGL_lose_context`)
   * is what frees its memory at once rather than whenever the browser collects the canvas,
   * and every session gets a new canvas and a new context.
   */
  destroy(): void;
}
