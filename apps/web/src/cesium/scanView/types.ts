import type { PickTile } from "@/lib/splatPick";
import type { TileNode } from "@/view/tiles";

import type { InstanceStyle } from "./scanInstances";
import type { ScanMotion } from "./scanMotion";

/** What draws a splat scan on the globe: CesiumJS itself, or a dedicated splat renderer. */
export type SplatRendererKind = "cesium" | "spark" | "playcanvas";

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
  /**
   * The tiles drawn now, in each tile's own splat order and the scan's frame, for selecting
   * objects in the scene (cesium/sceneSelect). Absent when the renderer cannot say.
   */
  pickTiles?(): readonly PickTile[];
  /** Tiles loaded now, and how many of them carry object ids: for tests and diagnostics. */
  instanceTiles?(): { tiles: number; matched: number };
  /**
   * Moves the scan's objects as `motion` says (scanMotion.ts: skins and rigid motions the
   * shared drivers set), on every tile loaded now or later; null draws every splat at rest.
   * Absent when the renderer cannot, and the panels then offer CesiumJS's renderer.
   */
  setMotion?(motion: ScanMotion | null): void;
  /** Tiles loaded now that carry skin weights, and those redrawn for motion so far. */
  motionTiles?(): { skinned: number; redrawn: number };
  /**
   * Draws `mesh` -- a split object's tile -- under `matrix` (column-major 4x4, the scan's
   * frame; a rigid motion), or where it was decoded with null.
   */
  place?(mesh: M, matrix: readonly number[] | null): void;
  destroy(): void;
}
