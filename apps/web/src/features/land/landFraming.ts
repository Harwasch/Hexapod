import { Cartesian3, Cartographic } from "cesium";

import type { Footprint } from "@twin/contracts";
import { boundsOf, outerRings } from "@twin/geo";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import {
  panelInsets,
  pointsInOpenView,
  rectangleInView,
  topDownDistance,
  type ScreenRect,
  type ViewInsets,
} from "@/lib/viewFraming";

/**
 * Framing a land beside the Land panel, wherever the panel is: beside it on a desktop, above it
 * on a phone, where the panel is a sheet over the lower half of the map (`lib/viewFraming.ts`
 * does the arithmetic). Used when a saved land is opened (from the panel's list or the command
 * box), by the panel's "Show boundary on map", and when a boundary goes to review and the sheet
 * would cover it.
 */

/** At most this many boundary points are projected to test what is on screen. */
const SAMPLE_POINTS = 64;

function rectOf(element: Element | null): ScreenRect | null {
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 ? rect : null;
}

/** The map's size, and what the Land panel and the HUD's top row cover of it now. */
function openView(scene: CesiumSceneManager): {
  view: { width: number; height: number };
  insets: ViewInsets;
} | null {
  const canvas = rectOf(scene.viewer.canvas);
  if (!canvas) return null;
  const panel = rectOf(document.querySelector('[data-testid="land-panel"]'));
  const topBar = rectOf(document.querySelector(".hud-top"))?.bottom ?? canvas.top;
  return {
    view: { width: canvas.right - canvas.left, height: canvas.bottom - canvas.top },
    insets: panelInsets(panel, canvas, topBar),
  };
}

/** Heights outside this band are a coarse tile or a miss, not the ground (CameraController). */
const PLAUSIBLE_M: readonly [number, number] = [-500, 9000];

/** Above this the world under a land is drawn too coarsely to read the ground from. */
const SAMPLE_BELOW_M = 50_000;

function plausible(height: number | undefined): height is number {
  return (
    height !== undefined &&
    Number.isFinite(height) &&
    height >= PLAUSIBLE_M[0] &&
    height <= PLAUSIBLE_M[1]
  );
}

/**
 * The ground's height at a point as the map has it drawn now: the 3D world or the terrain under
 * it. Undefined where neither is loaded, or the camera is too high for either to be detailed
 * there (from orbit a sample of the 3D world's coarsest tiles came back 40 km underground).
 */
function groundNow(scene: CesiumSceneManager, longitude: number, latitude: number) {
  if (scene.viewer.camera.positionCartographic.height > SAMPLE_BELOW_M) return undefined;
  const at = Cartographic.fromDegrees(longitude, latitude);
  const sampled = scene.scene.sampleHeightSupported ? scene.scene.sampleHeight(at) : undefined;
  if (plausible(sampled)) return sampled;
  const terrain = scene.scene.globe.getHeight(at);
  return plausible(terrain) ? terrain : undefined;
}

/** Vertical field of view of the camera, radians (a perspective camera's; 60 degrees else). */
function fovy(scene: CesiumSceneManager): number {
  const frustum = scene.viewer.camera.frustum as { fovy?: number };
  return typeof frustum.fovy === "number" && frustum.fovy > 0 ? frustum.fovy : Math.PI / 3;
}

/**
 * Flies the camera to the boundary, looking straight down, framed in the part of the map the
 * panel leaves open. Its height is measured from the ground under the land (the world drawn
 * there now, else the terrain), so hills do not push the land under the panel or the top row.
 */
export function frameLand(scene: CesiumSceneManager | null, boundary: Footprint): void {
  if (!scene) return;
  const bounds = boundsOf(boundary);
  const open = openView(scene);
  const view = open?.view ?? { width: 1, height: 1 };
  const r = open ? rectangleInView(bounds, view, open.insets) : bounds;
  const longitude = (r.west + r.east) / 2;
  const latitude = (r.south + r.north) / 2;
  const fly = (ground: number) =>
    scene.camera.flyTo(longitude, latitude, ground + topDownDistance(r, view, fovy(scene)), {
      heading: 0,
      pitch: -90,
    });
  const centre = [(bounds.west + bounds.east) / 2, (bounds.south + bounds.north) / 2] as const;
  const ground = groundNow(scene, ...centre);
  if (ground !== undefined) {
    fly(ground);
    return;
  }
  // Somewhere not loaded yet (a land opened from the command box): ask the terrain.
  void scene.camera.groundHeightAt(...centre).then((height) => {
    if (!scene.isDestroyed) fly(height ?? 0);
  });
}

/**
 * Whether any of the boundary is hidden: off screen, behind the globe, or under the Land panel
 * or the top row. Tests a sample of its outer rings' vertices, at the ground's height.
 */
export function landHidden(scene: CesiumSceneManager, boundary: Footprint): boolean {
  const open = openView(scene);
  if (!open) return false;
  const vertices = outerRings(boundary).flat();
  // One height for the whole land, sampled once: a pick pass per vertex is too slow to repeat.
  const b = boundsOf(boundary);
  const height = groundNow(scene, (b.west + b.east) / 2, (b.south + b.north) / 2) ?? 0;
  const step = Math.max(1, Math.ceil(vertices.length / SAMPLE_POINTS));
  const points = vertices
    .filter((_, i) => i % step === 0)
    .map(([lon = 0, lat = 0]) => {
      const at = scene.scene.cartesianToCanvasCoordinates(Cartesian3.fromDegrees(lon, lat, height));
      return at ? { x: at.x, y: at.y } : null;
    });
  return !pointsInOpenView(points, open.view, open.insets);
}

/** Runs `fn` two frames on, once a panel that just opened or changed has laid out. */
function afterLayout(fn: () => void): () => void {
  let frame = requestAnimationFrame(() => {
    frame = requestAnimationFrame(fn);
  });
  return () => cancelAnimationFrame(frame);
}

/**
 * Frames a boundary that has just gone to review when the panel (on a phone, the sheet) or the
 * screen's edge hides any of it. A boundary already in view is left where the person drew it.
 */
export function revealLand(scene: CesiumSceneManager | null, boundary: Footprint): () => void {
  if (!scene) return () => undefined;
  return afterLayout(() => {
    if (!scene.isDestroyed && landHidden(scene, boundary)) frameLand(scene, boundary);
  });
}

/** Frames a land once the Land panel opening with it has laid out (the command box). */
export function frameLandSoon(scene: CesiumSceneManager | null, boundary: Footprint): void {
  if (!scene) return;
  afterLayout(() => {
    if (!scene.isDestroyed) frameLand(scene, boundary);
  });
}
