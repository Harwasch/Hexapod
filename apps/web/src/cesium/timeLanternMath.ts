/**
 * The Time Lantern's geometry and input, as pure functions (TimeLantern.ts uses them; tested in
 * timeLanternMath.test.ts): where the lens is in the drawing buffer, how large it is, the ground
 * window under it, how a wheel becomes steps through dates, and which record covers a point.
 */
import {
  Cartesian2,
  Cartesian3,
  IntersectionTests,
  Plane,
  Ray,
  type Cartesian3 as Cartesian3Type,
} from "cesium";

import { footprintContains } from "@twin/geo";

import type { LanternPart } from "./lanternTypes";

/** The canvas sizes that map CSS pixels to drawing-buffer pixels. */
export interface CanvasSize {
  clientWidth: number;
  clientHeight: number;
  drawingBufferWidth: number;
  drawingBufferHeight: number;
}

/**
 * A CSS point on the canvas (origin top left) in drawing-buffer pixels, GL origin bottom left:
 * what `gl_FragCoord` measures. Recomputed every frame, since the adaptive quality policy
 * changes the resolution scale (PerformanceManager) and with it the buffer's size.
 */
export function toDrawingBuffer(
  cssX: number,
  cssY: number,
  canvas: CanvasSize,
): { x: number; y: number; scale: number } {
  const sx = canvas.drawingBufferWidth / Math.max(canvas.clientWidth, 1);
  const sy = canvas.drawingBufferHeight / Math.max(canvas.clientHeight, 1);
  return { x: cssX * sx, y: canvas.drawingBufferHeight - cssY * sy, scale: sy };
}

/** The lens's outer and inner radius (CSS px): a sixth of the shorter side, 110 to 200 px. */
export function lensRadii(
  clientWidth: number,
  clientHeight: number,
): { outer: number; inner: number } {
  const outer = Math.min(200, Math.max(110, 0.16 * Math.min(clientWidth, clientHeight)));
  return { outer, inner: 0.72 * outer };
}

/** What `groundWindow` needs of a camera: a ray through a CSS point. */
export interface PickRayCamera {
  getPickRay(windowPosition: Cartesian2, result?: Ray): Ray | undefined;
}

/** A plane by a point on it and its normal (Earth-fixed). */
export interface GroundPlane {
  point: Cartesian3Type;
  normal: Cartesian3Type;
}

/** How far past the cursor's ground a window ray may land, in cursor distances. */
export const WINDOW_REACH = 5;

/**
 * The terrain's height above the window's plane at a point (m), where the terrain is known:
 * the globe's loaded tiles, which are what it draws.
 */
export type HeightAbovePlane = (point: Cartesian3Type) => number | undefined;

/** Height refinements per ray: terrain tiles come in steps of metres, three is plenty. */
const REFINE_STEPS = 3;

/**
 * The ground under a screen circle: rays through `n` points on a circle of `radiusCss * margin`
 * around the cursor, met with the plane through the cursor's ground -- and, given the terrain
 * (`heightAbove`), with the terrain itself, refined a few times along each ray, so the window's
 * edge lands on the screen circle over hills as well as on flat ground. A ray that misses the
 * ground (above the horizon) or meets it beyond WINDOW_REACH cursor distances is stopped at that
 * reach, so a lens near the horizon gets a long window, never an endless one. Null when the
 * cursor's own ray misses the plane: the lens is over the sky.
 */
export function groundWindow(
  camera: PickRayCamera,
  cursor: { x: number; y: number },
  radiusCss: number,
  plane: GroundPlane,
  n = 32,
  margin = 1.35,
  heightAbove?: HeightAbovePlane,
): Cartesian3Type[] | null {
  const ground = Plane.fromPointNormal(plane.point, plane.normal);
  const window = new Cartesian2(cursor.x, cursor.y);
  const centreRay = camera.getPickRay(window, new Ray());
  if (!centreRay) return null;
  const centre = IntersectionTests.rayPlane(centreRay, ground, new Cartesian3());
  if (!centre) return null;
  const reach = WINDOW_REACH * Cartesian3.distance(centreRay.origin, centre);
  const ring: Cartesian3Type[] = [];
  const ray = new Ray();
  const radius = radiusCss * margin;
  const lifted = new Cartesian3();
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * Math.PI * 2;
    window.x = cursor.x + radius * Math.cos(angle);
    window.y = cursor.y + radius * Math.sin(angle);
    if (!camera.getPickRay(window, ray)) continue;
    let hit = IntersectionTests.rayPlane(ray, ground, new Cartesian3());
    let offset = 0;
    for (let step = 0; hit && heightAbove && step < REFINE_STEPS; step++) {
      const height = heightAbove(hit);
      if (height === undefined || Math.abs(height - offset) < 0.5) break;
      offset = height;
      Cartesian3.add(
        plane.point,
        Cartesian3.multiplyByScalar(plane.normal, offset, lifted),
        lifted,
      );
      hit = IntersectionTests.rayPlane(
        ray,
        Plane.fromPointNormal(lifted, plane.normal),
        new Cartesian3(),
      );
    }
    if (hit && Cartesian3.distance(ray.origin, hit) <= reach) ring.push(hit);
    else ring.push(Ray.getPoint(ray, reach, new Cartesian3()));
  }
  return ring.length >= 3 ? ring : null;
}

/** The wheel's running total between steps, and when it last stepped (ms). */
export interface WheelState {
  accumulated: number;
  lastStepAt: number;
}

export const WHEEL_IDLE: WheelState = { accumulated: 0, lastStepAt: Number.NEGATIVE_INFINITY };
/** A trackpad's pixels per step through time. */
export const WHEEL_PIXELS_PER_STEP = 60;
/** Steps at most this often (ms): one notch, one date, however the wheel bounces. */
export const WHEEL_COOLDOWN_MS = 120;
/** A pixel delta this large in one event is a mouse notch, not a trackpad's drift. */
const NOTCH_PIXELS = 50;

/**
 * A wheel event as steps through dates: +1 is older (wheel down, `deltaY > 0`), -1 newer. A
 * mouse notch is one step (lines or pages, or a pixel delta as large as a notch); a trackpad's
 * pixels add up, a step per 60. Never more than one step per WHEEL_COOLDOWN_MS, and the total
 * waiting meanwhile is held at one step, so a fling does not run on through every year.
 */
export function wheelSteps(
  state: WheelState,
  deltaY: number,
  deltaMode: number,
  now: number,
): { steps: number; state: WheelState } {
  if (deltaY === 0) return { steps: 0, state };
  const notch = deltaMode !== 0 || Math.abs(deltaY) >= NOTCH_PIXELS;
  let accumulated = notch
    ? Math.sign(deltaY) * WHEEL_PIXELS_PER_STEP
    : Math.sign(state.accumulated) === -Math.sign(deltaY)
      ? deltaY
      : state.accumulated + deltaY;
  accumulated = Math.max(-WHEEL_PIXELS_PER_STEP, Math.min(WHEEL_PIXELS_PER_STEP, accumulated));
  if (Math.abs(accumulated) < WHEEL_PIXELS_PER_STEP || now - state.lastStepAt < WHEEL_COOLDOWN_MS)
    return { steps: 0, state: { accumulated, lastStepAt: state.lastStepAt } };
  return { steps: Math.sign(accumulated), state: { accumulated: 0, lastStepAt: now } };
}

/** The record of `parts` that covers a point, or null where the source has none. */
export function coverageAt(
  parts: readonly LanternPart[],
  longitude: number,
  latitude: number,
): LanternPart | null {
  for (const part of parts)
    if (footprintContains(part.footprint, { longitude, latitude })) return part;
  return null;
}
