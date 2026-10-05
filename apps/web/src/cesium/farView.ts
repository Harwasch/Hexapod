/**
 * Whether a splat scan is drawn while the camera is too far out for its site to be engaged
 * (SiteManager.shouldEngage): small, as it is on the map, rather than not at all.
 *
 * Engagement is about working with a model -- it takes over from the world, its objects can be
 * selected, it is walked on -- and it hands back within a few footprint radii: about 100 m up
 * for a scan the size of a house, which is where the scan used to vanish. Seen from further out
 * it is still something on the map. It is drawn while it is at least `FAR_SHOW_PX` across on
 * screen, until it is under `FAR_HIDE_PX` (hysteresis, so it does not flicker at the
 * threshold), never from further than `FAR_MAX_DISTANCE_M`, and not while the globe hides it.
 * A dedicated renderer then draws it with a quarter of its budget and nothing culled for size
 * (scanView/ScanRendererHost.ts, scanView/quality.ts); CesiumJS by its own level of detail.
 *
 * Pure: everything it needs is passed in, so it is tested without a viewer.
 */

import { Cartesian3, Cartographic, Ellipsoid } from "cesium";

/** A scan seen from afar is drawn once it is at least this many CSS pixels across... */
export const FAR_SHOW_PX = 10;
/** ...and stays drawn until it is under this many. */
export const FAR_HIDE_PX = 6;
/** Never drawn from further than this (m, camera to the scan's centre)... */
export const FAR_MAX_DISTANCE_M = 25_000;
/** ...and drawn again only once within this, so a camera at the limit does not flicker it. */
export const FAR_SHOW_DISTANCE_M = 22_500;
/**
 * The globe the horizon is tested against is the ellipsoid lowered to just under the camera
 * and the scan (m): a scan in a valley below the ellipsoid is not behind it, nor is one seen
 * from there. A couple of metres more, as the lowered ellipsoid is not quite a parallel surface.
 */
const HORIZON_SLACK_M = 2;

/** What `farVisible` is decided from: where the camera and the scan are, and the view. */
export interface FarView {
  /** The camera, Earth-fixed (m). */
  readonly camera: Cartesian3;
  /**
   * The scan's bounding sphere, Earth-fixed: the tileset's own, not the footprint radius
   * engagement measures in (which is never under 30 m), so a 6 m tree is as small as it is.
   */
  readonly center: Cartesian3;
  readonly radiusM: number;
  /** The camera's vertical field of view (radians). */
  readonly fovy: number;
  /** The canvas's height (CSS pixels). */
  readonly heightPx: number;
}

/**
 * How many CSS pixels across the scan's bounding sphere is drawn, from the camera: its angular
 * size against the vertical field of view. Infinite with the camera inside the sphere.
 */
export function projectedDiameterPx(view: FarView): number {
  const { radiusM, fovy, heightPx } = view;
  if (!(radiusM > 0 && fovy > 0 && heightPx > 0)) return 0;
  const distance = Cartesian3.distance(view.camera, view.center);
  if (distance <= radiusM) return Number.POSITIVE_INFINITY;
  // The tangent of the sphere's angular radius: exact, and radius over distance once small.
  const tangent = radiusM / Math.sqrt(distance * distance - radiusM * radiusM);
  return (tangent / Math.tan(fovy / 2)) * heightPx;
}

/** Height above the WGS84 ellipsoid (m); 0 for the Earth's centre, which has none. */
function heightOf(point: Cartesian3): number {
  return Cartographic.fromCartesian(point)?.height ?? 0;
}

/**
 * Whether the globe hides `point` from `camera`: the horizon test CesiumJS culls with
 * (`EllipsoidalOccluder`), in the scaled space where the ellipsoid is the unit sphere, against
 * the ellipsoid lowered to just under the lower of the two (`HORIZON_SLACK_M`). Terrain is not
 * tested: a hill between the camera and the scan hides CesiumJS's own splats (they are
 * depth-tested against the globe), but not a dedicated renderer's, drawn over the globe.
 */
export function hiddenByGlobe(camera: Cartesian3, point: Cartesian3): boolean {
  const lowered = Math.max(0, -Math.min(heightOf(camera), heightOf(point))) + HORIZON_SLACK_M;
  const { x: radiusX, y: radiusY, z: radiusZ } = Ellipsoid.WGS84.radii;
  const rx = radiusX - lowered;
  const ry = radiusY - lowered;
  const rz = radiusZ - lowered;
  const cx = camera.x / rx;
  const cy = camera.y / ry;
  const cz = camera.z / rz;
  const tx = point.x / rx - cx;
  const ty = point.y / ry - cy;
  const tz = point.z / rz - cz;
  // Squared distance from the camera to the limb, and how far the point is along the camera's
  // way down (both in scaled space).
  const limb = cx * cx + cy * cy + cz * cz - 1;
  const down = -(tx * cx + ty * cy + tz * cz);
  // A camera under the surface sees nothing below its own horizontal plane.
  if (limb < 0) return down > 0;
  return down > limb && (down * down) / (tx * tx + ty * ty + tz * tz) > limb;
}

/**
 * Whether a scan is drawn from afar now, given whether it is drawn already (`current`): within
 * the distance limit, big enough on screen, and not behind the globe.
 */
export function farVisible(view: FarView, current: boolean): boolean {
  const distance = Cartesian3.distance(view.camera, view.center);
  // Also false for NaN: a camera or a scan with no position is not drawn from afar.
  if (!(distance <= (current ? FAR_MAX_DISTANCE_M : FAR_SHOW_DISTANCE_M))) return false;
  if (projectedDiameterPx(view) < (current ? FAR_HIDE_PX : FAR_SHOW_PX)) return false;
  return !hiddenByGlobe(view.camera, view.center);
}
