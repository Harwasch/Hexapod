/**
 * Framing a place in the part of the map a panel leaves open. Pure, so the arithmetic is tested
 * without a scene (`__tests__/viewFraming.test.ts`); `features/land/landFraming.ts` measures the
 * panel and flies the camera.
 *
 * A camera flown to a rectangle fills the whole view with it, so a boundary flown to that way
 * sits under whatever panel is open: on a phone the Land sheet covers the lower half of the
 * screen, and the boundary being reviewed was under it. Instead the rectangle handed to the
 * camera is grown, and moved, so that the boundary lands in the open part of the view.
 */

import type { Bounds } from "@twin/geo";

/** CSS pixels covered at each edge of the view. */
export interface ViewInsets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A rectangle on screen, as `getBoundingClientRect` gives one. */
export interface ScreenRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export const NO_INSETS: ViewInsets = { top: 0, right: 0, bottom: 0, left: 0 };

/** Metres per degree of latitude (and of longitude at the equator), near enough for framing. */
const M_PER_DEG = 111_320;

/** A view this narrow or short in its open part is not framed into any further. */
const MIN_OPEN_SHARE = 0.1;

/**
 * Which edges a panel covers. A panel as wide as most of the view is a bottom sheet (a phone,
 * or any narrow window, where the HUD's docks become one sheet): it covers the bottom, from its
 * top edge down. Anything narrower is a side panel and covers the side it is on, up to its far
 * edge. `topBar` is where the HUD's top row ends (the site badge and the command box), which
 * nothing should be framed under.
 */
export function panelInsets(panel: ScreenRect | null, view: ScreenRect, topBar = 0): ViewInsets {
  const top = Math.max(0, topBar - view.top);
  if (!panel) return { ...NO_INSETS, top };
  const sheet = panel.right - panel.left > (view.right - view.left) * 0.6;
  if (sheet) return { ...NO_INSETS, top, bottom: Math.max(0, view.bottom - panel.top) };
  const onLeft = (panel.left + panel.right) / 2 < (view.left + view.right) / 2;
  return onLeft
    ? { ...NO_INSETS, top, left: Math.max(0, panel.right - view.left) }
    : { ...NO_INSETS, top, right: Math.max(0, view.right - panel.left) };
}

/**
 * The rectangle to fly the camera to so that `bounds`, with `margin` of its size around it,
 * fills the part of a `width` x `height` view the insets leave open, centred there. Looking
 * straight down, which is how a camera flies to a rectangle; a local flat approximation, which
 * is all framing needs.
 */
export function rectangleInView(
  bounds: Bounds,
  view: { width: number; height: number },
  insets: ViewInsets,
  margin = 0.15,
): Bounds {
  const lat0 = (bounds.south + bounds.north) / 2;
  const mPerDegLon = M_PER_DEG * Math.max(0.01, Math.cos((lat0 * Math.PI) / 180));
  const widthM = Math.max(1, (bounds.east - bounds.west) * mPerDegLon);
  const heightM = Math.max(1, (bounds.north - bounds.south) * M_PER_DEG);
  const openW = Math.max(view.width * MIN_OPEN_SHARE, view.width - insets.left - insets.right);
  const openH = Math.max(view.height * MIN_OPEN_SHARE, view.height - insets.top - insets.bottom);
  // Metres per CSS pixel at which the bounds and their margin fit the open part.
  const scale = Math.max((widthM * (1 + 2 * margin)) / openW, (heightM * (1 + 2 * margin)) / openH);
  // Where the open part's centre is from the view's centre, in pixels (x right, y down). The
  // camera looks that far the other way from the bounds' centre.
  const dx = Math.min(insets.left, view.width - openW) + openW / 2 - view.width / 2;
  const dy = Math.min(insets.top, view.height - openH) + openH / 2 - view.height / 2;
  const lon = (bounds.west + bounds.east) / 2 - (dx * scale) / mPerDegLon;
  const lat = lat0 + (dy * scale) / M_PER_DEG;
  const halfLon = (view.width * scale) / 2 / mPerDegLon;
  const halfLat = (view.height * scale) / 2 / M_PER_DEG;
  return {
    west: Math.max(-180, lon - halfLon),
    south: Math.max(-90, lat - halfLat),
    east: Math.min(180, lon + halfLon),
    north: Math.min(90, lat + halfLat),
  };
}

/**
 * How far above the ground a camera looking straight down must be for `r` (sized to the view's
 * aspect by `rectangleInView`) to fill a `width` x `height` view, given its vertical field of
 * view in radians. Measured from the ground, not the ellipsoid: a ranch 300 m up flown to as a
 * rectangle on the ellipsoid comes out a fifth too close, and its top slid under the top row.
 */
export function topDownDistance(
  r: Bounds,
  view: { width: number; height: number },
  fovy: number,
): number {
  const lat0 = (r.south + r.north) / 2;
  const mPerDegLon = M_PER_DEG * Math.max(0.01, Math.cos((lat0 * Math.PI) / 180));
  const halfH = ((r.north - r.south) / 2) * M_PER_DEG;
  const halfW = ((r.east - r.west) / 2) * mPerDegLon;
  const tanY = Math.tan(fovy / 2);
  const tanX = tanY * (view.width / Math.max(1, view.height));
  return Math.max(halfH / tanY, halfW / tanX, 1);
}

/**
 * Whether every point is on screen and clear of the insets. A point the camera cannot see
 * (behind the globe) is passed as null and counts as hidden.
 */
export function pointsInOpenView(
  points: readonly ({ x: number; y: number } | null)[],
  view: { width: number; height: number },
  insets: ViewInsets,
): boolean {
  return points.every(
    (p) =>
      p !== null &&
      p.x >= insets.left &&
      p.x <= view.width - insets.right &&
      p.y >= insets.top &&
      p.y <= view.height - insets.bottom,
  );
}
