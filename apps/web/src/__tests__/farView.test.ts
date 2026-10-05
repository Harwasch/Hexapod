/**
 * When a splat scan is drawn from afar (cesium/farView.ts): big enough on screen, within the
 * distance limit, and not behind the globe -- with hysteresis on both thresholds.
 */

import { Cartesian3, Math as CesiumMath, Matrix4, Transforms } from "cesium";
import { describe, expect, it } from "vitest";

import {
  FAR_HIDE_PX,
  FAR_MAX_DISTANCE_M,
  FAR_SHOW_DISTANCE_M,
  FAR_SHOW_PX,
  farVisible,
  hiddenByGlobe,
  projectedDiameterPx,
  type FarView,
} from "@/cesium/farView";

/** A 960 x 600 canvas under Cesium's 60 degree field of view (across the wider side). */
const HEIGHT_PX = 600;
const FOVY = 2 * Math.atan(Math.tan(CesiumMath.toRadians(30)) / 1.6);

/** A scan on the ground at the Living Survey fixture's place (Florida, near sea level). */
const CENTER = Cartesian3.fromDegrees(-82.6966, 28.0389, 4);
const ENU = Transforms.eastNorthUpToFixedFrame(CENTER);

/** A camera `north` metres north of the scan's centre and `up` metres above its plane. */
function cameraAt(north: number, up: number): Cartesian3 {
  return Matrix4.multiplyByPoint(ENU, new Cartesian3(0, north, up), new Cartesian3());
}

function view(camera: Cartesian3, radiusM: number, center = CENTER): FarView {
  return { camera, center, radiusM, fovy: FOVY, heightPx: HEIGHT_PX };
}

/** The camera distance at which a scan of `radiusM` is `px` CSS pixels across. */
function distanceFor(radiusM: number, px: number): number {
  const tangent = (px * Math.tan(FOVY / 2)) / HEIGHT_PX;
  return radiusM * Math.sqrt(1 + 1 / (tangent * tangent));
}

describe("a splat scan seen from afar", () => {
  // The synthetic tree's bounds: some 8 m in radius.
  const tree = 7.84;

  it("is as many pixels across as its bounding sphere subtends", () => {
    const camera = cameraAt(-400, 0);
    // Radius over distance, against the vertical field of view, for a small sphere.
    const simple = (tree / 400 / Math.tan(FOVY / 2)) * HEIGHT_PX;
    expect(projectedDiameterPx(view(camera, tree))).toBeCloseTo(simple, 1);
    expect(projectedDiameterPx(view(cameraAt(-5, 0), tree))).toBe(Number.POSITIVE_INFINITY);
    expect(projectedDiameterPx({ ...view(camera, tree), heightPx: 0 })).toBe(0);
  });

  it("is drawn from 400 m and not from 5 km", () => {
    // About 33 px across from 400 m, under 3 px from 5 km.
    expect(farVisible(view(cameraAt(-300, 260), tree), false)).toBe(true);
    expect(farVisible(view(cameraAt(-4000, 3000), tree), true)).toBe(false);
  });

  it("shows at 10 px and hides under 6 px, holding in between", () => {
    const show = distanceFor(tree, FAR_SHOW_PX);
    const hide = distanceFor(tree, FAR_HIDE_PX);
    const between = distanceFor(tree, (FAR_SHOW_PX + FAR_HIDE_PX) / 2);
    expect(farVisible(view(cameraAt(-(show - 1), 0), tree), false)).toBe(true);
    expect(farVisible(view(cameraAt(-(show + 1), 0), tree), false)).toBe(false);
    // Between the two thresholds it stays as it was: no flicker as the camera hovers there.
    expect(farVisible(view(cameraAt(-between, 0), tree), true)).toBe(true);
    expect(farVisible(view(cameraAt(-between, 0), tree), false)).toBe(false);
    expect(farVisible(view(cameraAt(-(hide - 1), 0), tree), true)).toBe(true);
    expect(farVisible(view(cameraAt(-(hide + 1), 0), tree), true)).toBe(false);
  });

  it("is measured on its own bounds, not on a 30 m floor", () => {
    // 3 km away the tree is some 4 px across; a 30 m sphere there would be some 17 px.
    const camera = cameraAt(-3000, 0);
    expect(farVisible(view(camera, tree), true)).toBe(false);
    expect(farVisible(view(camera, 30), true)).toBe(true);
  });

  it("is never drawn from beyond 25 km, however big", () => {
    // A 2 km campus: well over 10 px from any of these distances.
    const campus = 2000;
    expect(projectedDiameterPx(view(cameraAt(-26_000, 0), campus))).toBeGreaterThan(FAR_SHOW_PX);
    expect(farVisible(view(cameraAt(-(FAR_MAX_DISTANCE_M + 500), 0), campus), true)).toBe(false);
    expect(farVisible(view(cameraAt(-(FAR_MAX_DISTANCE_M - 500), 0), campus), true)).toBe(true);
    // Back inside the limit, it is drawn again only a little further in.
    const between = (FAR_MAX_DISTANCE_M + FAR_SHOW_DISTANCE_M) / 2;
    expect(farVisible(view(cameraAt(-between, 0), campus), false)).toBe(false);
    expect(farVisible(view(cameraAt(-(FAR_SHOW_DISTANCE_M - 500), 0), campus), false)).toBe(true);
  });

  it("is not drawn while the globe hides it", () => {
    // A 300 m scan at sea level, 20 km off: 25 px across, but under the horizon of a camera
    // 2 m up (some 5 km away), and over it from 500 m up.
    const scan = Cartesian3.fromDegrees(-82.6966, 28.0389, 0);
    const low = Cartesian3.fromDegrees(-82.6966, 28.0389 + 20_000 / 110_800, 2);
    const high = Cartesian3.fromDegrees(-82.6966, 28.0389 + 20_000 / 110_800, 500);
    expect(projectedDiameterPx(view(low, 300, scan))).toBeGreaterThan(FAR_SHOW_PX);
    expect(farVisible(view(low, 300, scan), true)).toBe(false);
    expect(farVisible(view(high, 300, scan), false)).toBe(true);
  });
});

describe("the horizon test", () => {
  const ground = Cartesian3.fromDegrees(10, 45, 0);

  it("sees the ground below and beside the camera, not the far side of the Earth", () => {
    expect(hiddenByGlobe(Cartesian3.fromDegrees(10, 45, 400), ground)).toBe(false);
    expect(hiddenByGlobe(Cartesian3.fromDegrees(10.05, 45, 400), ground)).toBe(false);
    expect(hiddenByGlobe(Cartesian3.fromDegrees(-170, -45, 400), ground)).toBe(true);
  });

  it("holds a scan below the ellipsoid as seen, from above it or beside it", () => {
    // Ground below the ellipsoid, as where the geoid dips (Death Valley is some 120 m under).
    const sunk = Cartesian3.fromDegrees(-116.8, 36.25, -120);
    expect(hiddenByGlobe(Cartesian3.fromDegrees(-116.8, 36.251, -100), sunk)).toBe(false);
    expect(hiddenByGlobe(Cartesian3.fromDegrees(-116.81, 36.25, 300), sunk)).toBe(false);
  });
});
