import { Cartesian3, Matrix4, type Camera } from "cesium";

import { uniformScale } from "../placement";
import type { ScanPose } from "./types";

const scratchPoint = new Cartesian3();
const scratchVector = new Cartesian3();

/** Far enough for any scan; a splat writes no depth, so this only bounds the projection. */
const MAX_FAR_M = 100_000;

/** How far from 1 a frame's scale must be to be a runtime scale rather than rounding. */
const SCALED_EPSILON = 1e-9;

/**
 * Cesium's camera as a dedicated renderer's, in the scan's frame: `toLocal` takes Earth-fixed
 * coordinates to the tileset root's east/north/up metres, where the splats are.
 *
 * Under a runtime scale (`renderConfig.scale`, tilesetScale.ts) a unit of that frame is
 * `scale` metres on the globe, so `toLocal` shrinks lengths by it: the eye lands where it should
 * among the splats, the direction and up come out `1 / scale` long and are made unit again, and
 * the near and far planes -- distances from the eye -- are converted the same way, so the
 * renderer clips where Cesium's camera does.
 */
export function scanPose(
  camera: Camera,
  toLocal: Matrix4,
  size: { width: number; height: number; pixelRatio: number },
): ScanPose {
  // Units of the scan's frame to a metre on the globe: 1, or 1 / scale.
  const perMetre = uniformScale(toLocal);
  const scaled = Math.abs(perMetre - 1) > SCALED_EPSILON;
  const eye = Matrix4.multiplyByPoint(toLocal, camera.positionWC, scratchPoint);
  const eyeTuple: [number, number, number] = [eye.x, eye.y, eye.z];
  const vector = (world: Cartesian3): [number, number, number] => {
    const local = Matrix4.multiplyByPointAsVector(toLocal, world, scratchVector);
    if (scaled) Cartesian3.divideByScalar(local, perMetre, local);
    return [local.x, local.y, local.z];
  };
  const frustum = camera.frustum as { fovy?: number; near?: number; far?: number };
  const near = Math.max(0.01, frustum.near ?? 0.1);
  const far = Math.min(MAX_FAR_M, frustum.far ?? MAX_FAR_M);
  return {
    eye: eyeTuple,
    direction: vector(camera.directionWC),
    up: vector(camera.upWC),
    fovy: frustum.fovy ?? Math.PI / 3,
    near: scaled ? near * perMetre : near,
    far: scaled ? far * perMetre : far,
    ...size,
  };
}
