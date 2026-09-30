import { Cartesian3, Matrix4, type Camera } from "cesium";

import type { ScanPose } from "./types";

const scratchPoint = new Cartesian3();
const scratchVector = new Cartesian3();

/** Far enough for any scan; a splat writes no depth, so this only bounds the projection. */
const MAX_FAR_M = 100_000;

/**
 * Cesium's camera as a dedicated renderer's, in the scan's frame: `toLocal` takes Earth-fixed
 * coordinates to the tileset root's east/north/up metres, where the splats are.
 */
export function scanPose(
  camera: Camera,
  toLocal: Matrix4,
  size: { width: number; height: number; pixelRatio: number },
): ScanPose {
  const eye = Matrix4.multiplyByPoint(toLocal, camera.positionWC, scratchPoint);
  const eyeTuple: [number, number, number] = [eye.x, eye.y, eye.z];
  const direction = Matrix4.multiplyByPointAsVector(toLocal, camera.directionWC, scratchVector);
  const directionTuple: [number, number, number] = [direction.x, direction.y, direction.z];
  const up = Matrix4.multiplyByPointAsVector(toLocal, camera.upWC, scratchVector);
  const upTuple: [number, number, number] = [up.x, up.y, up.z];
  const frustum = camera.frustum as { fovy?: number; near?: number; far?: number };
  return {
    eye: eyeTuple,
    direction: directionTuple,
    up: upTuple,
    fovy: frustum.fovy ?? Math.PI / 3,
    near: Math.max(0.01, frustum.near ?? 0.1),
    far: Math.min(MAX_FAR_M, frustum.far ?? MAX_FAR_M),
    ...size,
  };
}
