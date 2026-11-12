/** Camera/metric math for the dev-only console check; never imported by an app entry. */
import { Matrix4, PerspectiveCamera } from "three";

export interface ReferenceCamera {
  viewmat: number[][];
  K: number[][];
  width: number;
  height: number;
  near?: number;
  far?: number;
}

/** gsplat camera axes right/down/forward -> Three right/up/back, without moving the solve. */
export function referenceCamera(spec: ReferenceCamera): PerspectiveCamera {
  const { width, height, viewmat, K } = spec;
  if (
    !(width > 0 && height > 0) ||
    viewmat.length !== 4 ||
    viewmat.some((r) => r.length !== 4) ||
    K.length !== 3 ||
    K.some((r) => r.length !== 3) ||
    ![...viewmat.flat(), ...K.flat(), width, height].every(Number.isFinite)
  ) {
    throw new Error("Invalid camera matrices or image dimensions");
  }
  const fx = K[0]?.[0] ?? 0,
    fy = K[1]?.[1] ?? 0;
  const cx = K[0]?.[2] ?? 0,
    cy = K[1]?.[2] ?? 0,
    skew = K[0]?.[1] ?? 0;
  const near = spec.near ?? 0.01,
    far = spec.far ?? 1e10;
  if (!(fx > 0 && fy > 0 && near > 0 && far > near)) throw new Error("Invalid camera intrinsics");
  const camera = new PerspectiveCamera(
    (2 * Math.atan(height / (2 * fy)) * 180) / Math.PI,
    width / height,
    near,
    far,
  );
  const worldToCamera = new Matrix4().fromArray(viewmat.flat()).transpose();
  worldToCamera.premultiply(new Matrix4().makeScale(1, -1, -1));
  camera.matrix.copy(worldToCamera).invert();
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixAutoUpdate = false;
  camera.updateMatrixWorld(true);
  // Pixel x = fx*X/Z + skew*Y/Z + cx; pixel y = fy*Y/Z + cy in gsplat axes.
  camera.projectionMatrix.set(
    (2 * fx) / width,
    (-2 * skew) / width,
    1 - (2 * cx) / width,
    0,
    0,
    (2 * fy) / height,
    (2 * cy) / height - 1,
    0,
    0,
    0,
    -(far + near) / (far - near),
    (-2 * far * near) / (far - near),
    0,
    0,
    -1,
    0,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return camera;
}

/** All RGB samples in [0,1], no gamma conversion, masking, alignment or colour fit. */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (!a.length || a.length !== b.length || a.length % 3 !== 0)
    throw new Error("RGB sizes differ or are empty");
  let squared = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? NaN,
      y = b[i] ?? NaN;
    if (!(x >= 0 && x <= 1 && y >= 0 && y <= 1)) throw new Error("RGB outside [0,1]");
    squared += (x - y) ** 2;
  }
  return squared === 0 ? Infinity : -10 * Math.log10(squared / a.length);
}

export function rgbFloats(rgba: Uint8ClampedArray): Float64Array {
  const rgb = new Float64Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) rgb[j++] = (rgba[i + channel] ?? 0) / 255;
  }
  return rgb;
}
