import type { DecodedCameras } from "./liveData";

/** Index in the existing path order; height is measured along the solve's estimated up. */
export function medianHeightCamera(cameras: DecodedCameras, up: readonly number[]): number | null {
  const order = Array.from({ length: cameras.count }, (_, i) => i);
  const height = (i: number): number =>
    (cameras.centres[i * 3] ?? 0) * (up[0] ?? 0) +
    (cameras.centres[i * 3 + 1] ?? 0) * (up[1] ?? 0) +
    (cameras.centres[i * 3 + 2] ?? 0) * (up[2] ?? 0);
  order.sort((a, b) => height(a) - height(b) || a - b);
  return order[Math.floor(order.length / 2)] ?? null;
}

export function nextTrainingCamera(current: number | null, count: number): number | null {
  return count > 0 ? ((current ?? -1) + 1) % count : null;
}

export function trainingFov(value?: number | null): number {
  return value != null && Number.isFinite(value) && value > 0 && value < 180 ? value : 50;
}

/** Median spacing between consecutive cameras: how big a frustum should be. */
export function cameraSpacing(cameras: DecodedCameras, fallback: number): number {
  const gaps: number[] = [];
  for (let i = 1; i < cameras.count; i += 1) {
    const dx = (cameras.centres[i * 3] ?? 0) - (cameras.centres[i * 3 - 3] ?? 0);
    const dy = (cameras.centres[i * 3 + 1] ?? 0) - (cameras.centres[i * 3 - 2] ?? 0);
    const dz = (cameras.centres[i * 3 + 2] ?? 0) - (cameras.centres[i * 3 - 1] ?? 0);
    gaps.push(Math.hypot(dx, dy, dz));
  }
  gaps.sort((a, b) => a - b);
  const median = gaps[Math.floor(gaps.length / 2)] ?? 0;
  return median > 0 ? median : fallback;
}

/** Reuse the frustum index buffer; nearby camera lines never cross the eye's image. */
export function frustumIndices(
  centres: Float32Array,
  eye: readonly number[],
  radius: number,
  indices: Uint16Array,
): number {
  let count = 0;
  for (let i = 0; i < centres.length / 3; i++) {
    const distance = Math.hypot(
      (centres[i * 3] ?? 0) - (eye[0] ?? 0),
      (centres[i * 3 + 1] ?? 0) - (eye[1] ?? 0),
      (centres[i * 3 + 2] ?? 0) - (eye[2] ?? 0),
    );
    if (distance <= radius) continue;
    for (let j = 0; j < 16; j++) indices[count++] = i * 16 + j;
  }
  return count;
}

export function overlayVisibility(
  hasSplat: boolean,
  final: boolean,
  enabled: boolean,
): {
  cameras: boolean;
  points: boolean;
} {
  return { cameras: enabled && (!hasSplat || final), points: enabled && !hasSplat };
}
