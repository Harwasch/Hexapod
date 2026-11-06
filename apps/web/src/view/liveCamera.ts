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
