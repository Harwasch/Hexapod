/**
 * Decodes one splat tile's SPZ off the main thread: spz-loader's WASM, as CesiumJS runs it,
 * then colours and spherical harmonics laid out for the glTF loader (lib/splatLayout.ts).
 * In a trace of the globe these took 3.6 s of the main thread over 20 s of looking around.
 */

import { loadSpz } from "@spz-loader/core";

import { rgbaOf, shCoefficientsOf } from "@/lib/splatLayout";

self.onmessage = async (event: MessageEvent<{ id: number; spz: Uint8Array }>): Promise<void> => {
  const { id, spz } = event.data;
  try {
    const cloud = await loadSpz(spz, { unpackOptions: { coordinateSystem: "UNSPECIFIED" } });
    const rgba = rgbaOf(cloud);
    const shCoefficients = shCoefficientsOf(cloud);
    const decoded = {
      numPoints: cloud.numPoints,
      shDegree: cloud.shDegree,
      antialiased: cloud.antialiased,
      positions: cloud.positions,
      scales: cloud.scales,
      rotations: cloud.rotations,
      alphas: cloud.alphas,
      rgba,
      shCoefficients,
    };
    // Each array its own buffer, once (a buffer listed twice cannot be transferred).
    const transfer = new Set<ArrayBufferLike>([
      cloud.positions.buffer,
      cloud.scales.buffer,
      cloud.rotations.buffer,
      cloud.alphas.buffer,
      rgba.buffer,
      ...Object.values(shCoefficients).map((values) => values.buffer),
    ]);
    self.postMessage({ id, decoded }, { transfer: [...transfer] as Transferable[] });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
