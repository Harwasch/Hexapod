import type { GrayFrame } from "./types";
/** Small-frame translation estimate. This measures pixel motion, not 3D camera pose. */
export function measureFrame(previous: GrayFrame | undefined, current: GrayFrame) {
  const { width, height, pixels } = current;
  if (width < 16 || height < 16 || pixels.length !== width * height)
    throw new Error("Invalid evaluation frame.");
  let laplacianSum = 0;
  let laplacianSquares = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const value =
        (pixels[i - 1] ?? 0) +
        (pixels[i + 1] ?? 0) +
        (pixels[i - width] ?? 0) +
        (pixels[i + width] ?? 0) -
        4 * (pixels[i] ?? 0);
      laplacianSum += value;
      laplacianSquares += value * value;
      count++;
    }
  const sharpness = laplacianSquares / count - (laplacianSum / count) ** 2;
  if (previous?.width !== width || previous?.height !== height) return { sharpness };
  const maxShift = 5;
  let baseline = 0;
  for (let i = 0; i < pixels.length; i++)
    baseline += Math.abs((pixels[i] ?? 0) - (previous.pixels[i] ?? 0));
  let best = Infinity;
  let zero = Infinity;
  let dx = 0;
  let dy = 0;
  for (let sy = -maxShift; sy <= maxShift; sy++)
    for (let sx = -maxShift; sx <= maxShift; sx++) {
      let error = 0;
      let samples = 0;
      for (let y = maxShift; y < height - maxShift; y += 2)
        for (let x = maxShift; x < width - maxShift; x += 2) {
          error += Math.abs(
            (pixels[y * width + x] ?? 0) - (previous.pixels[(y - sy) * width + x - sx] ?? 0),
          );
          samples++;
        }
      const average = error / samples;
      if (sx === 0 && sy === 0) zero = average;
      if (average < best || (average === best && sx * sx + sy * sy < dx * dx + dy * dy)) {
        best = average;
        dx = sx;
        dy = sy;
      }
    }
  return {
    sharpness,
    pixelChange: baseline / pixels.length,
    motionX: dx,
    motionY: dy,
    motionConfidence: zero > 0.001 ? Math.max(0, (zero - best) / zero) : 0,
  };
}
export function frameSimilarity(first: GrayFrame, last: GrayFrame): number {
  if (first.width !== last.width || first.height !== last.height) return 0;
  let error = 0;
  for (let i = 0; i < first.pixels.length; i++)
    error += Math.abs((first.pixels[i] ?? 0) - (last.pixels[i] ?? 0));
  return Math.max(0, 1 - error / first.pixels.length);
}
export async function grayscale(blob: Blob): Promise<GrayFrame> {
  const image = await createImageBitmap(blob);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 40;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Frame evaluation needs Canvas 2D.");
    context.drawImage(image, 0, 0, 64, 40);
    const rgba = context.getImageData(0, 0, 64, 40).data;
    const pixels = new Float32Array(64 * 40);
    for (let i = 0; i < pixels.length; i++)
      pixels[i] =
        ((rgba[i * 4] ?? 0) * 0.2126 +
          (rgba[i * 4 + 1] ?? 0) * 0.7152 +
          (rgba[i * 4 + 2] ?? 0) * 0.0722) /
        255;
    return { width: 64, height: 40, pixels, sourceWidth: image.width, sourceHeight: image.height };
  } finally {
    image.close();
  }
}
