/**
 * The packed splat buffers captured from the engine, and how a primitive finds its own.
 *
 * Pure: no Cesium. `splatCapture.ts` does the interception and feeds this; the deformer only
 * reads. Splitting them keeps `SplatDeformer` importable — and unit-testable — without loading
 * CesiumJS.
 *
 * Why a registry at all: the interception point
 * (`GaussianSplatTextureGenerator.generateFromAttributes`) is handed attributes and a count and
 * nothing else — no primitive, no tileset. The capture therefore has to be matched back to the
 * primitive that caused it, which is what the digest below is for.
 */

/** One packed attribute buffer, exactly as the engine uploaded it. */
export interface SplatTextureCapture {
  /** Splats in the buffer, from the generator's own `count` parameter. */
  readonly count: number;
  /** Matching key over the positions the buffer was packed from. See `digestSplatPositions`. */
  readonly digest: string;
  /**
   * The packed RGBA32UI words. Frequently *shorter* than the texture the engine derives from it,
   * which zero-pads the tail — `createStagingBuffer` handles that.
   */
  readonly data: Uint32Array;
  /** `performance.now()` at capture, for diagnostics only. */
  readonly at: number;
}

/**
 * Captures kept. Small on purpose: each entry pins a multi-megabyte buffer alive, and only the
 * most recent generation of each loaded splat tileset is ever wanted.
 */
const MAX_CAPTURES = 4;

/** Float32 words sampled by `digestSplatPositions`. */
const DIGEST_SAMPLES = 4096;

const captures: SplatTextureCapture[] = [];

/**
 * A cheap matching key over a splat position array: FNV-1a over the count plus up to 4096
 * evenly-spaced raw words.
 *
 * Deliberately *not* a proof of identity — that is `checksumPositions`, run bit-exactly over the
 * un-baked positions inside the deformer. This one only has to tell two concurrently loaded
 * splat tilesets apart, and it has to be O(1): a full digest of a three-million-splat site is a
 * 36 MB pass in the middle of the engine's update, which is not a cost this feature is allowed
 * to impose on scenes that never deform anything.
 */
export function digestSplatPositions(positions: Float32Array, count: number): string {
  const words = Math.max(0, Math.min(positions.length, count * 3));
  // A uint32 view over the same bytes: the digest is over raw bits, so -0 and +0 differ.
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const stride = Math.max(1, Math.floor(words / DIGEST_SAMPLES));
  let h = 0x811c9dc5;
  h = Math.imul(h ^ (count & 0xffff), 0x01000193) >>> 0;
  h = Math.imul(h ^ ((count >>> 16) & 0xffff), 0x01000193) >>> 0;
  for (let i = 0; i < words; i += stride) {
    const word = bits[i] ?? 0;
    h = Math.imul(h ^ (word & 0xff), 0x01000193) >>> 0;
    h = Math.imul(h ^ ((word >>> 8) & 0xff), 0x01000193) >>> 0;
    h = Math.imul(h ^ ((word >>> 16) & 0xff), 0x01000193) >>> 0;
    h = Math.imul(h ^ ((word >>> 24) & 0xff), 0x01000193) >>> 0;
  }
  return `splat:${count}:${h.toString(16).padStart(8, "0")}`;
}

/** Files a capture, evicting the oldest once the ring is full. */
export function recordSplatCapture(capture: SplatTextureCapture): void {
  captures.push(capture);
  while (captures.length > MAX_CAPTURES) captures.shift();
}

/** The most recent capture matching both the splat count and the position digest. */
export function findSplatCapture(count: number, digest: string): SplatTextureCapture | undefined {
  for (let i = captures.length - 1; i >= 0; i -= 1) {
    const capture = captures[i];
    if (capture === undefined) continue;
    if (capture.count === count && capture.digest === digest) return capture;
  }
  return undefined;
}

/** How many captures are held. Diagnostics — a zero here means the interception ran too late. */
export function splatCaptureCount(): number {
  return captures.length;
}

/** Drops every capture and the buffers they pin. */
export function clearSplatCaptures(): void {
  captures.length = 0;
}
