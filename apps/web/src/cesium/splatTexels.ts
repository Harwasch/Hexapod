/**
 * Texel arithmetic for the Gaussian splat attribute texture.
 *
 * Pure: no Cesium, no WebGL, no DOM. Everything here is exercised headlessly, because this is
 * the half of the deformer where a mistake is invisible — an addressing error writes plausible
 * floats into the wrong splat and the tree simply looks wrong.
 *
 * The layout is fixed by `Shaders/PrimitiveGaussianSplatVS.glsl:144-166` (CesiumJS 1.145):
 *
 *     ivec2 posCoord = ivec2(int((texIdx & rowMask) << 1), int(texIdx >> rowShift));
 *     ivec2 covCoord = ivec2(int(((texIdx & rowMask) << 1) | 1u), int(texIdx >> rowShift));
 *
 * so each splat owns two adjacent RGBA32UI texels on one row: an even column holding the
 * position as raw float32 bits, and the odd column beside it holding packed covariance halves
 * and an RGBA8 colour. We write three words per splat and never touch the odd column — that
 * property is what makes rewriting positions safe without reimplementing the engine's packer,
 * and it is asserted in `splatTexels.test.ts`.
 *
 * `rowMask` and `rowShift` are read from the primitive, never recomputed from an assumed
 * maximum texture size: the same value differs between machines (8192 under SwiftShader here,
 * 16384 on a typical desktop GPU), and a wrong guess addresses every splat into the wrong texel
 * while still producing a picture.
 */

/** Words per texel in an RGBA32UI texture. */
const WORDS_PER_TEXEL = 4;
/** Texels per splat: one position texel, one covariance/colour texel. */
const TEXELS_PER_SPLAT = 2;

/** The largest `rowShift` that keeps `width = 2 << rowShift` inside any plausible GL limit. */
const MAX_ROW_SHIFT = 20;

/** Where a splat's two texels sit, and the shape of the texture that holds them. */
export interface SplatTextureLayout {
  /** `splatsPerRow - 1`, straight from the primitive. Masks the column part of a splat index. */
  readonly rowMask: number;
  /** `log2(splatsPerRow)`, straight from the primitive. Shifts out the row part. */
  readonly rowShift: number;
  readonly splatsPerRow: number;
  /** Texture width in texels: `2 * splatsPerRow`, i.e. the device's maximum texture size. */
  readonly width: number;
  /** Texture height in texels: enough rows for `numSplats`. */
  readonly height: number;
  readonly numSplats: number;
}

/**
 * Validates the addressing parameters the primitive reports and derives the texture shape.
 *
 * Throws rather than guessing. The caller turns that into a refusal: a deformer that writes to
 * a layout it does not understand corrupts the geometry it was supposed to preserve.
 */
export function splatTextureLayout(
  numSplats: number,
  rowMask: number,
  rowShift: number,
): SplatTextureLayout {
  if (!Number.isInteger(numSplats) || numSplats <= 0) {
    throw new Error(`splat layout: numSplats must be a positive integer, got ${numSplats}`);
  }
  if (!Number.isInteger(rowShift) || rowShift < 1 || rowShift > MAX_ROW_SHIFT) {
    throw new Error(`splat layout: rowShift ${rowShift} is outside 1..${MAX_ROW_SHIFT}`);
  }
  const splatsPerRow = 1 << rowShift;
  if (rowMask !== splatsPerRow - 1) {
    throw new Error(
      `splat layout: rowMask ${rowMask} does not match rowShift ${rowShift} (expected ${splatsPerRow - 1})`,
    );
  }
  const width = splatsPerRow * TEXELS_PER_SPLAT;
  const height = Math.ceil(numSplats / splatsPerRow);
  return { rowMask, rowShift, splatsPerRow, width, height, numSplats };
}

/** The row a splat's texels live on. */
export function splatRow(splatIndex: number, layout: SplatTextureLayout): number {
  return splatIndex >>> layout.rowShift;
}

/** The even column holding a splat's position, as `[x, y]` texel coordinates. */
export function positionTexel(
  splatIndex: number,
  layout: SplatTextureLayout,
): readonly [number, number] {
  return [(splatIndex & layout.rowMask) << 1, splatIndex >>> layout.rowShift];
}

/** The odd column holding a splat's covariance and colour. We never write here. */
export function covarianceTexel(
  splatIndex: number,
  layout: SplatTextureLayout,
): readonly [number, number] {
  return [((splatIndex & layout.rowMask) << 1) | 1, splatIndex >>> layout.rowShift];
}

/** Index of a splat's first position word in a staging buffer laid out as the texture is. */
export function positionWordOffset(splatIndex: number, layout: SplatTextureLayout): number {
  const [x, y] = positionTexel(splatIndex, layout);
  return (y * layout.width + x) * WORDS_PER_TEXEL;
}

/** The raw IEEE-754 bits of a float32, as an unsigned 32-bit integer. */
export function float32Bits(value: number): number {
  const scratch = new Float32Array(1);
  scratch[0] = value;
  return new Uint32Array(scratch.buffer)[0] ?? 0;
}

/** The float32 a word of raw bits denotes. Inverse of `float32Bits`. */
export function bitsToFloat32(bits: number): number {
  const scratch = new Uint32Array(1);
  scratch[0] = bits;
  return new Float32Array(scratch.buffer)[0] ?? 0;
}

/**
 * The buffer we stage a frame's texture contents in.
 *
 * `words` and `floats` are two views of one `ArrayBuffer`, so writing a position is a float32
 * store rather than a bit-twiddling conversion — the packing is the aliasing, and it is exact
 * by construction.
 */
export interface SplatStagingBuffer {
  readonly words: Uint32Array;
  readonly floats: Float32Array;
  readonly layout: SplatTextureLayout;
}

/**
 * Allocates a staging buffer of exactly `width * height * 4` words and copies the engine's
 * packed data into it.
 *
 * The captured WASM buffer is routinely *shorter* than the texture needs — 466,944 words against
 * 491,520 for the 57,410-splat capture this was measured on — because the packer sizes its
 * output to the splat count and the engine zero-pads the tail (`GaussianSplatPrimitive.js`,
 * `processGeneratedSplatTextureData`). Staging the short buffer directly would upload garbage
 * for the last row, so the allocation is always the full texture and the copy is clamped.
 */
export function createStagingBuffer(
  layout: SplatTextureLayout,
  packed: Uint32Array,
): SplatStagingBuffer {
  const words = new Uint32Array(layout.width * layout.height * WORDS_PER_TEXEL);
  const copied = Math.min(packed.length, words.length);
  words.set(packed.subarray(0, copied));
  return { words, floats: new Float32Array(words.buffer), layout };
}

/**
 * Writes `count` splat positions from `positions` (flat xyz, metres, baked frame) into the
 * staging buffer's position lanes, starting at splat `first`.
 *
 * Only the R, G and B words of each even-column texel are touched. A is left as the engine
 * packed it, and the whole odd column — covariance and colour — is left alone.
 */
export function writeSplatPositions(
  staging: SplatStagingBuffer,
  positions: Float32Array,
  first = 0,
  count = staging.layout.numSplats - first,
): void {
  const { layout, floats } = staging;
  const available = Math.min(layout.numSplats, Math.floor(positions.length / 3));
  const last = Math.min(first + count, available);
  const rowWords = layout.width * WORDS_PER_TEXEL;
  for (let i = Math.max(0, first); i < last; i += 1) {
    const offset =
      (i >>> layout.rowShift) * rowWords + ((i & layout.rowMask) << 1) * WORDS_PER_TEXEL;
    const base = i * 3;
    floats[offset] = positions[base] ?? 0;
    floats[offset + 1] = positions[base + 1] ?? 0;
    floats[offset + 2] = positions[base + 2] ?? 0;
  }
}

/** A contiguous band of texture rows, the unit of upload. */
export interface SplatRowRange {
  readonly firstRow: number;
  readonly rowCount: number;
}

/** Every row of the texture. */
export function fullRowRange(layout: SplatTextureLayout): SplatRowRange {
  return { firstRow: 0, rowCount: layout.height };
}

/**
 * The smallest band of rows containing every splat whose node moves.
 *
 * `nodeMoves[n]` is non-zero when node `n`'s transform is not the identity. Returns `undefined`
 * when nothing moves, which is how a zero-wind frame skips its upload entirely and leaves an
 * idle scene idle.
 *
 * Nearest-node assignment gives no index contiguity, so in practice a moving tree covers most
 * of the rows it occupies; this is a cheap bound, not a clever one. For an isolated single-tile
 * tree at 8192-wide texels the whole tree is one or two rows anyway.
 */
export function rowRangeForMovingNodes(
  assignment: Uint16Array,
  nodeMoves: Uint8Array,
  layout: SplatTextureLayout,
): SplatRowRange | undefined {
  const count = Math.min(assignment.length, layout.numSplats);
  let firstRow = Number.POSITIVE_INFINITY;
  let lastRow = -1;
  for (let i = 0; i < count; i += 1) {
    if ((nodeMoves[assignment[i] ?? 0] ?? 0) === 0) continue;
    const row = i >>> layout.rowShift;
    if (row < firstRow) firstRow = row;
    if (row > lastRow) lastRow = row;
    // Once the band spans everything there is nothing left to narrow.
    if (firstRow === 0 && lastRow === layout.height - 1) break;
  }
  if (lastRow < 0) return undefined;
  return { firstRow, rowCount: lastRow - firstRow + 1 };
}

/** The words of a row band, as a view into the staging buffer. No copy. */
export function rowSlice(staging: SplatStagingBuffer, range: SplatRowRange): Uint32Array {
  const rowWords = staging.layout.width * WORDS_PER_TEXEL;
  return staging.words.subarray(
    range.firstRow * rowWords,
    (range.firstRow + range.rowCount) * rowWords,
  );
}

/** Words uploaded for a row band — the number the upload budget is actually spent on. */
export function rowRangeWordCount(layout: SplatTextureLayout, range: SplatRowRange): number {
  return range.rowCount * layout.width * WORDS_PER_TEXEL;
}
