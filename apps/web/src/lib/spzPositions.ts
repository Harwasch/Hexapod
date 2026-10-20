/**
 * The positions of an SPZ file, exactly as the pipeline digests them (`checksumPositions`):
 * the 24-bit fixed-point centres over `2^fractionalBits`, as float32, in the file's order.
 *
 * For a renderer that hands the SPZ to a decoder it cannot read back from (Spark), so it can
 * still find a tile's object ids in `instances.json` by the tile's checksum
 * (cesium/scanView/scanInstances.ts). Only the positions are read; nothing else is decoded.
 */

const SPZ_MAGIC = 0x5053474e; // "NGSP"
const HEADER_BYTES = 16;

/** The SPZ in `bytes` unzipped: off the main thread (the browser's `DecompressionStream`). */
export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const body = new Response(bytes as BodyInit).body;
  if (!body) return new Uint8Array(0);
  const stream = body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * The centres of the SPZ in `bytes` (gzip-compressed, as a tile carries it), or undefined for
 * a version that stores them otherwise (v1's half floats) or a file that is not SPZ.
 */
export async function spzPositions(bytes: Uint8Array): Promise<Float32Array | undefined> {
  return positionsOf(await gunzip(bytes));
}

/** What picking needs of each splat (lib/splatPick.ts): centre, largest axis, opacity. */
export interface SpzPickData {
  positions: Float32Array;
  radii: Float32Array;
  opacity: Float32Array;
}

/**
 * The centres of the SPZ in `bytes` as `spzPositions` reads them, with each splat's largest
 * axis and its opacity: after the centres, SPZ stores an alpha byte per splat (opacity after
 * the sigmoid, over 255), three colour bytes, then three log-scale bytes (`b / 16 - 10`).
 * Undefined where `spzPositions` is.
 */
export async function spzPickData(bytes: Uint8Array): Promise<SpzPickData | undefined> {
  return spzPickDataOf(await gunzip(bytes));
}

/**
 * `spzPickData` of an SPZ already unzipped (`gunzip`): the synchronous part, a pass over every
 * splat, which a renderer runs within its main-thread frame budget.
 */
export function spzPickDataOf(raw: Uint8Array): SpzPickData | undefined {
  const positions = positionsOf(raw);
  if (!positions) return undefined;
  const count = positions.length / 3;
  const alphaAt = HEADER_BYTES + count * 9;
  const scaleAt = alphaAt + count * 4;
  if (raw.byteLength < scaleAt + count * 3) return undefined;
  const radii = new Float32Array(count);
  const opacity = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    opacity[i] = (raw[alphaAt + i] ?? 0) / 255;
    const s = scaleAt + i * 3;
    radii[i] = Math.exp(Math.max(raw[s] ?? 0, raw[s + 1] ?? 0, raw[s + 2] ?? 0) / 16 - 10);
  }
  return { positions, radii, opacity };
}

/**
 * A tile's decoded positions in the scan's (site) frame: `local` moved by the tile's `offset`
 * (tiles.ts `TileNode.offset`, the composed translation of its frame), the basis every position
 * checksum is taken over (rig `tileChecksums`, `instances.json`, `skin.json`, plants), as
 * `tools/captures/splat_tiles.py` `site_positions` defines it: `float32(local + offset)`, the
 * sum in doubles. Exact wherever |site| < 4096 m (an offset is whole metres and `local` is on
 * the 2^-12 grid), the correctly rounded float32 beyond -- the same bits numpy gives.
 *
 * `local` itself, unchanged, when there is no offset or a zero one: every tile of a scan that
 * fits SPZ's +/-2048 m.
 */
export function siteFramePositions(
  local: Float32Array,
  offset?: readonly [number, number, number],
): Float32Array {
  if (!offset || (offset[0] === 0 && offset[1] === 0 && offset[2] === 0)) return local;
  const out = new Float32Array(local.length);
  for (let i = 0; i < local.length; i += 1) {
    // `+ 0` folds a negative zero into the positive one the pipeline writes.
    out[i] = (local[i] ?? 0) + (offset[i % 3] ?? 0) + 0;
  }
  return out;
}

function positionsOf(raw: Uint8Array): Float32Array | undefined {
  if (raw.byteLength < HEADER_BYTES) return undefined;
  const header = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (header.getUint32(0, true) !== SPZ_MAGIC) return undefined;
  const version = header.getUint32(4, true);
  if (version < 2) return undefined;
  const count = header.getUint32(8, true);
  const fractionalBits = header.getUint8(13);
  if (raw.byteLength < HEADER_BYTES + count * 9) return undefined;
  const scale = 1 / (1 << fractionalBits);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i += 1) {
    const o = HEADER_BYTES + i * 3;
    let fixed = (raw[o] ?? 0) | ((raw[o + 1] ?? 0) << 8) | ((raw[o + 2] ?? 0) << 16);
    if (fixed & 0x800000) fixed -= 0x1000000;
    // `+ 0` folds a negative zero into the positive one the pipeline writes.
    out[i] = fixed * scale + 0;
  }
  return out;
}
