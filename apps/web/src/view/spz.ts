/**
 * Where a tile's splats are and how opaque, read straight from its SPZ bytes: what the scan
 * viewer's collision grid is built from (lib/occupancy.ts).
 *
 * Spark keeps a level-of-detail mesh as its merged LoD tree only (`packedSplats.lodSplats`),
 * whose inner nodes sit at the centroids of clusters -- in the air between a table top and
 * the floor, say -- so collision is built from the tile's own splats instead: the positions
 * and alphas at the head of the SPZ (nianticlabs/spz, versions 2 and 3, gzip-framed; as
 * `tools/captures/splat_tiles.py` `pack_spz` writes them). Colours, scales, rotations and
 * spherical harmonics follow and are not read.
 */

const SPZ_MAGIC = 0x5053474e; // "NGSP"
const HEADER_BYTES = 16;

export interface SpzPoints {
  count: number;
  /** x, y, z per splat, in the tile's own frame (east/north/up metres for this pipeline). */
  positions: Float32Array;
  /** Opacity in [0, 1] per splat. */
  alphas: Float32Array;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Positions and opacities of a gzip-framed SPZ (versions 2 and 3). */
export async function spzPoints(spz: Uint8Array): Promise<SpzPoints> {
  const raw = await gunzip(spz);
  return spzPointsRaw(raw);
}

/** The same, from the decompressed bytes. */
export function spzPointsRaw(raw: Uint8Array): SpzPoints {
  if (raw.byteLength < HEADER_BYTES) throw new Error("The SPZ is shorter than its header.");
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (view.getUint32(0, true) !== SPZ_MAGIC) throw new Error("The data is not an SPZ.");
  const version = view.getUint32(4, true);
  if (version !== 2 && version !== 3) {
    throw new Error(`SPZ version ${String(version)} is not read here.`);
  }
  const count = view.getUint32(8, true);
  const fractionalBits = view.getUint8(13);
  const positionBytes = count * 9;
  if (raw.byteLength < HEADER_BYTES + positionBytes + count) {
    throw new Error("The SPZ is shorter than its splats.");
  }
  const positions = new Float32Array(count * 3);
  const scale = 1 / (1 << fractionalBits);
  let at = HEADER_BYTES;
  for (let i = 0; i < count * 3; i++) {
    let fixed = (raw[at] ?? 0) | ((raw[at + 1] ?? 0) << 8) | ((raw[at + 2] ?? 0) << 16);
    if (fixed & 0x800000) fixed -= 0x1000000;
    positions[i] = fixed * scale;
    at += 3;
  }
  const alphas = new Float32Array(count);
  for (let i = 0; i < count; i++) alphas[i] = (raw[at + i] ?? 0) / 255;
  return { count, positions, alphas };
}
