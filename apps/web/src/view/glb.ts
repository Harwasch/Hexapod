/**
 * The SPZ bytes inside one of this pipeline's splat tiles.
 *
 * `tools/captures/splat_tiles.py` packs a splat as a glTF binary whose one primitive
 * carries `KHR_gaussian_splatting` with `KHR_gaussian_splatting_compression_spz_2`, the
 * data being a standard SPZ (v2) file stored whole in a buffer view. That is what Cesium
 * renders on the globe; the standalone viewer wants the same bytes as a plain `.spz`, so
 * it reads them back out of the `.glb` rather than asking the pipeline to publish the
 * scan twice. Every scan already on the map is viewable this way, with no re-run.
 */

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"
const SPZ_EXTENSION = "KHR_gaussian_splatting_compression_spz_2";

interface GltfJson {
  meshes?: {
    primitives?: {
      extensions?: Record<string, { extensions?: Record<string, { bufferView?: number }> }>;
    }[];
  }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number }[];
}

export function spzFromGlb(glb: ArrayBuffer): Uint8Array {
  const view = new DataView(glb);
  if (glb.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error("The scan's tile is not a glTF binary.");
  }
  let json: GltfJson | null = null;
  let bin: Uint8Array | null = null;
  for (let offset = 12; offset + 8 <= glb.byteLength;) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const body = new Uint8Array(glb, offset + 8, length);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(body)) as GltfJson;
    if (type === CHUNK_BIN) bin = body;
    offset += 8 + length;
  }
  if (!json || !bin) throw new Error("The scan's tile has no data in it.");

  const index = json.meshes
    ?.flatMap((mesh) => mesh.primitives ?? [])
    .map(
      (primitive) =>
        primitive.extensions?.KHR_gaussian_splatting?.extensions?.[SPZ_EXTENSION]?.bufferView,
    )
    .find((found) => found !== undefined);
  const bufferView = index === undefined ? undefined : json.bufferViews?.[index];
  if (!bufferView) throw new Error("The scan's tile holds no SPZ splat data.");
  const start = bufferView.byteOffset ?? 0;
  return bin.slice(start, start + bufferView.byteLength);
}
