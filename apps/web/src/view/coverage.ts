/**
 * The quality bar's coverage cloud: `coverage_enu.ply` from the pipeline's `place` stage.
 *
 * A sample of the splat's gaussian centres, each coloured by the tier the quality stage
 * put it in -- green kept, amber context, grey dropped -- and the camera centres in capture
 * order (tier 3), all east/north/up in the same frame as the published splat. The file is
 * a fixed, tiny binary PLY (`tools/pipeline/quality.py` `write_coverage`), so it is read
 * with a DataView here rather than with a PLY loader.
 */

export const TIER_DROP = 0;
export const TIER_CONTEXT = 1;
export const TIER_KEEP = 2;
export const TIER_CAMERA = 3;

export interface Coverage {
  /** x, y, z per point, cameras excluded. */
  positions: Float32Array;
  /** r, g, b in 0..1 per point, the file's own tier colours. */
  colors: Float32Array;
  /** x, y, z per camera, in the order the frames were taken. */
  cameraPath: Float32Array;
  counts: { keep: number; context: number; drop: number; cameras: number };
}

const SIZES: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

export function parseCoverage(buffer: ArrayBuffer): Coverage {
  const bytes = new Uint8Array(buffer);
  const marker = "end_header\n";
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 4096)));
  const end = head.indexOf(marker);
  if (!head.startsWith("ply") || end < 0) throw new Error("The coverage file is not a PLY.");
  if (!head.includes("format binary_little_endian")) {
    throw new Error("The coverage file is not binary little-endian.");
  }
  let count = 0;
  let stride = 0;
  const offsets: Record<string, number> = {};
  const types: Record<string, string> = {};
  for (const line of head.slice(0, end).split("\n")) {
    const words = line.trim().split(/\s+/);
    if (words[0] === "element" && words[1] === "vertex") count = Number(words[2]);
    if (words[0] === "property" && words.length === 3) {
      const [, type = "", name = ""] = words;
      const size = SIZES[type];
      if (size === undefined) throw new Error(`The coverage file has a ${type} property.`);
      offsets[name] = stride;
      types[name] = type;
      stride += size;
    }
  }
  for (const name of ["x", "y", "z", "red", "green", "blue", "tier"]) {
    if (offsets[name] === undefined) throw new Error(`The coverage file has no ${name}.`);
  }
  if (types.x !== "float" || types.y !== "float" || types.z !== "float") {
    throw new Error("The coverage file's positions are not floats.");
  }
  const body = end + marker.length;
  if (body + count * stride > bytes.length) throw new Error("The coverage file is truncated.");
  const view = new DataView(buffer, body);
  const at = (index: number, name: string): number => index * stride + (offsets[name] ?? 0);

  const counts = { keep: 0, context: 0, drop: 0, cameras: 0 };
  for (let i = 0; i < count; i += 1) {
    const tier = view.getUint8(at(i, "tier"));
    if (tier === TIER_CAMERA) counts.cameras += 1;
    else if (tier === TIER_KEEP) counts.keep += 1;
    else if (tier === TIER_CONTEXT) counts.context += 1;
    else counts.drop += 1;
  }
  const points = count - counts.cameras;
  const positions = new Float32Array(points * 3);
  const colors = new Float32Array(points * 3);
  const cameraPath = new Float32Array(counts.cameras * 3);
  let p = 0;
  let c = 0;
  for (let i = 0; i < count; i += 1) {
    const x = view.getFloat32(at(i, "x"), true);
    const y = view.getFloat32(at(i, "y"), true);
    const z = view.getFloat32(at(i, "z"), true);
    if (view.getUint8(at(i, "tier")) === TIER_CAMERA) {
      cameraPath.set([x, y, z], c * 3);
      c += 1;
      continue;
    }
    positions.set([x, y, z], p * 3);
    colors.set(
      [
        view.getUint8(at(i, "red")) / 255,
        view.getUint8(at(i, "green")) / 255,
        view.getUint8(at(i, "blue")) / 255,
      ],
      p * 3,
    );
    p += 1;
  }
  return { positions, colors, cameraPath, counts };
}
