import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { spzFromGlb } from "@/view/glb";

/** A GLB with one primitive whose SPZ lives at `offset` in the binary chunk. */
function glb(spz: Uint8Array, offset = 0): ArrayBuffer {
  const json = new TextEncoder().encode(
    JSON.stringify({
      meshes: [
        {
          primitives: [
            {
              extensions: {
                KHR_gaussian_splatting: {
                  extensions: { KHR_gaussian_splatting_compression_spz_2: { bufferView: 1 } },
                },
              },
            },
          ],
        },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: offset },
        { buffer: 0, byteOffset: offset, byteLength: spz.length },
      ],
    }),
  );
  const jsonLength = Math.ceil(json.length / 4) * 4;
  const binLength = Math.ceil((offset + spz.length) / 4) * 4;
  const out = new ArrayBuffer(12 + 8 + jsonLength + 8 + binLength);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(json, 20);
  const bin = 20 + jsonLength;
  view.setUint32(bin, binLength, true);
  view.setUint32(bin + 4, 0x004e4942, true);
  bytes.set(spz, bin + 8 + offset);
  return out;
}

describe("the SPZ inside a splat tile", () => {
  it("is found through the extension's buffer view, wherever it sits", () => {
    const spz = new Uint8Array([1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(spzFromGlb(glb(spz, 12)))).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("refuses something that is not a glTF binary", () => {
    expect(() => spzFromGlb(new ArrayBuffer(32))).toThrow("not a glTF binary");
  });

  it("reads the pipeline's own committed tile as gzip-compressed SPZ", () => {
    // vitest runs from apps/web; the fixture is the repository's committed tile.
    const path = resolve(process.cwd(), "../../data/tiles/synthetic-tree/splat/splat.glb");
    const file = readFileSync(path);
    const spz = spzFromGlb(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
    // SPZ is a gzip stream: the viewer hands exactly this to Spark.
    expect([spz[0], spz[1]]).toEqual([0x1f, 0x8b]);
    expect(spz.length).toBeGreaterThan(1000);
  });
});
