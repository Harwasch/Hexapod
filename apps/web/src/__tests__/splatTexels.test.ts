/**
 * The texel arithmetic, against the shader that actually reads it.
 *
 * `PrimitiveGaussianSplatVS.glsl:144-166` addresses with shifts and a mask; every test here
 * checks that against the same thing written with division and modulo, so a transposed shift
 * would have to be made twice, differently, to slip through.
 *
 * The addressing is exercised at **both** row widths that matter. `maximumTextureSize` is 8192
 * under the SwiftShader used for headless runs here and 16384 on a typical desktop GPU, and a
 * deformer that assumed either one would write every splat into the wrong texel on the other
 * machine while still producing a picture.
 */

import { describe, expect, it } from "vitest";

import {
  bitsToFloat32,
  covarianceTexel,
  createStagingBuffer,
  float32Bits,
  fullRowRange,
  positionTexel,
  positionWordOffset,
  rowRangeForMovingNodes,
  rowRangeWordCount,
  rowSlice,
  splatRow,
  splatTextureLayout,
  writeSplatPositions,
  type SplatTextureLayout,
} from "@/cesium/splatTexels";

/** The two maximum texture sizes this runs on, and the mask/shift each implies. */
const DEVICES = [
  { maximumTextureSize: 8192, rowShift: 12, rowMask: 4095 },
  { maximumTextureSize: 16384, rowShift: 13, rowMask: 8191 },
] as const;

/** `layout` and `numSplats` for a tree-sized and a site-sized capture on each device. */
function layoutsUnderTest(): { label: string; layout: SplatTextureLayout }[] {
  const cases: { label: string; layout: SplatTextureLayout }[] = [];
  for (const device of DEVICES) {
    for (const numSplats of [2000, 57410, 250_000]) {
      cases.push({
        label: `${device.maximumTextureSize}px / ${numSplats} splats`,
        layout: splatTextureLayout(numSplats, device.rowMask, device.rowShift),
      });
    }
  }
  return cases;
}

describe("splatTextureLayout", () => {
  it.each(DEVICES)(
    "derives the texture shape from the primitive's own mask and shift ($maximumTextureSize)",
    ({ maximumTextureSize, rowMask, rowShift }) => {
      const layout = splatTextureLayout(100_000, rowMask, rowShift);
      expect(layout.width).toBe(maximumTextureSize);
      expect(layout.splatsPerRow).toBe(maximumTextureSize / 2);
      expect(layout.height).toBe(Math.ceil(100_000 / (maximumTextureSize / 2)));
    },
  );

  it("refuses a mask and shift that disagree, rather than addressing into nowhere", () => {
    // The exact bug S0 warned about: 16384's shift with 8192's mask.
    expect(() => splatTextureLayout(2000, 4095, 13)).toThrow(/rowMask/);
    expect(() => splatTextureLayout(2000, 0, 0)).toThrow(/rowShift/);
    expect(() => splatTextureLayout(0, 4095, 12)).toThrow(/numSplats/);
  });
});

describe("texel addressing matches the shader formula", () => {
  it.each(layoutsUnderTest())("$label", ({ layout }) => {
    const { splatsPerRow, width, numSplats } = layout;
    const probes = [
      0,
      1,
      splatsPerRow - 1,
      splatsPerRow,
      splatsPerRow + 1,
      2 * splatsPerRow - 1,
      numSplats - 1,
      Math.floor(numSplats / 3),
      Math.floor(numSplats / 2),
    ].filter((i) => i >= 0 && i < numSplats);

    for (const i of probes) {
      // The shader does `(texIdx & rowMask) << 1` and `texIdx >> rowShift`; this is the same
      // addressing expressed without bit operations.
      const expectedX = 2 * (i % splatsPerRow);
      const expectedY = Math.floor(i / splatsPerRow);
      expect(positionTexel(i, layout)).toEqual([expectedX, expectedY]);
      expect(covarianceTexel(i, layout)).toEqual([expectedX + 1, expectedY]);
      expect(splatRow(i, layout)).toBe(expectedY);
      expect(positionWordOffset(i, layout)).toBe((expectedY * width + expectedX) * 4);
      // Adjacent texels, one row, never straddling a row boundary.
      expect(covarianceTexel(i, layout)[1]).toBe(positionTexel(i, layout)[1]);
    }
  });

  it("gives every splat its own pair of texels", () => {
    const layout = splatTextureLayout(9000, 4095, 12);
    const seen = new Set<number>();
    for (let i = 0; i < layout.numSplats; i += 1) {
      const offset = positionWordOffset(i, layout);
      expect(seen.has(offset)).toBe(false);
      seen.add(offset);
      expect(offset + 8).toBeLessThanOrEqual(layout.width * layout.height * 4);
    }
  });
});

describe("float32 ↔ uint32 packing", () => {
  const values = [
    0,
    -0,
    1,
    -1,
    0.1,
    -1234.5678,
    7.612548828125,
    1 / 4096,
    -(1 / 4096),
    3.4028234663852886e38,
    1.401298464324817e-45,
    Number.POSITIVE_INFINITY,
  ];

  it("round-trips exactly through raw bits", () => {
    for (const value of values) {
      expect(bitsToFloat32(float32Bits(value))).toBe(Math.fround(value));
    }
    expect(Object.is(bitsToFloat32(float32Bits(-0)), -0)).toBe(true);
  });

  it("round-trips exactly through the staging buffer", () => {
    const layout = splatTextureLayout(4, 4095, 12);
    const staging = createStagingBuffer(layout, new Uint32Array(0));
    const positions = Float32Array.from([
      ...[values[0] ?? 0, values[1] ?? 0, values[2] ?? 0],
      ...[values[3] ?? 0, values[4] ?? 0, values[5] ?? 0],
      ...[values[6] ?? 0, values[7] ?? 0, values[8] ?? 0],
      ...[values[9] ?? 0, values[10] ?? 0, values[11] ?? 0],
    ]);
    writeSplatPositions(staging, positions);
    for (let i = 0; i < 4; i += 1) {
      const offset = positionWordOffset(i, layout);
      for (let c = 0; c < 3; c += 1) {
        expect(bitsToFloat32(staging.words[offset + c] ?? 0)).toBe(positions[i * 3 + c]);
      }
    }
  });
});

describe("the covariance and colour lanes are never touched", () => {
  // The whole approach rests on this. Position texels sit at even columns and covariance at
  // odd, `copyFrom` takes a contiguous rectangle, and we supply the engine's own packed bytes
  // for everything we do not own. If a position write ever reached an odd column, the geometry
  // and colour of the splat next to it would change with no error anywhere.
  it.each(DEVICES)("at $maximumTextureSize", ({ rowMask, rowShift }) => {
    const numSplats = 5000;
    const layout = splatTextureLayout(numSplats, rowMask, rowShift);
    const packed = new Uint32Array(layout.width * layout.height * 4);
    for (let w = 0; w < packed.length; w += 1) packed[w] = (w * 2654435761) >>> 0;
    const staging = createStagingBuffer(layout, packed);

    const positions = new Float32Array(numSplats * 3);
    for (let i = 0; i < positions.length; i += 1) positions[i] = Math.sin(i) * 1000;
    writeSplatPositions(staging, positions);

    for (let y = 0; y < layout.height; y += 1) {
      for (let x = 0; x < layout.width; x += 1) {
        const base = (y * layout.width + x) * 4;
        const splatIndex = y * layout.splatsPerRow + (x >> 1);
        // Ours: the R, G and B words of an even column belonging to a real splat. Everything
        // else — the position texel's alpha, the whole odd column, and the padding past the
        // last splat — is the engine's and must come back byte-identical.
        const ours = x % 2 === 0 && splatIndex < numSplats;
        for (let c = 0; c < 4; c += 1) {
          const after = staging.words[base + c] ?? 0;
          if (ours && c < 3) {
            expect(after).toBe(float32Bits(positions[splatIndex * 3 + c] ?? 0));
          } else {
            expect(after).toBe(packed[base + c] ?? 0);
          }
        }
      }
    }
  });
});

describe("staging a buffer the engine packed short", () => {
  // Measured on the 57,410-splat capture: 466,944 words supplied against 491,520 the texture
  // needs. Staging the short buffer directly renders the last row as garbage.
  it("allocates the whole texture and zero-pads the tail", () => {
    const layout = splatTextureLayout(57_410, 4095, 12);
    const needed = layout.width * layout.height * 4;
    const packed = new Uint32Array(466_944).fill(0xabcdef01);
    expect(packed.length).toBeLessThan(needed);

    const staging = createStagingBuffer(layout, packed);
    expect(staging.words.length).toBe(needed);
    expect(staging.floats.length).toBe(needed);
    expect(staging.words[packed.length - 1]).toBe(0xabcdef01);
    expect(staging.words[packed.length]).toBe(0);
    expect(staging.words[needed - 1]).toBe(0);
  });

  it("clamps a buffer longer than the texture instead of overflowing", () => {
    const layout = splatTextureLayout(100, 4095, 12);
    const needed = layout.width * layout.height * 4;
    const packed = new Uint32Array(needed + 1024).fill(7);
    const staging = createStagingBuffer(layout, packed);
    expect(staging.words.length).toBe(needed);
    expect(staging.words[needed - 1]).toBe(7);
  });
});

describe("row ranges cover every moved splat", () => {
  const layout = splatTextureLayout(12_000, 4095, 12); // 3 rows of 4096
  const assignment = new Uint16Array(layout.numSplats);
  for (let i = 0; i < assignment.length; i += 1) assignment[i] = i % 5;

  it("returns undefined when nothing moves, so a still scene uploads nothing", () => {
    expect(rowRangeForMovingNodes(assignment, new Uint8Array(5), layout)).toBeUndefined();
  });

  it("returns a band containing every splat on a moving node", () => {
    for (const movingNode of [0, 1, 2, 3, 4]) {
      const moves = new Uint8Array(5);
      moves[movingNode] = 1;
      const range = rowRangeForMovingNodes(assignment, moves, layout);
      expect(range).toBeDefined();
      if (range === undefined) continue;
      for (let i = 0; i < assignment.length; i += 1) {
        if (assignment[i] !== movingNode) continue;
        const row = splatRow(i, layout);
        expect(row).toBeGreaterThanOrEqual(range.firstRow);
        expect(row).toBeLessThan(range.firstRow + range.rowCount);
      }
    }
  });

  it("narrows to the rows a node actually occupies", () => {
    // One node owning only splats in the middle row must not drag the whole texture along.
    const sparse = new Uint16Array(layout.numSplats);
    for (let i = 4096; i < 8192; i += 1) sparse[i] = 1;
    const moves = Uint8Array.from([0, 1]);
    expect(rowRangeForMovingNodes(sparse, moves, layout)).toEqual({ firstRow: 1, rowCount: 1 });
  });

  it("slices exactly the words of its band", () => {
    const staging = createStagingBuffer(layout, new Uint32Array(0));
    const range = { firstRow: 1, rowCount: 2 };
    const slice = rowSlice(staging, range);
    expect(slice.length).toBe(rowRangeWordCount(layout, range));
    expect(slice.byteOffset).toBe(layout.width * 4 * 4);
    expect(rowSlice(staging, fullRowRange(layout)).length).toBe(staging.words.length);
  });
});
