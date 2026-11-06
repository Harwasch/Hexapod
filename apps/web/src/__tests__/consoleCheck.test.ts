import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { Matrix4, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { psnr, referenceCamera, rgbFloats, type ReferenceCamera } from "@/dev/consoleCheck";

const spec: ReferenceCamera = {
  viewmat: [
    [0, 0, 1, -3],
    [0, 1, 0, -2],
    [-1, 0, 0, 1],
    [0, 0, 0, 1],
  ],
  K: [
    [500, 12, 300],
    [0, 700, 210],
    [0, 0, 1],
  ],
  width: 800,
  height: 600,
  near: 0.01,
  far: 1000,
};

describe("console check camera", () => {
  it("preserves a translated/rotated solve and the right/down/forward camera axes", () => {
    const camera = referenceCamera(spec);
    expect(camera.position.toArray()).toEqual([1, 2, 3]);
    const gsView = new Matrix4().fromArray(spec.viewmat.flat()).transpose();
    for (const point of [new Vector3(-2, 3, 4), new Vector3(-10, -4, 1)]) {
      const expected = point.clone().applyMatrix4(gsView);
      const actual = point.clone().applyMatrix4(camera.matrixWorldInverse);
      expect(actual.x).toBeCloseTo(expected.x, 12);
      expect(actual.y).toBeCloseTo(-expected.y, 12);
      expect(actual.z).toBeCloseTo(-expected.z, 12);
      const ndc = point.clone().project(camera);
      expect(((ndc.x + 1) * spec.width) / 2).toBeCloseTo(
        (500 * expected.x) / expected.z + (12 * expected.y) / expected.z + 300,
        10,
      );
      expect(((1 - ndc.y) * spec.height) / 2).toBeCloseTo(
        (700 * expected.y) / expected.z + 210,
        10,
      );
    }
    const opticalAxis = new Vector3(-4, 2, 3).project(camera);
    expect((opticalAxis.x + 1) * 400).toBeCloseTo(300);
    expect((1 - opticalAxis.y) * 300).toBeCloseTo(210);
  });
  it("sets the exact near/far projection and rejects malformed intrinsics", () => {
    const camera = referenceCamera(spec);
    const world = (z: number) =>
      new Vector3(0, 0, -z).applyMatrix4(camera.matrixWorld).project(camera).z;
    expect(world(0.01)).toBeCloseTo(-1, 10);
    expect(world(1000)).toBeCloseTo(1, 10);
    expect(() => referenceCamera({ ...spec, width: 0 })).toThrow();
    expect(() =>
      referenceCamera({
        ...spec,
        K: [
          [0, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      }),
    ).toThrow();
  });
});

describe("console check PSNR", () => {
  it("uses every RGB sample, not alpha, with no mask or colour transform", () => {
    const black = rgbFloats(new Uint8ClampedArray([0, 0, 0, 0, 0, 0, 0, 255]));
    const redPixel = rgbFloats(new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 0, 0]));
    expect(psnr(black, redPixel)).toBeCloseTo(10 * Math.log10(6), 12);
    expect(psnr([0, 0, 0], [0.1, 0.1, 0.1])).toBeCloseTo(20, 12);
    expect(psnr(black, black)).toBe(Infinity);
    expect(() => psnr([0], [])).toThrow();
    expect(() => psnr([0, 0, 0], [NaN, 0, 0])).toThrow();
  });
  it("keeps the harness outside production HTML inputs and public assets", () => {
    const config = readFileSync(resolve("vite.config.ts"), "utf8");
    expect(config).not.toContain("console-check.html");
    expect(existsSync(resolve("console-check.html"))).toBe(true);
    expect(existsSync(resolve("public/console-check.html"))).toBe(false);
  });
});
