import { describe, expect, it } from "vitest";
import {
  articulationPixels,
  decodeArticulation,
  evaluateArticulation,
  orderedArticulationBinding,
  parseArticulationHeader,
  rotatePoint,
  type ArticulationHeader,
  type Quat,
} from "@/cesium/scanView/livingArticulation";
import { useAsCaptured } from "@/state/asCaptured";
import { effectiveLivingLab, useLivingLab } from "@/state/livingLab";
import { playcanvasModifierGlsl } from "@/cesium/scanView/playcanvasBackend";

const header: ArticulationHeader = {
  format: "hexapod.living-articulation",
  version: 1,
  motionEvidence: "simulated",
  nodes: [
    { parent: -1, position: [0, 0, 0] },
    { parent: 0, position: [1, 0, 0] },
    { parent: 1, position: [2, 0, 0] },
  ],
  frequenciesHz: [0],
  referenceBearingDeg: 90,
  motionBinary: "motion.bin",
  leafBinary: "leaves.bin",
  bindingBinary: "bindings.bin",
  leafCount: 1,
  tiles: { fixture: { offset: 0, count: 2 } },
};
function fixture() {
  const motion = new Float32Array(18);
  motion[10] = Math.PI / 2; // node 1, z DC rotation about root joint
  const leaf = new Float32Array([1.5, 0, 0, 0, 1, 0, 3, Math.PI / 2, 0.08]);
  const binding = new Uint32Array([3, 2, 65535, 1, 2, 1, 32768, 0]);
  return decodeArticulation(header, motion.buffer, leaf.buffer, binding.buffer);
}

describe("articulated branch spectra", () => {
  it("composes proximal joint rotations and preserves rigid distances/covariance axes", () => {
    const a = fixture(),
      poses = articulationPixels(3),
      leaves = articulationPixels(1);
    evaluateArticulation(a, 0, 1, 90, poses, leaves);
    const q = Array.from(poses.slice(24, 28)) as Quat;
    const moved = rotatePoint(q, [2, 0, 0]);
    expect(moved[0]).toBeCloseTo(0, 5);
    expect(moved[1]).toBeCloseTo(2, 5);
    expect(Math.hypot(...rotatePoint(q, [0, 0.2, 0]))).toBeCloseTo(0.2, 6);
    expect(poses[8 + 3]).toBe(1); // root pinned
    expect(Array.from(poses.slice(28, 31))).toEqual([0, 0, 0]);
    expect(leaves[13]).not.toBe(0); // leaf has actual rotation, not only colour shimmer
    expect(leaves[8]).toBe(1.5); // pivot immutable
  });
  it("is identity at zero strength and repeats the ten-second flutter period", () => {
    const a = fixture(),
      p = articulationPixels(3),
      l = articulationPixels(1),
      later = articulationPixels(1);
    evaluateArticulation(a, 0.23, 1, 60, p, l);
    evaluateArticulation(a, 10.23, 1, 60, p, later);
    for (let i = 0; i < 16; i++) expect(later[i]).toBeCloseTo(l[i] ?? 0, 6);
    evaluateArticulation(a, 0, 0, 0, p, l);
    for (let i = 0; i < 4; i++)
      expect(Array.from(p.slice(i * 8, i * 8 + 8))).toEqual([0, 0, 0, 1, 0, 0, 0, 0]);
    expect(l.slice(12, 15).every((v) => v === 0)).toBe(true);
    expect(l[15]).toBe(1);
  });
  it("matches checksums and reorders bindings exactly once, leaving unknown tiles static", () => {
    const a = fixture(),
      out = new Uint32Array(12).fill(99);
    expect(orderedArticulationBinding(a, "fixture", new Uint32Array([1, 0]), out)).toBe(true);
    expect(Array.from(out)).toEqual([2, 1, 32768, 0, 3, 2, 65535, 1, 0, 0, 0, 0]);
    expect(orderedArticulationBinding(a, "other", new Uint32Array([1, 0]), out)).toBe(false);
    expect(out.every((v) => v === 0)).toBe(true);
    expect(playcanvasModifierGlsl(false, false, false, false, false, false, false, true)).toContain(
      "writeHexapodArticulation(loadHexapodArticulation())",
    );
  });
  it("rejects invalid hierarchies, truncated payloads, and invalid node bindings", () => {
    expect(() =>
      parseArticulationHeader({ ...header, nodes: [{ parent: 0, position: [0, 0, 0] }] }),
    ).toThrow();
    const a = fixture();
    expect(() =>
      decodeArticulation(
        header,
        new ArrayBuffer(1),
        a.leaves.buffer as ArrayBuffer,
        a.bindings.buffer as ArrayBuffer,
      ),
    ).toThrow(/length/);
    const binding = new Uint32Array(a.bindings);
    binding[0] = 4;
    expect(() =>
      decodeArticulation(
        header,
        a.coefficients.buffer as ArrayBuffer,
        a.leaves.buffer as ArrayBuffer,
        binding.buffer,
      ),
    ).toThrow(/binding/);
  });
  it("respects As captured for the articulated arm", () => {
    useLivingLab.getState().set({ motion: "articulated", bench: false });
    useAsCaptured.setState({ on: true });
    expect(effectiveLivingLab()).toBeNull();
    useAsCaptured.setState({ on: false });
    expect(effectiveLivingLab()?.motion).toBe("articulated");
    useLivingLab.getState().set({ motion: "off" });
  });
});
