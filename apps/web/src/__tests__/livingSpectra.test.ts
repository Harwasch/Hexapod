import { describe, expect, it } from "vitest";
import {
  decodeSpectra,
  evaluateSpectra,
  parseSpectraHeader,
  type SpectraHeader,
} from "@/cesium/scanView/livingSpectra";
import { useAsCaptured } from "@/state/asCaptured";
import { effectiveLivingLab, useLivingLab } from "@/state/livingLab";

const header: SpectraHeader = {
  format: "hexapod.living-spectra",
  version: 1,
  encoding: "float16-le",
  motionEvidence: "simulated",
  binary: "spectra.bin",
  origin: [0, 0, 0],
  size: [1, 1, 1],
  resolution: 2,
  occupied: 1,
  frequenciesHz: [1],
  referenceBearingDeg: 90,
  coefficientOffset: 4,
  byteLength: 16,
};
function fixture() {
  const binary = new ArrayBuffer(16);
  const view = new DataView(binary);
  view.setUint32(0, 7, true);
  view.setUint16(4, 0x3c00, true); // x real = 1
  view.setUint16(10, 0xbc00, true); // y imag = -1 (positive sine)
  return { binary, view };
}

describe("living spectra", () => {
  it("decodes x-fastest lookup and plays real cosine, imaginary sine, direction and calm", () => {
    const s = decodeSpectra(header, fixture().binary);
    expect(s.index[7]).toBe(1);
    expect(s.index[0]).toBe(0);
    const out = new Float32Array(4);
    evaluateSpectra(s, 0.25, 2, 90, out);
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(2);
    evaluateSpectra(s, 0, 1, 0, out);
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(1);
    evaluateSpectra(s, 1.25, 0, 60, out);
    expect(
      Array.from(out)
        .slice(0, 3)
        .every((v) => v === 0),
    ).toBe(true);
  });
  it("rejects malformed headers and unsafe or non-finite payloads", () => {
    for (const patch of [
      { occupied: 50_001 },
      { resolution: 129 },
      { frequenciesHz: [1, 1] },
      { size: [0, 1, 1] },
      { version: 2 },
    ])
      expect(() => parseSpectraHeader({ ...header, ...patch })).toThrow();
    expect(() => decodeSpectra(header, new ArrayBuffer(4))).toThrow(/length/);
    const { binary, view } = fixture();
    view.setUint32(0, 8, true);
    expect(() => decodeSpectra(header, binary)).toThrow(/voxel/);
    view.setUint32(0, 7, true);
    view.setUint16(4, 0x7c00, true);
    expect(() => decodeSpectra(header, binary)).toThrow(/Non-finite/);
  });
  it("hides spectra under As captured including benchmark mode", () => {
    useLivingLab.getState().set({ motion: "spectra", bench: false });
    useAsCaptured.setState({ on: false });
    expect(effectiveLivingLab()?.motion).toBe("spectra");
    useAsCaptured.setState({ on: true });
    expect(effectiveLivingLab()).toBeNull();
    useLivingLab.getState().set({ bench: true });
    expect(effectiveLivingLab()?.motion).toBe("off");
    useLivingLab.getState().set({ motion: "off", bench: false });
    useAsCaptured.setState({ on: false });
  });
});
