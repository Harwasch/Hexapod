import { describe, expect, it } from "vitest";
import { leafResolutionScore, motionTier, type LivingDetail } from "./livingDetail";
const d: LivingDetail = {
  vegetation: true,
  fittedRig: false,
  separableCrownShare: 0.9,
  attachedCrownShare: 0.9,
  smallSplatShare: 0.9,
  leafBindings: false,
};
describe("automatic scan motion tier", () => {
  it("requires vegetation, uses clumps without a rig, and branches with one", () => {
    expect(motionTier({ ...d, vegetation: false })).toBe("off");
    expect(motionTier(d)).toBe("A");
    expect(motionTier({ ...d, fittedRig: true })).toBe("B");
  });
  it("never calls spatial packets leaves and requires every measurement to pass", () => {
    expect(motionTier({ ...d, fittedRig: true, leafBindings: true })).toBe("C");
    expect(motionTier({ ...d, fittedRig: true, leafBindings: true, sampled: true })).toBe("B");
    for (const key of ["separableCrownShare", "attachedCrownShare", "smallSplatShare"] as const)
      expect(motionTier({ ...d, fittedRig: true, leafBindings: true, [key]: 0.64 })).toBe("B");
    expect(leafResolutionScore({ ...d, attachedCrownShare: NaN })).toBe(0);
  });
});
