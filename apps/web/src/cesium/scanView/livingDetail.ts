/** Conservative geometric permission, not semantic leaf identification. */
export interface LivingDetail {
  vegetation: boolean;
  fittedRig: boolean;
  separableCrownShare: number;
  attachedCrownShare: number;
  smallSplatShare: number;
  /** Actual piece-specific leaf bindings are required, never 12 cm spatial packets. */
  leafBindings: boolean;
  sampled?: boolean;
}
export const LEAF_DETAIL_THRESHOLD = 0.65;
export function leafResolutionScore(d: LivingDetail): number {
  const shares = [d.separableCrownShare, d.attachedCrownShare, d.smallSplatShare];
  return shares.every((x) => Number.isFinite(x) && x >= 0 && x <= 1) ? Math.min(...shares) : 0;
}
export function motionTier(d: LivingDetail): "off" | "A" | "B" | "C" {
  if (!d.vegetation) return "off";
  if (
    d.sampled !== true &&
    d.leafBindings &&
    d.fittedRig &&
    leafResolutionScore(d) >= LEAF_DETAIL_THRESHOLD
  )
    return "C";
  return d.fittedRig ? "B" : "A";
}
