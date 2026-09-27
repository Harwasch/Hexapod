/**
 * What the phone says about a finished run's quality bar (apps/api `CaptureQuality`).
 *
 * After a preview it is a forecast -- how much of the region the cameras were pointed at
 * the capture supports at high quality, and the held-out PSNR the short training reached
 * -- because that is what decides whether Refine is worth the GPU time. After a refine it
 * is the result.
 */
import type { CaptureQuality, Job } from "@twin/contracts";

const BAR_NAME: Record<string, string> = {
  strict: "Strict",
  balanced: "Balanced",
  everything: "Everything",
};

/** The verdict that describes `job`, or null: a capture's verdict is its latest run's. */
export function qualityOf(
  quality: CaptureQuality | null | undefined,
  job: Pick<Job, "id"> | undefined,
): CaptureQuality | null {
  return quality && quality.jobId === job?.id ? quality : null;
}

function share(quality: CaptureQuality): string {
  if (quality.keepPct === null) return "nothing met the high-quality bar";
  return `${String(Math.round(quality.keepPct))}% of the scene met the high-quality bar`;
}

function psnr(quality: CaptureQuality): string | null {
  return quality.heldOutPsnr === null ? null : `held-out ${quality.heldOutPsnr.toFixed(1)} dB`;
}

/** "62% of the scene met the high-quality bar · held-out 23.0 dB" */
export function forecast(quality: CaptureQuality): string {
  return [share(quality), psnr(quality)].filter(Boolean).join(" · ");
}

/** "Refined · 71% of the scene met the high-quality bar · held-out 25.3 dB · Strict bar" */
export function summary(quality: CaptureQuality): string {
  const bar = BAR_NAME[quality.barApplied] ?? quality.barApplied;
  const fellBack = quality.barApplied !== quality.bar ? " (the bar asked for kept too little)" : "";
  return ["Refined", share(quality), psnr(quality), `${bar} bar${fellBack}`]
    .filter(Boolean)
    .join(" · ");
}

/**
 * Below this share of the scene at the high-quality bar, a Refine is not offered as the
 * next step. The keep tier is decided by how many frames saw a point, over what spread
 * of angles, at what pixel size (tools/pipeline/quality.py): geometry of the capture,
 * which a longer training does not change. So the preview's share is close to the most
 * a Refine of the same frames can reach, and under a fifth of the scene is a capture to
 * redo rather than a result to polish. A judgement, not a measurement: revisit it once
 * previews and their Refines have been compared.
 */
export const REFINE_MIN_KEEP_PCT = 20;

/**
 * With less than this share kept *and* a tip that says views are missing, the same
 * holds: the tips name what the capture lacks, and Refine cannot add it.
 */
export const REFINE_WITH_GAPS_MIN_KEEP_PCT = 40;

/** The tips (quality.py `capture_tips`) that mean frames are missing, not training. */
export const COVERAGE_TIP_IDS: ReadonlySet<string> = new Set([
  "all-around",
  "from-above",
  "from-level",
  "more-frames",
  "more-angles",
  "get-closer",
]);

export interface RefineAdvice {
  /** Offer Refine as the primary action. */
  worthIt: boolean;
  /** Why not, in a sentence the phone shows above the tips; null when it is. */
  reason: string | null;
}

/**
 * Whether a finished preview is worth refining: the one place that rule lives. The phone
 * shows Refine as the primary action when it is, and otherwise the reason, the tips and
 * a quieter "Refine anyway".
 */
export function refineAdvice(
  quality: Pick<CaptureQuality, "mode" | "keepPct" | "tips">,
): RefineAdvice {
  if (quality.mode !== "preview") return { worthIt: false, reason: null };
  const keep = quality.keepPct;
  if (keep === null || keep < REFINE_MIN_KEEP_PCT) {
    const share = keep === null ? "None" : `Only ${String(Math.round(keep))}%`;
    return {
      worthIt: false,
      reason: `${share} of the scene met the high-quality bar. Refine trains longer; it can't add the views this capture is missing, so capturing again will do more.`,
    };
  }
  const gaps = quality.tips.filter((tip) => COVERAGE_TIP_IDS.has(tip.id));
  if (keep < REFINE_WITH_GAPS_MIN_KEEP_PCT && gaps.length > 0) {
    return {
      worthIt: false,
      reason: `${String(Math.round(keep))}% of the scene met the high-quality bar, and the capture is missing views (see Next time). Refine can't add them, so capturing again will do more.`,
    };
  }
  return { worthIt: true, reason: null };
}
