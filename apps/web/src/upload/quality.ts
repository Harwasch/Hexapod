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
  if (quality.keepPct === null) return "none of the circled area reached high quality";
  return `${String(Math.round(quality.keepPct))}% of the circled area reached high quality`;
}

function psnr(quality: CaptureQuality): string | null {
  return quality.heldOutPsnr === null ? null : `held-out ${quality.heldOutPsnr.toFixed(1)} dB`;
}

/** "62% of the circled area reached high quality · held-out 23.0 dB" */
export function forecast(quality: CaptureQuality): string {
  return [share(quality), psnr(quality)].filter(Boolean).join(" · ");
}

/** "Refined · 71% of the circled area reached high quality · held-out 25.3 dB · Strict bar" */
export function summary(quality: CaptureQuality): string {
  const bar = BAR_NAME[quality.barApplied] ?? quality.barApplied;
  const fellBack = quality.barApplied !== quality.bar ? " (the bar asked for kept too little)" : "";
  return ["Refined", share(quality), psnr(quality), `${bar} bar${fellBack}`]
    .filter(Boolean)
    .join(" · ");
}
