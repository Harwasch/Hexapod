/**
 * The Inspector's one line on how well the shown model was captured, from its root's
 * `extras.confidence.summary` (hexapod.cellgrid v1, written by the pipeline's confidence map).
 * Small and parse-free on purpose: the Inspector is on the globe's first load, the grid decoder
 * (`lib/cellGrid.ts`) is not.
 */

const pct = (share: number): string => `${String(Math.round(share * 100))}%`;

/** What the summary says, or why there is none; `extras` undefined: the model is not loaded. */
export function confidenceFact(extras: unknown): string {
  if (extras === undefined) return "Known once the model has loaded";
  const confidence =
    typeof extras === "object" && extras !== null
      ? (extras as { confidence?: Record<string, unknown> }).confidence
      : undefined;
  if (confidence?.format !== "hexapod.cellgrid") {
    return "Not measured: this model came without capture evidence (a splat upload, or a run from before confidence maps)";
  }
  const policy = typeof confidence.policy === "string" ? confidence.policy : "weak-v1";
  const summary = (confidence.summary ?? {}) as {
    scored?: number;
    weak?: number;
    weakShare?: number;
    reasons?: Record<string, number>;
  };
  const scored = summary.scored ?? 0;
  if (scored <= 0) return `Nothing scored · ${policy}`;
  const share = summary.weakShare ?? (summary.weak ?? 0) / scored;
  if (share <= 0) return `Nothing weak · ${policy}`;
  const reasons = Object.entries(summary.reasons ?? {})
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, 3)
    .map(([name, n]) => `${name} ${pct(n / scored)}`);
  return `${pct(share)} weak${reasons.length > 0 ? `: ${reasons.join(", ")}` : ""} · ${policy}`;
}
