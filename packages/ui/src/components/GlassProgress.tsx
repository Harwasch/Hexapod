import { clsx } from "clsx";

export type GlassProgressTone = "accent" | "warn" | "run" | "idle" | "danger" | "success";

export interface GlassProgressProps {
  /** Completed amount, in whatever unit `max` is in. Clamped into `[0, max]`. */
  value: number;
  /** Total. Defaults to 100, so a percentage needs no `max`. */
  max?: number;
  /** Accessible name. Required unless the bar is named by `aria-labelledby`. */
  label?: string;
  "aria-labelledby"?: string;
  /** What to announce instead of the bare number ("2.1 GB of 4.4 GB"). */
  valueText?: string;
  tone?: GlassProgressTone;
  /**
   * Unknown total: a sweeping bar and no `aria-valuenow`, which is what ARIA asks
   * for when progress cannot be quantified.
   */
  indeterminate?: boolean;
  /** A second mark over the track, in the same unit as `value` (e.g. where a schedule expects to be). */
  expected?: number;
  /** Tooltip for the `expected` mark. */
  expectedTitle?: string;
  className?: string;
}

function clamp(value: number, max: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, 0), max);
}

/**
 * One progress bar for the whole app: uploads, jobs, plan completion.
 *
 * The fill colour is a CSS custom property (`--progress-tone`), so a surface with its
 * own palette can re-skin the bar by setting it, rather than by hand-rolling a fourth
 * `role="progressbar"`.
 */
export function GlassProgress({
  value,
  max = 100,
  label,
  valueText,
  tone = "accent",
  indeterminate = false,
  expected,
  expectedTitle,
  className,
  ...aria
}: GlassProgressProps) {
  const total = max > 0 ? max : 100;
  const current = clamp(value, total);
  const pct = (current / total) * 100;
  const expectedPct = expected === undefined ? null : (clamp(expected, total) / total) * 100;
  return (
    <div
      className={clsx(
        "glass-progress",
        tone !== "accent" && `glass-progress--${tone}`,
        indeterminate && "glass-progress--indeterminate",
        expectedPct !== null && "glass-progress--marked",
        className,
      )}
      role="progressbar"
      aria-label={label}
      aria-labelledby={aria["aria-labelledby"]}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={indeterminate ? undefined : current}
      aria-valuetext={valueText}
    >
      <div
        className="glass-progress__bar"
        style={indeterminate ? undefined : { width: `${pct}%` }}
      />
      {expectedPct !== null && (
        <div
          className="glass-progress__marker"
          style={{ left: `${expectedPct}%` }}
          title={expectedTitle}
        />
      )}
    </div>
  );
}
