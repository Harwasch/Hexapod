/** Pure presentation of measured training data. Missing measurements stay missing. */
import type {
  LiveState,
  LiveTrain,
  LiveTrainPlan,
  LiveTrainQuality,
  LiveTrainSnapshot,
  LiveTrainSeeds,
} from "@twin/contracts";

export const count = (n: number | null | undefined): string =>
  n == null ? "–" : Math.round(n).toLocaleString("en-US");
export const blockCountText = (n: number | null | undefined): string =>
  `${count(n)} ${n === 1 ? "block" : "blocks"}`;

/** A block advances the charts; only an attempt retry invalidates cached splats and scrubs. */
export function isNewAttempt(
  previous: LiveTrain | null | undefined,
  next: LiveTrain | null | undefined,
): boolean {
  if (!previous || !next) return false;
  if (next.planId != null && next.planId !== previous.planId) return true;
  return (
    next.run != null &&
    previous.run != null &&
    next.run !== previous.run &&
    (next.block ?? null) === (previous.block ?? null)
  );
}

export function qualityReadouts(
  snapshot: LiveTrainSnapshot | null | undefined,
  quality: LiveTrainQuality | null | undefined,
): string[] {
  // TODO(codex/feat-splat-quality-stack): prefer snapshot measurements when emitted.
  const values = [
    ["Far-field", snapshot?.farFieldShare ?? quality?.farFieldShare],
    ["Elongated", snapshot?.elongationShare ?? quality?.elongationShare],
  ] as const;
  return values.flatMap(([label, value]) =>
    value == null ? [] : [`${label} ${(value * 100).toFixed(1)}%`],
  );
}

export const compact = (n: number | null | undefined): string =>
  n == null
    ? "–"
    : n >= 1e6
      ? `${(n / 1e6).toFixed(2)}M`
      : n >= 1000
        ? `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`
        : count(n);
export function effectiveCap(plan?: LiveTrainPlan | null): number | null {
  const caps = [plan?.cap, plan?.capMax].filter((n): n is number => n != null);
  return caps.length ? Math.min(...caps) : null;
}
const endLabels = {
  preempted: "Pre-empted",
  oom: "Out of memory",
  failed: "Failed",
} as const;
export function phaseOf(
  step: number,
  plan?: LiveTrainPlan | null,
  gaussians?: number | null,
  ended?: LiveTrain["ended"],
): string {
  if (ended) return endLabels[ended.reason];
  const cap = effectiveCap(plan);
  if (
    cap != null &&
    gaussians != null &&
    gaussians >= cap &&
    plan?.refineStop != null &&
    step < plan.refineStop
  )
    return "Relocating";
  if (plan?.refineStart == null) return "–";
  if (step < plan.refineStart) return "Warm-up";
  return plan.refineStop != null && step >= plan.refineStop ? "Settling" : "Densifying";
}
export function lastStep(train?: LiveTrain | null): number {
  if (train?.ended) return train.ended.step;
  return Math.max(
    train?.series?.step?.at(-1) ?? 0,
    train?.evals?.at(-1)?.step ?? 0,
    train?.stop?.step ?? 0,
  );
}
export function pillText(state: LiveState): string {
  const train = state.train;
  if (train?.ended) return `${endLabels[train.ended.reason]} · step ${count(lastStep(train))}`;
  if (train?.stop) return "Finished · converged";
  if (state.status === "error" || state.status === "cancelled") return "Failed";
  const ceiling = train?.plan?.stepsMax;
  if (
    state.status === "complete" ||
    (ceiling != null && lastStep(train) >= ceiling) ||
    (state.stage && state.stage.stageId !== "train" && train)
  )
    return "Finished";
  const phase = phaseOf(lastStep(train), train?.plan, train?.series?.gs?.at(-1));
  if (phase === "Relocating") return "Relocating · at cap";
  return phase === "–" ? "Training" : `Training · ${phase.toLowerCase()}`;
}
export function ceilingText(plan?: LiveTrainPlan | null): string {
  return plan?.scheduleFactor == null ? "–" : `30k × ${String(plan.scheduleFactor)}`;
}
export function timeText(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  if (seconds < 60) return `${count(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 60
    ? `${String(minutes)} min`
    : `${String(Math.floor(minutes / 60))} h ${String(minutes % 60)} min`;
}
export function cost(
  seconds: number | null | undefined,
  rate: number | null | undefined,
): number | null {
  return seconds == null || rate == null ? null : (seconds / 3600) * rate;
}
export function money(value: number | null): string {
  return value == null ? "–" : `$${value.toFixed(2)}`;
}

export function stopRule(train?: LiveTrain | null): {
  line: string;
  checks: { label: string; value: string; ok: boolean }[];
  gain: number | null;
  fraction: number;
  threshold: number;
} {
  const plan = train?.plan,
    rule = plan?.rule;
  const step = lastStep(train),
    stop = plan?.refineStop;
  const evaluation = train?.evals?.findLast((e) => e.step <= step);
  const gain = evaluation?.gain ?? null;
  const threshold = rule?.minGainDb ?? 0;
  const full = evaluation?.decision === "improving" || evaluation?.decision === "converged";
  return {
    line: rule
      ? `stop if Δ best PSNR (last ${compact(rule.window)} steps) < ${String(threshold)} dB`
      : "–",
    checks: [
      {
        label: `Densify done · ${count(stop)}`,
        value: stop == null ? "–" : step >= stop ? "✓" : `${count(stop - step)} to go`,
        ok: stop != null && step >= stop,
      },
      {
        label: `Window filled · ${compact(rule?.window)}`,
        value: full
          ? "✓"
          : stop != null && rule
            ? `${count(Math.max(0, stop + rule.window - step))} to go`
            : "–",
        ok: full,
      },
      {
        label: `Gain < ${rule ? String(threshold) : "–"} dB`,
        value: gain == null ? "–" : `${gain.toFixed(3)} dB`,
        ok: gain != null && rule != null && gain < threshold,
      },
    ],
    gain,
    fraction: gainBarFraction(gain, threshold),
    threshold: threshold > 0 ? 0.2 : 0,
  };
}
export function gainBarFraction(gain: number | null, threshold: number): number {
  return gain == null ? 0 : Math.max(0, Math.min(1, gain / Math.max(threshold * 5, 0.001)));
}
export function nearestIndex(steps: readonly number[], at: number): number {
  if (!steps.length) return -1;
  let best = 0;
  steps.forEach((step, i) => {
    if (Math.abs(step - at) < Math.abs((steps[best] ?? 0) - at)) best = i;
  });
  return best;
}
export function snapshotAt(
  snapshots: readonly LiveTrainSnapshot[],
  step: number,
): LiveTrainSnapshot | null {
  return (
    [...snapshots]
      .sort((a, b) => a.step - b.step)
      .filter((s) => s.step <= step)
      .at(-1) ?? null
  );
}
export type ChartId =
  | "gs"
  | "loss"
  | "psnr"
  | "lr"
  | "its"
  | "mem"
  | "far"
  | "elongation"
  | "refine"
  | "gain"
  | "ceiling";
export interface ChartPoint {
  step: number;
  value: number;
}
export function chartPoints(train: LiveTrain | null | undefined, id: ChartId): ChartPoint[] {
  const quality = train?.quality?.series;
  const end = train?.ended?.step ?? Infinity;
  if (
    id === "far" ||
    id === "elongation" ||
    (id === "mem" && quality?.peakAllocatedGb?.some((v) => v != null))
  ) {
    const values =
      id === "far"
        ? quality?.farFieldShare
        : id === "elongation"
          ? quality?.elongationShare
          : quality?.peakAllocatedGb;
    return (quality?.step ?? []).flatMap((step, i) =>
      step > end || values?.[i] == null
        ? []
        : [{ step, value: values[i] * (id === "mem" ? 1 : 100) }],
    );
  }
  if (id === "refine" || id === "ceiling")
    return (train?.series?.step ?? [])
      .filter((step) => step <= end)
      .map((step) => ({ step, value: step }));
  if (id === "gain")
    return (train?.evals ?? []).flatMap((e) =>
      e.step > end || e.gain == null ? [] : [{ step: e.step, value: e.gain }],
    );
  if (id === "psnr")
    return (train?.evals ?? []).flatMap((e) =>
      e.step > end || e.psnr == null ? [] : [{ step: e.step, value: e.psnr }],
    );
  const series = train?.series;
  const values = id === "loss" ? series?.loss?.total : series?.[id];
  return (series?.step ?? []).flatMap((step, i) => {
    const value = values?.[i];
    return step > end || value == null || !Number.isFinite(value) || (id === "lr" && value <= 0)
      ? []
      : [{ step, value }];
  });
}
export function chartDomain(
  points: readonly ChartPoint[],
  log = false,
  reference?: number | null,
  zero = false,
): [number, number] {
  const values = points.map((p) => p.value).filter((v) => Number.isFinite(v) && (!log || v > 0));
  if (reference != null && (!log || reference > 0)) values.push(reference);
  if (!values.length) return log ? [-6, -2] : [0, 1];
  const transformed = log ? values.map(Math.log10) : values;
  const lo = zero ? 0 : Math.min(...transformed),
    hi = Math.max(...transformed);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  // A quantity that is never negative (loss, speed, shares) never gets a negative axis.
  const floor = !log && lo >= 0 ? 0 : -Infinity;
  return [zero ? 0 : Math.max(floor, lo - pad), hi + pad];
}
export function nearestCameraDistance(
  x: number,
  y: number,
  z: number,
  centres: ArrayLike<number>,
): number | null {
  let nearest = Infinity;
  for (let i = 0; i + 2 < centres.length; i += 3)
    nearest = Math.min(
      nearest,
      Math.hypot(x - (centres[i] ?? 0), y - (centres[i + 1] ?? 0), z - (centres[i + 2] ?? 0)),
    );
  return Number.isFinite(nearest) ? nearest : null;
}
export function distanceColour(metres: number): [number, number, number] {
  const near = [127, 216, 192],
    middle = [216, 195, 154],
    far = [224, 106, 90];
  const a = metres <= 40 ? near : middle,
    b = metres <= 40 ? middle : far;
  const t = Math.max(0, Math.min(1, metres <= 40 ? metres / 40 : (metres - 40) / 80));
  return [0, 1, 2].map((i) => ((a[i] ?? 0) + ((b[i] ?? 0) - (a[i] ?? 0)) * t) / 255) as [
    number,
    number,
    number,
  ];
}
export function eventsDisplay(
  train?: LiveTrain | null,
): { step: string; kind: string; text: string }[] {
  return [...(train?.events ?? [])]
    .reverse()
    .map((e) => ({ step: count(e.step), kind: e.kind, text: e.text }));
}

export function snapshotLag(shown: number, current: number): string | null {
  return shown < 0 || shown >= current ? null : `−${compact(current - shown)}`;
}
export function scrubPosition(
  step: number,
  latest: number,
  ceiling: number,
  width: number,
  release = false,
  ended = false,
): number | null {
  if (ended) return Math.max(0, Math.min(latest, step));
  const margin = release ? (ceiling * 6) / Math.max(1, width - 12) : 0;
  return step >= latest - margin ? null : Math.max(0, step);
}
export function seedsLabel(seeds?: LiveTrainSeeds | null): string {
  if (!seeds) return "";
  return (
    `${compact(seeds.of)} seeds · ${compact(seeds.sources?.sfm ?? 0)} SfM · ${compact(seeds.sources?.dense ?? 0)} dense` +
    (seeds.sources?.seed ? ` · ${compact(seeds.sources.seed)} seed` : "")
  );
}
export function decodeSeeds(
  bytes: ArrayBuffer,
  seeds: LiveTrainSeeds,
): { positions: Float32Array; colors: Float32Array } {
  if (bytes.byteLength !== seeds.count * 10 || seeds.origin.length !== 3 || seeds.scale <= 0)
    throw new Error("Invalid seeds");
  const view = new DataView(bytes),
    positions = new Float32Array(seeds.count * 3),
    colors = new Float32Array(seeds.count * 3);
  const palette = [
    [127, 216, 192],
    [216, 195, 154],
    [185, 163, 240],
  ];
  for (let i = 0; i < seeds.count; i++) {
    const color = palette[view.getUint8(i * 10 + 9)] ?? palette[0];
    for (let axis = 0; axis < 3; axis++) {
      positions[i * 3 + axis] =
        (view.getInt16(i * 10 + axis * 2, true) * seeds.scale) / 32767 + (seeds.origin[axis] ?? 0);
      colors[i * 3 + axis] = (color?.[axis] ?? 0) / 255;
    }
  }
  return { positions, colors };
}
export function lossLines(
  train?: LiveTrain | null,
): { name: string; color: string; points: ChartPoint[] }[] {
  const lines = [{ name: "total", color: "#d8c39a", points: chartPoints(train, "loss") }];
  const series = train?.quality?.series;
  for (const [name, color] of [
    ["depth", "#7fd8c0"],
    ["normal", "#b9a3f0"],
    ["anisotropy", "#e06a5a"],
    ["gradient", "#f0b254"],
  ]) {
    if (!name || !color) continue;
    const values = series?.loss?.[name];
    if (!values?.some((v) => v != null && (name !== "gradient" || v !== 0))) continue;
    lines.push({
      name,
      color,
      points: (series?.step ?? []).flatMap((step, i) =>
        step > (train?.ended?.step ?? Infinity) || values[i] == null
          ? []
          : [{ step, value: values[i] }],
      ),
    });
  }
  return lines.filter((line) => line.points.length);
}
export function finalReadouts(train?: LiveTrain | null): string[] {
  const final = train?.final;
  if (!final) return [];
  return (
    [
      ["PSNR", final.psnr],
      ["SSIM", final.ssim],
      ["LPIPS", final.lpips],
      ["cc_PSNR", final.ccPsnr],
      ["cc_SSIM", final.ccSsim],
      ["cc_LPIPS", final.ccLpips],
    ] as const
  ).flatMap(([label, value]) =>
    value == null ? [] : [`${label} ${value.toFixed(label.endsWith("PSNR") ? 2 : 3)}`],
  );
}
