/**
 * Two or more runs, side by side — the Phase B instrument.
 *
 * B3 runs one windy capture through `mask: none`, `robust` and `imc` and asks what
 * changed. B5 needs a null control against the same capture. Both are this: take the
 * runs, line up every parameter either of them set, and mark the ones that differ. The
 * parameters are real because A8 made `jobs.params` the resolved set the run actually
 * used (`Recipe.with_params`), not a summary written down beside it.
 *
 * Pure functions, no React: this is the part worth testing directly, and it is also the
 * part that would be quietly wrong forever if it were buried in a component.
 */
import type { Job } from "@twin/contracts";

import { elapsed, formatCost, formatDuration, formatValue } from "./format";

export interface ComparisonRow {
  /** "package · max_gaussians", or the fact's name for a run-level row. */
  label: string;
  /** Stable key: `stageId.param` for a parameter, `fact:<name>` otherwise. */
  key: string;
  /** One cell per run, in the order the runs were given. */
  values: string[];
  /** True when the runs do not agree — the only rows a comparison is really about. */
  differs: boolean;
}

export interface Comparison {
  runs: Job[];
  /** Recipe, version, provider, tier, status, wall time, cost. */
  facts: ComparisonRow[];
  /** Every parameter any of the runs set, whether or not the others did. */
  params: ComparisonRow[];
  /** Per-stage wall time, for the stage that got slower. */
  stages: ComparisonRow[];
  /** How many rows differ across facts, params and stages together. */
  differences: number;
  /**
   * True when every run is over the same capture. A comparison across captures is not
   * refused — comparing the same recipe on two different trees is a real question — but
   * it is not a controlled experiment and the view says so.
   */
  sameCapture: boolean;
}

type ParamRecord = Record<string, Record<string, unknown>>;

/** `jobs.params`, which is JSONB and therefore `unknown` until it is looked at. */
function stageParams(job: Job): ParamRecord {
  const params: unknown = job.params;
  if (typeof params !== "object" || params === null) return {};
  const out: ParamRecord = {};
  for (const [stageId, overrides] of Object.entries(params as Record<string, unknown>)) {
    if (typeof overrides === "object" && overrides !== null && !Array.isArray(overrides)) {
      out[stageId] = overrides as Record<string, unknown>;
    }
  }
  return out;
}

function row(label: string, key: string, values: string[]): ComparisonRow {
  return { label, key, values, differs: new Set(values).size > 1 };
}

function factRows(runs: Job[]): ComparisonRow[] {
  return [
    row(
      "Recipe",
      "fact:recipe",
      runs.map((job) => `${job.recipe} v${job.recipeVersion}`),
    ),
    row(
      "Status",
      "fact:status",
      runs.map((job) => job.status),
    ),
    row(
      "Provider",
      "fact:provider",
      runs.map((job) => job.provider ?? "—"),
    ),
    row(
      "Tier",
      "fact:tier",
      runs.map((job) => job.tier ?? "—"),
    ),
    row(
      "Wall time",
      "fact:duration",
      runs.map((job) => formatDuration(job.durationS)),
    ),
    row(
      "Cost",
      "fact:cost",
      runs.map((job) => formatCost(job.costUsd)),
    ),
    row(
      "Attempts",
      "fact:attempts",
      runs.map((job) => String(Math.max(0, ...job.steps.map((step) => step.attempt)))),
    ),
    row(
      "Preemptions",
      "fact:preemptions",
      runs.map((job) => String(job.steps.filter((step) => step.preemptedAt !== null).length)),
    ),
  ];
}

function paramRows(runs: Job[]): ComparisonRow[] {
  const perRun = runs.map(stageParams);
  const keys = new Set<string>();
  for (const params of perRun) {
    for (const [stageId, overrides] of Object.entries(params)) {
      for (const name of Object.keys(overrides)) keys.add(`${stageId}.${name}`);
    }
  }
  return [...keys].sort().map((key) => {
    const dot = key.indexOf(".");
    const stageId = key.slice(0, dot);
    const name = key.slice(dot + 1);
    return row(
      `${stageId} · ${name}`,
      key,
      perRun.map((params) => formatValue(params[stageId]?.[name])),
    );
  });
}

function stageRows(runs: Job[]): ComparisonRow[] {
  const stages = new Set<string>();
  for (const job of runs) for (const step of job.steps) stages.add(step.stageId);
  return [...stages].map((stageId) =>
    row(
      stageId,
      `stage:${stageId}`,
      runs.map((job) => {
        const step = job.steps.find((candidate) => candidate.stageId === stageId);
        if (!step) return "—";
        // The step's own start and finish, not the run's: what a comparison wants to
        // know is which stage got slower, not that the run did.
        return formatDuration(elapsed(step.startedAt, step.finishedAt));
      }),
    ),
  );
}

export function compareRuns(runs: Job[]): Comparison {
  const facts = factRows(runs);
  const params = paramRows(runs);
  const stages = stageRows(runs);
  return {
    runs,
    facts,
    params,
    stages,
    differences: [...facts, ...params, ...stages].filter((entry) => entry.differs).length,
    sameCapture: new Set(runs.map((job) => job.captureId)).size <= 1,
  };
}
