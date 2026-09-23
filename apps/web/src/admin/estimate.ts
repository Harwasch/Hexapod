/**
 * What a run will cost, before launching it.
 *
 * Every figure here comes from the sprint plan's own cost table, which is the only
 * measured data there is:
 *
 * | Capture                                | Tier | Wall time       | On Modal | Interruptible |
 * | One object / one tree                  | L4   | ~20-30 min      | under $1 | cents         |
 * | Site, 500-1500 drone frames            | L4   | ~1-2 h          | $1-2     | well under $1 |
 * | Same, with IMC motion compensation     | L4   | 3-5x the train  | $3-10    | $1-4          |
 *
 * It is deliberately a range rather than a number. A per-tier price table nobody has
 * verified would put invented precision in front of somebody deciding how to spend money;
 * B1's `ProviderAdapter` brings the real rates and records what a run actually cost in
 * `jobs.cost_usd`, which is the figure that settles it. Until then this says what is
 * known, at the precision it is known to.
 */
import type { RecipeRead } from "@twin/contracts";

export type Workload = "no-gpu" | "object" | "site" | "site-motion";

export interface RunEstimate {
  workload: Workload;
  /** Whether any stage of this recipe asks for a GPU at all. */
  gpu: boolean;
  /** The tiers the recipe's GPU stages ask for, in recipe order. */
  tiers: string[];
  wall: string;
  cost: string;
  /** Where the figure comes from, shown beside it so it is not mistaken for a quote. */
  basis: string;
}

/**
 * Above this, a capture is a site rather than an object.
 *
 * 2 GiB is roughly where an iPhone clip stops being one tree and starts being a walk
 * around a paddock; the plan's two rows are "one object" and "500-1500 drone frames" and
 * this is the line between them. It only chooses which row of the table to show.
 */
const SITE_BYTES = 2 * 1024 ** 3;

const TABLE: Record<Exclude<Workload, "no-gpu">, { wall: string; modal: string; cheap: string }> = {
  object: { wall: "about 20-30 minutes", modal: "under $1", cheap: "cents" },
  site: { wall: "about 1-2 hours", modal: "$1-2", cheap: "well under $1" },
  "site-motion": { wall: "3-5x the train", modal: "$3-10", cheap: "$1-4" },
};

/** A stage that is switched off. `impl: none` is how the recipes spell "skip this". */
function isOff(impl: string): boolean {
  return impl === "none";
}

export function estimateRun(options: {
  recipe: RecipeRead | undefined;
  /** Total bytes of the capture's uploaded source files. */
  bytes: number;
  /** The provider chosen for this run, if any — an interruptible one is the cheap column. */
  interruptible: boolean;
  /** Stage overrides as the form currently has them, keyed by stage id. */
  overrides?: Record<string, Record<string, unknown>>;
  /** Stage impl overrides, when the form offers them (B3 swaps `mask` and `compensate`). */
  impls?: Record<string, string>;
}): RunEstimate {
  const { recipe, bytes, interruptible, impls = {} } = options;
  const stages = recipe?.stages ?? [];
  const gpuStages = stages.filter(
    (stage) => stage.gpu !== null && !isOff(impls[stage.id] ?? stage.impl),
  );
  const tiers = gpuStages.map((stage) => stage.gpu?.tier ?? "gpu");

  if (gpuStages.length === 0) {
    return {
      workload: "no-gpu",
      gpu: false,
      tiers: [],
      // Lane 1: normalise, georeference, package, describe, register. The plan's own
      // words, and A8 measured 0.07 s for the 12,000-gaussian fixture.
      wall: "under a minute",
      cost: "no GPU time",
      basis: "Lane 1 runs on CPU wherever the worker happens to be.",
    };
  }

  const compensate = stages.find((stage) => stage.id === "compensate");
  const motion = compensate !== undefined && !isOff(impls.compensate ?? compensate.impl);
  const workload: Exclude<Workload, "no-gpu"> = motion
    ? "site-motion"
    : bytes >= SITE_BYTES
      ? "site"
      : "object";
  const entry = TABLE[workload];
  return {
    workload,
    gpu: true,
    tiers,
    wall: entry.wall,
    cost: interruptible ? entry.cheap : entry.modal,
    basis: interruptible
      ? "Interruptible tier. A0 measured Vast's unverified tier running 20-40% above its listed price once restarts are priced in."
      : "Reliable tier, per-second billed.",
  };
}
