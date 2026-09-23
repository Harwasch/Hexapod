/**
 * The data console's rules, tested without rendering anything.
 *
 * Three things here are logic rather than layout: what "these two runs differ" means,
 * what a typed-in parameter becomes on the wire, and which row of the plan's cost table
 * applies. All three would be quietly wrong forever if they lived inside a component.
 */
import { describe, expect, it } from "vitest";

import type { Job, JobStep, RecipeRead } from "@twin/contracts";

import { compareRuns } from "@/admin/compare";
import { estimateRun } from "@/admin/estimate";
import { coerce, overridesFrom } from "@/admin/runDraft";

const CAPTURE = "11111111-1111-4111-8111-111111111111";

function step(overrides: Partial<JobStep> = {}): JobStep {
  return {
    id: "step-1",
    jobId: "job-1",
    stageId: "package",
    ordinal: 1,
    impl: "splat_tiles",
    status: "complete",
    startedAt: "2026-09-20T10:00:00Z",
    finishedAt: "2026-09-20T10:00:30Z",
    metrics: {},
    logKey: null,
    attempt: 1,
    checkpointKey: null,
    preemptedAt: null,
    artifacts: [],
    createdAt: "2026-09-20T10:00:00Z",
    updatedAt: "2026-09-20T10:00:30Z",
    ...overrides,
  };
}

function job(id: string, params: Record<string, unknown>, overrides: Partial<Job> = {}): Job {
  return {
    id,
    captureId: CAPTURE,
    recipe: "photo-reconstruct",
    recipeVersion: "2",
    params,
    status: "complete",
    provider: "modal",
    tier: "l4",
    finishedAt: "2026-09-20T10:30:00Z",
    durationS: 1800,
    costUsd: "0.82",
    error: null,
    claimedBy: "worker-1",
    claimedAt: "2026-09-20T10:00:00Z",
    leaseExpiresAt: null,
    steps: [step()],
    createdAt: "2026-09-20T09:59:00Z",
    updatedAt: "2026-09-20T10:30:00Z",
    ...overrides,
  };
}

describe("comparing runs", () => {
  it("lines up the parameter that differs and leaves the rest alone", () => {
    // B3, exactly: one capture, one recipe, one changed stage parameter.
    const none = job("job-none", { mask: { impl: "none" }, train: { iterations: 30000 } });
    const robust = job("job-robust", { mask: { impl: "robust" }, train: { iterations: 30000 } });

    const comparison = compareRuns([none, robust]);
    const masked = comparison.params.find((row) => row.key === "mask.impl");
    const iterations = comparison.params.find((row) => row.key === "train.iterations");

    expect(masked?.values).toEqual(["none", "robust"]);
    expect(masked?.differs).toBe(true);
    expect(iterations?.differs).toBe(false);
    expect(comparison.sameCapture).toBe(true);
    expect(comparison.differences).toBe(1);
  });

  it("shows a parameter one run set and the other did not, rather than hiding it", () => {
    const control = job("job-a", {});
    const treatment = job("job-b", { compensate: { impl: "imc" } });

    const row = compareRuns([control, treatment]).params.find(
      (entry) => entry.key === "compensate.impl",
    );
    // The null control's cell says "not set", which is the difference B5 is measuring.
    expect(row?.values).toEqual(["—", "imc"]);
    expect(row?.differs).toBe(true);
  });

  it("compares more than two runs at once", () => {
    const runs = ["none", "robust", "imc"].map((impl) => job(`job-${impl}`, { mask: { impl } }));
    const comparison = compareRuns(runs);
    expect(comparison.params[0]?.values).toEqual(["none", "robust", "imc"]);
    expect(comparison.runs).toHaveLength(3);
  });

  it("times each stage from the step, not from the run", () => {
    const fast = job("job-fast", {});
    const slow = job(
      "job-slow",
      {},
      {
        steps: [step({ finishedAt: "2026-09-20T10:05:00Z" })],
        // Same run-level duration: only the stage got slower, which is the point.
        durationS: 1800,
      },
    );
    const row = compareRuns([fast, slow]).stages.find((entry) => entry.key === "stage:package");
    expect(row?.values).toEqual(["30.0 s", "5m 0s"]);
    expect(row?.differs).toBe(true);
  });

  it("says when the runs are not over the same capture", () => {
    const here = job("job-a", {});
    const elsewhere = job("job-b", {}, { captureId: "22222222-2222-4222-8222-222222222222" });
    expect(compareRuns([here, elsewhere]).sameCapture).toBe(false);
  });

  it("counts attempts and preemptions, because on a cheap tier they are the story", () => {
    const clean = job("job-a", {});
    const preempted = job(
      "job-b",
      {},
      {
        steps: [step({ attempt: 3, preemptedAt: "2026-09-20T10:02:00Z" })],
      },
    );
    const comparison = compareRuns([clean, preempted]);
    expect(comparison.facts.find((row) => row.key === "fact:attempts")?.values).toEqual(["1", "3"]);
    expect(comparison.facts.find((row) => row.key === "fact:preemptions")?.values).toEqual([
      "0",
      "1",
    ]);
  });

  it("ignores a params value that is not an object of overrides", () => {
    // `jobs.params` is JSONB: nothing in the database stops a flat value getting in.
    const odd = job("job-a", { sh: 3 });
    expect(compareRuns([odd]).params).toEqual([]);
  });
});

describe("the new-run draft", () => {
  const recipe: RecipeRead = {
    name: "photo-reconstruct",
    version: "2",
    description: "Reconstruct a splat from a raw capture.",
    inputs: ["upload"],
    stages: [
      {
        id: "normalize",
        impl: "ffmpeg_frames",
        params: { fps: 4, select: "sharpness" },
        gpu: null,
      },
      {
        id: "train",
        impl: "gsplat",
        params: { iterations: 30000 },
        gpu: { tier: "l4", preemptible: true },
      },
      { id: "compensate", impl: "none", params: {}, gpu: null },
    ],
  };

  it("keeps the type the recipe's default had", () => {
    expect(coerce("7000", 30000)).toBe(7000);
    expect(coerce("sharpness", "blur")).toBe("sharpness");
    expect(coerce("true", false)).toBe(true);
    // No default to learn a type from: JSON where it parses, text where it does not.
    expect(coerce("[1, 2]", undefined)).toEqual([1, 2]);
    expect(coerce("l4", undefined)).toBe("l4");
  });

  it("sends only what was changed", () => {
    const params = overridesFrom(
      { train: { iterations: "7000" }, normalize: { fps: "4" } },
      recipe,
    );
    // `fps` was retyped to the same value it already had, so it is not an override and
    // does not make this run look different from one that left it alone.
    expect(params).toEqual({ train: { iterations: 7000 } });
  });

  it("is empty when nothing was touched", () => {
    expect(overridesFrom({}, recipe)).toEqual({});
    expect(overridesFrom({ train: {} }, undefined)).toEqual({});
  });
});

describe("estimating a run", () => {
  const lane1: RecipeRead = {
    name: "splat-ingest",
    version: "2",
    description: "",
    inputs: ["upload"],
    stages: [{ id: "package", impl: "splat_tiles", params: {}, gpu: null }],
  };
  const lane2: RecipeRead = {
    name: "photo-reconstruct",
    version: "2",
    description: "",
    inputs: ["upload"],
    stages: [
      { id: "train", impl: "gsplat", params: {}, gpu: { tier: "l4", preemptible: true } },
      { id: "compensate", impl: "none", params: {}, gpu: null },
    ],
  };

  it("charges nothing for a lane that never asks for a GPU", () => {
    const estimate = estimateRun({ recipe: lane1, bytes: 2_000_000, interruptible: false });
    expect(estimate.gpu).toBe(false);
    expect(estimate.cost).toBe("no GPU time");
  });

  it("reads the plan's object row for a small capture and its site row for a big one", () => {
    expect(estimateRun({ recipe: lane2, bytes: 200e6, interruptible: false }).workload).toBe(
      "object",
    );
    expect(estimateRun({ recipe: lane2, bytes: 11e9, interruptible: false }).workload).toBe("site");
  });

  it("is cheaper on an interruptible tier, and says why", () => {
    const reliable = estimateRun({ recipe: lane2, bytes: 200e6, interruptible: false });
    const cheap = estimateRun({ recipe: lane2, bytes: 200e6, interruptible: true });
    expect(reliable.cost).toBe("under $1");
    expect(cheap.cost).toBe("cents");
    expect(cheap.basis).toContain("20-40%");
  });

  it("moves to the motion row only once compensate actually runs", () => {
    expect(estimateRun({ recipe: lane2, bytes: 200e6, interruptible: false }).workload).toBe(
      "object",
    );
    const withImc = estimateRun({
      recipe: lane2,
      bytes: 200e6,
      interruptible: false,
      impls: { compensate: "imc" },
    });
    expect(withImc.workload).toBe("site-motion");
    expect(withImc.cost).toBe("$3-10");
  });
});
