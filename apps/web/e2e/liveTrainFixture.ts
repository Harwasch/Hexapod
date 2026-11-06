/** The read-only bench excerpt supplies chart samples; file URLs are test-local. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { LiveTrain, LiveTrainPlan } from "@twin/contracts";

type Row = LiveTrainPlan & {
  k?: string;
  run?: string;
  step?: number;
  t?: number;
  gs?: number;
  lr?: number;
  sh?: number;
  its?: number;
  mem?: number;
  loss?: Record<string, number>;
  maxSteps?: number;
  key?: string;
  count?: number;
  of?: number;
  bytes?: number;
  farFieldShare?: number;
  elongationShare?: number;
  lossTerms?: Record<string, number>;
  trainMemoryCurve?: { step: number; peakAllocatedGb?: number }[];
};
export function benchTrain(bucket: string): LiveTrain {
  const records = readFileSync(
    resolve(import.meta.dirname, "../../../tools/pipeline/tests/fixtures/run6fb-train.log"),
    "utf8",
  )
    .trim()
    .split("\n")
    .map((line) => {
      const split = line.indexOf(": ");
      return { tag: line.slice(0, split), row: JSON.parse(line.slice(split + 2)) as Row };
    });
  const plan = records.find((r) => r.tag === "live-plan")?.row ?? {};
  const resolved = records.find((r) => r.tag === "live-train" && r.row.k === "p")?.row ?? {};
  const samples = records
    .filter((r) => r.tag === "live-train" && r.row.k === "s")
    .map((r) => r.row);
  const quality = records.filter((r) => r.tag === "live-metrics").map((r) => r.row);
  const last = quality.at(-1);
  return {
    run: resolved.run,
    plan: { ...plan, ...resolved, stepsMax: resolved.maxSteps ?? plan.stepsMax },
    series: {
      step: samples.map((r) => r.step ?? 0),
      t: samples.map((r) => r.t ?? null),
      gs: samples.map((r) => r.gs ?? null),
      lr: samples.map((r) => r.lr ?? null),
      sh: samples.map((r) => r.sh ?? null),
      its: samples.map((r) => r.its ?? null),
      mem: samples.map((r) => r.mem ?? null),
      loss: { total: samples.map((r) => r.loss?.total ?? null) },
    },
    quality: {
      step: last?.step,
      farFieldShare: last?.farFieldShare,
      elongationShare: last?.elongationShare,
      series: {
        step: quality.map((r) => r.step ?? 0),
        farFieldShare: quality.map((r) => r.farFieldShare ?? null),
        elongationShare: quality.map((r) => r.elongationShare ?? null),
        peakAllocatedGb: quality.map((r) => r.trainMemoryCurve?.at(-1)?.peakAllocatedGb ?? null),
        loss: Object.fromEntries(
          ["depth", "normal", "gradient", "anisotropy"].map((key) => [
            key,
            quality.map((r) => r.lossTerms?.[key] ?? null),
          ]),
        ),
      },
    },
    snapshots: records
      .filter((r) => r.tag === "live-splat")
      .map(({ row }) => ({
        step: row.step ?? 0,
        count: row.count ?? 0,
        of: row.of,
        bytes: row.bytes,
        key: row.key ?? "",
        url: `${bucket}/${row.key?.split("/").at(-1) ?? ""}?signature=test`,
      })),
  };
}
