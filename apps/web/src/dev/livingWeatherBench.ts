/** Benchmark statistics kept separate from the page so pairing and drift are testable. */
export interface WeatherBenchBlock {
  round: number;
  arm: string;
  samples: number[];
}
export const quantile = (v: number[], q: number) => {
  const sorted = [...v].sort((a, b) => a - b);
  const at = Math.max(0, sorted.length - 1) * q;
  const low = sorted[Math.floor(at)] ?? 0,
    high = sorted[Math.ceil(at)] ?? low;
  return low + (high - low) * (at - Math.floor(at));
};
export function armOrder(round: number, seed: number): string[] {
  let state = (seed + Math.imul(round + 1, 2654435761)) >>> 0;
  const arms = ["calm", "light", "fresh", "storm"];
  for (let i = arms.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1);
    [arms[i], arms[j]] = [arms[j] ?? "storm", arms[i] ?? "storm"];
  }
  return ["off", ...arms.flatMap((arm) => [arm, "off"])];
}
export function summarizeWeatherBench(blocks: WeatherBenchBlock[]) {
  const offMedians = blocks.filter((b) => b.arm === "off").map((b) => quantile(b.samples, 0.5));
  const driftRatio = Math.max(...offMedians) / Math.min(...offMedians);
  const rounds = [...new Set(blocks.map((b) => b.round))].map((round) => {
    const off = blocks
      .filter((b) => b.round === round && b.arm === "off")
      .map((b) => quantile(b.samples, 0.5));
    return { round, offMedians: off, driftRatio: Math.max(...off) / Math.min(...off) };
  });
  const inconclusive = driftRatio > 1.3;
  const summary = ["off", "calm", "light", "fresh", "storm"].map((arm) => {
    const selected = blocks.filter((b) => b.arm === arm).flatMap((b) => b.samples);
    const pairs = blocks.flatMap((b, i) => {
      const before = blocks[i - 1],
        after = blocks[i + 1];
      if (
        arm === "off" ||
        b.arm !== arm ||
        before?.arm !== "off" ||
        after?.arm !== "off" ||
        before.round !== b.round ||
        after.round !== b.round
      )
        return [];
      const baseline = (quantile(before.samples, 0.5) + quantile(after.samples, 0.5)) / 2;
      return [
        {
          round: b.round,
          addedMs: quantile(b.samples, 0.5) - baseline,
          ratio: quantile(b.samples, 0.5) / baseline,
        },
      ];
    });
    const added = pairs.map((p) => p.addedMs);
    return {
      arm,
      medianMs: quantile(selected, 0.5),
      p95Ms: quantile(selected, 0.95),
      pairs,
      addedMs: quantile(added, 0.5),
      addedP10Ms: quantile(added, 0.1),
      addedP90Ms: quantile(added, 0.9),
      ratio:
        arm === "off"
          ? 1
          : quantile(
              pairs.map((p) => p.ratio),
              0.5,
            ),
    };
  });
  return {
    summary,
    rounds,
    driftRatio,
    inconclusive,
    verdict: inconclusive
      ? "inconclusive, rerun with the laptop plugged in and nothing else open"
      : "Stable off blocks; inspect paired differences and their spread.",
  };
}
