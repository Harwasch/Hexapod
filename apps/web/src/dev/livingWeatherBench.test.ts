import { expect, it } from "vitest";
import { armOrder, summarizeWeatherBench } from "./livingWeatherBench";
it("randomizes reproducibly and brackets each arm with off in every round", () => {
  expect(armOrder(0, 42)).toEqual(armOrder(0, 42));
  expect(new Set(Array.from({ length: 8 }, (_, i) => armOrder(i, 42).join())).size).toBeGreaterThan(
    1,
  );
  const order = armOrder(0, 42);
  expect(order.filter((a) => a !== "off").sort()).toEqual(["calm", "fresh", "light", "storm"]);
  for (let i = 0; i < order.length; i += 2) expect(order[i]).toBe("off");
});
it("reports paired per-round differences and rejects a drifting baseline", () => {
  const blocks = [10, 12, 10, 20, 23, 20].map((n, i) => ({
    round: Math.floor(i / 3),
    arm: i % 3 === 1 ? "storm" : "off",
    samples: [n, n, n],
  }));
  const r = summarizeWeatherBench(blocks);
  expect(r.inconclusive).toBe(true);
  expect(r.verdict).toContain("plugged in");
  expect(r.summary.find((s) => s.arm === "storm")?.pairs.map((p) => p.addedMs)).toEqual([2, 3]);
  expect(r.driftRatio).toBe(2);
  expect(r.summary.find((s) => s.arm === "storm")?.addedMs).toBe(2.5);
});
