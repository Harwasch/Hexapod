import { describe, expect, it } from "vitest";

import { MOVING_BUDGET_MS, RESTING_BUDGET_MS, TileWork } from "@/cesium/scanView/tileWork";

/** A clock the jobs advance by what they cost, and frames the test steps. */
function rig() {
  let now = 0;
  let next: (() => void)[] = [];
  const work = new TileWork({
    now: () => now,
    nextFrame: (callback) => next.push(callback),
  });
  const job = (ms: number, log: number[], id: number) => () => {
    now += ms;
    log.push(id);
    return id;
  };
  const frame = async (): Promise<void> => {
    const run = next;
    next = [];
    for (const callback of run) callback();
    await Promise.resolve();
  };
  return { work, job, frame };
}

describe("main-thread tile work, within a frame budget", () => {
  it("runs one costly job a frame while the camera moves, so frames stay short", async () => {
    const { work, job, frame } = rig();
    work.moving = true;
    const log: number[] = [];
    const done = [1, 2, 3].map((id) => work.run(job(MOVING_BUDGET_MS + 2, log, id)));
    expect(log).toEqual([1]);
    await frame();
    expect(log).toEqual([1, 2]);
    await frame();
    expect(log).toEqual([1, 2, 3]);
    expect(await Promise.all(done)).toEqual([1, 2, 3]);
    expect(work.deferred).toBeGreaterThan(0);
  });

  it("runs as many as fit the frame at rest", async () => {
    const { work, job, frame } = rig();
    const log: number[] = [];
    for (let id = 1; id <= 5; id++) void work.run(job(RESTING_BUDGET_MS / 4, log, id));
    // Four fit the resting budget; the fifth waits for the next frame.
    expect(log).toEqual([1, 2, 3, 4]);
    await frame();
    expect(log).toEqual([1, 2, 3, 4, 5]);
  });

  it("passes a job's failure on, and refuses work once stopped", async () => {
    const { work } = rig();
    await expect(
      work.run(() => {
        throw new Error("aborted");
      }),
    ).rejects.toThrow("aborted");
    work.stop();
    await expect(work.run(() => 1)).rejects.toThrow();
  });
});
