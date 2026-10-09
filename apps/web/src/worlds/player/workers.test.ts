import { describe, expect, it } from "vitest";
import { selectRetainedWorker, type ReusableWorker } from "./workers";

const worker = (id: string, overrides: Partial<ReusableWorker> = {}): ReusableWorker => ({
  id,
  provider: "runpod",
  status: "ready",
  managed: true,
  ...overrides,
});
describe("retained worker reuse", () => {
  it("does not reuse a different model's warm worker", () => {
    const old = worker("astronex");
    const forge = worker("forge", { modelId: "forge-wm" });
    expect(selectRetainedWorker([old, forge], [], "runpod", "forge-wm")).toEqual(forge);
    expect(selectRetainedWorker([old], [], "runpod", "forge-wm")).toBeUndefined();
    expect(selectRetainedWorker([old], [], "runpod", "astronex-world")).toEqual(old);
  });
  it("prefers an idle ready worker over one still starting", () => {
    const starting = worker("starting", { status: "starting" });
    const ready = worker("ready");
    expect(
      selectRetainedWorker(
        [starting, ready],
        [{ id: "old", workerId: ready.id, status: "stopped" }],
        "runpod",
      ),
    ).toEqual(ready);
  });
  it("does not reuse workers with active or uncertain failed sessions", () => {
    expect(
      selectRetainedWorker(
        [worker("active"), worker("uncertain")],
        [
          { id: "a", workerId: "active", status: "running" },
          { id: "b", workerId: "uncertain", status: "error" },
        ],
        "runpod",
      ),
    ).toBeUndefined();
  });
  it("excludes external, different-provider, and terminal/unknown workers", () => {
    const workers = [
      worker("external", { managed: false }),
      worker("local", { provider: "local" }),
      ...["unknown", "error", "stopped", "destroyed"].map((status) => worker(status, { status })),
    ];
    expect(selectRetainedWorker(workers, [], "runpod")).toBeUndefined();
  });
});
