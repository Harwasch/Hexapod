import * as CesiumBarrel from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { installSplatSorter, whenGone } from "@/cesium/splatSorter";

/** The worker the sorter talks to, recording what it is sent. */
class FakeWorker {
  static last: FakeWorker | null = null;
  readonly sent: { kind: string; owner?: number }[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  constructor() {
    FakeWorker.last = this;
  }
  postMessage(message: { kind: string; owner?: number }): void {
    this.sent.push(message);
  }
  terminate(): void {
    /* nothing to stop */
  }
}

interface Hook {
  write?: (owner: object, capacity: number, start: number, positions: Float32Array) => void;
  hide?: (owner: object, capacity: number, start: number, count: number) => void;
}
const sortHookOf = (): Hook | undefined =>
  (CesiumBarrel as unknown as { GaussianSplatPrimitive?: { sortHook?: Hook } })
    .GaussianSplatPrimitive?.sortHook;

describe("the sort worker lets go of a scan that is gone", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("calls back once, before the primitive or its tileset is destroyed", () => {
    const order: string[] = [];
    const tileset = { destroy: () => order.push("tileset destroyed") };
    const primitive = { _tileset: tileset, destroy: () => order.push("primitive destroyed") };
    whenGone(primitive, () => order.push("gone"));
    tileset.destroy();
    primitive.destroy();
    expect(order).toEqual(["gone", "tileset destroyed", "primitive destroyed"]);
  });

  it("tells the worker to forget a primitive's positions when its tileset is destroyed", () => {
    vi.stubGlobal("Worker", FakeWorker);
    const uninstall = installSplatSorter();
    const hook = sortHookOf();
    expect(hook?.write).toBeTypeOf("function");
    let destroyed = false;
    const primitive = { _tileset: { destroy: () => (destroyed = true) } };
    hook?.write?.(primitive, 1000, 0, new Float32Array(30));
    // Hidden (its tile left the view): still resident, nothing forgotten.
    hook?.hide?.(primitive, 1000, 0, 10);
    const sent = FakeWorker.last?.sent ?? [];
    expect(sent.some((m) => m.kind === "forget")).toBe(false);
    const owner = sent[0]?.owner;
    primitive._tileset.destroy();
    expect(destroyed).toBe(true);
    expect(sent.filter((m) => m.kind === "forget")).toEqual([{ kind: "forget", owner }]);
    uninstall();
  });
});
