/**
 * The SPZ decode pool starts no worker until a splat tile asks for one (splatDecoder.ts):
 * installing it is part of the globe's boot, and every worker it used to make there loaded
 * and started the 245 kB decoder before the first frame, for a page that may never show a
 * splat. The pool then grows only when every running worker is full.
 */
import { GltfSpzLoader } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { installSplatDecoder } from "@/cesium/splatDecoder";

type DecodeHook = (spz: Uint8Array) => Promise<object> | undefined;

class FakeWorker {
  static made: FakeWorker[] = [];
  onmessage: ((event: { data: { id: number; decoded?: object } }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  posted: { id: number }[] = [];
  terminated = false;
  constructor() {
    FakeWorker.made.push(this);
  }
  postMessage(message: { id: number }): void {
    this.posted.push(message);
  }
  terminate(): void {
    this.terminated = true;
  }
  /** Answers the oldest request this worker holds. */
  answer(): void {
    const message = this.posted.shift();
    if (message) this.onmessage?.({ data: { id: message.id, decoded: { id: message.id } } });
  }
}

/** A request whose answer the test does not wait for; the uninstall rejects it. */
function pending(request: Promise<object> | undefined): Promise<object> | undefined {
  void request?.catch(() => undefined);
  return request;
}

function hook(): DecodeHook {
  const module = GltfSpzLoader as { decodeHook?: DecodeHook };
  if (!module.decodeHook) throw new Error("no decode hook installed");
  return module.decodeHook;
}

describe("installSplatDecoder", () => {
  let uninstall: () => void = () => undefined;

  beforeEach(() => {
    FakeWorker.made = [];
    vi.stubGlobal("Worker", FakeWorker);
    // Twelve cores: a pool of three.
    vi.spyOn(window.navigator, "hardwareConcurrency", "get").mockReturnValue(12);
  });

  afterEach(() => {
    uninstall();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("starts no worker at install", () => {
    uninstall = installSplatDecoder();
    expect(FakeWorker.made).toHaveLength(0);
  });

  it("starts the first worker for the first tile and another only when every one is full", async () => {
    uninstall = installSplatDecoder();
    const decode = hook();
    const first = pending(decode(new Uint8Array(8)));
    expect(FakeWorker.made).toHaveLength(1);
    // Two tiles at a time per worker before another is started.
    void pending(decode(new Uint8Array(8)));
    expect(FakeWorker.made).toHaveLength(1);
    void pending(decode(new Uint8Array(8)));
    expect(FakeWorker.made).toHaveLength(2);

    FakeWorker.made[0]?.answer();
    await expect(first).resolves.toEqual({ id: 1 });
  });

  it("answers 'not now' once the whole pool is full, so the loader asks again", () => {
    uninstall = installSplatDecoder();
    const decode = hook();
    for (let i = 0; i < 6; i += 1) {
      expect(pending(decode(new Uint8Array(8)))).toBeInstanceOf(Promise);
    }
    expect(FakeWorker.made).toHaveLength(3);
    expect(decode(new Uint8Array(8))).toBeUndefined();
  });

  it("stops the workers it started, and only those, on uninstall", () => {
    uninstall = installSplatDecoder();
    void pending(hook()(new Uint8Array(8)));
    uninstall();
    uninstall = () => undefined;
    expect(FakeWorker.made.map((worker) => worker.terminated)).toEqual([true]);
    expect((GltfSpzLoader as { decodeHook?: DecodeHook }).decodeHook).toBeUndefined();
  });
});
