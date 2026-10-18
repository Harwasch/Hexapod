import { afterEach, describe, expect, it, vi } from "vitest";
import { canUseContinuousControls, SessionActionQueue } from "./actions";

function deferred() {
  let resolve: (() => void) | undefined;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve: () => resolve?.() };
}
afterEach(() => vi.restoreAllMocks());

describe("session control lifecycle", () => {
  it("does not allow a queued command to cross an aborted session boundary", async () => {
    const queue = new SessionActionQueue();
    const old = new AbortController();
    const gate = deferred();
    const first = queue.enqueue(() => gate.promise, old.signal);
    const staleSend = vi.fn(() => Promise.resolve());
    const stale = queue.enqueue(staleSend, old.signal).catch((error: unknown) => error);
    old.abort();
    gate.resolve();
    await first.catch(() => {
      /* The first command may not have started before cancellation. */
    });
    expect(await stale).toMatchObject({ name: "AbortError" });
    expect(staleSend).not.toHaveBeenCalled();
    const nextSend = vi.fn(() => Promise.resolve("new-session"));
    await expect(queue.enqueue(nextSend, new AbortController().signal)).resolves.toBe(
      "new-session",
    );
  });

  it("discards delayed movement but still delivers its release in order", async () => {
    let clock = 0;
    vi.spyOn(performance, "now").mockImplementation(() => clock);
    const queue = new SessionActionQueue();
    const signal = new AbortController().signal;
    const gate = deferred();
    const initial = queue.enqueue(() => gate.promise, signal);
    const movement = vi.fn(() => Promise.resolve());
    const held = queue.enqueue(movement, signal, 1000).catch((error: unknown) => error);
    const release = vi.fn(() => Promise.resolve("released"));
    const released = queue.enqueue(release, signal);
    clock = 1500;
    gate.resolve();
    await initial;
    expect(await held).toBeInstanceOf(Error);
    expect(movement).not.toHaveBeenCalled();
    await expect(released).resolves.toBe("released");
    expect(release).toHaveBeenCalledOnce();
  });

  it("bounds queued operations and recovers capacity after completion", async () => {
    const queue = new SessionActionQueue(2);
    const signal = new AbortController().signal;
    const gate = deferred();
    const first = queue.enqueue(() => gate.promise, signal);
    const second = queue.enqueue(() => Promise.resolve(), signal);
    await expect(queue.enqueue(() => Promise.resolve(), signal)).rejects.toThrow("queue is full");
    gate.resolve();
    await Promise.all([first, second]);
    await expect(queue.enqueue(() => Promise.resolve("ready"), signal)).resolves.toBe("ready");
  });

  it("prevents held/gamepad movement when the visible tab loses focus or becomes hidden", () => {
    const engaged = { visible: true, focused: true, paused: false, menuOpen: false, typing: false };
    expect(canUseContinuousControls(engaged)).toBe(true);
    expect(canUseContinuousControls({ ...engaged, focused: false })).toBe(false);
    expect(canUseContinuousControls({ ...engaged, visible: false })).toBe(false);
    expect(canUseContinuousControls({ ...engaged, paused: true })).toBe(false);
    expect(canUseContinuousControls({ ...engaged, typing: true })).toBe(false);
  });
});
