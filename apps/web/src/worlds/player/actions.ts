/** A stalled connection must not accumulate an unlimited control history. */
export class SessionActionQueue {
  private tail: Promise<unknown> = Promise.resolve();
  private pending = 0;

  constructor(private readonly capacity = 32) {}

  async enqueue<T>(
    operation: () => Promise<T>,
    signal: AbortSignal,
    maxQueueAgeMs?: number,
  ): Promise<T> {
    signal.throwIfAborted();
    if (this.pending >= this.capacity)
      throw new Error("Control queue is full. Wait for the connection to recover.");
    const queuedAt = performance.now();
    this.pending++;
    const result = this.tail
      .catch(() => {
        /* Later releases must survive a rejected action. */
      })
      .then(() => {
        signal.throwIfAborted();
        if (maxQueueAgeMs !== undefined && performance.now() - queuedAt > maxQueueAgeMs)
          throw new Error("A delayed movement was discarded. Try the control again.");
        return operation();
      });
    this.tail = result;
    try {
      return await result;
    } finally {
      this.pending--;
    }
  }
}

export function canUseContinuousControls(state: {
  visible: boolean;
  focused: boolean;
  paused: boolean;
  menuOpen: boolean;
  typing: boolean;
}): boolean {
  return state.visible && state.focused && !state.paused && !state.menuOpen && !state.typing;
}
