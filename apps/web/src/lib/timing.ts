/** Lightweight performance instrumentation: named spans recorded in memory. */

export interface TimingSpan {
  name: string;
  durationMs: number;
  at: number;
  detail?: Record<string, unknown>;
}

const spans: TimingSpan[] = [];
const listeners = new Set<(span: TimingSpan) => void>();

export function recordSpan(
  name: string,
  durationMs: number,
  detail?: Record<string, unknown>,
): void {
  const span: TimingSpan = { name, durationMs, at: Date.now(), detail };
  spans.push(span);
  if (spans.length > 100) spans.shift();
  for (const listener of listeners) listener(span);
}

export function onSpan(listener: (span: TimingSpan) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function recentSpans(): readonly TimingSpan[] {
  return spans;
}

export async function timed<T>(
  name: string,
  work: () => Promise<T>,
  detail?: Record<string, unknown>,
): Promise<T> {
  const started = performance.now();
  try {
    return await work();
  } finally {
    recordSpan(name, performance.now() - started, detail);
  }
}
