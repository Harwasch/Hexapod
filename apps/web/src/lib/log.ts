/**
 * Minimal structured logger with an instrumentation seam. Swap `sink` for a
 * vendor later without touching call sites.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface LogEvent {
  level: LogLevel;
  scope: string;
  message: string;
  data?: Record<string, unknown>;
  at: number;
}

export type LogSink = (event: LogEvent) => void;

const consoleSink: LogSink = (event) => {
  const line = `[${event.scope}] ${event.message}`;
  if (event.level === "error") console.error(line, event.data ?? "");
  else if (event.level === "warn") console.warn(line, event.data ?? "");
  else if (event.level === "info") console.info(line, event.data ?? "");
};

let sink: LogSink = consoleSink;
const recent: LogEvent[] = [];

export function setLogSink(next: LogSink): void {
  sink = next;
}

export function recentLogs(): readonly LogEvent[] {
  return recent;
}

export function createLogger(scope: string) {
  const emit = (level: LogLevel, message: string, data?: Record<string, unknown>) => {
    const event: LogEvent = { level, scope, message, data, at: Date.now() };
    recent.push(event);
    if (recent.length > 200) recent.shift();
    sink(event);
  };
  return {
    debug: (message: string, data?: Record<string, unknown>) => emit("debug", message, data),
    info: (message: string, data?: Record<string, unknown>) => emit("info", message, data),
    warn: (message: string, data?: Record<string, unknown>) => emit("warn", message, data),
    error: (message: string, data?: Record<string, unknown>) => emit("error", message, data),
  };
}

export function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
