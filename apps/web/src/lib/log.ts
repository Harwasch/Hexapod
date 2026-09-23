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

/**
 * One readable sentence for an error of any shape.
 *
 * Cesium rejects with plain objects (a `RequestErrorEvent`, an aborted request) whose JSON
 * is `{}`; showing that to a person says nothing, so empty shapes fall back to a sentence.
 */
export function describeError(error: unknown): string {
  if (error instanceof Error) return error.message || "Something went wrong.";
  if (typeof error === "string") return error || "Something went wrong.";
  if (error && typeof error === "object") {
    const { message, statusCode } = error as { message?: unknown; statusCode?: unknown };
    if (typeof message === "string" && message) return message;
    if (typeof statusCode === "number") return `The server answered ${statusCode}.`;
    // A class with its own `toString` (Cesium's `RequestErrorEvent` has one) says more than JSON.
    const toText = (error as { toString?: () => string }).toString;
    if (typeof toText === "function" && toText !== Object.prototype.toString) {
      const text = toText.call(error);
      if (text) return text;
    }
  }
  try {
    const json = JSON.stringify(error);
    if (json && json !== "{}" && json !== "null") return json;
  } catch {
    // fall through
  }
  return "The request failed — check the connection.";
}
