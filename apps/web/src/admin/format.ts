/** Formatting shared by the console's tables. Tabular numerals are set in admin.css. */

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${String(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : String(Math.round(value))} ${units[unit] ?? "TB"}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 1) return `${String(Math.round(seconds * 1000))} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${String(minutes)}m ${String(rest)}s`;
  return `${String(Math.floor(minutes / 60))}h ${String(minutes % 60)}m`;
}

/** Seconds between two ISO timestamps, or null when either is missing. */
export function elapsed(from: string | null | undefined, to: string | null | undefined) {
  if (!from || !to) return null;
  const span = (Date.parse(to) - Date.parse(from)) / 1000;
  return Number.isFinite(span) ? span : null;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "—";
  return at.toISOString().replace("T", " ").slice(0, 16);
}

export function formatCost(usd: number | string | null | undefined): string {
  if (usd === null || usd === undefined || usd === "") return "—";
  const value = typeof usd === "string" ? Number(usd) : usd;
  if (!Number.isFinite(value)) return "—";
  return value < 0.01 ? "<$0.01" : `$${value.toFixed(2)}`;
}

/** First segment of a uuid: enough to tell rows apart, short enough for a column. */
export function shortId(id: string): string {
  return id.split("-")[0] ?? id;
}

/** A parameter value as a table cell. `undefined` means the run did not set it at all. */
export function formatValue(value: unknown): string {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value) ?? "—";
}
