/**
 * Treatment rates learned from the work log: acres per machine-hour per task family, so the
 * planner's estimates rest on what the fleet actually did rather than a default.
 */

import type { WorkLogRow } from "./types";

export interface LearnedRate {
  task: string;
  acresPerMachineHour: number;
  samples: number;
  machineIds: string[];
}

/** "Mow pass 2" and "Mow thistle" share the family "mow". */
export function taskFamily(task: string): string {
  return /[a-z]+/i.exec(task)?.[0]?.toLowerCase() ?? "";
}

/** "5:40" → 5.67 hours; "6.5" and "6.5 h" also accepted. */
export function parseHours(text: string): number | null {
  const clock = /^(\d+):(\d{1,2})$/.exec(text.trim());
  if (clock) return Number(clock[1]) + Number(clock[2]) / 60;
  const plain = /^(\d+(?:\.\d+)?)\s*h?$/i.exec(text.trim());
  return plain ? Number(plain[1]) : null;
}

export function learnedRates(rows: readonly WorkLogRow[]): LearnedRate[] {
  const byFamily = new Map<
    string,
    { acres: number; hours: number; samples: number; machines: Set<string> }
  >();
  for (const row of rows) {
    const hours = parseHours(row.hours);
    const family = taskFamily(row.task);
    if (!family || row.acres === null || row.acres <= 0 || hours === null || hours <= 0) continue;
    const entry = byFamily.get(family) ?? { acres: 0, hours: 0, samples: 0, machines: new Set() };
    entry.acres += row.acres;
    entry.hours += hours;
    entry.samples += 1;
    entry.machines.add(row.machine);
    byFamily.set(family, entry);
  }
  return [...byFamily.entries()]
    .map(([task, e]) => ({
      task,
      acresPerMachineHour: Math.round((e.acres / e.hours) * 10) / 10,
      samples: e.samples,
      machineIds: [...e.machines].sort(),
    }))
    .sort((a, b) => b.samples - a.samples);
}
