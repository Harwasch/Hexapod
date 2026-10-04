/**
 * What the command box lists for a query, and which row Enter runs. Pure, so the rules are
 * tested without a scene (`__tests__/commandBox.test.tsx`).
 *
 * One box replaced three inputs (the search pill, the ⌘K palette and the agent bar), so it
 * must answer each of their questions without making the operator say which one they meant:
 * a place name flies there, "settings" opens Settings, "Z-21" selects the zone, and a
 * sentence of work goes to the agent. The groups are listed in a fixed order — what the
 * console already knows (sites, zones, plans, layers, actions) above what the geocoder finds
 * (places), so the best local match is the top row — and what changes with the words is
 * which row is highlighted for Enter (`defaultActiveId`).
 */

import type { LucideIcon } from "lucide-react";

import { isWorkRequest } from "@/lib/intents";

export type CommandGroupId =
  "places" | "sites" | "zones" | "plans" | "layers" | "actions" | "agent";

export interface CommandRow {
  /** Unique across every group: the option's DOM id and the highlight's key. */
  id: string;
  label: string;
  sub?: string;
  /** Extra words the row answers to besides its label. */
  keywords?: string;
  /** Key caps to show, from the hotkey registry (`app/hotkeys.ts`). */
  shortcut?: readonly string[];
  /** Drawn before the label; rows without one take their group's. */
  icon?: LucideIcon;
  run: () => void;
}

export interface CommandGroup {
  id: CommandGroupId;
  heading: string;
  rows: CommandRow[];
}

/**
 * Local matches first, then places, then the agent. Places arrive a beat after the keystroke;
 * listed below the local groups, they never push the highlighted row down the list.
 */
export const GROUP_ORDER: readonly CommandGroupId[] = [
  "sites",
  "zones",
  "plans",
  "layers",
  "actions",
  "places",
  "agent",
];

export const GROUP_HEADINGS: Record<CommandGroupId, string> = {
  places: "Places",
  sites: "Sites",
  zones: "Zones",
  plans: "Plans",
  layers: "Layers",
  actions: "Actions",
  agent: "Agent",
};

/** The most rows a group shows; the agent row is always one. */
export const GROUP_LIMITS: Record<CommandGroupId, number> = {
  places: 4,
  sites: 4,
  zones: 4,
  plans: 4,
  layers: 5,
  actions: 6,
  agent: 1,
};

export const AGENT_ROW_ID = "agent";

/** Lower-case words of a query: every one must appear in a row for it to match. */
function tokens(query: string): string[] {
  return query.toLowerCase().split(/\s+/).filter(Boolean);
}

/** Every word of the query appears somewhere in the row's label or keywords. */
export function rowMatches(query: string, row: Pick<CommandRow, "label" | "keywords">): boolean {
  const words = tokens(query);
  if (words.length === 0) return true;
  const haystack = `${row.label} ${row.keywords ?? ""}`.toLowerCase();
  return words.every((word) => haystack.includes(word));
}

/** Rows whose label starts with the query come before rows that merely contain it. */
function rank(query: string, rows: CommandRow[]): CommandRow[] {
  const q = query.trim().toLowerCase();
  const starts = (row: CommandRow) =>
    row.label.toLowerCase().startsWith(q) ||
    (row.keywords ?? "").toLowerCase().split(/\s+/).includes(q);
  return [...rows.filter(starts), ...rows.filter((row) => !starts(row))];
}

/**
 * The groups to show for a query.
 *
 * With nothing typed the box is a menu: every action with its key. With words, each group is
 * filtered to the rows that match (places come from the geocoder already matched), and the
 * last row asks the agent with the words as typed — everything the agent understands is
 * still one Enter away when no row matches.
 */
export function buildCommandGroups(
  query: string,
  candidates: Partial<Record<Exclude<CommandGroupId, "agent">, CommandRow[]>>,
  ask: ((text: string) => void) | null,
): CommandGroup[] {
  const text = query.trim();
  if (!text) {
    const actions = candidates.actions ?? [];
    return actions.length
      ? [{ id: "actions", heading: GROUP_HEADINGS.actions, rows: actions }]
      : [];
  }
  const groups: CommandGroup[] = [];
  for (const id of GROUP_ORDER) {
    if (id === "agent") continue;
    const rows = candidates[id] ?? [];
    const matched =
      id === "places"
        ? rows
        : rank(
            text,
            rows.filter((row) => rowMatches(text, row)),
          );
    const limited = matched.slice(0, GROUP_LIMITS[id]);
    if (limited.length) groups.push({ id, heading: GROUP_HEADINGS[id], rows: limited });
  }
  if (ask) {
    groups.push({
      id: "agent",
      heading: GROUP_HEADINGS.agent,
      rows: [{ id: AGENT_ROW_ID, label: `Ask the agent: “${text}”`, run: () => ask(text) }],
    });
  }
  return groups;
}

/** Words that open an instruction rather than name a thing ("show zones", "where is TR-07"). */
const INSTRUCTION =
  /^(?:show|hide|turn|enable|disable|toggle|select|find|locate|where|fly|go|take|jump|navigate|measure|reset|look|open|back|zoom|explore|draft|create|make|write|start|new|plan|tell|what|which|how|why|when|can|could|please)\b/;

/**
 * Whether Enter should ask the agent rather than run the best match: the words read as an
 * instruction or a question, or as a change to the plan draft that is open.
 */
export function prefersAgent(
  query: string,
  context: { draftOpen: boolean; localMatches: number },
): boolean {
  const lower = query.trim().toLowerCase();
  if (!lower) return false;
  if (isWorkRequest(lower) || lower.startsWith("plan:")) return true;
  if (INSTRUCTION.test(lower) || lower.endsWith("?")) return true;
  if (lower.split(/\s+/).length >= 4) return true;
  // With a draft open, plain words are a change to it ("two drones", "finish by Friday") —
  // unless they name something the box can find, like a zone.
  return context.draftOpen && context.localMatches === 0;
}

/**
 * The row Enter runs before the operator moves the highlight.
 *
 * The highlight goes to the agent when the words read as an instruction, else to the first
 * local match (a site, zone, plan, layer or action) — the top row, since the local groups are
 * listed first — else to the first place. Local matches beat places because they are there the
 * moment a key is pressed, while places arrive a beat later: Enter must not change meaning
 * under a fast typist.
 */
export function defaultActiveId(
  query: string,
  groups: CommandGroup[],
  context: { draftOpen: boolean },
): string | null {
  const local = groups.filter((g) => g.id !== "places" && g.id !== "agent");
  const firstLocal = local[0]?.rows[0]?.id ?? null;
  const localMatches = local.reduce((sum, g) => sum + g.rows.length, 0);
  const agent = groups.find((g) => g.id === "agent")?.rows[0]?.id ?? null;
  if (agent && prefersAgent(query, { draftOpen: context.draftOpen, localMatches })) return agent;
  const firstPlace = groups.find((g) => g.id === "places")?.rows[0]?.id ?? null;
  return firstLocal ?? firstPlace ?? agent ?? groups[0]?.rows[0]?.id ?? null;
}

/** The next row id when the highlight moves by `step`, wrapping at either end. */
export function moveActive(
  rows: CommandRow[],
  current: string | null,
  step: 1 | -1,
): string | null {
  if (rows.length === 0) return null;
  const index = rows.findIndex((row) => row.id === current);
  if (index < 0) return rows[step === 1 ? 0 : rows.length - 1]?.id ?? null;
  return rows[(index + step + rows.length) % rows.length]?.id ?? null;
}
