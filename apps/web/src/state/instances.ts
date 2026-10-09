import { create } from "zustand";

import { indexCategories, matchCategory, type CategoryIndex } from "@/lib/categories";
import { parseQuery, searchInstances, withDescendants, type Instance } from "@/lib/instances";
import { selectionLabel } from "@/lib/sceneSelect";

import { canExtend, currentTask, discard, record, touched, type UndoStep } from "./history";

/** What the highlight is of, so the panel can mark it and a second click clears it. */
export type Focus =
  | { kind: "category"; id: string }
  | { kind: "object"; id: number }
  /** The query's matches, or only those in one category. */
  | { kind: "matches"; category?: string }
  | { kind: "ids" };

/** One scan's objects (lib/instances.ts, lib/categories.ts) and what the viewer does with them. */
export interface AssetInstances {
  instances: Instance[];
  /** The instances by broad category and object (`indexCategories`). */
  index: CategoryIndex;
  /**
   * Instance ids not drawn. Exact: the renderers hide these ids and nothing else, so a
   * category's or an object's members (never another category's instances under them).
   */
  hidden: ReadonlySet<number>;
  /** Instance ids tinted (exact, as `hidden`); while any are, the rest are dimmed. */
  highlighted: ReadonlySet<number>;
  /** What `highlighted` is of, or null when nothing is. */
  focus: Focus | null;
  query: string;
  /** Object ids the query matches, best first: what "Hide all" and "Show only" act on. */
  matches: readonly number[];
}

/** A splat renderer that cannot hide or highlight this scan's objects, and why. */
export interface RendererGap {
  /** The renderer drawing the scan now. */
  renderer: string;
  reason: string;
}

interface InstancesState {
  /** Per asset id, present while the asset's tileset declares instances and they loaded. */
  assets: Record<string, AssetInstances>;
  /** Whether everything but a highlight is dimmed while one is active (always, in the app). */
  dimOthers: boolean;
  /**
   * Per asset id, set while the splat renderer drawing it cannot apply hide and highlight
   * (`cesium/scanView`): the panel says so and offers the CesiumJS renderer.
   */
  gaps: Record<string, RendererGap>;
  setGap: (assetId: string, gap: RendererGap | null) => void;
  /**
   * Per asset id, set while the splat renderer drawing it cannot move its objects (skins in
   * the wind, telemetry; `cesium/scanView/scanMotion.ts`): the objects panel and the wind
   * control say so and offer the CesiumJS renderer.
   */
  motionGaps: Record<string, RendererGap>;
  setMotionGap: (assetId: string, gap: RendererGap | null) => void;
  setTable: (assetId: string, table: { instances: Instance[] } | null) => void;
  setQuery: (assetId: string, query: string) => void;
  // What changes `hidden` (below, up to `reset`) is one undoable step each (`state/history.ts`,
  // `changeHidden`); the query and the highlight are not recorded.
  /** Hides or shows instances with everything below them (`withDescendants`). */
  setHidden: (assetId: string, ids: readonly number[], hidden: boolean) => void;
  /** Hides or shows objects: their members (`SceneObject.members`). */
  setObjectsHidden: (assetId: string, objectIds: readonly number[], hidden: boolean) => void;
  /** Hides or shows every object of a category. */
  setCategoryHidden: (assetId: string, categoryId: string, hidden: boolean) => void;
  /** Hides every object the current query matches. */
  hideMatches: (assetId: string) => void;
  /** Hides everything but the objects the current query matches. */
  showOnlyMatches: (assetId: string) => void;
  showAll: (assetId: string) => void;
  /** Highlights instances with everything below them; an empty list clears the highlight. */
  highlight: (assetId: string, ids: readonly number[]) => void;
  /** Highlights a category, an object or the query's matches; the same again (or null) clears. */
  toggleFocus: (assetId: string, focus: Focus | null) => void;
  /** Nothing hidden, nothing highlighted (undo brings back what was hidden). */
  reset: (assetId: string) => void;
  setDimOthers: (dim: boolean) => void;
}

const EMPTY: ReadonlySet<number> = new Set();

function patch(
  state: InstancesState,
  assetId: string,
  change: (current: AssetInstances) => Partial<AssetInstances>,
): Pick<InstancesState, "assets"> | InstancesState {
  const current = state.assets[assetId];
  if (!current) return state;
  return { assets: { ...state.assets, [assetId]: { ...current, ...change(current) } } };
}

/** `gaps` with `assetId`'s set to `gap` (or cleared), or `gaps` itself when nothing changes. */
function withGap(
  gaps: Record<string, RendererGap>,
  assetId: string,
  gap: RendererGap | null,
): Record<string, RendererGap> {
  const current = gaps[assetId];
  if (gap === null) {
    if (!current) return gaps;
    return Object.fromEntries(Object.entries(gaps).filter(([id]) => id !== assetId));
  }
  if (current?.renderer === gap.renderer && current.reason === gap.reason) return gaps;
  return { ...gaps, [assetId]: gap };
}

// ---- Undo --------------------------------------------------------------------------------

/**
 * A second change of the same thing this soon after the first (an eye flicked off and on)
 * folds into one step: the net change, or none when it came back to where it was.
 */
export const TOGGLE_COALESCE_MS = 800;

/** One undoable change of a scan's hidden set: the exact sets before and after. */
interface HiddenStep extends UndoStep {
  assetId: string;
  /** The table it is of: a reloaded scan's ids are another scan's, and the step dies. */
  table: readonly Instance[];
  before: ReadonlySet<number>;
  after: ReadonlySet<number>;
  /** What was changed ("objects:3", "category:trees"), for folding a quick repeat in. */
  key: string;
  at: number;
  task: number;
}

let lastHidden: HiddenStep | null = null;

function sameMembers(a: ReadonlySet<number>, b: ReadonlySet<number>): boolean {
  if (a === b) return true;
  if (a.size !== b.size) return false;
  for (const id of a) if (!b.has(id)) return false;
  return true;
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** What a person calls instance `id`: its object's name as the panel lists it, else the card's. */
function instanceName(entry: AssetInstances, id: number): string {
  const object = entry.index.objects.get(id);
  if (object) return capitalise(object.name);
  const instance = entry.instances.find((i) => i.id === id);
  // Not in the file: an object painted in this browser (lib/customSets.ts).
  if (!instance) return "Painted object";
  return selectionLabel(instance, id, entry.index.categoryOf.get(id));
}

function namesOf(entry: AssetInstances, ids: readonly number[]): string {
  const [only] = ids;
  return ids.length === 1 && only !== undefined
    ? instanceName(entry, only)
    : `${ids.length.toLocaleString()} objects`;
}

/** Some objects in words: "Pumpkin 3", "Trees" (all of them), "3 objects in Trees". */
function objectsName(entry: AssetInstances, objectIds: readonly number[]): string {
  const objects = objectIds.flatMap((id) => entry.index.objects.get(id) ?? []);
  const [first] = objects;
  if (!first) return `${objectIds.length.toLocaleString()} objects`;
  if (objects.length === 1) return capitalise(first.name);
  const group = entry.index.groups.find((g) => g.category.id === first.category);
  if (!group || objects.some((o) => o.category !== first.category))
    return `${objects.length.toLocaleString()} objects`;
  return objects.length === group.objects.length
    ? group.category.name
    : `${objects.length.toLocaleString()} objects in ${group.category.name}`;
}

/** Of `ids`, those whose parent is not among them: the tops of what they cover. */
function topsOf(entry: AssetInstances, ids: ReadonlySet<number>): number[] {
  const parentOf = new Map(entry.instances.map((i) => [i.id, i.parent]));
  return [...ids].filter((id) => {
    const parent = parentOf.get(id);
    return parent === null || parent === undefined || !ids.has(parent);
  });
}

/** The tops of what is wholly drawn with `hidden` hidden: drawn, and all below it too. */
function wholeShownTops(entry: AssetInstances, hidden: ReadonlySet<number>): number[] {
  const children = new Map<number, number[]>();
  for (const i of entry.instances) {
    if (i.parent === null) continue;
    const list = children.get(i.parent) ?? [];
    list.push(i.id);
    children.set(i.parent, list);
  }
  const whole = new Map<number, boolean>();
  const isWhole = (id: number): boolean => {
    const known = whole.get(id);
    if (known !== undefined) return known;
    const value = !hidden.has(id) && (children.get(id) ?? []).every(isWhole);
    whole.set(id, value);
    return value;
  };
  return entry.instances
    .filter((i) => isWhole(i.id) && (i.parent === null || !isWhole(i.parent)))
    .map((i) => i.id);
}

/**
 * A change of the hidden set in words, from what it did: for a change made by ids (the
 * selection card's Hide and Show only, through `setHidden`) or by several actions in one
 * gesture ("Show only" is show all, then hide the rest).
 */
function describeHiddenChange(
  entry: AssetInstances,
  before: ReadonlySet<number>,
  after: ReadonlySet<number>,
): string {
  const added = new Set([...after].filter((id) => !before.has(id)));
  const removed = new Set([...before].filter((id) => !after.has(id)));
  if (added.size === 0) {
    const shownTops = topsOf(entry, removed);
    return after.size === 0 && shownTops.length > 1
      ? "Show all objects"
      : `Show ${namesOf(entry, shownTops)}`;
  }
  const hiddenTops = topsOf(entry, added);
  if (removed.size === 0 && hiddenTops.length === 1) return `Hide ${namesOf(entry, hiddenTops)}`;
  const shown = wholeShownTops(entry, after);
  if (shown.length > 0 && shown.length < hiddenTops.length)
    return `Show only ${namesOf(entry, shown)}`;
  return removed.size === 0 ? `Hide ${namesOf(entry, hiddenTops)}` : "Change what is shown";
}

function putHidden(assetId: string, hidden: ReadonlySet<number>): void {
  useInstances.setState((s) => patch(s, assetId, () => ({ hidden })));
}

/**
 * Records that `assetId`'s hidden set went from `before` to `after`: a step of its own, or
 * folded into the last one when it continues it -- the same gesture (one task), or the same
 * thing changed again within `TOGGLE_COALESCE_MS`. `label` null describes the change.
 */
function noteHidden(
  entry: AssetInstances,
  assetId: string,
  key: string,
  label: string | null,
  before: ReadonlySet<number>,
  after: ReadonlySet<number>,
): void {
  const now = Date.now();
  const task = currentTask();
  const last = lastHidden;
  if (
    last &&
    canExtend(last) &&
    last.assetId === assetId &&
    last.table === entry.instances &&
    (last.task === task || (last.key === key && now - last.at < TOGGLE_COALESCE_MS))
  ) {
    const gesture = last.task === task && last.key !== key;
    Object.assign(last, { after, at: now, task, key });
    last.label =
      gesture || label === null ? describeHiddenChange(entry, last.before, after) : label;
    if (sameMembers(last.before, after)) {
      discard(last);
      lastHidden = null;
    } else {
      touched(last);
    }
    return;
  }
  const step: HiddenStep = {
    label: label ?? describeHiddenChange(entry, before, after),
    scope: "site",
    assetId,
    table: entry.instances,
    before,
    after,
    key,
    at: now,
    task,
    alive: () => useInstances.getState().assets[assetId]?.instances === step.table,
    undo: () => putHidden(assetId, step.before),
    redo: () => putHidden(assetId, step.after),
  };
  record(step);
  lastHidden = step;
}

/**
 * Sets `assetId`'s hidden set to what `change` makes of it (null: leave it) as one undoable
 * step; a change that changes nothing is neither made nor recorded.
 */
function changeHidden(
  assetId: string,
  key: string,
  label: ((entry: AssetInstances) => string) | null,
  change: (current: AssetInstances) => ReadonlySet<number> | null,
): void {
  const current = useInstances.getState().assets[assetId];
  if (!current) return;
  const hidden = change(current);
  if (hidden === null || sameMembers(hidden, current.hidden)) return;
  putHidden(assetId, hidden);
  noteHidden(current, assetId, key, label ? label(current) : null, current.hidden, hidden);
}

/** `hidden` with `ids` added (`hide`) or taken out. */
function withIds(
  hidden: ReadonlySet<number>,
  ids: Iterable<number>,
  hide: boolean,
): ReadonlySet<number> {
  const next = new Set(hidden);
  for (const id of ids) {
    if (hide) next.add(id);
    else next.delete(id);
  }
  return next;
}

/** "Hide 12 matches for “pumpkin”". */
function matchesLabel(verb: string, entry: AssetInstances): string {
  const n = entry.matches.length;
  return `${verb} ${n.toLocaleString()} ${n === 1 ? "match" : "matches"} for “${entry.query.trim()}”`;
}

/** Whether two highlights are of the same thing (a second click on it clears it). */
export function sameFocus(a: Focus | null, b: Focus | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  if (a.kind === "matches" && b.kind === "matches") return a.category === b.category;
  return !("id" in a) || a.id === (b as typeof a).id;
}

/** The instance ids of some objects. */
function membersOf(index: CategoryIndex, objectIds: Iterable<number>): number[] {
  const out: number[] = [];
  for (const id of objectIds) out.push(...(index.objects.get(id)?.members ?? []));
  return out;
}

/** The instance ids a focus lights. */
function focusMembers(entry: AssetInstances, focus: Focus): readonly number[] {
  switch (focus.kind) {
    case "category":
      return entry.index.groups.find((g) => g.category.id === focus.id)?.members ?? [];
    case "object":
      return entry.index.objects.get(focus.id)?.members ?? [];
    case "matches":
      return membersOf(
        entry.index,
        focus.category === undefined
          ? entry.matches
          : entry.matches.filter((id) => entry.index.objects.get(id)?.category === focus.category),
      );
    case "ids":
      return [...entry.highlighted];
  }
}

/**
 * The objects a query matches, best first: objects whose tags (or property filters) match,
 * each found through any of its parts; and, for words that name a category ("trees",
 * "water"), every object in it, first.
 */
export function matchObjects(
  entry: Pick<AssetInstances, "instances" | "index">,
  query: string,
): number[] {
  const parsed = parseQuery(query);
  if (parsed.terms.length === 0 && parsed.filters.length === 0) return [];
  const best = new Map<number, number>();
  const note = (objectId: number | undefined, score: number): void => {
    if (objectId === undefined) return;
    if (score > (best.get(objectId) ?? -1)) best.set(objectId, score);
  };
  if (parsed.filters.length === 0) {
    for (const group of entry.index.groups) {
      if (matchCategory(parsed.terms, group.category) <= 0) continue;
      for (const object of group.objects) note(object.id, 2 + object.splats * 1e-12);
    }
  }
  for (const result of searchInstances(entry.instances, parsed, Number.POSITIVE_INFINITY)) {
    note(entry.index.objectOf.get(result.id), result.score);
  }
  const splats = (id: number): number => entry.index.objects.get(id)?.splats ?? 0;
  return [...best].sort((a, b) => b[1] - a[1] || splats(b[0]) - splats(a[0])).map(([id]) => id);
}

export const useInstances = create<InstancesState>()((set) => ({
  assets: {},
  dimOthers: true,
  gaps: {},
  setGap: (assetId, gap) =>
    set((s) => {
      const gaps = withGap(s.gaps, assetId, gap);
      return gaps === s.gaps ? s : { gaps };
    }),
  motionGaps: {},
  setMotionGap: (assetId, gap) =>
    set((s) => {
      const motionGaps = withGap(s.motionGaps, assetId, gap);
      return motionGaps === s.motionGaps ? s : { motionGaps };
    }),
  setTable: (assetId, table) =>
    set((s) => {
      const next: Record<string, AssetInstances> = {};
      for (const [id, entry] of Object.entries(s.assets)) if (id !== assetId) next[id] = entry;
      if (table && table.instances.length > 0) {
        next[assetId] = {
          instances: table.instances,
          index: indexCategories(table.instances),
          hidden: EMPTY,
          highlighted: EMPTY,
          focus: null,
          query: "",
          matches: [],
        };
      }
      return { assets: next };
    }),
  setQuery: (assetId, query) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const matches = matchObjects(current, query);
        // A highlight of the old matches follows the query.
        if (current.focus?.kind === "matches") {
          const lit = focusMembers({ ...current, matches }, current.focus);
          return {
            query,
            matches,
            highlighted: lit.length ? new Set(lit) : EMPTY,
            focus: lit.length ? current.focus : null,
          };
        }
        return { query, matches };
      }),
    ),
  setHidden: (assetId, ids, hidden) =>
    changeHidden(assetId, `ids:${String(ids[0])}:${String(ids.length)}`, null, (current) =>
      withIds(current.hidden, withDescendants(current, ids), hidden),
    ),
  setObjectsHidden: (assetId, objectIds, hidden) =>
    changeHidden(
      assetId,
      `objects:${objectIds.join(",")}`,
      (entry) => `${hidden ? "Hide" : "Show"} ${objectsName(entry, objectIds)}`,
      (current) => withIds(current.hidden, membersOf(current.index, objectIds), hidden),
    ),
  setCategoryHidden: (assetId, categoryId, hidden) =>
    changeHidden(
      assetId,
      `category:${categoryId}`,
      (entry) => {
        const group = entry.index.groups.find((g) => g.category.id === categoryId);
        return `${hidden ? "Hide" : "Show"} ${group?.category.name ?? "a category"}`;
      },
      (current) => {
        const group = current.index.groups.find((g) => g.category.id === categoryId);
        return group ? withIds(current.hidden, group.members, hidden) : null;
      },
    ),
  hideMatches: (assetId) =>
    changeHidden(
      assetId,
      "matches",
      (entry) => matchesLabel("Hide", entry),
      (current) =>
        current.matches.length === 0
          ? null
          : withIds(current.hidden, membersOf(current.index, current.matches), true),
    ),
  showOnlyMatches: (assetId) =>
    changeHidden(
      assetId,
      "matches",
      (entry) => matchesLabel("Show only", entry),
      (current) => {
        if (current.matches.length === 0) return null;
        const keep = new Set(membersOf(current.index, current.matches));
        return new Set(current.instances.filter((i) => !keep.has(i.id)).map((i) => i.id));
      },
    ),
  showAll: (assetId) =>
    changeHidden(
      assetId,
      "all",
      () => "Show all objects",
      () => EMPTY,
    ),
  highlight: (assetId, ids) =>
    set((s) =>
      patch(s, assetId, (current) =>
        ids.length
          ? { highlighted: withDescendants(current, ids), focus: { kind: "ids" } }
          : { highlighted: EMPTY, focus: null },
      ),
    ),
  toggleFocus: (assetId, focus) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (focus === null || sameFocus(current.focus, focus)) {
          return { highlighted: EMPTY, focus: null };
        }
        const lit = focusMembers(current, focus);
        return lit.length
          ? { highlighted: new Set(lit), focus }
          : { highlighted: EMPTY, focus: null };
      }),
    ),
  reset: (assetId) => {
    changeHidden(
      assetId,
      "reset",
      () => "Reset objects",
      () => EMPTY,
    );
    set((s) => patch(s, assetId, () => ({ highlighted: EMPTY, focus: null })));
  },
  setDimOthers: (dimOthers) => set({ dimOthers }),
}));
