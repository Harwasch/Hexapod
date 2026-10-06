import { create } from "zustand";

import { loadCustomSets, saveCustomSets, withCustomSets, type CustomSet } from "@/lib/customSets";
import type { InstancesDoc } from "@/lib/instances";
import { cycleIndex } from "@/lib/sceneSelect";
import { withSupersedes } from "@/lib/supersedes";
import { record } from "@/state/history";
import { useInstances } from "@/state/instances";
import { activeSupersedes, onSupersedesChange } from "@/state/supersedes";

/** What a painted area matched (lib/sceneSelect.ts `bestSet`). */
export interface PaintResult {
  /**
   * The best match's instances (ids of the scan's document with its painted objects): one, or
   * a combination; empty when nothing painted carries an instance.
   */
  ids: readonly number[];
  /** The first of `ids`, or null: the best instance when one is the match. */
  best: number | null;
  iou: number;
  /** Splats painted (visible and under the brush). */
  painted: number;
  /**
   * Set while the stroke is still being painted: the match so far, highlighted but not yet
   * selected (the selection is made when the stroke ends).
   */
  live?: boolean;
}

/** What a plain stroke of the brush does: start again, add to the painted area, take away. */
export type StrokeMode = "replace" | "add" | "subtract";

/**
 * A combination of instances the brush selected (lib/sceneSelect.ts `bestSet`): instances of
 * disjoint subtrees whose union matched the painted area better than any one of them.
 */
export interface Combination {
  /** The members, the largest first (`PaintSetMatch.ids`). */
  ids: readonly number[];
  /** Their union's overlap with the painted area (IoU). */
  iou: number;
}

/**
 * Selecting a scan's objects in the scene (cesium/sceneSelect/): what a click offered, which
 * of it is chosen (one instance, or a combination the brush matched), where it was clicked, the
 * brush, and per scan the objects painted in this browser (lib/customSets.ts). The highlight
 * itself is the objects store's (`state/instances.ts`); the controller keeps it on what is
 * selected (`selectedIds`), and while a stroke is painted on its match so far. The HUD's
 * selection card (features/sites/ObjectCard.tsx) shows all of it.
 */
interface SceneSelectState {
  /** The scan the candidates are of. */
  assetId: string | null;
  /**
   * Instance ids, leaf → top of the main hit first (lib/sceneSelect.ts `buildCandidates`). For
   * a combination, its first member, then what its members are parts of together, coarser one
   * by one (`selectSet`).
   */
  candidates: number[];
  /** How many of `candidates` are the main hit's chain. */
  chain: number;
  /** The chosen candidate, or -1. */
  index: number;
  /**
   * Set when the brush selected a combination: candidate 0 is then the combination, not its
   * first member alone (`selectedIds`); cycling up reaches what holds it, and back.
   */
  combination: Combination | null;
  /** Where the click that made the selection was (CSS px in the viewport). */
  anchor: { x: number; y: number } | null;
  /** Clicking picks; painting collects splats under a brush. */
  mode: "pick" | "paint";
  /** Brush radius, CSS px. */
  brush: number;
  /**
   * What a stroke does with no modifier held: the card's New / Add / Remove, for a touch
   * screen, which has no Shift or Alt. Back to "replace" whenever the brush is taken up.
   */
  strokeMode: StrokeMode;
  /** What the last stroke matched, while painting. */
  paint: PaintResult | null;
  /** Per asset id, its painted objects, once read from storage. */
  custom: Record<string, CustomSet[]>;
  select: (
    assetId: string,
    candidates: readonly number[],
    chain: number,
    index: number,
    anchor: { x: number; y: number } | null,
  ) => void;
  /**
   * Selects a combination (two members or more): `above` is what its members are parts of
   * together, the lowest first (lib/sceneSelect.ts `commonChain`), offered as the coarser
   * candidates.
   */
  selectSet: (
    assetId: string,
    combination: Combination,
    above: readonly number[],
    anchor: { x: number; y: number } | null,
  ) => void;
  cycle: (step: number) => void;
  clear: () => void;
  setMode: (mode: "pick" | "paint") => void;
  setBrush: (radius: number) => void;
  setStrokeMode: (strokeMode: StrokeMode) => void;
  setPaint: (paint: PaintResult | null) => void;
  /** The scan's painted objects, read from storage the first time. */
  customOf: (assetId: string) => CustomSet[];
  /** Keeps a painted object; one undoable step (`state/history.ts`, `recordCustom`). */
  addCustom: (assetId: string, set: CustomSet) => void;
  /** Forgets a painted object; one undoable step, which puts it back where it was. */
  removeCustom: (assetId: string, key: string) => void;
}

const NONE: CustomSet[] = [];

export const MIN_BRUSH = 4;
export const MAX_BRUSH = 120;

/** The scan's painted objects as `sets`, stored. */
function putCustom(assetId: string, sets: CustomSet[]): void {
  saveCustomSets(assetId, sets);
  useSceneSelect.setState((s) => ({ custom: { ...s.custom, [assetId]: sets } }));
}

/**
 * Records a change of the scan's painted objects from `before` to what they are now: undo
 * and redo put the whole list back, so a painted object returns at its place and every later
 * one keeps its id (ids follow the order, lib/customSets.ts `customId`). A selection of a
 * painted object is cleared when they change under it, as Delete clears it.
 */
function recordCustom(assetId: string, label: string, before: CustomSet[]): void {
  const after = useSceneSelect.getState().custom[assetId] ?? [];
  const put = (sets: CustomSet[]): void => {
    const state = useSceneSelect.getState();
    const table = useInstances.getState().assets[assetId]?.instances ?? [];
    const maxId = table.reduce((max, i) => Math.max(max, i.id), 0);
    if (state.assetId === assetId && state.candidates.some((id) => id > maxId)) state.clear();
    putCustom(assetId, sets);
  };
  record({
    label,
    scope: "site",
    alive: () => useInstances.getState().assets[assetId] !== undefined,
    undo: () => put(before),
    redo: () => put(after),
  });
}

export const useSceneSelect = create<SceneSelectState>()((set, get) => ({
  assetId: null,
  candidates: [],
  chain: 0,
  index: -1,
  combination: null,
  anchor: null,
  mode: "pick",
  brush: 18,
  strokeMode: "replace",
  paint: null,
  custom: {},
  select: (assetId, candidates, chain, index, anchor) =>
    set({
      assetId,
      candidates: [...candidates],
      chain,
      index: candidates.length === 0 ? -1 : Math.max(0, Math.min(candidates.length - 1, index)),
      combination: null,
      ...(anchor ? { anchor } : {}),
    }),
  selectSet: (assetId, combination, above, anchor) => {
    const first = combination.ids[0];
    if (first === undefined) return;
    const candidates = [first, ...above.filter((id) => !combination.ids.includes(id))];
    set({
      assetId,
      candidates,
      chain: candidates.length,
      index: 0,
      combination: { ids: [...combination.ids], iou: combination.iou },
      ...(anchor ? { anchor } : {}),
    });
  },
  cycle: (step) =>
    set((s) =>
      s.candidates.length < 2 ? s : { index: cycleIndex(s.index, s.candidates.length, step) },
    ),
  clear: () =>
    set({ candidates: [], chain: 0, index: -1, combination: null, paint: null, anchor: null }),
  setMode: (mode) => set({ mode, paint: null, strokeMode: "replace" }),
  setBrush: (radius) => set({ brush: Math.max(MIN_BRUSH, Math.min(MAX_BRUSH, radius)) }),
  setStrokeMode: (strokeMode) => set({ strokeMode }),
  setPaint: (paint) => set({ paint }),
  customOf: (assetId) => {
    const known = get().custom[assetId];
    if (known) return known;
    const loaded = loadCustomSets(assetId);
    // Read once, kept: a store update inside a selector's render is avoided by deferring.
    queueMicrotask(() => {
      if (!get().custom[assetId]) set((s) => ({ custom: { ...s.custom, [assetId]: loaded } }));
    });
    return loaded.length ? loaded : NONE;
  },
  addCustom: (assetId, entry) => {
    const before = get().custom[assetId] ?? loadCustomSets(assetId);
    putCustom(assetId, [...before, entry]);
    recordCustom(assetId, `Create ${entry.name}`, before);
  },
  removeCustom: (assetId, key) => {
    const before = get().custom[assetId] ?? loadCustomSets(assetId);
    const gone = before.find((c) => c.key === key);
    if (!gone) return;
    putCustom(
      assetId,
      before.filter((c) => c !== gone),
    );
    recordCustom(assetId, `Delete ${gone.name}`, before);
  },
}));

/** The chosen candidate's id, or null; a combination's first member while it is chosen. */
export function selectedId(state: Pick<SceneSelectState, "candidates" | "index">): number | null {
  return state.candidates[state.index] ?? null;
}

/** The combination, while it is the chosen candidate; else null. */
export function chosenCombination(
  state: Pick<SceneSelectState, "combination" | "index">,
): Combination | null {
  return state.index === 0 ? state.combination : null;
}

const NO_IDS: readonly number[] = [];

/**
 * What is selected, as instance ids: a combination's members, else the chosen candidate alone;
 * none when nothing is. What Hide, Show only, Fly to and the highlight act on.
 */
export function selectedIds(
  state: Pick<SceneSelectState, "candidates" | "index" | "combination">,
): readonly number[] {
  const combination = chosenCombination(state);
  if (combination) return combination.ids;
  const id = selectedId(state);
  return id === null ? NO_IDS : [id];
}

/** Whether the selection card is about a scan object: one is chosen, or the brush is out. */
export function objectSelected(
  state: Pick<SceneSelectState, "candidates" | "index" | "mode">,
): boolean {
  return state.mode === "paint" || selectedId(state) !== null;
}

/**
 * `base` (a scan's `instances.json`) with the scan's painted objects drawn as instances of
 * their own (lib/customSets.ts `withCustomSets`), and the measured splats its chosen fill
 * supersedes under their reserved id (lib/supersedes.ts, state/supersedes.ts): what the
 * renderers draw from. The same object while neither changes.
 */
export function effectiveDoc(assetId: string, base: InstancesDoc): InstancesDoc {
  const painted = withCustomSets(base, useSceneSelect.getState().customOf(assetId));
  return withSupersedes(painted, activeSupersedes(assetId));
}

/**
 * Calls `listener` whenever what `effectiveDoc` draws for the scan changes: its painted
 * objects, or the splats its fill supersedes. Returns the unsubscriber.
 */
export function onCustomSetsChange(assetId: string, listener: () => void): () => void {
  const offPainted = useSceneSelect.subscribe((state, previous) => {
    if (state.custom[assetId] !== previous.custom[assetId]) listener();
  });
  const offSuperseded = onSupersedesChange(assetId, listener);
  return () => {
    offPainted();
    offSuperseded();
  };
}
