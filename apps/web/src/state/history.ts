import { create } from "zustand";

import { createLogger, describeError } from "@/lib/log";

import { useSites } from "./sites";

const log = createLogger("history");

/**
 * Undo and redo (Ctrl+Z / ⌘Z; Ctrl+Shift+Z / ⌘⇧Z or Ctrl+Y): one linear history of the
 * things a person does to what is shown, newest last, as every editor keeps it.
 *
 * A step is recorded where the change is made -- the stores' actions (`state/instances.ts`,
 * `state/sceneSelect.ts`) or the one helper a scene-side change goes through (layers,
 * measurements, an area's corners) -- so every caller of an action gets undo for free. A new
 * kind of change opts in with `recordAction(label, apply, revert)`, or `record` when it has
 * already been applied.
 *
 * What is recorded changes what the map shows and is unsurprising to take back: objects
 * hidden and shown, painted objects made and deleted, layers on and off, the swipe
 * comparison, measurements removed, an area's corners moved. Camera moves, selections and
 * highlights, settings, and writes to the server (renames, a scan's scale) are not.
 *
 * Steps about the active site's things (its scans' objects, its painted objects, its areas)
 * are `scope: "site"` and are dropped when the active site changes (`bindHistoryScope`):
 * Ctrl+Z at site B never reaches into site A, which may still be loaded but is not on
 * screen, and coming back to A starts afresh (its scans reload, so the old ids mean nothing).
 * Layers, the comparison and measurements are the globe's and stay. A step can also say it
 * no longer applies (`alive`: its scan was reloaded), and is then dropped when reached.
 */
export interface UndoStep {
  /** What it did, as the toast says it: "Hide Pumpkin 3". */
  label: string;
  undo: () => void;
  redo: () => void;
  /** "site": dropped when the active site changes. The default, "global", stays. */
  scope?: "site" | "global";
  /** False once the step can no longer apply (what it changed is gone). */
  alive?: () => boolean;
}

/** Steps kept; the oldest goes first. */
export const HISTORY_LIMIT = 50;

interface HistoryState {
  /** Done, oldest first: Ctrl+Z takes back the last. */
  past: readonly UndoStep[];
  /** Undone, the next to redo last. Emptied by any new step. */
  future: readonly UndoStep[];
}

export const useHistory = create<HistoryState>()(() => ({ past: [], future: [] }));

/** True while a step is being undone or redone: what it changes is not recorded again. */
let replaying = false;
/** The step last recorded, while nothing has been undone, redone or cleared since. */
let lastRecorded: UndoStep | null = null;
/** Counts tasks (event handlers): changes made in one are one gesture (`currentTask`). */
let task = 0;
let taskPending = false;

/** Whether an undo or redo is being applied now. */
export function isReplaying(): boolean {
  return replaying;
}

/**
 * The task (an event handler, a timer) running now, as a number that changes once it ends.
 * Changes made in one task are one gesture: a "Show only" that shows all and then hides the
 * rest is one step, not two.
 */
export function currentTask(): number {
  if (!taskPending) {
    taskPending = true;
    queueMicrotask(() => {
      taskPending = false;
      task += 1;
    });
  }
  return task;
}

/** Records a change already made. Ignored while undoing or redoing. Returns the step. */
export function record(step: UndoStep): UndoStep {
  if (replaying) return step;
  const past = [...useHistory.getState().past, step];
  useHistory.setState({ past: past.slice(-HISTORY_LIMIT), future: [] });
  lastRecorded = step;
  return step;
}

/** Applies a change and records it: `apply` again is its redo, `revert` its undo. */
export function recordAction(
  label: string,
  apply: () => void,
  revert: () => void,
  options: Pick<UndoStep, "scope" | "alive"> = {},
): UndoStep {
  apply();
  return record({ label, undo: revert, redo: apply, ...options });
}

/**
 * Whether `step` is the last thing done, so a change that continues it (the same toggle
 * again a moment later, the second half of one gesture) may fold into it instead of
 * recording another: nothing has been undone, redone or recorded since.
 */
export function canExtend(step: UndoStep): boolean {
  const { past, future } = useHistory.getState();
  return step === lastRecorded && past.at(-1) === step && future.length === 0;
}

/** Tells the history that `step` (the last one) changed in place: its label, its result. */
export function touched(step: UndoStep): void {
  if (!canExtend(step)) return;
  useHistory.setState((s) => ({ past: [...s.past] }));
}

/** Drops `step` (a change that came to nothing: a toggle and its toggle back). */
export function discard(step: UndoStep): void {
  if (lastRecorded === step) lastRecorded = null;
  useHistory.setState((s) => ({
    past: s.past.filter((p) => p !== step),
    future: s.future.filter((p) => p !== step),
  }));
}

/** Forgets everything, or only the site's steps. */
export function clearHistory(scope?: "site"): void {
  lastRecorded = null;
  if (!scope) {
    useHistory.setState({ past: [], future: [] });
    return;
  }
  const keep = (step: UndoStep): boolean => (step.scope ?? "global") !== scope;
  useHistory.setState((s) => ({ past: s.past.filter(keep), future: s.future.filter(keep) }));
}

function isAlive(step: UndoStep): boolean {
  try {
    return step.alive?.() ?? true;
  } catch {
    return false;
  }
}

/** The stacks with `from` replaced by `source` and, when given, `moved` put on the other. */
function moveTo(
  from: "past" | "future",
  source: readonly UndoStep[],
  moved?: UndoStep,
): Partial<HistoryState> {
  const { past, future } = useHistory.getState();
  if (from === "past") return { past: source, future: moved ? [...future, moved] : future };
  return { future: source, past: moved ? [...past, moved] : past };
}

/**
 * Runs the newest step on `from` and moves it to the other stack. A step that no longer
 * applies, or fails, is dropped and the one before it is tried. The step that ran, or null
 * when there was none.
 */
function replay(from: "past" | "future", run: (step: UndoStep) => void): UndoStep | null {
  lastRecorded = null;
  const source = [...useHistory.getState()[from]];
  const before = source.length;
  for (let next = source.pop(); next; next = source.pop()) {
    if (!isAlive(next)) continue;
    replaying = true;
    try {
      run(next);
    } catch (error) {
      log.warn("a step could not be replayed; dropped", {
        label: next.label,
        error: describeError(error),
      });
      continue;
    } finally {
      replaying = false;
    }
    useHistory.setState(moveTo(from, source, next));
    return next;
  }
  if (source.length !== before) useHistory.setState(moveTo(from, source));
  return null;
}

/** Takes back the last step. Returns it, or null when there was nothing to undo. */
export function undo(): UndoStep | null {
  return replay("past", (step) => step.undo());
}

/** Does again the last step undone. Returns it, or null when there was nothing to redo. */
export function redo(): UndoStep | null {
  return replay("future", (step) => step.redo());
}

let bound = false;

/**
 * Drops the site's steps whenever the active site changes (see the module comment). Called
 * once by the shell; returns the unsubscriber.
 */
export function bindHistoryScope(): () => void {
  if (bound) return () => undefined;
  bound = true;
  const off = useSites.subscribe((state, previous) => {
    if (state.activeSiteId !== previous.activeSiteId) clearHistory("site");
  });
  return () => {
    off();
    bound = false;
  };
}
