import { useEffect } from "react";

import { HOTKEYS } from "@/app/hotkeys";
import { isTextEntry, useHotkey } from "@/lib/hotkeys";
import { bindHistoryScope, redo, undo } from "@/state/history";
import { useToasts, type Toast } from "@/state/toasts";

/** The toast that says what was undone or redone: one at a time, each replacing the last. */
export const HISTORY_TOAST = "history";

function say(toast: Pick<Toast, "title" | "action">): void {
  useToasts.getState().push({ id: HISTORY_TOAST, tone: "info", ...toast });
}

/** Takes back the last step (`state/history.ts`) and says so: "Undid: Hide Pumpkin 3 · Redo". */
export function undoAndSay(): void {
  const step = undo();
  say(
    step
      ? { title: `Undid: ${step.label}`, action: { label: "Redo", run: redoAndSay } }
      : { title: "Nothing to undo" },
  );
}

/** Does again the last step undone and says so: "Redid: Hide Pumpkin 3 · Undo". */
export function redoAndSay(): void {
  const step = redo();
  say(
    step
      ? { title: `Redid: ${step.label}`, action: { label: "Undo", run: undoAndSay } }
      : { title: "Nothing to redo" },
  );
}

/** A key that means undo or redo, except in a text field, where the browser's own undo is. */
function onKey(run: () => void): (event: KeyboardEvent) => void {
  return (event) => {
    if (isTextEntry(event.target)) return;
    event.preventDefault();
    run();
  };
}

const onUndo = onKey(undoAndSay);
const onRedo = onKey(redoAndSay);

/**
 * Undo and redo from the keyboard (`HOTKEYS.undo`, `HOTKEYS.redo` and its `also`), bound by
 * `GlobalHotkeys`; and the history kept to the active site (`bindHistoryScope`). Unlike the
 * app's other keys they work from a button, a switch or a select, and only a text field,
 * a text area or editable content keeps Ctrl+Z for itself.
 */
export function useUndoHotkeys(): void {
  useEffect(() => bindHistoryScope(), []);
  useHotkey(HOTKEYS.undo.combo, onUndo, { allowInInputs: true });
  useHotkey(HOTKEYS.redo.combo, onRedo, { allowInInputs: true });
  useHotkey(HOTKEYS.redo.also[0], onRedo, { allowInInputs: true });
}
