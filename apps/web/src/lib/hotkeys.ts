import { useEffect } from "react";

type Handler = (event: KeyboardEvent) => void;

/**
 * Registers a keyboard shortcut like "mod+k", "escape", "shift+/" or "n".
 * Ignores keystrokes typed into inputs unless `allowInInputs` is set, and keys something else
 * already acted on (`defaultPrevented`): a dialog, popover or tooltip closing on Escape, the
 * measuring tool or explore mode taking theirs. One press does one thing.
 */
export function useHotkey(
  combo: string,
  handler: Handler,
  options: { allowInInputs?: boolean; enabled?: boolean } = {},
): void {
  const { allowInInputs = false, enabled = true } = options;
  useEffect(() => {
    if (!enabled) return;
    const parts = combo.toLowerCase().split("+");
    const key = parts.at(-1) ?? "";
    const wantMod = parts.includes("mod");
    const wantShift = parts.includes("shift");
    const wantAlt = parts.includes("alt");
    const listener = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (!allowInInputs && isTyping(event.target)) return;
      const mod = event.metaKey || event.ctrlKey;
      if (wantMod !== mod) return;
      if (wantShift !== event.shiftKey) return;
      if (wantAlt !== event.altKey) return;
      if (event.key.toLowerCase() !== key) return;
      handler(event);
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [combo, handler, allowInInputs, enabled]);
}

export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/** Input types whose text the browser edits, with its own undo. */
const TEXT_INPUT_TYPES = new Set([
  "text",
  "search",
  "email",
  "url",
  "tel",
  "password",
  "number",
  "date",
  "datetime-local",
  "month",
  "time",
  "week",
]);

/**
 * Whether `target` is where text is typed -- a text field, a text area, editable content --
 * so the browser's own undo is what Ctrl+Z means there. Narrower than `isTyping`: a checkbox
 * or a select has no text to take back.
 */
export function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLInputElement) return TEXT_INPUT_TYPES.has(target.type);
  return (
    target.isContentEditable ||
    target.closest('[contenteditable]:not([contenteditable="false"])') !== null
  );
}

export const MOD_LABEL =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl";
