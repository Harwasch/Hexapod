import { useEffect } from "react";

type Handler = (event: KeyboardEvent) => void;

/**
 * Registers a keyboard shortcut like "mod+k", "escape", "shift+/" or "n".
 * Ignores keystrokes typed into inputs unless `allowInInputs` is set.
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

export const MOD_LABEL =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl";
