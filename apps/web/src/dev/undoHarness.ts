/**
 * The app's undo keys and its toasts on a dev harness page, as `GlobalHotkeys` binds them in
 * the app (features/shell/undoHotkeys.ts): e2e/undoYard.spec.ts mounts it beside the scene
 * selection harness (`sceneSelectHarness.ts`) to take back the selection card's Hide and Show
 * only with the real keyboard. The toasts sit in the top right, where the app's right dock has
 * them.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle.
 */

import { createElement } from "react";
import { createRoot } from "react-dom/client";

import { Toasts } from "@/features/notices/Toasts";
import { useUndoHotkeys } from "@/features/shell/undoHotkeys";

function UndoKeys() {
  useUndoHotkeys();
  return createElement(Toasts);
}

export function mountUndo(): void {
  const root = document.createElement("div");
  Object.assign(root.style, {
    position: "fixed",
    top: "12px",
    right: "12px",
    width: "min(22rem, calc(100vw - 24px))",
    zIndex: "10",
  });
  root.dataset.testid = "undo-harness";
  document.body.appendChild(root);
  createRoot(root).render(createElement(UndoKeys));
}
