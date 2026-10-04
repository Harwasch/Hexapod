import { useSyncExternalStore } from "react";

/**
 * The phone breakpoint: the same width as the `@media (max-width: 640px)` blocks in `app.css`.
 * CSS lays the HUD out; this is for the rare part that has to move a node rather than restyle
 * it (the data credits, `MapCorner`). Keep the two in step.
 */
export const PHONE_MEDIA = "(max-width: 640px)";

/** Whether `query` matches now, re-rendering when that changes (a resize, a rotation). */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/**
 * A touch screen as the main pointer (a phone, a tablet): no hover, a finger. There is no
 * Shift, Alt or wheel, so what those keys do is offered as buttons there instead
 * (features/sites/ObjectCard.tsx). A laptop with a touch screen has a fine main pointer and is
 * not one.
 */
export const TOUCH_MEDIA = "(hover: none) and (pointer: coarse)";
