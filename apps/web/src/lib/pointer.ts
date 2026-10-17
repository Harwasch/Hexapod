/**
 * The verb for touching the map. On a phone or tablet nobody clicks: a finger taps. Read from
 * the primary pointer (`(pointer: coarse)`), so a laptop with a touch screen and a mouse still
 * says "Click", and from where the copy is drawn, so it follows a device that changes.
 */

/** Whether the primary pointer is a finger. False where the browser cannot say (tests, SSR). */
export function touchFirst(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}

/** "Tap" on a touch screen, "Click" with a mouse; lower case for mid-sentence use. */
export function tapOrClick(lower = false): string {
  const verb = touchFirst() ? "Tap" : "Click";
  return lower ? verb.toLowerCase() : verb;
}
