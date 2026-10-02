/**
 * Counters for what the splat overlay costs, read by e2e specs and by hand from the console
 * (`window.__twinStats`). They exist so that "the overlay draws nothing at rest" is a number a
 * test can hold, not a claim: `overlayDraws` counts every frame a dedicated renderer drew
 * (`ScanBackend.render`), `overlayLoopTicks` every tick of a renderer's own update loop
 * (PlayCanvas runs one beside ours), and `overlayWakes` every time something asked the overlay
 * for a frame, by reason.
 *
 * Exposed on `window` in a development build, or in any build opened with `?stats` in the URL,
 * and never otherwise: a production page keeps the counters (a few integer increments a frame)
 * but publishes nothing.
 */

export interface OverlayStats {
  overlayDraws: number;
  overlayLoopTicks: number;
  overlayWakes: Record<string, number>;
  /** The overlay canvas's backing size and the pixel ratio it was drawn at, last frame. */
  overlayCanvas: { width: number; height: number; pixelRatio: number };
}

export const overlayStats: OverlayStats = {
  overlayDraws: 0,
  overlayLoopTicks: 0,
  overlayWakes: {},
  overlayCanvas: { width: 0, height: 0, pixelRatio: 0 },
};

/** Whether the counters are published on `window.__twinStats`. */
export function statsExposed(): boolean {
  try {
    if (import.meta.env.DEV) return true;
    return new URLSearchParams(window.location.search).has("stats");
  } catch {
    return false;
  }
}

if (typeof window !== "undefined" && statsExposed()) {
  const target = window as unknown as { __twinStats?: Record<string, unknown> };
  // Shared with any other subsystem's counters: added to, never replaced; read live, so later
  // increments show without publishing again.
  const published = (target.__twinStats ??= {});
  for (const key of Object.keys(overlayStats) as (keyof OverlayStats)[]) {
    Object.defineProperty(published, key, {
      get: () => overlayStats[key],
      enumerable: true,
      configurable: true,
    });
  }
}

/** One frame drawn by a dedicated renderer onto `canvas`, at `pixelRatio`. */
export function countOverlayDraw(canvas: HTMLCanvasElement | null, pixelRatio: number): void {
  overlayStats.overlayDraws += 1;
  if (canvas) {
    overlayStats.overlayCanvas = { width: canvas.width, height: canvas.height, pixelRatio };
  }
}

/** One tick of a renderer's own update loop. */
export function countOverlayLoopTick(): void {
  overlayStats.overlayLoopTicks += 1;
}

/** Something asked the overlay for a frame. */
export function countOverlayWake(reason: string): void {
  overlayStats.overlayWakes[reason] = (overlayStats.overlayWakes[reason] ?? 0) + 1;
}
