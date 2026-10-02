/**
 * How many gaussians a view may draw, from how fast frames come while the camera moves.
 *
 * The device budget (lib/detail.ts) is a ceiling set from memory and the Detail choice; what
 * a GPU can actually sort and blend at a smooth rate is only known by trying. So, like
 * SuperSplat's viewer and Google Maps, the budget follows measured frame time: a window of
 * slow motion frames while the view draws near its budget cuts it by a fifth, and a window
 * of fast ones while the budget is what limits the view raises it back towards the ceiling.
 * Only motion frames count: at rest nothing renders, and a still frame's time says nothing
 * about how a gesture will feel.
 *
 * The budget reaches the tiles through SiteManager's splat memory source: past 125% of it
 * the PerformanceManager coarsens the splats' screen-space error, below 70% it refines.
 */

/** Motion frames per decision. */
export const BUDGET_WINDOW_FRAMES = 45;
/** A window whose median frame is slower than this (30 fps) cuts the budget. */
export const SLOW_FRAME_MS = 1000 / 30;
/** A window whose median frame is faster than this (50 fps) may raise it. */
export const FAST_FRAME_MS = 1000 / 50;
const CUT = 0.8;
const RAISE = 1.15;
/** The budget never drops below this share of the ceiling (nor below `MIN_SPLATS`). */
const FLOOR_SHARE = 0.3;
const MIN_SPLATS = 300_000;

export class AdaptiveSplatBudget {
  private current: number;
  private readonly floor: number;
  private readonly frames: number[] = [];

  /** Starts at `start` (the device budget; the ceiling if not given) and moves between a
   *  floor and `ceiling` from there. */
  constructor(
    readonly ceiling: number,
    start = ceiling,
  ) {
    this.current = Math.min(start, ceiling);
    this.floor = Math.min(ceiling, Math.max(MIN_SPLATS, Math.round(ceiling * FLOOR_SHARE)));
  }

  get budget(): number {
    return this.current;
  }

  /**
   * One motion frame's interval, with the gaussians it drew. Returns true when the budget
   * changed.
   */
  frame(intervalMs: number, drawn: number): boolean {
    if (!(intervalMs > 0 && intervalMs < 2000)) return false;
    this.frames.push(intervalMs);
    if (this.frames.length < BUDGET_WINDOW_FRAMES) return false;
    const sorted = [...this.frames].sort((a, b) => a - b);
    this.frames.length = 0;
    const median = sorted[sorted.length >> 1] ?? 0;
    let next = this.current;
    // Slow while drawing a good share of the budget: the splats are (part of) the cost.
    if (median > SLOW_FRAME_MS && drawn > this.current * 0.6) next = this.current * CUT;
    // Fast while the budget is what holds the view back.
    else if (median < FAST_FRAME_MS && drawn > this.current * 0.8) next = this.current * RAISE;
    next = Math.round(Math.min(this.ceiling, Math.max(this.floor, next)));
    if (next === this.current) return false;
    this.current = next;
    return true;
  }
}
