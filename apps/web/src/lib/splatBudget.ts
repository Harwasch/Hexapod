/**
 * How many gaussians a view may draw, from how fast frames come while the camera moves.
 *
 * The device budget (lib/detail.ts) is a ceiling set from memory and the Detail choice; what
 * a GPU can actually sort and blend at a smooth rate is only known by trying. So, like
 * SuperSplat's viewer and Google Maps, the budget follows measured frame time: a window of
 * slow motion frames while the view draws near its budget cuts it (phones halve sooner), and a window
 * of fast ones while the budget is what limits the view raises it back towards the ceiling.
 * Only motion frames count: at rest nothing renders, and a still frame's time says nothing
 * about how a gesture will feel.
 *
 * The budget reaches the tiles through SiteManager's splat memory source: past 125% of it
 * the PerformanceManager coarsens the splats' screen-space error, below 70% it refines.
 */

import { isHandheld, SCAN_DETAIL_CONFIG } from "./detail";
const POLICY = SCAN_DETAIL_CONFIG.adaptive;
export const BUDGET_WINDOW_FRAMES = POLICY.windowFrames;
export const SLOW_FRAME_MS = 1000 / POLICY.slowFps;
export const FAST_FRAME_MS = 1000 / POLICY.fastFps;

export class AdaptiveSplatBudget {
  private current: number;
  private readonly floor: number;
  private readonly frames: number[] = [];
  private slowFrames = 0;

  /** Starts at `start` (the device budget; the ceiling if not given) and moves between a
   *  floor and `ceiling` from there. */
  constructor(
    readonly ceiling: number,
    start = ceiling,
    private readonly handheld = isHandheld(),
  ) {
    this.current = Math.min(start, ceiling);
    this.floor = Math.min(
      ceiling,
      Math.max(
        handheld ? POLICY.phoneMinSplats : POLICY.minSplats,
        Math.round((handheld ? start : ceiling) * POLICY.floorShare),
      ),
    );
  }

  get budget(): number {
    return this.current;
  }

  /**
   * One motion frame's interval, with the gaussians it drew. Returns true when the budget
   * changed.
   */
  frame(intervalMs: number, drawn: number): boolean {
    if (!(intervalMs > 0 && intervalMs < POLICY.maxIntervalMs)) return false;
    if (this.handheld) {
      this.slowFrames =
        intervalMs > SLOW_FRAME_MS && drawn > this.current * POLICY.slowDrawShare
          ? this.slowFrames + 1
          : 0;
      if (this.slowFrames >= POLICY.phoneSlowFrames) {
        this.slowFrames = 0;
        this.frames.length = 0;
        const next = Math.max(this.floor, Math.round(this.current * POLICY.phoneCut));
        if (next === this.current) return false;
        this.current = next;
        return true;
      }
    }
    this.frames.push(intervalMs);
    if (this.frames.length < BUDGET_WINDOW_FRAMES) return false;
    const sorted = [...this.frames].sort((a, b) => a - b);
    this.frames.length = 0;
    const median = sorted[sorted.length >> 1] ?? 0;
    let next = this.current;
    // Slow while drawing a good share of the budget: the splats are (part of) the cost.
    if (median > SLOW_FRAME_MS && drawn > this.current * POLICY.slowDrawShare)
      next = this.current * (this.handheld ? POLICY.phoneCut : POLICY.cut);
    // Fast while the budget is what holds the view back.
    else if (median < FAST_FRAME_MS && drawn > this.current * POLICY.fastDrawShare)
      next = this.current * POLICY.raise;
    next = Math.round(Math.min(this.ceiling, Math.max(this.floor, next)));
    if (next === this.current) return false;
    this.current = next;
    return true;
  }
}
