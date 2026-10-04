/**
 * How fast the splat overlay draws while the camera moves: the number to compare renderers by
 * in person (docs/WEBGPU_TRIAL.md), shown in the developer readouts.
 *
 * Only motion frames are counted -- frames drawn because the camera moved since the last one
 * (ScanRendererHost marks them). The overlay draws nothing at rest (overlayFrames.ts), and
 * what it does draw then -- a tile arriving, the full-resolution frame after the camera rests,
 * a sort result -- comes at the pace of the network or a deadline, not of the GPU: counted, a
 * still view would read as 5 fps and every gesture's end as a 200 ms stall. While the camera
 * moves the overlay draws in the globe's own frame (its `postRender`), so the rate of motion
 * frames is the rate the user sees, the globe and the scan together.
 *
 * A reading is the latest run of motion frames -- consecutive frames no more than `GAP_MS`
 * apart, the last `WINDOW_MS` of it at most -- and it stays after the camera stops, marked
 * not live, so a phone's numbers can be read once the finger is off the screen:
 *
 * - `fps`: frames in the run over its duration;
 * - `p95Ms`: the 95th percentile of the time between frames, which is where a fast turn's
 *   hitches show and an average hides them;
 * - `cpuMs`: the median main-thread time of the renderer's draw call. A WebGPU (and, on most
 *   drivers, WebGL) draw call only records and submits work: it is the renderer's cost to the
 *   main thread, not its GPU time, which shows only as a lower `fps`.
 */

/** Frames further apart than this belong to different gestures (ms). */
export const GAP_MS = 500;
/** The longest stretch a reading covers (ms): recent enough to follow a change of view. */
export const WINDOW_MS = 2000;
/** Fewer intervals than this are no reading (a nudge, not a gesture). */
export const MIN_INTERVALS = 8;
/** A reading is live while its last frame is this recent (ms). */
export const LIVE_MS = 500;
/** Frames kept: several seconds of motion at 60 fps. */
const CAPACITY = 240;

export interface FrameReading {
  fps: number;
  p95Ms: number;
  cpuMs: number;
  /** Intervals the reading is made of. */
  frames: number;
  /** The camera is still moving: the last frame is under `LIVE_MS` old. */
  live: boolean;
}

/** The value below which a share `q` (0 to 1) of `values` lies (nearest rank). */
function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[index] ?? 0;
}

export class FrameMeter {
  private readonly at: number[] = [];
  private readonly cpu: number[] = [];

  /** A motion frame drawn at `at` (`performance.now()` ms), its draw call taking `cpuMs`. */
  record(at: number, cpuMs: number): void {
    this.at.push(at);
    this.cpu.push(cpuMs);
    if (this.at.length > CAPACITY) {
      this.at.shift();
      this.cpu.shift();
    }
  }

  /** The latest gesture's numbers, or null before there is one. */
  reading(now: number): FrameReading | null {
    // From the newest frame back, the first run with enough intervals in it.
    let end = this.at.length - 1;
    while (end > 0) {
      let start = end;
      while (start > 0) {
        const previous = this.at[start - 1] ?? 0;
        const current = this.at[start] ?? 0;
        if (current - previous > GAP_MS || (this.at[end] ?? 0) - previous > WINDOW_MS) break;
        start -= 1;
      }
      if (end - start >= MIN_INTERVALS) return this.summarise(start, end, now);
      end = start - 1;
    }
    return null;
  }

  private summarise(start: number, end: number, now: number): FrameReading {
    const intervals: number[] = [];
    for (let i = start + 1; i <= end; i++)
      intervals.push((this.at[i] ?? 0) - (this.at[i - 1] ?? 0));
    const span = (this.at[end] ?? 0) - (this.at[start] ?? 0);
    return {
      fps: span > 0 ? (intervals.length * 1000) / span : 0,
      p95Ms: quantile(intervals, 0.95),
      cpuMs: quantile(this.cpu.slice(start, end + 1), 0.5),
      frames: intervals.length,
      live: now - (this.at[end] ?? 0) < LIVE_MS,
    };
  }
}
