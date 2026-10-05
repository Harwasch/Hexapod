import { Cartesian3, Event as CesiumEvent, Matrix4, PerspectiveFrustum, type Camera } from "cesium";
import { describe, expect, it } from "vitest";

import {
  MAX_WAIT_FOR_GLOBE,
  OverlayFrames,
  OverlayInputs,
  type FrameClock,
  type FrameOutcome,
} from "@/cesium/scanView/overlayFrames";
import {
  globePixelRatio,
  maxShDegree,
  MIN_MOTION_PIXEL_RATIO,
  overlayPixelRatio,
  splatMinPixelSize,
} from "@/cesium/scanView/quality";
import { overlayStats } from "@/cesium/scanView/stats";

/**
 * A display, a globe and an overlay: `frame()` is one display frame (timers due, then the
 * globe's render if it renders, then the overlay's animation-frame callback, or the other way
 * round), `globe()` a globe render on its own.
 */
function rig(outcomes: FrameOutcome[] = []) {
  let now = 0;
  let callbacks: (() => void)[] = [];
  let timers: { at: number; run: () => void; id: number }[] = [];
  let nextTimer = 1;
  const listeners = new Set<() => void>();
  const clock: FrameClock = {
    now: () => now,
    requestFrame: (callback) => {
      callbacks.push(callback);
      return callbacks.length;
    },
    cancelFrame: () => {
      callbacks = [];
    },
    setTimer: (run, ms) => {
      const id = nextTimer++;
      timers.push({ at: now + ms, run, id });
      return id;
    },
    clearTimer: (id) => {
      timers = timers.filter((t) => t.id !== id);
    },
  };
  const state = {
    changed: false,
    draws: 0,
    drawnAt: [] as number[],
    during: null as (() => void) | null,
  };
  const frames = new OverlayFrames(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    {
      changed: () => state.changed,
      draw: () => {
        state.draws += 1;
        state.drawnAt.push(now);
        // What a frame drew from is what the next compares with.
        state.changed = false;
        state.during?.();
        return outcomes.shift() ?? { again: false, by: null };
      },
    },
    clock,
  );
  const globe = (): void => {
    for (const listener of listeners) listener();
  };
  const frame = (options: { globe?: "before" | "after" } = {}): void => {
    now += 16;
    const due = timers.filter((t) => t.at <= now);
    timers = timers.filter((t) => t.at > now);
    for (const t of due) t.run();
    if (options.globe === "before") globe();
    const run = callbacks;
    callbacks = [];
    for (const callback of run) callback();
    if (options.globe === "after") globe();
  };
  return { frames, state, globe, frame, listeners };
}

describe("the overlay draws only when something changes", () => {
  it("draws nothing while nothing changes, however often the globe renders", () => {
    const { state, frame } = rig();
    // Wind on another site: the globe renders every frame, the overlay's inputs never move.
    for (let i = 0; i < 120; i++) frame({ globe: "before" });
    expect(state.draws).toBe(0);
  });

  it("draws in step with the globe when the camera, canvas or scan frame moved", () => {
    const { state, frame } = rig();
    state.changed = true;
    frame({ globe: "after" });
    expect(state.draws).toBe(1);
    // Drawn from the same inputs as the globe's frame: nothing more until they move again.
    for (let i = 0; i < 10; i++) frame({ globe: "before" });
    expect(state.draws).toBe(1);
  });

  it("draws a woken frame on the next display frame, once", () => {
    const { frames, state, frame } = rig();
    frames.wake("tiles");
    expect(frames.waiting).toBe(true);
    frame();
    expect(state.draws).toBe(1);
    for (let i = 0; i < 10; i++) frame();
    expect(state.draws).toBe(1);
    expect(overlayStats.overlayWakes.tiles).toBeGreaterThan(0);
  });

  it("draws a woken frame with the globe's when the globe renders that frame", () => {
    const { frames, state, frame } = rig();
    frames.wake("renderer");
    frame({ globe: "before" });
    frame();
    expect(state.draws).toBe(1);
  });

  it("waits for the globe's render while the camera moves, up to a limit", () => {
    const { frames, state, frame } = rig();
    state.changed = true;
    frames.wake("tiles");
    for (let i = 0; i < MAX_WAIT_FOR_GLOBE; i++) frame();
    expect(state.draws).toBe(0);
    frame();
    expect(state.draws).toBe(1);
  });

  it("keeps drawing while a frame asks for another (a fade), and stops once none does", () => {
    const again = { again: true, by: null };
    const { frames, state, frame } = rig([again, again, again]);
    frames.wake("start");
    for (let i = 0; i < 20; i++) frame();
    expect(state.draws).toBe(4);
  });

  it("draws by a deadline (the rested full-resolution frame, a held-back re-plan)", () => {
    const { frames, state, frame } = rig([{ again: false, by: 200 }]);
    frames.wake("start");
    frame();
    expect(state.draws).toBe(1);
    // Nothing until 200 ms, then exactly one frame.
    while (state.drawnAt.length < 2 && (state.drawnAt[0] ?? 0) < 1000) frame();
    expect(state.draws).toBe(2);
    expect(state.drawnAt[1]).toBeGreaterThanOrEqual(200);
    expect(state.drawnAt[1]).toBeLessThan(200 + 3 * 16);
    for (let i = 0; i < 60; i++) frame();
    expect(state.draws).toBe(2);
  });

  it("keeps the earliest deadline when several are asked for", () => {
    const { frames, state, frame } = rig();
    frames.wakeBy(500, "late");
    frames.wakeBy(100, "early");
    frames.wakeBy(300, "middle");
    for (let i = 0; i < 10; i++) frame();
    expect(state.draws).toBe(1);
    expect(state.drawnAt[0]).toBeLessThan(200);
  });

  it("a wake while drawing (the renderer asking again) is not lost", () => {
    const { frames, state, frame } = rig();
    let first = true;
    state.during = () => {
      if (first) frames.wake("renderer");
      first = false;
    };
    frames.wake("start");
    frame();
    frame();
    expect(state.draws).toBe(2);
  });

  it("a frame that throws stops the overlay and never reaches the globe's render", () => {
    // CesiumJS's own event, raised as Scene.render raises `postRender`: outside the try that
    // turns a render error into `renderError`, so a listener's throw would stop its render loop.
    const postRender = new CesiumEvent();
    const failures: unknown[] = [];
    let draws = 0;
    const frames = new OverlayFrames((listener) => postRender.addEventListener(listener), {
      changed: () => true,
      draw: () => {
        draws += 1;
        // The renderer threw before the frame recorded its inputs: still "changed".
        throw new Error("Cannot read properties of null (reading 'hasCenters')");
      },
      failed: (error) => failures.push(error),
    });
    expect(() => postRender.raiseEvent()).not.toThrow();
    expect(draws).toBe(1);
    expect(failures).toHaveLength(1);
    expect((failures[0] as Error).message).toContain("hasCenters");
    // Stopped: off the globe's event, and nothing more is drawn however often it renders.
    expect(postRender.numberOfListeners).toBe(0);
    frames.wake("tiles");
    for (let i = 0; i < 5; i++) postRender.raiseEvent();
    expect(draws).toBe(1);
    expect(frames.waiting).toBe(false);

    // The same from the animation frame, and from `changed` itself.
    let callbacks: (() => void)[] = [];
    const clock: FrameClock = {
      now: () => 0,
      requestFrame: (callback) => callbacks.push(callback),
      cancelFrame: () => {
        callbacks = [];
      },
      setTimer: () => 0,
      clearTimer: () => undefined,
    };
    const told: unknown[] = [];
    let drawn = 0;
    const other = new OverlayFrames(
      () => () => undefined,
      {
        changed: () => {
          throw new Error("tileset destroyed");
        },
        draw: () => {
          drawn += 1;
          return { again: false, by: null };
        },
        failed: (error) => told.push(error),
      },
      clock,
    );
    other.wake("tiles");
    const run = callbacks;
    callbacks = [];
    expect(() => run.forEach((callback) => callback())).not.toThrow();
    expect(told).toHaveLength(1);
    other.wake("tiles");
    expect(callbacks).toHaveLength(0);
    expect(drawn).toBe(0);
  });

  it("stops: no frames, no deadlines, and off the globe's event", () => {
    const { frames, state, frame, listeners } = rig();
    frames.wakeBy(50, "deadline");
    frames.wake("tiles");
    frames.stop();
    for (let i = 0; i < 10; i++) frame({ globe: "before" });
    expect(state.draws).toBe(0);
    expect(listeners.size).toBe(0);
  });
});

describe("what an overlay frame is drawn from", () => {
  const camera = (x: number, dir = new Cartesian3(0, 1, 0)): Camera =>
    ({
      positionWC: new Cartesian3(x, -30, 5),
      directionWC: dir,
      upWC: new Cartesian3(0, 0, 1),
      frustum: new PerspectiveFrustum({ fov: 1, aspectRatio: 1.6, near: 0.1, far: 1e4 }),
    }) as unknown as Camera;
  const size = { width: 960, height: 600, pixelRatio: 2 };

  it("changes with the camera, the canvas, the resolution and the scan's frame", () => {
    const inputs = new OverlayInputs();
    expect(inputs.changed(camera(0), size, Matrix4.IDENTITY)).toBe(true);
    inputs.commit(camera(0), size, Matrix4.IDENTITY);
    expect(inputs.changed(camera(0), size, Matrix4.IDENTITY)).toBe(false);
    // Under a millimetre is no move (the overlay's motion test always said so); 2 mm is.
    expect(inputs.changed(camera(0.0005), size, Matrix4.IDENTITY)).toBe(false);
    expect(inputs.changed(camera(0.002), size, Matrix4.IDENTITY)).toBe(true);
    const turned = Cartesian3.normalize(new Cartesian3(0.001, 1, 0), new Cartesian3());
    expect(inputs.changed(camera(0, turned), size, Matrix4.IDENTITY)).toBe(true);
    expect(inputs.changed(camera(0), { ...size, width: 961 }, Matrix4.IDENTITY)).toBe(true);
    expect(inputs.changed(camera(0), { ...size, pixelRatio: 1.5 }, Matrix4.IDENTITY)).toBe(true);
    // Clamped to the ground after an asynchronous terrain sample: the scan's frame moved.
    const lifted = Matrix4.fromTranslation(new Cartesian3(0, 0, 0.3));
    expect(inputs.changed(camera(0), size, lifted)).toBe(true);
  });
});

describe("the overlay's resolution follows the globe's", () => {
  const at = (devicePixelRatio: number, browserRecommended: boolean, resolutionScale: number) => ({
    devicePixelRatio,
    browserRecommended,
    resolutionScale,
  });
  const desk = { handheld: false };

  it("performance: one device pixel per CSS pixel, whatever the display", () => {
    expect(globePixelRatio(at(2, true, 1))).toBe(1);
    expect(overlayPixelRatio(at(2, true, 1), { ...desk, moving: false })).toBe(1);
    expect(overlayPixelRatio(at(2, true, 1), { ...desk, moving: true })).toBe(
      MIN_MOTION_PIXEL_RATIO,
    );
  });

  it("balanced: the globe's 1.5 while it is, every device pixel once sharpened", () => {
    // Moving at the preset's base scale (1.5 / 2): the overlay's own cut is lower.
    expect(overlayPixelRatio(at(2, false, 0.75), { ...desk, moving: true })).toBeCloseTo(1.2);
    expect(overlayPixelRatio(at(2, false, 0.75), { ...desk, moving: false })).toBeCloseTo(1.5);
    expect(overlayPixelRatio(at(2, false, 1), { ...desk, moving: false })).toBe(2);
  });

  it("the adaptive ladder's cuts reach the overlay", () => {
    // Half resolution on balanced at 2x: 2 * 0.5 * 0.75.
    expect(overlayPixelRatio(at(2, false, 0.375), { ...desk, moving: true })).toBeCloseTo(0.75);
    expect(overlayPixelRatio(at(2, false, 0.65), { ...desk, moving: false })).toBeCloseTo(1.3);
    expect(overlayPixelRatio(at(1, false, 0.5), { ...desk, moving: false })).toBeCloseTo(0.5);
  });

  it("ultra: every device pixel, under the overlay's own caps", () => {
    expect(overlayPixelRatio(at(2, false, 1), { ...desk, moving: false })).toBe(2);
    expect(overlayPixelRatio(at(3, false, 1), { ...desk, moving: false })).toBe(2);
    expect(overlayPixelRatio(at(3, false, 1), { handheld: true, moving: false })).toBe(1.5);
    expect(overlayPixelRatio(at(3, false, 1), { handheld: true, moving: true })).toBeCloseTo(0.9);
  });

  it("keeps the smallest splat at half a CSS pixel, and a phone to one SH band", () => {
    expect(splatMinPixelSize(2)).toBe(1);
    expect(splatMinPixelSize(1)).toBe(0.5);
    // From afar the whole scan is a few dozen pixels: nothing in it is culled for size.
    expect(splatMinPixelSize(2, true)).toBe(0);
    expect(splatMinPixelSize(1, false)).toBe(0.5);
    expect(maxShDegree(true)).toBe(1);
    expect(maxShDegree(false)).toBe(3);
  });
});
