import { Event as CesiumEvent } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OverlayFrames, type FrameClock } from "@/cesium/scanView/overlayFrames";
import { MOVING_BUDGET_MS, RESTING_BUDGET_MS, TileWork } from "@/cesium/scanView/tileWork";
import {
  HELD_FRAME_MS,
  heldRedrawDelay,
  holdGlobeRenders,
  UI_PRIORITY_MS,
  UiActivity,
} from "@/cesium/uiActivity";

/** Timers and animation frames on a clock the test moves (16 ms a display frame). */
function clock() {
  let now = 0;
  let frames: (() => void)[] = [];
  let timers: { at: number; run: () => void; id: number }[] = [];
  let nextId = 1;
  const api: FrameClock = {
    now: () => now,
    requestFrame: (callback) => {
      frames.push(callback);
      return frames.length;
    },
    cancelFrame: () => {
      frames = [];
    },
    setTimer: (run, ms) => {
      const id = nextId++;
      timers.push({ at: now + ms, run, id });
      return id;
    },
    clearTimer: (id) => {
      timers = timers.filter((t) => t.id !== id);
    },
  };
  /** One display frame: due timers, then `beforeFrames` (a globe render), then the frames. */
  const frame = (beforeFrames?: () => void): void => {
    now += 16;
    const due = timers.filter((t) => t.at <= now);
    timers = timers.filter((t) => t.at > now);
    for (const t of due) t.run();
    beforeFrames?.();
    const run = frames;
    frames = [];
    for (const callback of run) callback();
  };
  return { api, frame, at: () => now };
}

describe("the interface holds the 3D view (the rule)", () => {
  it("never holds when the interface does not", () => {
    expect(heldRedrawDelay(false, 1000, 1001)).toBe(0);
  });

  it("holds a redraw until HELD_FRAME_MS after the last one, then lets it through", () => {
    expect(heldRedrawDelay(true, 1000, 1010)).toBe(HELD_FRAME_MS - 10);
    expect(heldRedrawDelay(true, 1000, 1000 + HELD_FRAME_MS)).toBe(0);
    expect(heldRedrawDelay(true, Number.NEGATIVE_INFINITY, 5)).toBe(0);
  });

  it("allows a handful of redraws a second, not sixty", () => {
    expect(1000 / HELD_FRAME_MS).toBeLessThanOrEqual(5);
  });
});

describe("what holds the view", () => {
  afterEach(() => {
    document.body.innerHTML = "";
    vi.useRealTimers();
  });

  it("an open popover holds it for as long as it is open", async () => {
    document.body.innerHTML = `<canvas id="globe"></canvas><div id="hud"></div>`;
    const activity = new UiActivity(document.getElementById("globe") as Element);
    expect(activity.holding).toBe(false);
    const menu = document.createElement("div");
    menu.setAttribute("data-hud-popover", "");
    document.getElementById("hud")?.appendChild(menu);
    await Promise.resolve();
    expect(activity.popoverOpen).toBe(true);
    expect(activity.holding).toBe(true);
    menu.remove();
    await Promise.resolve();
    expect(activity.holding).toBe(false);
    activity.destroy();
  });

  it("pointing at a control holds it briefly, without counting as a press", () => {
    vi.useFakeTimers({ toFake: ["performance"] });
    document.body.innerHTML = `<canvas id="globe"></canvas><button id="b">Layers</button>`;
    const canvas = document.getElementById("globe") as Element;
    const activity = new UiActivity(canvas);
    document.getElementById("b")?.dispatchEvent(new Event("pointermove", { bubbles: true }));
    expect(activity.holding).toBe(true);
    // The splat decoders keep their pace: only a press is `active`.
    expect(activity.active).toBe(false);
    vi.advanceTimersByTime(UI_PRIORITY_MS + 1);
    expect(activity.holding).toBe(false);
    // The pointer over the globe is the camera's business.
    canvas.dispatchEvent(new Event("pointermove", { bubbles: true }));
    expect(activity.holding).toBe(false);
    activity.destroy();
  });
});

describe("the globe's requested renders, while the interface holds the view", () => {
  /** A scene with CesiumJS's render decision: a request, or a moved camera. */
  function scene() {
    const s = {
      preUpdate: new CesiumEvent(),
      postRender: new CesiumEvent(),
      _renderRequested: false,
      requestRender() {
        s._renderRequested = true;
      },
      renders: 0,
      /** One `Scene.render` call: true when it rendered. */
      render(cameraMoved = false): boolean {
        s.preUpdate.raiseEvent();
        const should = s._renderRequested || cameraMoved;
        if (!should) return false;
        s._renderRequested = false;
        s.renders += 1;
        s.postRender.raiseEvent();
        return true;
      },
    };
    return s;
  }

  it("draws requests at most every HELD_FRAME_MS, and the one held back once it may", () => {
    const c = clock();
    const s = scene();
    let holding = true;
    const off = holdGlobeRenders(s, () => holding, c.api);
    // Tiles land every frame for a second.
    for (let i = 0; i < 62; i++) {
      s.requestRender();
      c.frame(() => s.render());
    }
    expect(s.renders).toBeLessThanOrEqual(Math.ceil(1000 / HELD_FRAME_MS) + 1);
    expect(s.renders).toBeGreaterThanOrEqual(3);
    // The last request is not lost: drawn once its wait is over.
    const before = s.renders;
    for (let i = 0; i < 20; i++) c.frame(() => s.render());
    expect(s.renders).toBe(before + 1);
    // Let go: every request draws at once again.
    holding = false;
    for (let i = 0; i < 10; i++) {
      s.requestRender();
      c.frame(() => s.render());
    }
    expect(s.renders).toBe(before + 11);
    off();
  });

  it("never holds a moved camera", () => {
    const c = clock();
    const s = scene();
    const off = holdGlobeRenders(s, () => true, c.api);
    for (let i = 0; i < 30; i++) c.frame(() => s.render(true));
    expect(s.renders).toBe(30);
    off();
  });
});

describe("the splat overlay's frames, while the interface holds the view", () => {
  function overlay(holding: () => boolean) {
    const c = clock();
    const listeners = new Set<() => void>();
    const state = { changed: false, draws: 0 };
    const frames = new OverlayFrames(
      (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      {
        changed: () => state.changed,
        draw: () => {
          state.draws += 1;
          state.changed = false;
          return { again: false, by: null };
        },
      },
      c.api,
      holding,
    );
    const globe = (): void => {
      for (const listener of listeners) listener();
    };
    return { c, frames, state, globe };
  }

  it("draws what tiles and sorts ask for at most every HELD_FRAME_MS", () => {
    const { c, frames, state } = overlay(() => true);
    for (let i = 0; i < 62; i++) {
      frames.wake("tiles");
      c.frame();
    }
    expect(state.draws).toBeLessThanOrEqual(Math.ceil(1000 / HELD_FRAME_MS) + 1);
    expect(state.draws).toBeGreaterThanOrEqual(3);
    // The last one is drawn once it may be.
    const before = state.draws;
    for (let i = 0; i < 20; i++) c.frame();
    expect(state.draws).toBe(before + 1);
    expect(frames.waiting).toBe(false);
  });

  it("draws a moved camera at once, in step with the globe", () => {
    const { c, state, globe } = overlay(() => true);
    for (let i = 0; i < 10; i++) {
      state.changed = true;
      c.frame(globe);
    }
    expect(state.draws).toBe(10);
  });

  it("draws every woken frame when the interface does not hold the view", () => {
    const { c, frames, state } = overlay(() => false);
    for (let i = 0; i < 30; i++) {
      frames.wake("tiles");
      c.frame();
    }
    expect(state.draws).toBe(30);
  });
});

describe("tile work, while the interface holds the view", () => {
  it("takes the moving budget, and pauses HELD_FRAME_MS between frames with work", async () => {
    let now = 0;
    let frames: (() => void)[] = [];
    const later: { at: number; run: () => void }[] = [];
    const work = new TileWork({
      now: () => now,
      nextFrame: (callback) => frames.push(callback),
      later: (run, ms) => later.push({ at: now + ms, run }),
    });
    work.holding = () => true;
    const log: number[] = [];
    for (let id = 1; id <= 3; id++)
      void work.run(() => {
        now += MOVING_BUDGET_MS + 1;
        log.push(id);
      });
    // One a frame (the moving budget; at rest several would fit `RESTING_BUDGET_MS`).
    expect(MOVING_BUDGET_MS + 1).toBeLessThan(RESTING_BUDGET_MS);
    expect(log).toEqual([1]);
    // The next frame with work comes only after the pause.
    expect(frames).toHaveLength(0);
    expect(later).toHaveLength(1);
    expect((later[0]?.at ?? 0) - now).toBe(HELD_FRAME_MS);
    later.shift()?.run();
    const run = frames;
    frames = [];
    for (const callback of run) callback();
    await Promise.resolve();
    expect(log).toEqual([1, 2]);
  });
});
