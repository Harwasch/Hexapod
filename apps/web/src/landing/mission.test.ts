/**
 * The mission section (mission.ts, scene.js startMissionScene): the scroll arithmetic, and the
 * story the scroll scrubs through, which must only ever go forward as the reader scrolls down.
 */
import { describe, expect, it } from "vitest";

import { progressThrough, scrollForStep, stepAt, STAGES } from "./mission";
import { missionStory, type StoryBot } from "./scene.js";

describe("the mission's scroll", () => {
  it("runs from 0 at the section's top to 1 when it unpins", () => {
    expect(progressThrough(300, 5000, 900)).toBe(0);
    expect(progressThrough(0, 5000, 900)).toBe(0);
    expect(progressThrough(-2050, 5000, 900)).toBeCloseTo(0.5);
    expect(progressThrough(-4100, 5000, 900)).toBe(1);
    expect(progressThrough(-9000, 5000, 900)).toBe(1);
    // A section no taller than the window has nowhere to scroll.
    expect(progressThrough(-10, 800, 900)).toBe(0);
  });

  it("gives each of the four steps an equal stretch", () => {
    expect([0, 0.24, 0.26, 0.5, 0.74, 0.76, 1].map(stepAt)).toEqual([0, 0, 1, 2, 2, 3, 3]);
  });

  it("jumps to a scroll position inside the step asked for", () => {
    for (let k = 0; k < STAGES.length; k++) {
      const y = scrollForStep(k, 5000, 900);
      expect(stepAt(progressThrough(-y, 5000, 900))).toBe(k);
    }
  });
});

describe("the mission's story", () => {
  const story = missionStory();
  /** Each status value as a number, or NaN for the ones in words. */
  const counts = (status: [string, string][]) => status.map(([, v]) => Number.parseInt(v, 10));
  const finite = (b: StoryBot | undefined) => !b?.p || b.p.every(Number.isFinite);

  it("runs plan, deploy, scan, operate, each for a stretch of time", () => {
    expect(story.spans.map((s) => s.name)).toEqual([...STAGES]);
    for (const s of story.spans) expect(s.t1 - s.t0, s.name).toBeGreaterThan(10);
  });

  for (const span of story.spans)
    it(`only moves forward through ${span.name}: its counts never go back`, () => {
      let was: number[] = [];
      for (let t = span.t0; t <= span.t1; t += 0.1) {
        const { m, status } = story.at(span.name, t);
        for (const b of [...m.atvs, ...(m.staged ?? []), ...m.hex, m.vtol, m.quad])
          expect(finite(b), `${span.name} at ${t.toFixed(1)} s`).toBe(true);
        const now = counts(status);
        now.forEach((n, i) => {
          const before = was[i] ?? Number.NaN;
          if (Number.isFinite(n) && Number.isFinite(before))
            expect(n, `${status[i]?.[0] ?? ""} at ${t.toFixed(1)} s`).toBeGreaterThanOrEqual(
              before,
            );
        });
        was = now;
      }
    });

  it("ends each stage with its job done", () => {
    const end = (name: (typeof STAGES)[number]) => {
      const s = story.spans.find((x) => x.name === name);
      return Object.fromEntries(story.at(name, s?.t1 ?? 0).status);
    };
    expect(end("plan")).toMatchObject({ Boundary: "160 ha", "Hub site": "by the road" });
    expect(end("deploy").Hub).toBe("online");
    const off = (end("deploy")["Hexapods off"] ?? "").split(" of ");
    expect(off[0]).toBe(off[1]);
    expect(end("scan").Mapped).toBe("100%");
    expect(Number(end("scan")["Broom flagged"])).toBeGreaterThan(0);
    const pulled = (end("operate")["Broom pulled"] ?? "").split(" of ");
    expect(pulled[0]).toBe(pulled[1]);
  });

  it("hands the camera from one stage to the next without a jump", () => {
    for (let i = 0; i + 1 < STAGES.length; i++) {
      const a = story.keys[STAGES[i] ?? "plan"].at(-1) ?? [];
      const b = story.keys[STAGES[i + 1] ?? "plan"][0] ?? [];
      const [, ax = 0, ay = 0, aw = 1] = a;
      const [, bx = 0, by = 0, bw = 1] = b;
      expect(Math.max(aw / bw, bw / aw), `${STAGES[i]} width`).toBeLessThan(1.6);
      expect(Math.hypot(ax - bx, ay - by), `${STAGES[i]} target`).toBeLessThan(0.6 * aw);
    }
  });
});
