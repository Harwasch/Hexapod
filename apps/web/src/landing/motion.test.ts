/**
 * How the hero's robots move (Harrison, 2026-10-26: "the robots keep doing weird things").
 *
 * Every hexapod in all nine crews, sampled 60 times a second through a whole work cycle, as the
 * hero draws them (scene.js `S.model("operate", t)`). What a viewer saw, and what these check:
 * - bodies sliding with their legs still (the crowd pass nudging robots apart while their plan
 *   said "standing");
 * - turning half way round in one frame;
 * - legs treading on the spot while the body stands still;
 * - jumps, the quadcopter's included.
 */
import { describe, expect, it } from "vitest";

import { crewStory, type StoryBot } from "./scene.js";

const ZONES: { id: string; c: [number, number] }[] = [
  { id: "nature", c: [-238, 172] },
  { id: "range", c: [-175, 158] },
  { id: "forest", c: [-112, 162] },
  { id: "infra", c: [-48, 172] },
  { id: "solar", c: [8, 170] },
  { id: "pipeline", c: [55, 195] },
  { id: "fuels", c: [-62, 238] },
  { id: "agri", c: [-132, 236] },
  { id: "reclaim", c: [-200, 232] },
];

const DT = 1 / 60;
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
/** Ground distance between two points, metres. */
const gap = (a: number[], b: number[]) =>
  Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0));

/** The worst of each fault over one crew's cycle, as numbers a failure can print. */
function faults(zone: (typeof ZONES)[number]) {
  const S = crewStory(zone);
  const prev = new Map<number, StoryBot & { p: number[] }>();
  const still = new Map<number, number>();
  const worst = { glide: 0, turn: 0, speed: 0, tread: 0, drone: 0 };
  let drone: number[] | undefined;
  for (let t = 5; t <= S.OPS.end + 1; t += DT) {
    const m = S.model("operate", t);
    if (drone) worst.drone = Math.max(worst.drone, gap(m.quad.p, drone) / DT);
    drone = m.quad.p;
    const hexes: (StoryBot | undefined)[] = m.hex;
    hexes.forEach((hx, k) => {
      if (!hx || hx.carried || !hx.p) {
        prev.delete(k);
        return;
      }
      const q = prev.get(k);
      prev.set(k, { ...hx, p: hx.p });
      if (!q) return;
      const v = gap(hx.p, q.p) / DT;
      const w = Math.abs(wrap(hx.h - q.h)) / DT;
      worst.speed = Math.max(worst.speed, v);
      worst.turn = Math.max(worst.turn, w);
      if (!hx.moving) worst.glide = Math.max(worst.glide, v);
      // Treading: legs going while the body neither moves nor turns, for longer than a step.
      const idle = v < 0.02 && w < 0.05 && hx.moving;
      const run = idle ? (still.get(k) ?? 0) + DT : 0;
      still.set(k, run);
      worst.tread = Math.max(worst.tread, run);
    });
  }
  return worst;
}

describe("the hero's hexapods", () => {
  for (const zone of ZONES)
    it(`step whenever they move, and turn and travel at a robot's pace (${zone.id})`, () => {
      const w = faults(zone);
      // A body creeping under 0.12 m/s with the legs at rest reads as settling, not sliding.
      expect(w.glide, "fastest slide with the legs still, m/s").toBeLessThan(0.12);
      // Half a turn in no less than about 0.7 s.
      expect(w.turn, "fastest turn, rad/s").toBeLessThan(4.6);
      // Walking pace is about 1 m/s, and the plan's eased walks peak near 2.4 m/s.
      expect(w.speed, "fastest travel, m/s").toBeLessThan(3);
      expect(w.tread, "longest treading on the spot, s").toBeLessThan(0.35);
      // The quadcopter's transits run a few metres a second; one frame's jump showed as 18 m/s.
      expect(w.drone, "fastest quadcopter move, m/s").toBeLessThan(6);
    });
});
