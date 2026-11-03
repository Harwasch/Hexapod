import { describe, expect, it } from "vitest";

import { approve, DRAFT_ID, figures, makeSim, step, type Sim } from "./mc";

/** The simulation after `seconds` of page time, a second at a time as the page runs it. */
function run(sim: Sim, seconds: number) {
  for (let s = 0; s < seconds; s++) sim = step(sim, 1);
  return sim;
}
const plan = (sim: Sim, id: string) => sim.plans.find((p) => p.id === id);

describe("Mission Control's simulation", () => {
  it("moves the running plans forward, never back, and never finishes the long ones", () => {
    const before = makeSim();
    const after = run(before, 3600);
    for (const p of before.plans) {
      const q = plan(after, p.id);
      expect(q?.progress, p.id).toBeGreaterThanOrEqual(p.progress);
      if (p.state !== "draft") expect(q?.progress, p.id).toBeLessThanOrEqual(99);
    }
    expect(plan(after, "survey")?.progress).toBeGreaterThan(62);
  });

  it("waits for the visitor to approve the agent's plan", () => {
    const sim = run(makeSim(), 120);
    expect(plan(sim, DRAFT_ID)?.state).toBe("draft");
    expect(plan(sim, DRAFT_ID)?.progress).toBe(0);
  });

  it("starts the plan on approval, and three hexapods take it up", () => {
    const sim = approve(makeSim());
    expect(plan(sim, DRAFT_ID)?.state).toBe("running");
    const crew = sim.machines.filter((m) => m.task === "Pulling broom");
    expect(crew.map((m) => m.id)).toEqual(["HX-07", "HX-09", "HX-15"]);
    expect(figures(sim).working).toBe(figures(makeSim()).working + 3);
    // Approving twice changes nothing more.
    expect(approve(sim)).toBe(sim);
  });

  it("finishes an approved plan in about a minute of page time", () => {
    const sim = run(approve(makeSim()), 70);
    expect(plan(sim, DRAFT_ID)?.state).toBe("done");
    expect(plan(sim, DRAFT_ID)?.progress).toBe(100);
  });

  it("keeps batteries in range: working machines draw down, docked ones charge", () => {
    const start = makeSim();
    const sim = run(start, 1800);
    for (const m of sim.machines) {
      expect(m.batt, m.id).toBeGreaterThanOrEqual(12);
      expect(m.batt, m.id).toBeLessThanOrEqual(100);
    }
    const was = (id: string) => start.machines.find((m) => m.id === id)?.batt ?? 0;
    const now = (id: string) => sim.machines.find((m) => m.id === id)?.batt ?? 0;
    expect(now("HX-04")).toBeLessThan(was("HX-04"));
    expect(now("VT-01")).toBeGreaterThanOrEqual(was("VT-01"));
  });

  it("brings the quadcopter back once the agent moves it", () => {
    const early = run(makeSim(), 5);
    expect(early.machines.find((m) => m.id === "QD-18")?.state).toBe("warn");
    const later = run(early, 10);
    expect(later.machines.find((m) => m.id === "QD-18")?.state).toBe("work");
    expect(figures(later).attention).toBe(figures(early).attention - 1);
  });

  it("turns the agent's line every few seconds", () => {
    const lines = new Set<string>();
    let sim = makeSim();
    for (let s = 0; s < 40; s++) {
      sim = step(sim, 1);
      lines.add(sim.agent);
    }
    expect(lines.size).toBeGreaterThanOrEqual(4);
  });
});
