import { describe, expect, it } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { DemoMissionProvider } from "@/missions/demo";
import { planFromDraft } from "@/missions/planDraft";
import { planProgress, replanRefinement } from "@/missions/progress";
import {
  planningApproved,
  planningDrafted,
  planningEdited,
  planningOpened,
  planningSummary,
  resetPlanningMetrics,
} from "@/lib/planningMetrics";

const project = new DemoMissionProvider().projectForSite("builtin-demo", null)!;

const draft: PlanDraft = {
  title: "Mow Z-14",
  objective: "",
  zoneIds: ["Z-14"],
  machineIds: ["TR-04", "TR-12"],
  cadence: "once",
  startDate: "2026-09-10",
  endDate: "2026-09-20",
  estimates: { acres: 310, machineHours: 206, calendarDays: 10 },
  steps: [
    {
      title: "Treat",
      detail: "",
      machineIds: ["TR-04", "TR-12"],
      zoneId: "Z-14",
      when: "",
      startDay: 0,
      days: 10,
    },
  ],
  assumptions: [],
  risks: [],
  questions: [],
  source: "rules",
  model: null,
  note: "",
};

describe("plan progress", () => {
  it("reads logged acres in the plan's zones by its machines since the start", () => {
    const plan = {
      ...planFromDraft(draft, project, { goal: "g", id: "p" }),
      state: "Dispatched",
      status: "run" as const,
    };
    const today = new Date("2026-09-15T12:00:00Z");
    const progress = planProgress(plan, project.workLog, today);
    // Demo rows dated 09-12 for TR-04 / TR-12 on "Z-14 west" count; earlier or other-zone rows do not.
    const expectedAcres = project.workLog
      .filter(
        (r) =>
          r.date >= "09-10" &&
          r.date <= "09-15" &&
          r.zone.includes("Z-14") &&
          ["TR-04", "TR-12"].includes(r.machine),
      )
      .reduce((s, r) => s + (r.acres ?? 0), 0);
    expect(progress.acresDone).toBe(expectedAcres);
    expect(progress.expectedPct).toBe(50);
    expect(progress.actualPct).toBe(Math.min(100, Math.round((expectedAcres / 310) * 100)));
    expect(progress.driftDays).toBe(Math.round(((50 - progress.actualPct) / 100) * 10));
    expect(progress.label).toMatch(/Behind by|Ahead by|On schedule/);
    expect(replanRefinement(progress, today)).toContain("Resequence");
  });

  it("is not started before dispatch and done when done", () => {
    const scheduled = planFromDraft(draft, project, { goal: "g", id: "p" });
    expect(planProgress(scheduled, project.workLog, new Date("2026-09-15T00:00:00Z")).label).toBe(
      "Not started",
    );
    const done = { ...scheduled, state: "Done", status: "ok" as const };
    expect(planProgress(done, project.workLog, new Date("2026-09-25T00:00:00Z")).actualPct).toBe(
      100,
    );
  });
});

describe("planning metrics", () => {
  it("summarises time to approve, redrafts and edits", () => {
    resetPlanningMetrics();
    planningOpened();
    planningDrafted(false);
    planningDrafted(true);
    planningEdited();
    const first = planningApproved();
    expect(first?.redrafts).toBe(1);
    expect(first?.edited).toBe(true);
    expect(first?.toFirstDraftMs).not.toBeNull();
    planningOpened();
    planningDrafted(false);
    planningApproved();
    const summary = planningSummary();
    expect(summary.plans).toBe(2);
    expect(summary.medianRedrafts).toBe(0.5);
    expect(summary.approvedUneditedPct).toBe(50);
    expect(planningApproved()).toBeNull();
    resetPlanningMetrics();
  });
});
