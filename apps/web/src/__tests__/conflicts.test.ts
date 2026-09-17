import { describe, expect, it } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { busyWindows, describeConflict, planConflicts } from "@/missions/conflicts";
import { DemoMissionProvider } from "@/missions/demo";
import { buildDraftRequest, planFromDraft } from "@/missions/planDraft";
import { learnedRates, parseHours, taskFamily } from "@/missions/rates";

const project = new DemoMissionProvider().projectForSite("builtin-demo", null)!;

const base: PlanDraft = {
  title: "Mow Z-14",
  objective: "",
  zoneIds: ["Z-14"],
  machineIds: ["TR-04"],
  cadence: "once",
  startDate: "2026-09-17",
  endDate: "2026-09-24",
  estimates: { acres: 310, machineHours: 206, calendarDays: 7 },
  steps: [
    {
      title: "Survey",
      detail: "",
      machineIds: ["TR-04"],
      zoneId: "Z-14",
      when: "",
      startDay: 0,
      days: 1,
    },
    {
      title: "Treat",
      detail: "",
      machineIds: ["TR-04"],
      zoneId: "Z-14",
      when: "",
      startDay: 1,
      days: 6,
    },
  ],
  assumptions: [],
  risks: [],
  questions: [],
  source: "rules",
  model: null,
  note: "",
};

describe("conflicts", () => {
  it("derives busy windows from steps", () => {
    const windows = busyWindows({
      startDate: "2026-09-17",
      zoneIds: [],
      machineIds: ["TR-04"],
      steps: base.steps,
    });
    expect(windows).toEqual([
      { machineId: "TR-04", startDate: "2026-09-17", endDate: "2026-09-18" },
      { machineId: "TR-04", startDate: "2026-09-18", endDate: "2026-09-24" },
    ]);
  });

  it("finds machine double-bookings and zone overlaps with active plans only", () => {
    const existing = planFromDraft(base, project, { goal: "x", id: "existing" });
    const done = {
      ...planFromDraft(base, project, { goal: "y", id: "done" }),
      state: "Done",
      status: "ok" as const,
    };
    const candidate = {
      startDate: "2026-09-20",
      zoneIds: ["Z-14", "Z-21"],
      machineIds: ["TR-04"],
      steps: [
        {
          title: "Treat",
          detail: "",
          machineIds: ["TR-04"],
          zoneId: "Z-21",
          when: "",
          startDay: 0,
          days: 3,
        },
      ],
    };
    const conflicts = planConflicts(candidate, [existing, done, ...project.plans]);
    const machine = conflicts.find((c) => c.kind === "machine");
    expect(machine).toMatchObject({
      id: "TR-04",
      planId: "existing",
      from: "2026-09-20",
      to: "2026-09-23",
    });
    expect(conflicts.filter((c) => c.kind === "zone").map((c) => c.planId)).toContain("existing");
    expect(conflicts.some((c) => c.planId === "done")).toBe(false);
    expect(describeConflict(machine!)).toContain("booked by");
    // Revising the same plan does not conflict with itself.
    expect(
      planConflicts({ ...candidate, id: "existing" }, [existing]).some(
        (c) => c.planId === "existing",
      ),
    ).toBe(false);
  });

  it("learns rates from the work log and sends them with booked windows", () => {
    expect(parseHours("5:40")).toBeCloseTo(5.667, 2);
    expect(parseHours("6.5 h")).toBe(6.5);
    expect(taskFamily("Mow pass 2")).toBe("mow");
    const rates = learnedRates(project.workLog);
    const mow = rates.find((r) => r.task === "mow");
    expect(mow).toBeDefined();
    expect(mow!.samples).toBeGreaterThan(1);
    expect(mow!.acresPerMachineHour).toBeGreaterThan(0);
    const existing = planFromDraft(base, project, { goal: "x", id: "existing" });
    const request = buildDraftRequest({ ...project, plans: [existing] }, { goal: "Mow Z-21" });
    expect(request.rates?.map((r) => r.task)).toContain("mow");
    expect(request.existingPlans?.[0]?.busy?.[0]?.machineId).toBe("TR-04");
    const revising = buildDraftRequest(
      { ...project, plans: [existing] },
      { goal: "Mow Z-21", replacePlanId: "existing" },
    );
    expect(revising.existingPlans).toHaveLength(0);
  });
});
