import { describe, expect, it } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { DemoMissionProvider } from "@/missions/demo";
import { buildDraftRequest, describeDraft, planFromDraft } from "@/missions/planDraft";
import { useMission } from "@/state/mission";

const project = new DemoMissionProvider().projectForSite("builtin-demo", null)!;

const draft: PlanDraft = {
  title: "Clear thistle on the west bench",
  objective: "Mow Z-14 with TR-04 at 1.5 acres per hour.",
  zoneIds: ["Z-14"],
  machineIds: ["TR-04"],
  cadence: "once",
  startDate: "2026-09-17",
  endDate: "2026-09-24",
  estimates: { acres: 310, machineHours: 206, calendarDays: 7 },
  steps: [
    { title: "Survey pass", detail: "", machineIds: ["TR-04"], zoneId: "Z-14", when: "Day 1" },
    { title: "Treat Z-14", detail: "Mow", machineIds: ["TR-04"], zoneId: "Z-14", when: "Day 2" },
  ],
  risks: ["TR-04 is at 22% battery."],
  questions: [],
  source: "rules",
  model: null,
  note: "Rule-based draft",
};

describe("plan drafting", () => {
  it("builds the planner request from the project", () => {
    const request = buildDraftRequest(project, {
      goal: " Mow Z-14 ",
      zoneIds: ["Z-14"],
      refinement: "use two machines",
      previous: draft,
    });
    expect(request.goal).toBe("Mow Z-14");
    expect(request.zones?.map((z) => z.id)).toContain("Z-14");
    expect(request.machines?.length).toBe(project.machines.length);
    expect(request.existingPlans?.map((p) => p.id)).toContain("thistle");
    expect(request.preferredZoneIds).toEqual(["Z-14"]);
    expect(request.refinement).toBe("use two machines");
    expect(request.previousDraft?.title).toBe(draft.title);
    expect(request.previousDraft).not.toHaveProperty("source");
  });

  it("turns an approved draft into a scheduled plan with steps and provenance", () => {
    const plan = planFromDraft(draft, project, { goal: "Mow Z-14" });
    expect(plan.status).toBe("idle");
    expect(plan.state).toBe("Scheduled");
    expect(plan.zoneIds).toEqual(["Z-14"]);
    expect(plan.machineIds).toEqual(["TR-04"]);
    expect(plan.steps).toHaveLength(2);
    expect(plan.ongoing).toBe(false);
    expect(plan.source).toEqual({ kind: "rules", model: null });
    expect(plan.facts.find((f) => f.k === "Risks")?.v).toContain("22%");
    expect(plan.facts.find((f) => f.k === "Area")?.v).toBe("Z-14 West bench");
    const recurring = planFromDraft({ ...draft, cadence: "monthly", endDate: null }, project, {
      goal: "x",
    });
    expect(recurring.ongoing).toBe(true);
    expect(recurring.end).toBe("Ongoing");
  });

  it("describes a draft for the agent stream", () => {
    expect(describeDraft(draft)).toContain("2 steps");
    expect(describeDraft({ ...draft, questions: ["Which zones?"] })).toContain("1 question");
  });

  it("keeps approved plans with the project and survives a project reload", () => {
    const store = useMission.getState();
    store.setProject(project);
    store.openComposer({ goal: "Mow Z-14" });
    expect(useMission.getState().composer?.goal).toBe("Mow Z-14");
    expect(useMission.getState().view).toBe("plan");
    const plan = planFromDraft(draft, project, { goal: "Mow Z-14", id: "plan-test" });
    store.approvePlan(plan);
    let state = useMission.getState();
    expect(state.composer).toBeNull();
    expect(state.planId).toBe("plan-test");
    expect(state.project?.plans.map((p) => p.id)).toContain("plan-test");
    store.setProject(project);
    state = useMission.getState();
    expect(state.project?.plans.filter((p) => p.id === "plan-test")).toHaveLength(1);
    store.approvePlan({ ...plan, title: "Revised" });
    expect(useMission.getState().project?.plans.filter((p) => p.id === "plan-test")).toHaveLength(
      1,
    );
    expect(useMission.getState().project?.plans.find((p) => p.id === "plan-test")?.title).toBe(
      "Revised",
    );
    store.removePlan("plan-test");
    expect(useMission.getState().project?.plans.some((p) => p.id === "plan-test")).toBe(false);
  });
});
