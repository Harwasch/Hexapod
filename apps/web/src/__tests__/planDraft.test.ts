import { describe, expect, it } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { DemoMissionProvider } from "@/missions/demo";
import {
  buildDraftRequest,
  describeDraft,
  diffDrafts,
  overlayFor,
  planFromDraft,
  scheduleLanes,
} from "@/missions/planDraft";
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
    {
      title: "Survey pass",
      detail: "",
      machineIds: ["TR-04"],
      zoneId: "Z-14",
      when: "Day 1",
      startDay: 0,
      days: 1,
    },
    {
      title: "Treat Z-14",
      detail: "Mow",
      machineIds: ["TR-04"],
      zoneId: "Z-14",
      when: "Day 2",
      startDay: 1,
      days: 6,
    },
  ],
  assumptions: ["1.5 acres per machine-hour"],
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

  it("builds the map overlay in step order and lanes per machine", () => {
    const overlay = overlayFor({
      zoneIds: ["Z-21", "Z-14"],
      machineIds: ["TR-12"],
      steps: [
        ...draft.steps,
        {
          title: "Treat Z-21",
          detail: "",
          machineIds: ["TR-12"],
          zoneId: "Z-21",
          when: "",
          startDay: 1,
          days: 3,
        },
      ],
    });
    expect(overlay.zones).toEqual([
      { zoneId: "Z-14", machineIds: ["TR-04"] },
      { zoneId: "Z-21", machineIds: ["TR-12"] },
    ]);
    const { lanes, totalDays } = scheduleLanes(draft.steps);
    expect(lanes).toHaveLength(1);
    expect(lanes[0]?.bars.map((b) => b.startDay)).toEqual([0, 1]);
    expect(totalDays).toBe(7);
    expect(scheduleLanes([{ ...draft.steps[0]!, machineIds: [] }]).lanes[0]?.machineId).toBe(
      "Fleet",
    );
  });

  it("describes what a redraft changed", () => {
    const next: PlanDraft = {
      ...draft,
      machineIds: ["TR-04", "TR-12"],
      zoneIds: ["Z-21"],
      estimates: { acres: 220, machineHours: 100, calendarDays: 3 },
      steps: draft.steps.slice(0, 1),
    };
    const changes = diffDrafts(draft, next);
    expect(changes).toContain("Added zone Z-21.");
    expect(changes).toContain("Dropped zone Z-14.");
    expect(changes).toContain("Added machine TR-12.");
    expect(changes).toContain("Machine-hours 206 → 100.");
    expect(changes).toContain("Duration 7 d → 3 d.");
    expect(changes).toContain("Steps 2 → 1.");
    expect(diffDrafts(draft, draft)).toEqual([]);
  });
});
