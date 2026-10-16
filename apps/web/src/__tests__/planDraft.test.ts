import { describe, expect, it } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { anywhereProject } from "@/missions/anywhere";
import { viewAreaZone } from "@/missions/areas";
import { demoRanchProject } from "@/missions/demo";
import { landProject } from "@/missions/landGround";
import {
  buildDraftRequest,
  describeDraft,
  diffDrafts,
  formatMachineHours,
  idLabel,
  overlayFor,
  planFromDraft,
  planFromRecord,
  presentStatus,
  recordBodyFromDraft,
  scheduleLanes,
  tickStep,
  zonesFromRecord,
} from "@/missions/planDraft";
import { useMission } from "@/state/mission";

import { ground, land, LAND_ID } from "./landGroundFixture";

const project = demoRanchProject();

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
  clarifications: [],
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
    const z14 = project.zones.find((z) => z.id === "Z-14")!;
    expect(plan.facts.find((f) => f.k === "Area")?.v).toBe(idLabel("Z-14", z14.name));
    const recurring = planFromDraft({ ...draft, cadence: "monthly", endDate: null }, project, {
      goal: "x",
    });
    expect(recurring.ongoing).toBe(true);
    expect(recurring.end).toBe("Ongoing");
  });

  it("describes a draft for the agent stream", () => {
    expect(describeDraft(draft)).toContain("2 steps");
    expect(describeDraft(draft)).toContain("tell me what to change");
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

  it("round-trips a draft through the API record shape", () => {
    const body = recordBodyFromDraft(draft, "Mow Z-14", "blackrock-mesa", null);
    expect(body.projectId).toBe("blackrock-mesa");
    expect(body.steps).toHaveLength(2);
    expect(body.source).toBe("rules");
    const record = {
      ...body,
      id: "8b4b7f3e-0d9e-4a1e-9c6d-1f2a3b4c5d6e",
      siteId: null,
      status: "dispatched" as const,
      revision: 2,
      revisions: [
        {
          revision: 1,
          note: "Approved",
          createdAt: "2026-09-17T00:00:00Z",
          title: "Mow Z-14",
          machineIds: ["TR-04"],
          zoneIds: ["Z-14"],
          estimates: draft.estimates,
        },
        {
          revision: 2,
          note: "Revised: skip nothing",
          createdAt: "2026-09-18T00:00:00Z",
          title: "Mow Z-14",
          machineIds: ["TR-04"],
          zoneIds: ["Z-14"],
          estimates: draft.estimates,
        },
      ],
      createdAt: "2026-09-17T00:00:00Z",
      updatedAt: "2026-09-18T00:00:00Z",
      landId: null,
      boundaryRevision: null,
      private: false,
      landActionId: null,
      areas: body.areas ?? [],
      zoneIds: body.zoneIds ?? [],
      machineIds: body.machineIds ?? [],
      steps: body.steps ?? [],
      assumptions: body.assumptions ?? [],
      risks: body.risks ?? [],
      questions: body.questions ?? [],
      cadence: body.cadence ?? ("once" as const),
      source: body.source ?? ("rules" as const),
      model: body.model ?? null,
      endDate: body.endDate ?? null,
    };
    const plan = planFromRecord(record, project);
    expect(plan.id).toBe(record.id);
    expect(plan.persisted).toBe(true);
    expect(plan.status).toBe("run");
    expect(plan.state).toBe("Dispatched");
    expect(plan.revision).toBe(2);
    expect(plan.revisions?.map((r) => r.note)).toEqual(["Approved", "Revised: skip nothing"]);
    expect(plan.steps).toHaveLength(2);
    expect(presentStatus("paused").action).toBe("Resume");
  });

  it("merges remote plans with local-only ones and drops stale console plans", () => {
    const store = useMission.getState();
    store.setProject(project);
    const local = planFromDraft(draft, project, { goal: "local", id: "plan-local" });
    store.approvePlan(local);
    const stale = planFromDraft(draft, project, { goal: "stale", id: "plan-stale" });
    store.approvePlan({ ...stale, persisted: true });
    const remote = {
      ...planFromDraft(draft, project, { goal: "remote", id: "plan-remote" }),
      persisted: true,
    };
    store.setRemotePlans(project.id, [remote]);
    const ids = useMission.getState().project?.plans.map((p) => p.id) ?? [];
    expect(ids).toContain("plan-remote");
    expect(ids).toContain("plan-local");
    expect(ids).not.toContain("plan-stale");
    expect(ids).toContain("thistle");
    store.removePlan("plan-local");
    store.setRemotePlans(project.id, []);
  });
});

describe("plans on a saved land", () => {
  const onLand = landProject(land, ground);
  const fenceDraft: PlanDraft = {
    ...draft,
    title: "Weekly fence check",
    zoneIds: ["L-aaaa0001"],
    machineIds: [],
    cadence: "weekly",
    endDate: null,
    estimates: { acres: 0, machineHours: 0.03, calendarDays: 2, pathM: 1000 },
  };

  it("sends outlines for drawn and land zones (a fence as its line), never for simulated ones", () => {
    const request = buildDraftRequest(onLand, { goal: "monitor the fence" });
    const byId = new Map(request.zones?.map((z) => [z.id, z]));
    expect(byId.get("L-aaaa0001")?.geometry?.type).toBe("LineString");
    expect(byId.get("L-0f1e2d3c")?.geometry?.type).toBe("Polygon");
    const anywhere = anywhereProject();
    const area = viewAreaZone("A-01", "View area 01", {
      center: { longitude: -120, latitude: 34.7 },
      metersPerPixel: 1,
      width: 800,
      height: 600,
      fraction: 0.5,
      dx: 0,
      dy: 0,
    });
    const drawn = buildDraftRequest({ ...anywhere, zones: [area] }, { goal: "mow it" });
    expect(drawn.zones?.[0]?.geometry?.type).toBe("Polygon");
    const demo = buildDraftRequest(project, { goal: "mow" });
    expect(demo.zones?.every((z) => z.geometry === undefined)).toBe(true);
  });

  it("stores land zones with their provenance and reads them back private", () => {
    const body = recordBodyFromDraft(fenceDraft, "monitor", onLand.id, null, onLand.zones);
    expect(body.areas).toEqual([
      {
        id: "L-aaaa0001",
        name: "North fence",
        footprint: onLand.zones[1]!.footprint,
        attribution: "Inventory · confirmed · revision 3 · Walked with a GPS",
      },
    ]);
    const record = {
      ...body,
      id: "rec-1",
      status: "scheduled" as const,
      revision: 1,
      revisions: [],
      createdAt: "",
      updatedAt: "",
      areas: body.areas ?? [],
      zoneIds: body.zoneIds ?? [],
      machineIds: body.machineIds ?? [],
      steps: body.steps ?? [],
      assumptions: body.assumptions ?? [],
      risks: body.risks ?? [],
      questions: body.questions ?? [],
      cadence: body.cadence ?? ("once" as const),
      source: body.source ?? ("rules" as const),
      model: body.model ?? null,
      endDate: body.endDate ?? null,
      siteId: null,
      landId: LAND_ID,
      boundaryRevision: 2,
      private: true,
      landActionId: null,
    };
    const [zone] = zonesFromRecord(record);
    expect(zone?.attribution).toBe("Inventory · confirmed · revision 3 · Walked with a GPS");
    expect(zone?.private).toBe(true);
    const plan = planFromRecord(record, onLand);
    expect(plan.scope).toBe("workspace");
    expect(plan.landId).toBe(LAND_ID);
    expect(plan.agentNote).toContain("Rule-based draft approved by you");
    // A line has a path, not acres; a short run reads in minutes.
    expect(plan.stats[0]).toEqual({ value: "1.0 km", label: "Path" });
    expect(plan.stats[2]?.value).toBe("2 min");
    expect(plan.meta).toBe("Weekly · North fence · 1.0 km");
    const scheduled = planFromRecord({ ...record, landActionId: "action-1" }, onLand);
    expect(scheduled.landActionId).toBe("action-1");
    expect(scheduled.agentNote).toContain("land action");
    expect(scheduled.agentNote).not.toContain("Rule-based draft approved by you");
    expect(scheduled.stats[2]?.value).toBe("Not estimated");
    expect(planFromRecord({ ...record, private: false }, onLand).scope).toBe("public");
  });

  it("names land zones without their opaque ids", () => {
    expect(idLabel("L-aaaa0001", "North fence")).toBe("North fence");
    expect(idLabel("LA-1234abcd", "Approved work area")).toBe("Approved work area");
    expect(idLabel("Z-14", "West bench")).toBe("Z-14 West bench");
    expect(formatMachineHours(0)).toBe("0 h");
    expect(formatMachineHours(0.004)).toBe("1 min");
    expect(formatMachineHours(12.4)).toBe("12 h");
    expect(describeDraft(fenceDraft)).toContain("1.0 km path");
  });
});

describe("schedule ticks", () => {
  it("spaces date labels so they never overlap", () => {
    // 448 viewBox units of plot: 21 days used to tick every 2 days, 42 units apart,
    // narrower than one "Oct 16" label.
    expect(tickStep(448 / 21)).toBe(3);
    expect(tickStep(448 / 4)).toBe(1);
    expect(tickStep(448 / 60)).toBe(7);
    expect(tickStep(448 / 120)).toBe(14);
    expect(tickStep(448 / 400)).toBe(28);
    for (const days of [1, 5, 10, 21, 30, 45, 90]) {
      expect(tickStep(448 / days) * (448 / days)).toBeGreaterThanOrEqual(52);
    }
  });
});
