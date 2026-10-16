import { beforeEach, describe, expect, it, vi } from "vitest";

import type { PlanDraft, PlanRecord } from "@twin/contracts";

import type * as Queries from "@/api/queries";

import { anywhereProject } from "@/missions/anywhere";
import { demoRanchProject } from "@/missions/demo";
import { landProject, landZones } from "@/missions/landGround";
import { planFromRecord } from "@/missions/planDraft";
import { useLand } from "@/state/land";
import { migrateMissionStorage, missionLandId, persistable, useMission } from "@/state/mission";

import { ground, land, LAND_ID } from "./landGroundFixture";

const api = vi.hoisted(() => ({
  publicCreate: vi.fn(),
  publicRevise: vi.fn(),
  workspaceCreate: vi.fn(),
  workspaceRevise: vi.fn(),
  workspaceStatus: vi.fn(),
  fetchLandGround: vi.fn(),
}));

vi.mock("@/api/queries", async (importOriginal) => ({
  ...(await importOriginal<typeof Queries>()),
  plansApi: {
    create: api.publicCreate,
    revise: api.publicRevise,
    setStatus: vi.fn(),
    remove: vi.fn(),
  },
  workspaceMissionsApi: {
    create: api.workspaceCreate,
    revise: api.workspaceRevise,
    setStatus: api.workspaceStatus,
  },
  fetchLandGround: api.fetchLandGround,
}));

const { approveDraft, setPlanStatus } = await import("@/features/mission/planDrafting");
const { engageLand, missionProjectName, planOnLand } =
  await import("@/features/mission/landPlanning");

const OTHER_ID = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";
const otherLand = { ...land, id: OTHER_ID, name: "Creek Field" };
const otherGround = {
  ...ground,
  landId: OTHER_ID,
  name: "Creek Field",
  items: ground.items.map((item) => ({ ...item, zoneId: item.zoneId.replace("L-", "L-b") })),
};

const draft = (zoneIds: string[]): PlanDraft => ({
  title: "Weekly fence check",
  objective: "Fly the north fence every week.",
  zoneIds,
  machineIds: [],
  cadence: "weekly",
  startDate: "2026-10-16",
  endDate: null,
  estimates: { acres: 0, machineHours: 0.03, calendarDays: 2, pathM: 1000 },
  steps: [],
  assumptions: ["Path from the outline: North fence 1.0 km at 8 m/s."],
  risks: [],
  questions: [],
  clarifications: [],
  source: "rules",
  model: null,
  note: "Rule-based draft",
});

function record(body: Record<string, unknown>, extra: Partial<PlanRecord> = {}): PlanRecord {
  return {
    ...(body as unknown as PlanRecord),
    id: "11111111-2222-4333-8444-555555555555",
    siteId: null,
    status: "scheduled",
    revision: 1,
    revisions: [],
    createdAt: "2026-10-16T00:00:00Z",
    updatedAt: "2026-10-16T00:00:00Z",
    landId: LAND_ID,
    boundaryRevision: 2,
    private: true,
    landActionId: null,
    ...extra,
  };
}

describe("plans on a saved land stay private", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useMission.setState({ composer: null, landGround: null, approvedPlans: {}, areas: {} });
    useLand.getState().clear();
  });

  it("sends a land project's plan to the workspace with the land and its boundary revision", async () => {
    useMission.getState().setProject(landProject(land, ground));
    api.workspaceCreate.mockImplementation((body: Record<string, unknown>) =>
      Promise.resolve(record(body)),
    );
    const plan = await approveDraft(draft(["L-aaaa0001"]), "monitor the fence every week");
    expect(api.publicCreate).not.toHaveBeenCalled();
    expect(api.workspaceCreate).toHaveBeenCalledTimes(1);
    const body = api.workspaceCreate.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(body).toMatchObject({
      projectId: `land:${LAND_ID}`,
      landId: LAND_ID,
      boundaryRevision: 2,
    });
    expect((body.areas as { id: string; attribution?: string }[])[0]).toMatchObject({
      id: "L-aaaa0001",
      attribution: "Inventory · confirmed · revision 3 · Walked with a GPS",
    });
    expect(plan?.scope).toBe("workspace");
    expect(plan?.landId).toBe(LAND_ID);
    // The status goes to the workspace route too.
    api.workspaceStatus.mockResolvedValue(record(body, { status: "dispatched" }));
    await setPlanStatus(plan!, "dispatched");
    expect(api.workspaceStatus).toHaveBeenCalledWith(plan!.id, "dispatched");
  });

  it("sends land zones layered on a fleet's project to the workspace, and the fleet's own to /plans", async () => {
    const demo = demoRanchProject();
    useMission.getState().setProject(demo);
    useMission.getState().engageLand({ landId: LAND_ID, zones: landZones(ground) });
    expect(missionLandId()).toBe(LAND_ID);
    // Ground engaged on another project says nothing about this one.
    expect(
      missionLandId({ landGround: useMission.getState().landGround, project: anywhereProject() }),
    ).toBeNull();
    api.workspaceCreate.mockImplementation((body: Record<string, unknown>) =>
      Promise.resolve(record(body)),
    );
    api.publicCreate.mockImplementation((body: Record<string, unknown>) =>
      Promise.resolve(record(body, { private: false, landId: null, boundaryRevision: null })),
    );
    await approveDraft(draft(["L-aaaa0001"]), "monitor the fence");
    expect(api.workspaceCreate.mock.calls[0]?.[0]).toMatchObject({
      projectId: demo.id,
      landId: LAND_ID,
    });
    await approveDraft(draft([demo.zones[0]!.id]), "mow it");
    expect(api.publicCreate).toHaveBeenCalledTimes(1);
  });

  it("never keeps a private plan in the browser when the API is offline", async () => {
    useMission.getState().setProject(landProject(land, ground));
    api.workspaceCreate.mockRejectedValue(new TypeError("Failed to fetch"));
    const plan = await approveDraft(draft(["L-aaaa0001"]), "monitor the fence");
    expect(plan).toBeNull();
    expect(useMission.getState().log.at(-1)?.text).toMatch(/^Not saved: the API is offline/);
    expect(useMission.getState().approvedPlans[`land:${LAND_ID}`] ?? []).toEqual([]);
  });

  it("writes no private plan or zone to browser storage, and forgets them on sign-out", () => {
    const demo = demoRanchProject();
    useMission.getState().setProject(demo);
    useMission.getState().engageLand({ landId: LAND_ID, zones: landZones(ground) });
    const project = useMission.getState().project!;
    const privatePlan = planFromRecord(
      record({ ...draft(["L-aaaa0001"]), goal: "g", projectId: demo.id, areas: [] }),
      project,
    );
    const publicPlan = planFromRecord(
      record(
        { ...draft([demo.zones[0]!.id]), goal: "g", projectId: demo.id, areas: [] },
        { id: "public-1", private: false, landId: null, boundaryRevision: null },
      ),
      project,
    );
    expect(privatePlan.scope).toBe("workspace");
    useMission.getState().setRemotePlans(demo.id, [privatePlan, publicPlan]);
    useMission.getState().mergeAreas(demo.id, [{ ...landZones(ground)[1]!, id: "LA-1234abcd" }]);
    const stored = persistable(useMission.getState());
    expect(stored.approvedPlans[demo.id]?.map((p) => p.id)).toEqual(["public-1"]);
    expect(stored.areas[demo.id]).toEqual([]);
    expect(JSON.stringify(stored)).not.toContain(LAND_ID);
    useMission.getState().dropPrivate();
    const after = useMission.getState();
    expect(after.landGround).toBeNull();
    expect(after.project?.plans.some((p) => p.scope === "workspace")).toBe(false);
    expect(after.project?.zones.some((z) => Boolean(z.private) || z.land !== undefined)).toBe(
      false,
    );
    useMission.getState().setProject(landProject(land, ground));
    useMission.getState().dropPrivate();
    expect(useMission.getState().project?.id).toBe(anywhereProject().id);
  });

  it("keeps one copy of a stored mission area however often the plans are refetched", () => {
    useMission.getState().setProject(landProject(land, ground));
    useMission
      .getState()
      .engageLand({ landId: LAND_ID, zones: landZones(ground) }, landProject(land, ground));
    const area = { ...landZones(ground)[0]!, id: "LA-c518a69e", land: undefined, private: true };
    const projectId = `land:${LAND_ID}`;
    for (let i = 0; i < 3; i++) {
      useMission
        .getState()
        .mergeAreas(projectId, [area, { ...landZones(ground)[1]!, land: undefined }]);
      useMission.getState().setRemotePlans(projectId, []);
    }
    const ids = useMission.getState().project?.zones.map((z) => z.id) ?? [];
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("LA-c518a69e");
    // The Land's own copy of a feature keeps its metadata over a stored copy of it.
    const fence = useMission.getState().project?.zones.find((z) => z.id === "L-aaaa0001");
    expect(fence?.land?.line).not.toBeNull();
  });

  it("drops land-shaped ground from storage written before version 2", () => {
    const zone = landZones(ground)[0]!;
    const migrated = migrateMissionStorage(
      {
        approvedPlans: { [`land:${LAND_ID}`]: [], demo: [] },
        areas: {
          demo: [
            { ...zone, private: undefined, land: undefined },
            { ...zone, id: "A-01", private: undefined, land: undefined },
          ],
        },
      },
      1,
    ) as { approvedPlans: Record<string, unknown[]>; areas: Record<string, { id: string }[]> };
    expect(Object.keys(migrated.approvedPlans)).toEqual(["demo"]);
    expect(migrated.areas.demo?.map((z) => z.id)).toEqual(["A-01"]);
  });

  it("opens the plan card on the land with its boundary as the ground", async () => {
    useMission.getState().setProject(anywhereProject());
    api.fetchLandGround.mockResolvedValue(ground);
    useLand.getState().select(land);
    await planOnLand(land);
    const state = useMission.getState();
    expect(state.project?.id).toBe(`land:${LAND_ID}`);
    expect(state.view).toBe("plan");
    expect(state.composer?.zoneIds).toEqual(["L-0f1e2d3c"]);
    expect(state.composer?.ground).toEqual({
      source: "land",
      note: "Oak Bench, 247.1 ac, saved boundary revision 2 + 2 mapped features.",
    });
    expect(state.log.at(-1)?.text).toContain("monitor the fence every week");
    expect(api.fetchLandGround).toHaveBeenCalledWith(LAND_ID, 2);
  });

  it("never layers one land's ground onto another land's project: the switch waits for the card", async () => {
    useMission
      .getState()
      .engageLand({ landId: LAND_ID, zones: landZones(ground) }, landProject(land, ground));
    useMission.getState().openComposer({ goal: "monitor the fence", zoneIds: ["L-0f1e2d3c"] });
    api.fetchLandGround.mockResolvedValue(otherGround);
    useLand.getState().select(otherLand);
    expect(await engageLand(otherLand)).toBeNull();
    let state = useMission.getState();
    expect(state.project?.id).toBe(`land:${LAND_ID}`);
    expect(state.landGround?.landId).toBe(LAND_ID);
    expect(state.project?.zones.some((z) => z.land?.landId === OTHER_ID)).toBe(false);
    expect(state.composer?.goal).toBe("monitor the fence");
    expect(state.log.at(-1)?.text).toBe(
      "Creek Field opens in Plan once you approve or close the plan card on Oak Bench.",
    );
    // Without the Land's own project too: the store never layers onto another Land's project.
    useMission.getState().engageLand({ landId: OTHER_ID, zones: landZones(otherGround) });
    expect(useMission.getState().landGround?.landId).toBe(LAND_ID);
    useMission.getState().closeComposer();
    await vi.waitFor(() => expect(useMission.getState().project?.id).toBe(`land:${OTHER_ID}`));
    state = useMission.getState();
    expect(state.landGround?.landId).toBe(OTHER_ID);
    expect(state.project?.zones.some((z) => z.land?.landId === LAND_ID)).toBe(false);
  });

  it("names a mission's project in the copy, never its id", () => {
    const demo = demoRanchProject();
    useMission.getState().setProject(landProject(land, ground));
    expect(missionProjectName(demo.id, undefined)).toBe(demo.name);
    expect(missionProjectName(`land:${LAND_ID}`, undefined)).toBe("Oak Bench");
    expect(missionProjectName(`land:${OTHER_ID}`, undefined)).toBeNull();
    const site = { id: "5d0c", slug: "west-yard", name: "West yard" };
    expect(missionProjectName("site-5d0c", [site as never])).toBe("West yard");
    expect(missionProjectName("site-nope", [site as never])).toBeNull();
  });
});
