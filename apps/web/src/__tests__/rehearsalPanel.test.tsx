/**
 * The dress rehearsal in the plan card: the zones and crews it rehearses, the run as state
 * (`features/mission/dressRehearsal.ts`) and the panel (`Rehearsal.tsx`), with the ground
 * reader mocked and the real RehearsalManager on a fake viewer.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { type CesiumWidget, Color, EntityCollection } from "cesium";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PlanDraft, WorkableGround } from "@twin/contracts";
import { DEG_TO_RAD, EARTH_RADIUS_M, type CoverageLane, type LonLat } from "@twin/geo";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { MissionManager } from "@/cesium/MissionManager";
import { RehearsalManager } from "@/cesium/RehearsalManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import type { SceneEvents } from "@/cesium/types";
import {
  approveLabel,
  rehearsalZones,
  useDressRehearsal,
  type DressRehearsal,
} from "@/features/mission/dressRehearsal";
import { RehearsalPanel } from "@/features/mission/Rehearsal";
import { rehearsalGround } from "@/features/mission/rehearsalGround";
import { startPlanDraft } from "@/features/mission/planDrafting";
import type * as WorkableModule from "@/features/mission/workableGround";
import { requestWorkableGround } from "@/features/mission/workableGround";
import { Emitter } from "@/lib/emitter";
import { planRehearsal } from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useWorkable } from "@/state/workable";

vi.mock("@/features/mission/rehearsalGround", () => ({ rehearsalGround: vi.fn() }));
vi.mock("@/features/mission/planDrafting", () => ({ startPlanDraft: vi.fn() }));
vi.mock("@/features/mission/workableGround", async (original) => ({
  ...(await original<typeof WorkableModule>()),
  requestWorkableGround: vi.fn(),
}));

const origin = { longitude: -120.04, latitude: 34.698 };
const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
const ky = DEG_TO_RAD * EARTH_RADIUS_M;
const at = (x: number, y: number): LonLat => ({
  longitude: origin.longitude + x / kx,
  latitude: origin.latitude + y / ky,
});

const LANES: CoverageLane[] = [
  { pass: 0, a: at(-200, -300), b: at(-200, 300), lengthM: 600 },
  { pass: 1, a: at(-180, 300), b: at(-180, -300), lengthM: 600 },
  { pass: 2, a: at(-160, -300), b: at(-160, 300), lengthM: 600 },
];

function zone(id: string, extra: Partial<Zone> = {}): Zone {
  return {
    id,
    name: `${id} West grassland`,
    short: id,
    footprint: { type: "Polygon", coordinates: [] },
    tone: "teal",
    treated: false,
    acres: 10,
    task: "",
    machines: "",
    progressPct: 0,
    note: "",
    anchor: origin,
    ...extra,
  };
}

const project = {
  id: "demo",
  zones: [
    zone("Z-14"),
    zone("L-fence", {
      name: "Perimeter fence",
      land: {
        kind: "feature",
        landId: "land-1",
        featureId: "f-1",
        line: {
          type: "LineString",
          coordinates: [
            [origin.longitude, origin.latitude],
            [origin.longitude + 0.001, origin.latitude],
          ],
        },
      },
    }),
  ],
  machines: [
    { id: "TR-04", name: "TR-04 Kestrel", model: "Ranger 2 · mower" },
    { id: "TR-12", name: "TR-12 Pipit", model: "Ranger 2 · mower" },
  ],
} as unknown as Project;

const draft = {
  title: "Mow Z-14",
  zoneIds: ["Z-14"],
  machineIds: ["TR-04"],
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
      machineIds: ["TR-04", "TR-12"],
      zoneId: "Z-14",
      when: "",
      startDay: 1,
      days: 6,
    },
  ],
  coverage: [{ zoneId: "Z-14", machineClass: "mower" }],
} as unknown as PlanDraft;

function fakeScene() {
  const mission = {
    previewLanes: vi.fn((id: string) => (id === "Z-14" ? LANES : [])),
    previewLaneNumbers: vi.fn((id: string) => (id === "Z-14" ? [1, 2, 4] : [])),
    machineColor: vi.fn(() => Color.CYAN),
    paletteColor: vi.fn(() => Color.ORANGE),
    leaveOutLane: vi.fn(),
    restoreLanes: vi.fn(),
    setPreviewPassesShown: vi.fn(),
    flyToPoint: vi.fn(),
  };
  const viewer = {
    entities: new EntityCollection(),
    scene: { requestRender: vi.fn() },
  } as unknown as CesiumWidget;
  const rehearsal = new RehearsalManager(viewer, mission as unknown as MissionManager);
  const events = new Emitter<SceneEvents>();
  return { mission, rehearsal, events, viewer };
}

describe("what the rehearsal drives", () => {
  it("takes the drawn lanes with their numbers, and every machine the zone's steps name", () => {
    const scene = fakeScene();
    const zones = rehearsalZones(draft, project, scene as unknown as CesiumSceneManager);
    expect(zones).toHaveLength(1);
    expect(zones[0]).toMatchObject({
      zoneId: "Z-14",
      label: "Z-14",
      machineIds: ["TR-04", "TR-12"],
      classId: "mower",
    });
    expect(zones[0]?.lanes.map((l) => l.number)).toEqual([1, 2, 4]);
    expect(zones[0]?.lanes[0]?.points).toEqual([LANES[0]?.a, LANES[0]?.b]);
  });

  it("follows a mapped line, and names a Land zone by its name", () => {
    const scene = fakeScene();
    const zones = rehearsalZones(
      { ...draft, zoneIds: ["L-fence"], steps: [], machineIds: [] },
      project,
      scene as unknown as CesiumSceneManager,
    );
    expect(zones[0]).toMatchObject({ zoneId: "L-fence", label: "Perimeter fence", machineIds: [] });
    expect(zones[0]?.lanes).toHaveLength(1);
    // Two clicks drew this fence: still the line the zone follows, not a lane to leave out.
    expect(zones[0]?.lanes[0]).toMatchObject({ followsLine: true });
    expect(zones[0]?.lanes[0]?.points).toHaveLength(2);
  });

  it("labels the approve button with the hands and conflicts", () => {
    expect(approveLabel(0, 0)).toBe("Approve plan");
    expect(approveLabel(2, 0)).toBe("Approve despite conflicts");
    expect(approveLabel(0, 1)).toBe("Approve despite 1 hand");
    expect(approveLabel(0, 7)).toBe("Approve despite 7 hands");
    expect(approveLabel(1, 3)).toBe("Approve despite conflicts and 3 hands");
  });
});

describe("useDressRehearsal", () => {
  let scene: ReturnType<typeof fakeScene>;
  let registry: SceneRegistry;
  let queries: QueryClient;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queries}>
      <SceneContext.Provider value={registry}>{children}</SceneContext.Provider>
    </QueryClientProvider>
  );

  beforeEach(() => {
    scene = fakeScene();
    queries = new QueryClient();
    registry = new SceneRegistry();
    registry.set(scene as unknown as CesiumSceneManager);
    vi.stubGlobal("requestAnimationFrame", () => 1);
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    vi.mocked(rehearsalGround).mockResolvedValue({
      elevation: null,
      elevationMissing: "Slope not checked: no terrain for this ground.",
      creeks: {
        label: "USGS NHD flowlines, 1:24,000",
        lines: [
          {
            id: "nhd-1",
            name: "Figueroa Creek",
            kind: "intermittent",
            coordinates: [at(-210, -50), at(-210, 50)],
          },
        ],
      },
      heightAt: () => 300,
      better: null,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(rehearsalGround).mockReset();
    vi.mocked(requestWorkableGround).mockReset();
    vi.mocked(startPlanDraft).mockReset();
    useMission.setState({ project: null, landGround: null, composer: null });
    useWorkable.getState().clear();
  });

  it("reads the ground once, plays, counts hands as they go up, and stops when the plan changes", async () => {
    const { result, rerender } = renderHook(
      ({ key }: { key: string }) => useDressRehearsal(draft, project, key),
      { wrapper, initialProps: { key: "a" } },
    );
    expect(result.current.status).toBe("idle");
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(rehearsalGround).toHaveBeenCalledWith(
      scene,
      expect.objectContaining({ landId: null, projectId: "demo" }),
    );
    expect(result.current.rehearsal?.machines.map((m) => m.id)).toEqual(["TR-04", "TR-12"]);
    expect(result.current.frame).toMatchObject({ playing: true, speed: 300 });
    expect(result.current.raised).toBe(0);
    act(() => result.current.skipToEnd());
    expect(result.current.raised).toBe(1);
    expect(result.current.frame?.ended).toBe(true);
    // A click on the hand on the map opens it in the card.
    const hand = result.current.rehearsal?.hands[0];
    act(() => scene.events.emit("rehearsal-hand", { id: hand?.id ?? "" }));
    expect(result.current.focused).toBe(hand?.id);
    expect(scene.mission.flyToPoint).toHaveBeenCalled();
    rerender({ key: "b" });
    expect(result.current.status).toBe("idle");
    expect(result.current.raised).toBe(0);
    expect(scene.rehearsal.loaded).toBeNull();
  });

  it("leaves a lane out of the preview, and its hands still count on Approve", async () => {
    const { result } = renderHook(() => useDressRehearsal(draft, project, "a"), { wrapper });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => result.current.skipToEnd());
    expect(result.current.raised).toBe(1);
    const hand = result.current.rehearsal?.hands[0];
    if (!hand) throw new Error("expected a hand");
    // Leaving it out redraws the preview without it, as MissionManager does.
    scene.mission.leaveOutLane.mockImplementation(() => {
      scene.mission.previewLanes.mockImplementation((id: string) =>
        id === "Z-14" ? LANES.slice(1) : [],
      );
      scene.mission.previewLaneNumbers.mockImplementation((id: string) =>
        id === "Z-14" ? [2, 4] : [],
      );
    });
    act(() => result.current.leaveOut(hand));
    expect(scene.mission.leaveOutLane).toHaveBeenCalledWith("Z-14", LANES[0]);
    expect(result.current.leftOut).toEqual([
      { zoneId: "Z-14", label: "Z-14", number: 1, hands: 1 },
    ]);
    // The preview rehearses without the lane, but the approved plan still drives it.
    expect(result.current.rehearsal?.lanes.map((l) => l.number)).toEqual([2, 4]);
    expect(result.current.rehearsal?.hands).toEqual([]);
    expect(result.current.raised).toBe(1);
    expect(approveLabel(0, result.current.raised)).toBe("Approve despite 1 hand");
    scene.mission.restoreLanes.mockImplementation(() => {
      scene.mission.previewLanes.mockImplementation((id: string) => (id === "Z-14" ? LANES : []));
      scene.mission.previewLaneNumbers.mockImplementation((id: string) =>
        id === "Z-14" ? [1, 2, 4] : [],
      );
    });
    act(() => result.current.restoreLanes());
    expect(scene.mission.restoreLanes).toHaveBeenCalled();
    expect(result.current.leftOut).toEqual([]);
    // Back in the run, and counted once.
    expect(result.current.raised).toBe(1);
  });

  it("colours generic ghosts after the project's machines, and reads the Land's outline", async () => {
    const landZone = zone("L-boundary", {
      name: "Oak Bench Ranch",
      land: { kind: "boundary", landId: "land-1", featureId: null },
    });
    const onLand = { ...project, id: "land:land-1", zones: [...project.zones, landZone] };
    const plan = { ...draft, steps: [], machineIds: [] } as unknown as PlanDraft;
    // The plan's Land comes from the mission: a Land's own project here.
    useMission.setState({ project: onLand, landGround: null });
    const { result } = renderHook(() => useDressRehearsal(plan, onLand, "a"), { wrapper });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(rehearsalGround).toHaveBeenCalledWith(
      scene,
      expect.objectContaining({
        landId: "land-1",
        land: { name: "Oak Bench Ranch", footprint: landZone.footprint },
      }),
    );
    expect(result.current.rehearsal?.machines.map((m) => m.id)).toEqual(["SIM-1"]);
    // SIM-1 takes the slot after the project's two machines, not the first machine's colour.
    expect(scene.mission.paletteColor).toHaveBeenCalledWith(2);
    expect(scene.mission.machineColor).not.toHaveBeenCalledWith("SIM-1");
  });

  it("says why when there is nothing to rehearse", () => {
    scene.mission.previewLanes.mockReturnValue([]);
    const { result } = renderHook(() => useDressRehearsal(draft, project, "a"), { wrapper });
    act(() => result.current.start());
    expect(result.current.status).toBe("error");
    expect(result.current.error).toContain("no lanes drawn");
    expect(rehearsalGround).not.toHaveBeenCalled();
  });
});

describe("work only the ground it can", () => {
  let scene: ReturnType<typeof fakeScene>;
  let registry: SceneRegistry;
  let queries: QueryClient;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queries}>
      <SceneContext.Provider value={registry}>{children}</SceneContext.Provider>
    </QueryClientProvider>
  );
  // A square around the lanes, as Z-14's outline.
  const square = {
    type: "Polygon" as const,
    coordinates: [
      [
        [at(-300, -400).longitude, at(-300, -400).latitude],
        [at(0, -400).longitude, at(-300, -400).latitude],
        [at(0, 400).longitude, at(0, 400).latitude],
        [at(-300, -400).longitude, at(0, 400).latitude],
        [at(-300, -400).longitude, at(-300, -400).latitude],
      ],
    ],
  };
  const workable = {
    type: "MultiPolygon" as const,
    coordinates: [
      [
        [
          [at(-200, -400).longitude, at(-200, -400).latitude],
          [at(0, -400).longitude, at(-200, -400).latitude],
          [at(0, 400).longitude, at(0, 400).latitude],
          [at(-200, -400).longitude, at(0, 400).latitude],
          [at(-200, -400).longitude, at(-200, -400).latitude],
        ],
      ],
    ],
  };
  const answer = {
    landId: "land-1",
    boundaryRevision: 3,
    workable,
    footprintM2: 240_000,
    workableM2: 160_000,
    removed: { steepM2: 60_000, creekBufferM2: 20_000, sliversM2: 0, slivers: 0, noDataM2: 0 },
    leftOut: [
      { reason: "steep", label: "Left out: too steep", areaM2: 60_000, geometry: workable },
    ],
    limits: {
      machineClass: "mower",
      label: "Robotic brush mower (assumed)",
      maxSlopePct: 25,
      creekBufferM: 15,
    },
    elevation: {
      rasterId: "raster-1",
      dataset: "usgs-3dep-13",
      label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
      published: "2025-08-27",
      resolutionM: 10,
      stale: false,
      fallback: false,
    },
    creeks: {
      status: "available",
      label: "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000",
      counted: 1,
      ephemeralIncluded: false,
      message: "1 mapped flowline",
    },
    sources: ["Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster raster-1"],
    assumptions: [],
    warnings: [],
  } as unknown as WorkableGround;
  const onRanch = {
    ...project,
    siteId: null,
    plans: [],
    zones: [zone("Z-14", { name: "West grassland", footprint: square, task: "Mow" })],
  } as unknown as Project;

  function grounded(rasterId: string | null) {
    vi.mocked(rehearsalGround).mockResolvedValue({
      elevation: rasterId
        ? {
            grid: {
              west: at(-400, -500).longitude,
              south: at(-400, -500).latitude,
              east: at(100, 500).longitude,
              north: at(100, 500).latitude,
              columns: 2,
              rows: 2,
              values: Float32Array.from([300, 300, 300, 300]),
            },
            label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
            ref: "raster raster-1",
            rasterId,
          }
        : null,
      creeks: {
        label: "USGS NHD flowlines, 1:24,000",
        lines: [
          {
            id: "nhd-1",
            name: "Figueroa Creek",
            kind: "intermittent",
            coordinates: [at(-210, -50), at(-210, 50)],
          },
        ],
      },
      heightAt: () => 300,
      better: null,
    });
  }

  beforeEach(() => {
    scene = fakeScene();
    queries = new QueryClient();
    registry = new SceneRegistry();
    registry.set(scene as unknown as CesiumSceneManager);
    vi.stubGlobal("requestAnimationFrame", () => 1);
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    vi.mocked(requestWorkableGround).mockResolvedValue(answer);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(rehearsalGround).mockReset();
    vi.mocked(requestWorkableGround).mockReset();
    vi.mocked(startPlanDraft).mockReset();
    useMission.setState({ project: null, landGround: null, composer: null });
    useWorkable.getState().clear();
  });

  it("says what is needed without a Land", async () => {
    grounded(null);
    useMission.setState({ project: onRanch, landGround: null });
    const { result } = renderHook(() => useDressRehearsal(draft, onRanch, "a"), { wrapper });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => result.current.skipToEnd());
    expect(result.current.workable.state).toBe("needs-land");
    expect(result.current.workable.message).toContain("Save this ground as a Land");
    expect(requestWorkableGround).not.toHaveBeenCalled();
  });

  it("asks for the Land's terrain when the slope came from the globe", async () => {
    grounded(null);
    useMission.setState({
      project: onRanch,
      landGround: { landId: "land-1", projectId: onRanch.id, zones: [] },
    });
    const { result } = renderHook(() => useDressRehearsal(draft, onRanch, "a"), { wrapper });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => result.current.skipToEnd());
    expect(result.current.workable.state).toBe("needs-terrain");
    expect(result.current.workable.message).toContain("Analyze bare-earth terrain (USGS 3DEP)");
  });

  it("offers the cut with its acres, redrafts on what is left, and rehearses that again", async () => {
    grounded("raster-1");
    useMission.setState({
      project: onRanch,
      landGround: { landId: "land-1", projectId: onRanch.id, zones: [] },
      composer: {
        goal: "Mow the west grassland",
        zoneIds: [],
        machineIds: [],
        replacePlanId: null,
        status: "ready",
        ground: null,
        draft,
        previousDraft: null,
        answers: {},
        asked: {},
        error: null,
      },
    });
    const { result, unmount } = renderHook(() => useDressRehearsal(draft, onRanch, "a"), {
      wrapper,
    });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.workable.state).toBe("ready"));
    // The strictest class on Z-14 (both mowers), with the raster the rehearsal read slope from.
    expect(requestWorkableGround).toHaveBeenCalledWith("land-1", {
      footprint: square,
      machineClass: "mower",
      demRasterId: "raster-1",
      includeEphemeralCreeks: false,
    });
    expect(result.current.workable.text).toBe("−15 ac too steep, −4.9 ac creek buffer → 40 ac");
    act(() => result.current.skipToEnd());
    const before = result.current.rehearsal?.hands.length ?? 0;
    expect(before).toBeGreaterThan(0);
    act(() => result.current.workable.apply());
    // A private area derived from the Land takes Z-14's place in the plan, machines kept.
    const state = useMission.getState();
    const area = state.project?.zones.find((z) => z.id === "A-01");
    expect(area).toMatchObject({
      name: "West grassland, workable",
      private: true,
      land: { kind: "derived", landId: "land-1", featureId: null, boundaryRevision: 3 },
    });
    expect(area?.attribution).toBe(
      "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed) on USGS 3DEP 1/3″ DEM, published 2025-08-27, 15 m creek buffer (assumed) on USGS NHD, from West grassland (Z-14)",
    );
    expect(state.composer?.zoneIds).toEqual(["A-01"]);
    // Z-14 is set aside while its workable ground is in scope: a goal naming it does not
    // bring all of it back.
    expect(state.composer?.setAside).toEqual({ "Z-14": "A-01" });
    expect(state.composer?.ground?.source).toBe("workable");
    expect(startPlanDraft).toHaveBeenCalledWith("Mow the west grassland", {
      zoneIds: ["A-01"],
      machineIds: ["TR-04"],
      reshaped: true,
    });
    expect(useWorkable.getState().records["A-01"]?.leftOut).toEqual([
      { reason: "steep", label: "Left out: too steep", footprint: workable },
    ]);
    unmount();
    // The redraft's review rehearses the new ground again, straight to its verdict.
    const redrafted = {
      ...draft,
      zoneIds: ["A-01"],
      steps: draft.steps?.map((step) => ({ ...step, zoneId: "A-01" })),
      coverage: [{ zoneId: "A-01", machineClass: "mower" }],
    } as unknown as PlanDraft;
    scene.mission.previewLanes.mockImplementation((id: string) => (id === "A-01" ? LANES : []));
    scene.mission.previewLaneNumbers.mockImplementation((id: string) =>
      id === "A-01" ? [1, 2, 3] : [],
    );
    const withArea = useMission.getState().project ?? onRanch;
    const again = renderHook(() => useDressRehearsal(redrafted, withArea, "b"), { wrapper });
    await waitFor(() => expect(again.result.current.status).toBe("ready"));
    expect(again.result.current.frame?.ended).toBe(true);
    expect(again.result.current.after).toEqual({ handsBefore: before, labels: ["Z-14"] });
    expect(useWorkable.getState().rerun).toBeNull();
  });
});

describe("RehearsalPanel", () => {
  const rehearsal = planRehearsal({
    zones: rehearsalZones(draft, project, fakeScene() as unknown as CesiumSceneManager),
    machines: project.machines,
    elevation: null,
    creeks: {
      label: "USGS NHD flowlines, 1:24,000",
      lines: [
        {
          id: "nhd-1",
          name: "Figueroa Creek",
          kind: "intermittent",
          coordinates: [at(-210, -50), at(-210, 50)],
        },
      ],
    },
  });
  const hand = rehearsal.hands[0];

  function run(overrides: Partial<DressRehearsal> = {}): DressRehearsal {
    return {
      status: "ready",
      error: null,
      rehearsal,
      ground: null,
      frame: {
        t: (hand?.t ?? 0) + 1,
        durationS: rehearsal.durationS,
        playing: true,
        speed: 300,
        raised: 1,
        ended: false,
      },
      focused: null,
      leftOut: [],
      includeEphemeral: false,
      raised: 1,
      workable: {
        state: "none",
        message: null,
        text: null,
        totals: null,
        labels: [],
        kept: [],
        sources: [],
        elevation: null,
        applying: false,
        apply: vi.fn(),
        retry: vi.fn(),
      },
      after: null,
      start: vi.fn(),
      stop: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      setSpeed: vi.fn(),
      seek: vi.fn(),
      skipToEnd: vi.fn(),
      focus: vi.fn(),
      leaveOut: vi.fn(),
      restoreLanes: vi.fn(),
      setIncludeEphemeral: vi.fn(),
      ...overrides,
    };
  }

  it("is nothing until Rehearse is pressed", () => {
    const { container } = render(<RehearsalPanel run={run({ status: "idle" })} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("says it is simulated, shows the clock and controls, and the hands raised so far", async () => {
    const controls = run();
    render(<RehearsalPanel run={controls} />);
    expect(screen.getByRole("status")).toHaveTextContent("SIMULATED REHEARSAL");
    expect(screen.getByTestId("rehearsal-clock")).toHaveTextContent(/of .* sim · 300×/);
    expect(screen.getByText("1 HAND RAISED")).toBeInTheDocument();
    // Hands are grouped by rule with their count; a group opens to every lane.
    const group = screen.getByTestId("rehearsal-hand-group");
    expect(group).toHaveTextContent("Inside a creek buffer1 lane");
    expect(screen.queryByTestId("rehearsal-hand")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /Inside a creek buffer/ }));
    expect(screen.getByTestId("rehearsal-hand")).toHaveTextContent(hand?.label ?? "");
    await userEvent.click(screen.getByTestId("rehearsal-pause"));
    expect(controls.pause).toHaveBeenCalled();
    await userEvent.click(screen.getByTestId("rehearsal-speed-600"));
    expect(controls.setSpeed).toHaveBeenCalledWith(600);
    await userEvent.click(screen.getByTestId("rehearsal-skip"));
    expect(controls.skipToEnd).toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /buffer 15 m \(assumed\)/ }));
    expect(controls.focus).toHaveBeenCalledWith(hand?.id);
  });

  it("offers Work only the ground it can with what it cuts, or says what it needs", async () => {
    const offer = run().workable;
    const ready = run({
      workable: {
        ...offer,
        state: "ready",
        text: "−137 ac too steep, −14 ac creek buffer → 263 ac",
        labels: ["Z-14"],
        kept: [{ label: "Perimeter fence", hands: 3 }],
        sources: ["Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster 1108dcf3"],
        elevation: "USGS 3DEP 1/3″ bare-earth DEM, 10 m (published 2025-08-27)",
      },
    });
    const { unmount } = render(<RehearsalPanel run={ready} />);
    const action = screen.getByTestId("rehearsal-workable-apply");
    expect(action).toHaveTextContent("Work only the ground it can");
    expect(action).toHaveTextContent("−137 ac too steep, −14 ac creek buffer → 263 ac");
    const box = screen.getByTestId("rehearsal-workable");
    expect(box).toHaveTextContent("Redrafts the plan on what is left of Z-14");
    expect(box).toHaveTextContent(
      "Derived from USGS 3DEP 1/3″ bare-earth DEM, 10 m (published 2025-08-27) and USGS NHD creeks; the limits are assumed.",
    );
    expect(box).toHaveTextContent("Perimeter fence follows a line, so its 3 hands stay");
    await userEvent.click(action);
    expect(ready.workable.apply).toHaveBeenCalled();
    unmount();
    render(
      <RehearsalPanel
        run={run({
          workable: {
            ...offer,
            state: "needs-land",
            message: "Save this ground as a Land to read its terrain.",
          },
        })}
      />,
    );
    expect(screen.queryByTestId("rehearsal-workable-apply")).toBeNull();
    expect(screen.getByTestId("rehearsal-workable")).toHaveTextContent(
      "Save this ground as a Land to read its terrain.",
    );
  });

  it("after a redraft onto workable ground, says how many hands remain and why", () => {
    render(
      <RehearsalPanel
        run={run({
          after: { handsBefore: 147, labels: ["Z-14"] },
          frame: {
            t: rehearsal.durationS,
            durationS: rehearsal.durationS,
            playing: false,
            speed: 300,
            raised: 1,
            ended: true,
          },
        })}
      />,
    );
    const after = screen.getByTestId("rehearsal-after");
    expect(after).toHaveTextContent(
      "Rehearsed again on the workable ground of Z-14: 1 hand, down from 147.",
    );
    expect(after).toHaveTextContent("so hands stay at its edges");
    expect(after).toHaveTextContent("The ground between drawn lanes is not checked.");
  });

  it("opens a hand with its rule, its source and Leave this lane out", async () => {
    const controls = run({ focused: hand?.id ?? null });
    render(<RehearsalPanel run={controls} />);
    expect(screen.getByText(hand?.rule ?? "")).toBeInTheDocument();
    expect(screen.getByText(hand?.source ?? "")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("rehearsal-leave-out"));
    expect(controls.leaveOut).toHaveBeenCalledWith(hand);
  });

  it("offers no Leave this lane out on a followed line (a fence)", () => {
    const fence = planRehearsal({
      zones: rehearsalZones(
        { ...draft, zoneIds: ["L-fence"], steps: [], machineIds: [] },
        project,
        fakeScene() as unknown as CesiumSceneManager,
      ),
      machines: project.machines,
      elevation: null,
      creeks: {
        label: "USGS NHD flowlines, 1:24,000",
        lines: [
          {
            id: "nhd-9",
            name: "Figueroa Creek",
            kind: "perennial",
            coordinates: [at(40, -50), at(40, 50)],
          },
        ],
      },
    });
    const crossing = fence.hands[0];
    expect(crossing?.label).toBe("Crosses Figueroa Creek · buffer 15 m (assumed)");
    render(
      <RehearsalPanel
        run={run({
          rehearsal: fence,
          focused: crossing?.id ?? null,
          frame: {
            t: fence.durationS,
            durationS: fence.durationS,
            playing: false,
            speed: 300,
            raised: 1,
            ended: true,
          },
        })}
      />,
    );
    expect(screen.getByText(crossing?.rule ?? "")).toBeInTheDocument();
    expect(screen.queryByTestId("rehearsal-leave-out")).toBeNull();
  });

  it("ends with the summary and where its numbers came from, and says what was left out", () => {
    render(
      <RehearsalPanel
        run={run({
          frame: {
            t: rehearsal.durationS,
            durationS: rehearsal.durationS,
            playing: false,
            speed: 300,
            raised: 1,
            ended: true,
          },
          leftOut: [{ zoneId: "Z-14", label: "Z-14", number: 4, hands: 2 }],
          ground: {
            elevation: null,
            creeks: null,
            heightAt: () => 0,
            better: "For 10 m bare-earth slope, analyze bare-earth terrain (USGS 3DEP).",
          },
        })}
      />,
    );
    const summary = screen.getByTestId("rehearsal-summary");
    for (const line of rehearsal.summary) expect(summary).toHaveTextContent(line.text);
    // A hand warns; a rule that could not be read warns too, dashed, so it never reads as clear.
    const hands = summary.querySelector('[data-tone="hand"]');
    expect(hands).toHaveClass("mc-note--warn");
    expect(hands).toHaveTextContent("1 of 3 checked lanes inside the 15 m buffer");
    const gap = summary.querySelector('[data-tone="gap"]');
    expect(gap).toHaveClass("mc-note--warn", "mc-rehearsal__gap");
    expect(gap).toHaveTextContent("Slope not checked");
    expect(summary).toHaveTextContent("Creek: USGS NHD flowlines, 1:24,000");
    expect(summary).toHaveTextContent("Limits: assumed per machine class");
    expect(screen.getByTestId("rehearsal-left-out")).toHaveTextContent(
      "Left out of this draft's preview: Z-14 lane 4. The approved plan still drives it, so its 2 hands still count on Approve",
    );
    expect(summary).toHaveTextContent("analyze bare-earth terrain (USGS 3DEP)");
    expect(screen.getByTestId("rehearsal-play")).toHaveTextContent("Play again");
    expect(screen.queryByTestId("rehearsal-skip")).toBeNull();
  });
});
