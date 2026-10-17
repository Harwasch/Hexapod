/**
 * The dress rehearsal in the plan card: the zones and crews it rehearses, the run as state
 * (`features/mission/dressRehearsal.ts`) and the panel (`Rehearsal.tsx`), with the ground
 * reader mocked and the real RehearsalManager on a fake viewer.
 */

import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { type CesiumWidget, Color, EntityCollection } from "cesium";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PlanDraft } from "@twin/contracts";
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
import { Emitter } from "@/lib/emitter";
import { planRehearsal } from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";

vi.mock("@/features/mission/rehearsalGround", () => ({ rehearsalGround: vi.fn() }));

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
    machineColor: () => Color.CYAN,
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
  const wrapper = ({ children }: { children: ReactNode }) => (
    <SceneContext.Provider value={registry}>{children}</SceneContext.Provider>
  );

  beforeEach(() => {
    scene = fakeScene();
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
    act(() => scene.events.emit("rehearsal-pick", { id: `hand:${hand?.id}` }));
    expect(result.current.focused).toBe(hand?.id);
    expect(scene.mission.flyToPoint).toHaveBeenCalled();
    rerender({ key: "b" });
    expect(result.current.status).toBe("idle");
    expect(result.current.raised).toBe(0);
    expect(scene.rehearsal.loaded).toBeNull();
  });

  it("leaves a lane out of the preview and rehearses without it", async () => {
    const { result } = renderHook(() => useDressRehearsal(draft, project, "a"), { wrapper });
    act(() => result.current.start());
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => result.current.skipToEnd());
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
    expect(result.current.leftOut).toEqual([{ zoneId: "Z-14", label: "Z-14", number: 1 }]);
    expect(result.current.rehearsal?.lanes.map((l) => l.number)).toEqual([2, 4]);
    expect(result.current.rehearsal?.hands).toEqual([]);
    act(() => result.current.restoreLanes());
    expect(scene.mission.restoreLanes).toHaveBeenCalled();
    expect(result.current.leftOut).toEqual([]);
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

  it("opens a hand with its rule, its source and Leave this lane out", async () => {
    const controls = run({ focused: hand?.id ?? null });
    render(<RehearsalPanel run={controls} />);
    expect(screen.getByText(hand?.rule ?? "")).toBeInTheDocument();
    expect(screen.getByText(hand?.source ?? "")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("rehearsal-leave-out"));
    expect(controls.leaveOut).toHaveBeenCalledWith(hand);
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
          leftOut: [{ zoneId: "Z-14", label: "Z-14", number: 4 }],
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
    for (const line of rehearsal.summary) expect(summary).toHaveTextContent(line);
    expect(summary).toHaveTextContent("Creek: USGS NHD flowlines, 1:24,000");
    expect(summary).toHaveTextContent("Limits: assumed per machine class");
    expect(screen.getByTestId("rehearsal-left-out")).toHaveTextContent(
      "Left out of this draft's preview: Z-14 lane 4.",
    );
    expect(summary).toHaveTextContent("analyze bare-earth terrain (USGS 3DEP)");
    expect(screen.getByTestId("rehearsal-play")).toHaveTextContent("Play again");
    expect(screen.queryByTestId("rehearsal-skip")).toBeNull();
  });
});
