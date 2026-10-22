/**
 * As captured on a simulated project (state/asCaptured.ts, ADR 0019): nothing its simulated
 * fleet did is drawn -- no machine marker or track, no zone's status or progress (the chip's
 * "mowing 58%", the treated zone's solid outline and stronger fill) -- while the zones'
 * outlines and names stay, as the plan's annotations. A project that is not simulated keeps
 * all of it: what its fleet reported is data.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import {
  type Color,
  type ColorMaterialProperty,
  EllipsoidTerrainProvider,
  EntityCollection,
  Event,
  JulianDate,
  PolylineDashMaterialProperty,
  type CesiumWidget,
} from "cesium";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { CameraController } from "@/cesium/CameraController";
import { MissionManager } from "@/cesium/MissionManager";
import { MissionOverlays } from "@/features/mission/MissionOverlays";
import type { Machine, Project, Zone } from "@/missions/types";
import { useAsCaptured } from "@/state/asCaptured";
import { useMission } from "@/state/mission";
import { initialCamera, useViewer } from "@/state/viewer";

const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [0, 0],
      [0, 0.001],
      [0.001, 0.001],
      [0, 0],
    ],
  ],
};

function zone(id: string, treated: boolean): Zone {
  return {
    id,
    name: `${id} field`,
    short: treated ? "mowed" : "mowing 58%",
    footprint: square,
    tone: "teal",
    treated,
    acres: 10,
    task: "Mow",
    machines: "",
    progressPct: treated ? 100 : 58,
    note: "",
    anchor: { longitude: 0, latitude: 0 },
  };
}

function machine(id: string): Machine {
  return {
    id,
    name: `${id} Kestrel`,
    model: "Ranger 2",
    task: "Mowing",
    status: "working",
    position: { longitude: 0.0005, latitude: 0.0005 },
    headingDeg: 0,
    batteryPct: 80,
    acresDone: 12,
    shift: "1:00",
    zoneLabel: "Z-1",
    primaryAction: "Pause",
    track: [
      { longitude: 0, latitude: 0 },
      { longitude: 0.0005, latitude: 0.0005 },
    ],
  };
}

function project(simulated: boolean): Project {
  return {
    id: "p",
    siteId: "s",
    name: "Test ranch",
    meta: "10 acres",
    simulated,
    machines: [machine("TR-04")],
    zones: [zone("Z-1", false), zone("Z-2", true)],
    plans: [],
    agent: { headline: "Agent idle", summary: "", footer: "", actions: [] },
    fleetStats: [],
    fleetNote: "",
    workLog: [],
    feeds: [],
  };
}

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  useMission.setState({
    project: null,
    selection: null,
    layers: { zones: true, tracks: true, vegetation: false },
  });
  useViewer.setState({ camera: { ...initialCamera, altitude: 1_200 } });
});

afterEach(() => {
  cleanup();
  useAsCaptured.getState().set(false);
});

describe("the map's overlays of a simulated project", () => {
  it("keep each zone's name and drop its simulated status, and no machine is drawn", () => {
    useMission.setState({ project: project(true) });
    render(wrap(<MissionOverlays />));
    expect(screen.getByTestId("zone-chip-Z-1")).toHaveTextContent("mowing 58%");
    expect(screen.getByTestId("machine-marker-TR-04")).toBeInTheDocument();
    act(() => useAsCaptured.getState().set(true));
    const chip = screen.getByTestId("zone-chip-Z-1");
    expect(chip).toHaveTextContent("Z-1");
    expect(chip).not.toHaveTextContent("mowing");
    expect(chip).toHaveAttribute("aria-label", "Z-1 field");
    expect(screen.queryByTestId("machine-marker-TR-04")).not.toBeInTheDocument();
    act(() => useAsCaptured.getState().set(false));
    expect(screen.getByTestId("zone-chip-Z-1")).toHaveTextContent("mowing 58%");
  });

  it("count no simulated machine in the site's pin", () => {
    useMission.setState({ project: project(true) });
    useViewer.setState({ camera: { ...initialCamera, altitude: 400_000 } });
    useAsCaptured.getState().set(true);
    render(wrap(<MissionOverlays />));
    expect(screen.getByTestId("site-pin")).toHaveAttribute(
      "aria-label",
      "Test ranch: 2 zones. Fly to the site",
    );
  });

  it("are all kept for a project whose fleet is real", () => {
    useMission.setState({ project: project(false) });
    useAsCaptured.getState().set(true);
    render(wrap(<MissionOverlays />));
    expect(screen.getByTestId("zone-chip-Z-1")).toHaveTextContent("mowing 58%");
    expect(screen.getByTestId("machine-marker-TR-04")).toBeInTheDocument();
  });
});

function manager() {
  const entities = new EntityCollection();
  const viewer = {
    entities,
    scene: { postRender: new Event(), requestRender: () => undefined },
    terrainProvider: new EllipsoidTerrainProvider(),
  } as unknown as CesiumWidget;
  const events = { on: () => () => undefined };
  const mission = new MissionManager(viewer, events as never, {} as CameraController);
  const now = JulianDate.now();
  const zoneLook = (id: string) => {
    const e = entities.getById(`mission:zone:${id}`);
    const fill = e?.polygon?.material as ColorMaterialProperty | undefined;
    return {
      alpha: (fill?.color?.getValue(now) as Color | undefined)?.alpha,
      width: e?.polyline?.width?.getValue(now) as number | undefined,
      dashed: e?.polyline?.material instanceof PolylineDashMaterialProperty,
    };
  };
  const tracks = () =>
    entities.values.filter((e) => e.id.startsWith("mission:track")).map((e) => e.show);
  return { mission, zoneLook, tracks };
}

describe("the zones and tracks of a simulated project", () => {
  it("are drawn as the plan alone while As captured hides the fleet's work, and as before after", () => {
    const { mission, zoneLook, tracks } = manager();
    mission.setProject(project(true));
    // Z-2 is drawn treated by the simulated fleet: solid, wider, a stronger fill.
    expect(zoneLook("Z-2")).toMatchObject({ dashed: false, width: 2.4 });
    expect(zoneLook("Z-2").alpha).toBeCloseTo(0.22, 5);
    expect(tracks().length).toBeGreaterThan(0);
    expect(tracks().every(Boolean)).toBe(true);
    mission.hideAgents(true);
    expect(zoneLook("Z-2")).toMatchObject({ dashed: true, width: 1.8 });
    expect(zoneLook("Z-2").alpha).toBeCloseTo(0.14, 5);
    expect(tracks().some(Boolean)).toBe(false);
    // A project set meanwhile is drawn the same way.
    mission.setProject(project(true));
    expect(zoneLook("Z-2")).toMatchObject({ dashed: true, width: 1.8 });
    mission.hideAgents(false);
    expect(zoneLook("Z-2")).toMatchObject({ dashed: false, width: 2.4 });
    expect(zoneLook("Z-2").alpha).toBeCloseTo(0.22, 5);
    expect(tracks().every(Boolean)).toBe(true);
  });

  it("are left alone for a project whose fleet is real", () => {
    const { mission, zoneLook, tracks } = manager();
    mission.setProject(project(false));
    mission.hideAgents(true);
    expect(zoneLook("Z-2")).toMatchObject({ dashed: false, width: 2.4 });
    expect(tracks().every(Boolean)).toBe(true);
  });
});
