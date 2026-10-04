import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it } from "vitest";

import { MissionOverlays } from "@/features/mission/MissionOverlays";
import {
  GLOBE_SCALE_ALTITUDE_M,
  collapsesToPin,
  pinCounts,
  pinLabel,
  siteAnchor,
} from "@/missions/sitePin";
import type { Machine, Project, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { initialCamera, useViewer } from "@/state/viewer";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [0, 0],
      [0, 1],
      [1, 1],
      [0, 0],
    ],
  ],
};

function zone(id: string, longitude: number, latitude: number): Zone {
  return {
    id,
    name: `${id} field`,
    short: "field",
    footprint: square,
    tone: "teal",
    treated: false,
    acres: 10,
    task: "Mow",
    machines: "",
    progressPct: 0,
    note: "",
    anchor: { longitude, latitude },
  };
}

function machine(id: string): Machine {
  return {
    id,
    name: `${id} Kestrel`,
    model: "Ranger 2",
    task: "Mowing",
    status: "working",
    position: { longitude: 1, latitude: 1 },
    headingDeg: 0,
    batteryPct: 80,
    acresDone: 12,
    shift: "1:00",
    zoneLabel: "Z-1",
    primaryAction: "Pause",
  };
}

const project: Project = {
  id: "p",
  siteId: "s",
  name: "Test ranch",
  meta: "10 acres",
  simulated: false,
  machines: [machine("TR-04"), machine("TR-07")],
  zones: [zone("Z-1", 0, 0), zone("Z-2", 2, 4)],
  plans: [],
  agent: { headline: "Agent idle", summary: "", footer: "", actions: [] },
  fleetStats: [],
  fleetNote: "",
  workLog: [],
  feeds: [],
};

beforeEach(() => {
  useMission.setState({
    project: null,
    selection: null,
    layers: { zones: true, tracks: true, vegetation: false },
  });
  useViewer.setState({ camera: initialCamera });
});

describe("the site pin at globe scale", () => {
  it("collapses above 50 km, stands in the middle of what it replaces, and counts it", () => {
    expect(GLOBE_SCALE_ALTITUDE_M).toBe(50_000);
    expect(collapsesToPin(400_000)).toBe(true);
    expect(collapsesToPin(49_000)).toBe(false);
    expect(collapsesToPin(Number.NaN)).toBe(false);
    expect(siteAnchor(project)).toEqual({ longitude: 1, latitude: 2 });
    expect(siteAnchor({ zones: [], machines: [machine("TR-01")] })).toEqual({
      longitude: 1,
      latitude: 1,
    });
    expect(siteAnchor({ zones: [], machines: [] })).toBeNull();
    expect(pinCounts(project, true)).toEqual({ machines: 2, zones: 2, total: 4 });
    expect(pinCounts(project, false)).toEqual({ machines: 2, zones: 0, total: 2 });
    expect(pinLabel("Test ranch", { machines: 1, zones: 2 })).toBe(
      "Test ranch: 1 machine, 2 zones",
    );
  });

  it("draws one pin from far away and the markers and chips up close", () => {
    useMission.setState({ project });
    useViewer.setState({ camera: { ...initialCamera, altitude: 400_000 } });
    const { rerender } = render(wrap(<MissionOverlays />));
    const pin = screen.getByTestId("site-pin");
    expect(pin).toHaveTextContent("Test ranch");
    expect(pin).toHaveTextContent("4");
    // Hidden until the scene places it, so its name is read off the attribute here.
    expect(pin).toHaveAttribute("aria-label", "Test ranch: 2 machines, 2 zones. Fly to the site");
    expect(screen.queryByTestId("machine-marker-TR-04")).not.toBeInTheDocument();

    useViewer.setState({ camera: { ...initialCamera, altitude: 1_200 } });
    rerender(wrap(<MissionOverlays />));
    expect(screen.queryByTestId("site-pin")).not.toBeInTheDocument();
    expect(screen.getByTestId("machine-marker-TR-04")).toBeInTheDocument();
    expect(screen.getByTestId("zone-chip-Z-2")).toBeInTheDocument();
  });
});
