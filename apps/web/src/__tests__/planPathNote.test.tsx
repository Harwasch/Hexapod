/**
 * The plan card's PATH note on the demo ranch: a scan the planner sized for a mapping drone,
 * in a fleet where nothing flies, says the drone has no crew, so the assumed class is never
 * read as one of the ranch's machines.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { PlanDraft, ZoneCoverage } from "@twin/contracts";

import { api } from "@/api/client";
import { PlanCard } from "@/features/mission/PlanCard";
import { demoRanchProject } from "@/missions/demo";
import { DEFAULT_SCOUT } from "@/missions/scout";
import type { Project } from "@/missions/types";
import { useMission } from "@/state/mission";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

const flight: ZoneCoverage = {
  zoneId: "Z-08",
  kind: "area",
  machineClass: "drone",
  areaM2: 177_000,
  lengthM: 5_500,
  transitM: 300,
  passes: 14,
  spans: 14,
  spacingM: 31,
  altitudeM: 70,
  speedMps: 8,
  turnS: 0,
  hours: 0.47,
  capped: false,
  basis: "Z-08 Creek draw: 5.5 km in 14 passes; Mapping drone, Mavic 3 Enterprise class (assumed)",
};

const scan: PlanDraft = {
  title: "3D scan the creek draw",
  objective: "3D scan the creek draw",
  zoneIds: ["Z-08"],
  machineIds: [],
  cadence: "once",
  startDate: "2026-10-17",
  endDate: "2026-10-21",
  estimates: { acres: 43.8, machineHours: 0.47, calendarDays: 4, pathM: 5_500 },
  steps: [],
  assumptions: [],
  risks: [],
  questions: [],
  clarifications: [],
  source: "rules",
  model: null,
  note: "",
  coverage: [flight],
};

function show(draft: PlanDraft, project: Project = demoRanchProject()) {
  useMission.setState({ composer: null });
  useMission.getState().setProject(project);
  useMission.getState().openComposer({ goal: draft.objective, zoneIds: draft.zoneIds });
  useMission.getState().updateComposer({ status: "ready", draft });
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <PlanCard project={project} />
    </QueryClientProvider>,
  );
  return screen.getByTestId("plan-path");
}

beforeEach(() => {
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { configured: false, provider: "rules", model: null },
    response: new Response(),
  });
});

afterEach(() => {
  cleanup();
  useMission.setState({ composer: null });
  vi.restoreAllMocks();
});

it("says the drone has no crew when nothing in the fleet flies", () => {
  expect(show(scan)).toHaveTextContent(
    "Mapping drone, Mavic 3 Enterprise class (assumed): 70 m above ground",
  );
  expect(screen.getByTestId("plan-path")).toHaveTextContent(
    "No machine in this fleet flies, so the drone has no crew yet.",
  );
});

it("says nothing of the kind when the flight has a crew", () => {
  expect(show({ ...scan, machineIds: ["TR-11"] })).not.toHaveTextContent(/no crew yet/);
});

it("says a scout area's height is above take-off, as its route flies it", () => {
  const demo = demoRanchProject();
  // A fog patch on the creek draw: the planner sized it from the capture it carries.
  const scouted = {
    ...demo,
    zones: demo.zones.map((z) => (z.id === "Z-08" ? { ...z, scout: { ...DEFAULT_SCOUT } } : z)),
  };
  const path = show({ ...scan, coverage: [{ ...flight, altitudeM: 60 }] }, scouted);
  expect(path).toHaveTextContent(
    "Mapping drone, Mavic 3 Enterprise class (assumed): 60 m above take-off (lines 27 m apart)",
  );
  expect(path).not.toHaveTextContent(/above ground/);
});
