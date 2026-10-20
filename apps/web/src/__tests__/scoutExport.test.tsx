/**
 * The scout card lists every height its KMZ writes: the route's and the return-home height,
 * which stays under the same 400 ft cap at the highest height offered. Drone missions are a
 * stub until there is a drone (`VITE_ENABLE_DRONE_EXPORT`, off): with the flag off the card
 * offers no export, only a note; the flight tests below run with it on.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { env } from "@/app/env";
import { DroneScoutExport } from "@/features/mission/DroneScoutExport";
import { ScoutExport } from "@/features/mission/ScoutExport";
import type * as Wpml from "@/missions/wpml";
import { DEFAULT_SCOUT } from "@/missions/scout";
import type { Zone } from "@/missions/types";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));
// The writer as it is, unless a test makes it fail (as a chunk that cannot load does).
const writer = vi.hoisted(() => ({ fail: null as Error | null }));
vi.mock("@/missions/wpml", async (importOriginal) => {
  const actual = await importOriginal<typeof Wpml>();
  return {
    ...actual,
    buildScoutKmz: (...args: Parameters<typeof actual.buildScoutKmz>) => {
      if (writer.fail) throw writer.fail;
      return actual.buildScoutKmz(...args);
    },
  };
});

const droneExport = env.droneExportEnabled;
beforeEach(() => {
  env.droneExportEnabled = true;
});
afterEach(() => {
  cleanup();
  env.droneExportEnabled = droneExport;
});

const patch = {
  id: "A-01",
  name: "Fog patch 1",
  task: "Scout",
  footprint: {
    type: "Polygon",
    coordinates: [
      [
        [-120.045, 34.695],
        [-120.04, 34.695],
        [-120.04, 34.699],
        [-120.045, 34.699],
        [-120.045, 34.695],
      ],
    ],
  },
  scout: { ...DEFAULT_SCOUT, altitudeM: 120 },
} as unknown as Zone;

it("shows the return-home height, capped at 120 m above take-off", () => {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <DroneScoutExport zones={[patch]} onAltitude={() => undefined} />
    </QueryClientProvider>,
  );
  const facts = screen.getByTestId("scout-facts");
  expect(facts).toHaveTextContent("Height120 m above take-off");
  expect(facts).toHaveTextContent(
    "Returnflies home at 120 m above take-off, at the end or if the controller link is lost",
  );
});

function card(zones: Zone[]) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <DroneScoutExport zones={zones} />
    </QueryClientProvider>,
  );
}

const planned = (pattern: "nadir" | "oblique" | "orbit", gimbalPitchDeg: number) =>
  ({
    ...patch,
    scout: { ...DEFAULT_SCOUT, altitudeM: 120, purpose: "re-scan-weak", pattern, gimbalPitchDeg },
  }) as unknown as Zone;

it("refuses to export a nadir capture pitched off straight down: the lanes fly at -90°", () => {
  card([patch, planned("nadir", -60)]);
  const refusal = screen.getByTestId("scout-unsupported");
  expect(refusal).toHaveTextContent("pitched −60°");
  expect(refusal).toHaveTextContent("only straight-down");
  expect(screen.queryByTestId("scout-download")).toBeNull();
  expect(screen.queryByTestId("scout-facts")).toBeNull();
});

it.each([
  ["orbit", -30],
  ["oblique", -45],
] as const)("exports a %s capture, which the writer flies as planned", (pattern, pitch) => {
  card([patch, planned(pattern, pitch)]);
  expect(screen.queryByTestId("scout-unsupported")).toBeNull();
  expect(screen.getByTestId("scout-download")).toBeInTheDocument();
});

it("still exports a straight-down capture", () => {
  card([patch]);
  expect(screen.queryByTestId("scout-unsupported")).toBeNull();
  expect(screen.getByTestId("scout-download")).toBeInTheDocument();
});

it("offers no download for a route with an area it could not plan, and says which", () => {
  const tiny = {
    ...patch,
    id: "A-02",
    name: "Weak area 2",
    footprint: {
      type: "Polygon",
      coordinates: [
        [
          [-120.03, 34.695],
          [-120.02995, 34.695],
          [-120.02995, 34.69505],
          [-120.03, 34.69505],
          [-120.03, 34.695],
        ],
      ],
    },
  } as unknown as Zone;
  card([patch, tiny]);
  expect(screen.getByTestId("scout-unsupported")).toHaveTextContent("Weak area 2: no lane fits");
  expect(screen.queryByTestId("scout-download")).toBeNull();
  expect(screen.queryByTestId("scout-facts")).toBeNull();
});

describe("drone missions behind their flag", () => {
  it("with the flag on, loads the drone card and its download", async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScoutExport zones={[patch]} />
      </QueryClientProvider>,
    );
    expect(await screen.findByTestId("scout-download")).toBeInTheDocument();
    expect(screen.queryByTestId("scout-drone-stub")).toBeNull();
  });

  it("with the flag off (the default), offers no drone export, and says drone missions are not available yet", () => {
    env.droneExportEnabled = false;
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScoutExport zones={[patch]} onAltitude={() => undefined} />
      </QueryClientProvider>,
    );
    expect(screen.getByTestId("scout-drone-stub")).toHaveTextContent(
      "Drone missions: not available yet",
    );
    expect(screen.queryByTestId("scout-download")).toBeNull();
    expect(screen.queryByTestId("scout-facts")).toBeNull();
    expect(screen.queryByRole("group", { name: "Height above take-off" })).toBeNull();
  });
});

it("says when areas are flown as their re-scan asks, and counts the crossing passes", () => {
  const facts = (scout: Zone["scout"]) => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <DroneScoutExport zones={[{ ...patch, scout }]} />
      </QueryClientProvider>,
    );
    const card = screen.getByTestId("scout-export").textContent ?? "";
    const photos = Number(/about ([\d,]+)/.exec(card)?.[1]?.replace(/,/g, "") ?? NaN);
    cleanup();
    return { card, photos };
  };
  const nadir = facts({ ...DEFAULT_SCOUT, altitudeM: 120 });
  expect(nadir.card).toContain("flying lines with the camera straight down.");
  const oblique = facts({
    ...DEFAULT_SCOUT,
    altitudeM: 120,
    purpose: "re-scan-weak",
    pattern: "oblique",
    gimbalPitchDeg: -45,
  });
  expect(oblique.card).toContain(
    "flying each area as it asks: lines straight down, crossing obliques or an orbit.",
  );
  // Both sets of lanes: about twice the photos of the nadir lines.
  expect(oblique.photos).toBeGreaterThan(1.5 * nadir.photos);
});

it("writes the KMZ on the download click, its writer loaded then", async () => {
  const made: Blob[] = [];
  vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
    made.push(blob as Blob);
    return "blob:kmz";
  });
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(
    <QueryClientProvider client={new QueryClient()}>
      <DroneScoutExport zones={[patch]} />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("scout-download"));
  await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
  expect(made[0]?.type).toBe("application/vnd.google-earth.kmz");
  expect(made[0]?.size).toBeGreaterThan(1000);
  expect(screen.queryByRole("alert")).toBeNull();
  vi.restoreAllMocks();
});

it("says when the route cannot be written, and downloads nothing", async () => {
  writer.fail = new Error("Failed to fetch dynamically imported module");
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(
    <QueryClientProvider client={new QueryClient()}>
      <DroneScoutExport zones={[patch]} />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("scout-download"));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Failed to fetch dynamically imported module",
  );
  expect(click).not.toHaveBeenCalled();
  writer.fail = null;
  vi.restoreAllMocks();
});
