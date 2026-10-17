/**
 * The scout card lists every height its KMZ writes: the route's and the return-home height,
 * which stays under the same 400 ft cap at the highest height offered.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ScoutExport } from "@/features/mission/ScoutExport";
import { DEFAULT_SCOUT } from "@/missions/scout";
import type { Zone } from "@/missions/types";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

afterEach(cleanup);

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
      <ScoutExport zones={[patch]} onAltitude={() => undefined} />
    </QueryClientProvider>,
  );
  const facts = screen.getByTestId("scout-facts");
  expect(facts).toHaveTextContent("Height120 m above take-off");
  expect(facts).toHaveTextContent(
    "Returnflies home at 120 m above take-off, at the end or if the controller link is lost",
  );
});
