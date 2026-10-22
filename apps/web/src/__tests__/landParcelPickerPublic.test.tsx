import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import type { CandidateResult, LandCandidate } from "@twin/contracts";

import { api } from "@/api/client";
import { LandParcelPicker, type PublicParcelPick } from "@/features/land/surface/LandParcelPicker";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

afterEach(() => {
  cleanup();
  useLand.getState().clear();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
});

const parcel: LandCandidate = {
  id: "ca-parcel/06001/2-51-1",
  label: "Parcel 2-51-1",
  geometry: {
    type: "Polygon",
    coordinates: [
      [
        [-122.272, 37.804],
        [-122.27, 37.804],
        [-122.27, 37.806],
        [-122.272, 37.806],
        [-122.272, 37.804],
      ],
    ],
  },
  distanceM: 0,
  properties: {},
  source: {
    method: "parcel",
    meaning: "recorded-parcel",
    label: "Alameda County assessor parcel",
    recordId: "2-51-1",
    attribution: "Alameda County Information Technology Department",
  },
};

const answer: CandidateResult = {
  status: "available",
  message: "Parcels from Alameda County records near your click.",
  candidates: [parcel],
  truncated: false,
};

it("asks the public lookup, offers parcels alone, and hands the choice back", async () => {
  const post = vi.spyOn(api, "POST");
  const pick: PublicParcelPick = {
    find: vi.fn(() => Promise.resolve(answer)),
    onUse: vi.fn(),
    onCancel: vi.fn(),
  };
  useLand.getState().begin("candidates");
  useLand.getState().addPoint([-122.2711, 37.8044]);
  render(<LandParcelPicker publicPick={pick} />);

  expect(pick.find).toHaveBeenCalledWith([-122.2711, 37.8044], expect.any(AbortSignal));
  const box = await screen.findByRole("checkbox", { name: "Parcel 2-51-1" });
  // Only what a visitor without a workspace can use: no kinds, no tracing, no description.
  expect(screen.queryByRole("radiogroup", { name: "What to pick" })).toBeNull();
  expect(screen.queryByText("Describe it instead")).toBeNull();
  fireEvent.click(box);
  expect(screen.queryByRole("button", { name: "Trace over them" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: /^Use this/ }));
  await waitFor(() => expect(pick.onUse).toHaveBeenCalledWith([parcel]));
  // Handed back, not proposed as a land, and nothing went through the workspace API.
  expect(useLand.getState().draft).toBeNull();
  expect(post).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(pick.onCancel).toHaveBeenCalled();
});
