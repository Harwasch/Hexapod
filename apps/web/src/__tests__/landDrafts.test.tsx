import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { LandCreate } from "@twin/contracts";
import { api } from "@/api/client";
import { LandDraftRecovery } from "@/features/land/LandDraftRecovery";
import { useLand } from "@/state/land";

const key = "living-world-land-draft:pilot";
const draft: LandCreate = {
  name: "Unfinished ranch",
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
      ],
    ],
  },
  source: { method: "drawn", label: "Drawn" },
};
beforeEach(() => {
  localStorage.clear();
  useLand.getState().clear();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("recovers an unsaved boundary after reload and removes it when discarded", async () => {
  const first = render(<LandDraftRecovery scope="pilot" />);
  act(() => useLand.getState().propose(draft));
  expect(localStorage.getItem(key)).toContain("Unfinished ranch");
  first.unmount();
  useLand.getState().clear();
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume boundary draft" }));
  await waitFor(() => expect(useLand.getState().draft?.name).toBe(draft.name));
  act(() => useLand.getState().cancel());
  expect(localStorage.getItem(key)).toBeNull();
});

it("keeps a newer saved revision intact until a separate draft is explicitly requested", async () => {
  localStorage.setItem(
    key,
    JSON.stringify({ version: 1, draft, activeId: "land", revision: 1, savedAt: "2026-10-09" }),
  );
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { ...draft, id: "land", revision: 2 },
    response: new Response(),
  });
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume boundary draft" }));
  await screen.findByRole("button", { name: "Restore as a separate area" });
  expect(useLand.getState().draft).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Restore as a separate area" }));
  await waitFor(() => expect(useLand.getState().draft?.name).toBe(draft.name));
  expect(useLand.getState().active).toBeNull();
});

it("does not display a different workspace's stored draft", () => {
  localStorage.setItem(
    "living-world-land-draft:someone%2Fprivate",
    JSON.stringify({ version: 1, draft }),
  );
  render(<LandDraftRecovery scope="pilot" />);
  expect(screen.queryByText(/Resume Unfinished ranch/)).toBeNull();
});
