/**
 * Capture date (features/timeline/TimelineControl.tsx): the active site's dated captures by
 * date, platform and alignment, the one the scene shows checked, and a choice handed to the
 * scene rather than kept by the control.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site } from "@twin/contracts";

import { queryKeys } from "@/api/queries";
import { TimelineControl } from "@/features/timeline/TimelineControl";
import { useSites } from "@/state/sites";

import { A1, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

const scene = vi.hoisted(() => ({ sites: { setEpoch: vi.fn(() => Promise.resolve()) } }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

function show(site: Site) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  return render(
    <QueryClientProvider client={client}>
      <TimelineControl />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  scene.sites.setEpoch.mockClear();
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
  });
});

afterEach(() => cleanup());

describe("the capture-date control", () => {
  it("shows nothing for a site with fewer than two epochs", () => {
    show(twoEpochSite({ epochs: [E1], assets: [A1] }));
    expect(screen.queryByTestId("timeline")).toBeNull();
    cleanup();
    // Nor for a site whose second date has nothing of the representation shown.
    useSites.setState({ representation: { [SITE_ID]: "mesh" } });
    show(twoEpochSite());
    expect(screen.queryByTestId("timeline")).toBeNull();
  });

  it("offers each epoch by date, platform and alignment, the latest checked", () => {
    show(twoEpochSite());
    const timeline = screen.getByTestId("timeline");
    expect(timeline).toHaveAttribute("aria-label", "Capture date");
    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(2);
    const mar = screen.getByRole("radio", { name: /Mar 1, 2026.*Drone.*Defines the site frame/ });
    const jun = screen.getByRole("radio", {
      name: /Jun 1, 2026.*Phone video.*Not aligned yet: placed by its own GPS/,
    });
    expect(jun).toHaveAttribute("aria-checked", "true");
    expect(mar).toHaveAttribute("aria-checked", "false");
    // On screen: the date over what captured it and the few words for its placement.
    expect(jun).toHaveTextContent("Jun 1, 2026");
    expect(jun).toHaveTextContent("Phone video · not aligned");
    expect(mar).toHaveTextContent("Drone · site frame");
  });

  it("choosing a date asks the scene for that epoch", async () => {
    show(twoEpochSite());
    await userEvent.click(screen.getByRole("radio", { name: /Mar 1, 2026/ }));
    expect(scene.sites.setEpoch).toHaveBeenCalledWith(E1.id);
    // The control does not check it itself: the scene says what it shows.
    expect(screen.getByRole("radio", { name: /Jun 1, 2026/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("follows the scene's choice", () => {
    show(twoEpochSite());
    act(() => useSites.setState({ epoch: { [SITE_ID]: E1.id } }));
    expect(screen.getByRole("radio", { name: /Mar 1, 2026/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("names an epoch M3 aligned by the label it wrote", () => {
    const aligned = {
      ...E2,
      alignment: {
        verdict: "coarse",
        uncertaintyM: 0.065,
        label: "Phone video, aligned visually, ±0.065 m",
      },
    };
    show(twoEpochSite({ epochs: [E1, aligned] }));
    const jun = screen.getByRole("radio", { name: /Jun 1, 2026/ });
    expect(jun.getAttribute("aria-label")).toContain("Phone video, aligned visually, ±0.065 m");
    expect(jun).toHaveTextContent("Phone video · aligned visually");
    expect(jun.getAttribute("aria-label")).not.toMatch(/survey/i);
  });
});
