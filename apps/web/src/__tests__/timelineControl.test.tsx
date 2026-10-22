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
import { useViewRule } from "@/state/viewRule";

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
    compareDates: null,
  });
});

afterEach(() => {
  cleanup();
  useViewRule.setState({ active: null, ended: null, legend: null });
});

describe("the capture-date control", () => {
  it("reads As of while a per-cell view rule draws the dates as one", () => {
    useViewRule.setState({ active: { siteId: SITE_ID, rule: "latest" } });
    show(twoEpochSite());
    expect(screen.getByTestId("timeline")).toHaveTextContent(/^As of/);
    expect(screen.getByTestId("timeline-epochs")).toHaveAttribute("aria-label", "As of");
    // Another site's rule says nothing here.
    act(() => useViewRule.setState({ active: { siteId: "elsewhere", rule: "latest" } }));
    expect(screen.getByTestId("timeline")).not.toHaveTextContent(/^As of/);
    expect(screen.getByTestId("timeline-epochs")).toHaveAttribute("aria-label", "Capture date");
  });

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

  it("says a date that is only the upload's is one, on screen and in its name", () => {
    show(twoEpochSite({ epochs: [E1, { ...E2, observedAtSource: "created" }] }));
    const jun = screen.getByRole("radio", {
      name: /^Uploaded Jun 1, 2026 \(capture time not recorded\), Phone video/,
    });
    expect(jun).toHaveTextContent("Uploaded Jun 1, 2026 (capture time not recorded)");
    // A date the capture recorded is said as before.
    expect(screen.getByRole("radio", { name: /^Mar 1, 2026, Drone/ })).toBeInTheDocument();
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

  it("while two dates are compared, checks none and says which side each is on", async () => {
    show(twoEpochSite());
    act(() => useSites.getState().setCompareDates(SITE_ID, { left: E1.id, right: E2.id }));
    const mar = screen.getByRole("radio", { name: /Mar 1, 2026/ });
    const jun = screen.getByRole("radio", { name: /Jun 1, 2026/ });
    expect(mar).toHaveAttribute("aria-checked", "false");
    expect(jun).toHaveAttribute("aria-checked", "false");
    expect(mar).toHaveTextContent("Drone · left of divider");
    expect(jun).toHaveTextContent("Phone video · right of divider");
    expect(jun.getAttribute("aria-label")).toMatch(/shown right of the divider$/);
    // Choosing one asks for it alone (the scene ends the comparison).
    await userEvent.click(jun);
    expect(scene.sites.setEpoch).toHaveBeenCalledWith(E2.id);
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    expect(jun).toHaveAttribute("aria-checked", "true");
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
