/**
 * Developer fixtures in the site switcher (`features/sites/siteFixtures.ts`): the synthetic
 * tree and audit or probe records are listed only with developer readouts on; a real capture is
 * never taken for one, and the site you are at is always listed.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SiteSummary } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { builtinRanchSite, toSummary } from "@/api/fallback";
import { ProjectCard } from "@/features/mission/ProjectCard";
import { isDeveloperFixture, listedSites } from "@/features/sites/siteFixtures";
import { DEMO_RANCH_SLUG } from "@/missions/demoRanchData";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

const base: SiteSummary = toSummary(builtinRanchSite());

function site(id: string, slug: string, name: string, extra: Partial<SiteSummary> = {}) {
  return { ...base, id, slug, name, description: null, latestObservedAt: null, ...extra };
}

const ranch = site("77777777-7777-4777-8777-777777777777", DEMO_RANCH_SLUG, "Oak Bench Ranch");
const synthetic = site(
  "88888888-8888-4888-8888-888888888888",
  "synthetic-tree",
  "Synthetic tree (Living Survey fixture)",
  { description: "A 6 m procedural tree with a known skeleton. Not a survey." },
);
const audit = site("99999999-9999-4999-8999-999999999999", "audit-probe-site", "Audit probe site");
const minnetonka = site(
  "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  "minnetonka-tree",
  "Minnetonka tree (real capture, Living Survey)",
);
/** A phone scan of a soil probe: its photos have a date, so it is a capture, not a fixture. */
const soilProbe = site("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "soil-probe", "Soil probe", {
  latestObservedAt: "2026-09-28T00:30:00Z",
});

describe("isDeveloperFixture", () => {
  it("knows the synthetic tree and audit or probe records", () => {
    expect(isDeveloperFixture(synthetic)).toBe(true);
    expect(isDeveloperFixture(audit)).toBe(true);
    expect(isDeveloperFixture(site("c", "probe-123", "Untitled"))).toBe(true);
  });

  it("never takes a real capture, or a place, for one", () => {
    expect(isDeveloperFixture(minnetonka)).toBe(false);
    expect(isDeveloperFixture(soilProbe)).toBe(false);
    expect(isDeveloperFixture(ranch)).toBe(false);
    expect(isDeveloperFixture(site("d", "cesium-splat-demo", "Cesium Gaussian splat demo"))).toBe(
      false,
    );
  });

  it("lists everything with developer readouts on, and always the site you are at", () => {
    const all = [ranch, synthetic, audit, minnetonka, soilProbe];
    expect(listedSites(all, { developer: false }).shown.map((s) => s.slug)).toEqual([
      DEMO_RANCH_SLUG,
      "minnetonka-tree",
      "soil-probe",
    ]);
    expect(listedSites(all, { developer: false }).hidden).toBe(2);
    expect(listedSites(all, { developer: true })).toEqual({ shown: all, hidden: 0 });
    const there = listedSites(all, { developer: false, hereId: synthetic.id });
    expect(there.shown).toContain(synthetic);
    expect(there.hidden).toBe(1);
  });
});

describe("the site switcher", () => {
  function wrap(children: ReactNode) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    return (
      <QueryClientProvider client={client}>
        <GlassTooltipProvider>{children}</GlassTooltipProvider>
      </QueryClientProvider>
    );
  }

  beforeEach(() => {
    const catalog = [ranch, synthetic, audit, minnetonka];
    vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path === "/api/v1/sites" ? catalog : null,
        response: new Response(null, { status: 200 }),
      })) as never);
    useSettings.getState().reset();
    useSites.setState({ activeSiteId: null });
    useUi.setState({ switcherFocus: "sites" });
    useMission.setState({ project: null, projectsOpen: true });
  });

  afterEach(() => {
    useMission.setState({ projectsOpen: false });
    vi.restoreAllMocks();
  });

  it("leaves developer fixtures out unless developer readouts are on", async () => {
    const { unmount } = render(wrap(<ProjectCard />));
    const switcher = screen.getByRole("dialog", { name: "Switch site" });
    await within(switcher).findByText("Oak Bench Ranch");
    expect(
      within(switcher).getByText("Minnetonka tree (real capture, Living Survey)"),
    ).toBeVisible();
    expect(within(switcher).queryByText(/Synthetic tree/)).not.toBeInTheDocument();
    expect(within(switcher).queryByText("Audit probe site")).not.toBeInTheDocument();
    expect(within(switcher).getByTestId("switcher-fixtures-hidden")).toHaveTextContent(
      "2 developer fixtures are not listed. Turn on Settings › Advanced › Show developer readouts",
    );
    unmount();

    useSettings.getState().set({ devReadouts: true });
    render(wrap(<ProjectCard />));
    const all = screen.getByRole("dialog", { name: "Switch site" });
    expect(
      await within(all).findByText("Synthetic tree (Living Survey fixture)"),
    ).toBeInTheDocument();
    expect(within(all).getByText("Audit probe site")).toBeInTheDocument();
    expect(within(all).queryByTestId("switcher-fixtures-hidden")).not.toBeInTheDocument();
  });
});
