import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SiteAsset } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { StatusLine } from "@/features/mission/StatusLine";
import {
  agentLine,
  fleetCounts,
  fleetLine,
  globeLine,
  isFresh,
  REPLY_HOLD_MS,
  sumFleet,
} from "@/features/mission/statusSummary";
import { shownAsset, siteLoad } from "@/features/sites/siteLoad";
import type { Machine, Project } from "@/missions/types";
import type { AgentLogEntry } from "@/state/mission";
import { useMission } from "@/state/mission";
import { defaultAssetRuntime, useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

function machine(id: string, status: Machine["status"]): Machine {
  return {
    id,
    name: id,
    model: "",
    task: "",
    status,
    position: { longitude: 0, latitude: 0 },
    headingDeg: 0,
    batteryPct: 80,
    acresDone: null,
    shift: "",
    zoneLabel: "",
    primaryAction: "Pause",
  };
}

function project(over: Partial<Project> = {}): Project {
  return {
    id: "p",
    siteId: "site-1",
    name: "Test ranch",
    meta: "",
    simulated: false,
    machines: [
      machine("TR-01", "working"),
      machine("TR-02", "attention"),
      machine("TR-03", "idle"),
    ],
    zones: [],
    plans: [],
    agent: {
      headline: "Agent idle",
      summary: "Working on the schedule",
      footer: "Review when ready",
      actions: [
        { id: "a1", text: "Sequencing passes against the forecast", status: "run" },
        { id: "a2", text: "Merging the buffer", status: "run" },
        { id: "a3", text: "Holding TR-12", status: "wait" },
        { id: "a4", text: "Drew the buffer", status: "done" },
      ],
    },
    fleetStats: [],
    fleetNote: "",
    workLog: [],
    feeds: [],
    ...over,
  };
}

function entry(role: AgentLogEntry["role"], text: string, at = Date.now()): AgentLogEntry {
  return { id: `${role}-${text}`, at, role, text };
}

describe("status line words", () => {
  it("counts the fleet the way the Fleet window does", () => {
    // No KPIs from the provider: counted by status; a machine needing attention is still out.
    expect(fleetCounts(project())).toEqual({ machines: 3, working: 2, attention: 1 });
    // The provider's KPIs win, so the line and the window never disagree.
    const kpis = project({
      fleetStats: [
        { value: "4", label: "Working now" },
        { value: "2", label: "Need attention", accent: true },
      ],
    });
    expect(fleetCounts(kpis)).toEqual({ machines: 3, working: 4, attention: 2 });
    expect(fleetLine(fleetCounts(kpis))).toBe("Fleet: 4 working · 2 need attention");
    expect(fleetLine({ machines: 0, working: 0, attention: 0 })).toBe(
      "Fleet: no machines registered",
    );
    expect(fleetLine({ machines: 2, working: 0, attention: 0 })).toBe("Fleet: 2 machines idle");
    expect(fleetLine({ machines: 2, working: 2, attention: 0 })).toBe("Fleet: 2 working");
  });

  it("sums sites at globe scale", () => {
    const counts = sumFleet([fleetCounts(project()), fleetCounts(project({ machines: [] }))]);
    expect(globeLine(3, counts)).toBe("3 sites · 3 machines · 1 alert");
    expect(globeLine(1, { machines: 0, working: 0, attention: 0 })).toBe(
      "1 site · 0 machines · no alerts",
    );
  });

  it("says what the agent is doing, or just said, in one line", () => {
    const p = project();
    expect(agentLine(p, [], false, false)).toEqual({
      text: "Sequencing passes against the forecast",
      more: 2,
      busy: true,
      idle: false,
    });
    expect(agentLine(p, [], true, false)).toMatchObject({ text: "Drafting a plan…", more: 3 });
    const replied = [entry("you", "where is TR-02"), entry("agent", "Locating TR-02")];
    expect(agentLine(p, replied, false, true)).toMatchObject({ text: "Locating TR-02", more: 3 });
    // Once the reply has had its moment, the running task is back.
    expect(agentLine(p, replied, false, false).text).toBe("Sequencing passes against the forecast");
    expect(agentLine(p, [entry("you", "hide zones")], false, true).text).toBe(
      "On it: “hide zones”",
    );
    expect(agentLine(project({ agent: { ...p.agent, actions: [] } }), [], false, false)).toEqual({
      text: "idle",
      more: 0,
      busy: false,
      idle: true,
    });
    expect(isFresh({ at: 0 }, REPLY_HOLD_MS - 1)).toBe(true);
    expect(isFresh({ at: 0 }, REPLY_HOLD_MS)).toBe(false);
  });
});

describe("site model load feedback", () => {
  const ready = { ...defaultAssetRuntime, loadState: "ready" as const };

  it("goes from loading, through a percentage, to nothing, and says when it failed", () => {
    expect(siteLoad(undefined, 0, false)).toBeNull();
    expect(siteLoad({ ...defaultAssetRuntime, loadState: "loading" }, 0, false)).toEqual({
      phase: "loading",
      percent: null,
    });
    const midway = { ...ready, progress: { pending: 30, processing: 10 } };
    expect(siteLoad(midway, 100, false)).toEqual({ phase: "loading", percent: 60 });
    expect(siteLoad(ready, 100, false)).toBeNull();
    // After the first load drains, tiles streaming in as the camera moves are not news.
    expect(siteLoad(midway, 100, true)).toBeNull();
    expect(
      siteLoad({ ...defaultAssetRuntime, loadState: "error", error: "ion refused" }, 0, false),
    ).toEqual({ phase: "error", message: "ion refused" });
  });

  it("picks the asset the scene shows, the way SiteManager does", () => {
    const asset = (id: string, representation: SiteAsset["representation"], extra = {}) =>
      ({
        id,
        representation,
        defaultVisible: false,
        observedAt: null,
        ...extra,
      }) as SiteAsset;
    const assets = [
      asset("old-mesh", "mesh", { observedAt: "2024-01-01" }),
      asset("new-mesh", "mesh", { observedAt: "2025-01-01" }),
      asset("splat", "gaussian-splat", { defaultVisible: true }),
    ];
    expect(shownAsset(assets, "mesh", undefined)?.id).toBe("new-mesh");
    expect(shownAsset(assets, "mesh", "old-mesh")?.id).toBe("old-mesh");
    expect(shownAsset(assets, "gaussian-splat", undefined)?.id).toBe("splat");
    expect(shownAsset(assets, undefined, undefined)).toBeNull();
  });
});

describe("StatusLine", () => {
  beforeEach(() => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useMission.setState({ project: project(), log: [], composer: null, streamOpen: false });
    useSites.setState({ activeSiteId: "site-1" });
    useUi.setState({ activityOpen: false });
    useViewer.getState().setTokenState("custom");
  });
  afterEach(() => {
    vi.restoreAllMocks();
    useSites.setState({ activeSiteId: null });
    useViewer.getState().setTokenState("unknown");
  });

  it("shows the fleet at the site and the agent's task, and opens the full log above", async () => {
    const user = userEvent.setup();
    render(wrap(<StatusLine />));
    expect(screen.getByTestId("status-fleet")).toHaveTextContent(
      "Fleet: 2 working · 1 need attention",
    );
    expect(screen.getByTestId("status-agent")).toHaveTextContent(
      "Agent: Sequencing passes against the forecast",
    );
    expect(screen.getByTestId("agent-stream-toggle")).toHaveTextContent("+2 tasks");
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
    expect(screen.queryByTestId("agent-stream")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Show agent activity/ }));
    const log = screen.getByRole("region", { name: "Agent activity" });
    expect(log).toHaveTextContent("Holding TR-12");
    expect(useUi.getState().activityOpen).toBe(true);
    // A press anywhere else closes it, like any popover.
    fireEvent.pointerDown(document.body);
    expect(useUi.getState().activityOpen).toBe(false);
  });

  it("summarises the catalog at globe scale, with no site engaged", async () => {
    useSites.setState({ activeSiteId: null });
    render(wrap(<StatusLine />));
    // The built-in catalog (API offline) holds the one demo site and its simulated fleet.
    await waitFor(
      () => expect(screen.getByTestId("status-fleet")).toHaveTextContent(/^1 site · 6 machines/),
      { timeout: 4000 },
    );
    expect(screen.getByTestId("status-fleet")).toHaveTextContent("simulated");
    expect(screen.getByTestId("notice-api-offline")).toHaveTextContent("Catalog API offline");
  });

  it("lights up the agent's line when a flow asks for it, instead of opening the log", async () => {
    render(wrap(<StatusLine />));
    act(() => {
      useMission.getState().appendLog("agent", "Tell me which machine should take it.");
      useMission.getState().setStreamOpen(true);
    });
    await waitFor(() => expect(useMission.getState().streamOpen).toBe(false));
    expect(useUi.getState().activityOpen).toBe(false);
    const line = screen.getByTestId("status-agent");
    expect(line).toHaveTextContent("Tell me which machine should take it.");
    expect(line).toHaveClass("is-flash");
  });
});
