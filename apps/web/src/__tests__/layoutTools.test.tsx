import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { useAppActions } from "@/features/command-palette/useAppActions";
import { LayersPanel } from "@/features/layers/LayersPanel";
import { ProjectCard } from "@/features/mission/ProjectCard";
import { SelectionCard } from "@/features/mission/SelectionCard";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { PhoneTabBar } from "@/features/shell/PhoneTabBar";
import { ToolRail } from "@/features/shell/ToolRail";
import { LOCAL_VIEWS_KEY, localViewsFor, unsitedViews } from "@/features/bookmarks/savedViews";
import type { Machine, Project, Zone } from "@/missions/types";
import { bindDockRules, rightDockBusy, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { initialCamera, useViewer } from "@/state/viewer";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
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

let unbind: () => void = () => undefined;

beforeEach(() => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  useSettings.getState().reset();
  useUi.setState({
    activePanel: null,
    layersMode: "browse",
    addTab: "upload",
    settingsOpen: false,
    moreOpen: false,
    compareActive: false,
    compareLeft: "",
    compareRight: "",
  });
  useMission.setState({
    project: null,
    view: "map",
    selection: null,
    projectsOpen: false,
    layers: { zones: true, tracks: true, vegetation: false },
  });
  useViewer.setState({ camera: initialCamera });
  useLayout.setState({ focus: null });
  window.localStorage.clear();
});

afterEach(() => {
  unbind();
  unbind = () => undefined;
  vi.restoreAllMocks();
});

describe("the tool rail", () => {
  it("has four labelled tools, and nothing that moved elsewhere", async () => {
    const user = userEvent.setup();
    render(wrap(<ToolRail />));
    const rail = screen.getByRole("toolbar", { name: "Tools" });
    const names = within(rail)
      .getAllByRole("button")
      .map((b) => b.textContent);
    expect(names).toEqual(["Layers", "Measure", "Add", "Settings"]);
    await user.click(within(rail).getByRole("button", { name: "Add" }));
    expect(useUi.getState().activePanel).toBe("add");
    expect(within(rail).getByRole("button", { name: "Add" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await user.click(within(rail).getByRole("button", { name: "Settings" }));
    expect(useUi.getState().settingsOpen).toBe(true);
  });

  it("keeps every entry that left the rail one command away", () => {
    let labels: string[] = [];
    function Probe() {
      labels = useAppActions().map((action) => action.label);
      return null;
    }
    render(wrap(<Probe />));
    for (const label of [
      "Switch site",
      "Saved views",
      "Upload a capture",
      "Link a hosted source",
      "Add a site",
      "Compare layers",
      "Scan gallery",
      "Data console",
      "Developer tools",
    ])
      expect(labels).toContain(label);
  });
});

describe("Layers", () => {
  it("leads with the favourites and holds Compare as a mode", async () => {
    const user = userEvent.setup();
    useMission.setState({ project });
    useUi.getState().openLayers("browse");
    render(wrap(<LayersPanel />));
    const favourites = screen.getByTestId("layer-favourites");
    expect(
      within(favourites)
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual(["Imagery", "Vegetation", "Zones", "Tracks"]);
    await user.click(screen.getByTestId("pill-zones"));
    expect(useMission.getState().layers.zones).toBe(false);

    const modes = screen.getByRole("radiogroup", { name: "Layers mode" });
    await user.click(within(modes).getByRole("radio", { name: /Compare/ }));
    expect(useUi.getState().layersMode).toBe("compare");
    expect(screen.queryByTestId("layers-filter")).not.toBeInTheDocument();
    // `c` (or the Compare layers action) opens straight into the mode, and again closes it.
    useUi.getState().setPanel(null);
    useUi.getState().openLayers("compare");
    expect(useUi.getState()).toMatchObject({ activePanel: "layers", layersMode: "compare" });
    useUi.getState().openLayers("compare");
    expect(useUi.getState().activePanel).toBeNull();
  });
});

describe("the site switcher", () => {
  it("lists sites by their one name, keeps saved views, and leads to the other pages", async () => {
    const user = userEvent.setup();
    useMission.setState({ project: { ...project, name: "Blackrock Mesa" }, projectsOpen: true });
    render(wrap(<ProjectCard />));
    const switcher = screen.getByRole("dialog", { name: "Switch site" });
    // The built-in catalog's demo site, under its project's name rather than the record's.
    expect(await within(switcher).findByText("Blackrock Mesa")).toBeInTheDocument();
    expect(within(switcher).queryByText("Cesium Gaussian splat demo")).not.toBeInTheDocument();

    // No site is engaged, so the view is kept in this browser; it can be deleted again.
    await user.type(within(switcher).getByRole("textbox", { name: "View name" }), "North gate");
    await user.click(within(switcher).getByRole("button", { name: /Save current view/ }));
    expect(await within(switcher).findByTestId("saved-view-North gate")).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem(LOCAL_VIEWS_KEY) ?? "[]")).toHaveLength(1);
    await user.click(within(switcher).getByRole("button", { name: "Delete North gate" }));
    expect(within(switcher).queryByTestId("saved-view-North gate")).not.toBeInTheDocument();

    expect(within(switcher).getByRole("link", { name: /Scan gallery/ })).toHaveAttribute(
      "href",
      "/view.html",
    );
    expect(within(switcher).getByRole("link", { name: /Data console/ })).toHaveAttribute(
      "href",
      "/admin.html",
    );
    await user.click(within(switcher).getByRole("button", { name: /Add a site/ }));
    expect(useUi.getState()).toMatchObject({ activePanel: "add", addTab: "link" });
    expect(useMission.getState().projectsOpen).toBe(false);
  });
});

describe("views saved with no site", () => {
  const unsited = {
    id: "local-1",
    name: "Before any site",
    longitude: 1,
    latitude: 2,
    height: 300,
    heading: 0,
    pitch: -45,
    roll: 0,
    isDefault: false,
    siteId: "",
    createdAt: "2026-10-01T00:00:00.000Z",
  };

  it("are kept, and listed beside a site's own once a site is current", () => {
    const elsewhere = { ...unsited, id: "local-2", name: "At the orchard", siteId: "orchard" };
    expect(unsitedViews([unsited, elsewhere], "yard")).toEqual([unsited]);
    // With no current site they are the list itself, not a second one.
    expect(unsitedViews([unsited, elsewhere], null)).toEqual([]);
    expect(localViewsFor([unsited, elsewhere], null)).toEqual([unsited]);
  });

  it("show under Other saved views at a site, and can still be deleted there", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem(LOCAL_VIEWS_KEY, JSON.stringify([unsited]));
    useSites.setState({ activeSiteId: "yard" });
    useMission.setState({ project, projectsOpen: true });
    render(wrap(<ProjectCard />));
    const switcher = screen.getByRole("dialog", { name: "Switch site" });
    const other = within(switcher).getByRole("list", { name: "Other saved views" });
    expect(within(other).getByTestId("saved-view-Before any site")).toBeInTheDocument();
    expect(within(switcher).getByText(/None for this site yet/)).toBeInTheDocument();
    await user.click(within(other).getByRole("button", { name: "Delete Before any site" }));
    expect(within(switcher).queryByText("Other saved views")).not.toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem(LOCAL_VIEWS_KEY) ?? "[]")).toEqual([]);
    useSites.setState({ activeSiteId: null });
  });
});

describe("closing the site switcher with Escape", () => {
  it("closes from the view-name field `v` focuses, and gives the keyboard back to the badge", async () => {
    const user = userEvent.setup();
    useMission.setState({ project, projectsOpen: true });
    useUi.setState({ switcherFocus: "views" });
    render(wrap(<ProjectCard />));
    const field = screen.getByRole("textbox", { name: "View name" });
    await waitFor(() => expect(field).toHaveFocus());
    await user.keyboard("{Escape}");
    expect(useMission.getState().projectsOpen).toBe(false);
    expect(screen.getByRole("button", { name: /switch site/ })).toHaveFocus();
  });

  it("closes the switcher, and nothing else the app's Escape would close first", async () => {
    const user = userEvent.setup();
    useUi.setState({ switcherFocus: "sites" });
    useMission.setState({ project, projectsOpen: true });
    // The app's own Escape (GlobalHotkeys) listens on the window: measuring, say, it would stop
    // that and leave the switcher open.
    const appEscape = vi.fn();
    window.addEventListener("keydown", appEscape);
    render(wrap(<ProjectCard />));
    const switcher = screen.getByRole("dialog", { name: "Switch site" });
    within(switcher)
      .getByRole("button", { name: /Whole Earth/ })
      .focus();
    await user.keyboard("{Escape}");
    window.removeEventListener("keydown", appEscape);
    expect(useMission.getState().projectsOpen).toBe(false);
    expect(appEscape).not.toHaveBeenCalled();
  });
});

describe("Plan and Fleet beside the map", () => {
  it("keeps the drawer open when a machine is picked, and the card beside it", () => {
    unbind = bindDockRules();
    useMission.setState({ project });
    useMission.getState().setView("fleet");
    useMission.getState().select({ kind: "machine", id: "TR-04" });
    expect(useMission.getState().view).toBe("fleet");
    expect(rightDockBusy()).toBe(true);
    render(wrap(<SelectionCard />));
    expect(screen.getByTestId("selection-card")).toHaveTextContent("TR-04 Kestrel");
    // A tool panel and the drawer still replace each other.
    useUi.getState().openLayers("browse");
    expect(useMission.getState().view).toBe("map");
    useMission.getState().setView("plan");
    expect(useUi.getState().activePanel).toBeNull();
  });
});

describe("the phone's tab bar", () => {
  it("switches views and opens the four tools under More", async () => {
    const user = userEvent.setup();
    render(wrap(<PhoneTabBar />));
    await user.click(screen.getByTestId("phone-tab-fleet"));
    expect(useMission.getState().view).toBe("fleet");
    expect(useLayout.getState().focus).toBe("left");
    expect(screen.getByTestId("phone-tab-fleet")).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByTestId("phone-tab-more"));
    const more = screen.getByRole("group", { name: "More tools" });
    expect(
      within(more)
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual(["Layers", "Measure", "Add", "Settings"]);
    await user.click(within(more).getByRole("button", { name: "Measure" }));
    expect(useUi.getState()).toMatchObject({ activePanel: "measure", moreOpen: false });
  });
});

describe("Settings › Advanced", () => {
  it("holds the developer console that left the rail", async () => {
    const user = userEvent.setup();
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));
    const advanced = screen.getByTestId("settings-advanced");
    await user.click(within(advanced).getByText("Advanced"));
    const tools = within(advanced).getByRole("switch", { name: "Developer tools" });
    expect(tools).toHaveAttribute("aria-checked", "false");
    await user.click(tools);
    await waitFor(() => expect(useSettings.getState().devToolsOpen).toBe(true));
  });
});
