import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { HOTKEYS, hotkeyKeys, hotkeySheet, type Hotkey } from "@/app/hotkeys";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import { CommandBox } from "@/features/command-palette/CommandBox";
import {
  AGENT_ROW_ID,
  buildCommandGroups,
  defaultActiveId,
  moveActive,
  prefersAgent,
  rowMatches,
  type CommandRow,
} from "@/features/command-palette/commandResults";
import { ShortcutSheet } from "@/features/command-palette/ShortcutSheet";
import { MOD_LABEL } from "@/lib/hotkeys";
import type { Project } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

function row(id: string, label: string, keywords?: string): CommandRow {
  return { id, label, ...(keywords ? { keywords } : {}), run: () => undefined };
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

/** A small project in the shape the demo provider gives, with nothing simulated about it. */
const project: Project = {
  id: "p",
  siteId: "s",
  name: "Test ranch",
  meta: "",
  simulated: false,
  machines: [],
  zones: [
    {
      id: "Z-21",
      name: "Z-21 North fence",
      short: "north",
      footprint: square,
      tone: "teal",
      treated: false,
      acres: 118,
      task: "Mow",
      machines: "TR-07",
      progressPct: 12,
      note: "",
      anchor: { longitude: 0, latitude: 0 },
    },
  ],
  plans: [],
  agent: { headline: "Agent idle", summary: "Nothing running", footer: "", actions: [] },
  fleetStats: [],
  fleetNote: "",
  workLog: [],
  feeds: [],
};

describe("command results", () => {
  it("matches every word of the query against the label and keywords", () => {
    expect(rowMatches("top down", row("a", "Top-down view", "camera overhead"))).toBe(true);
    expect(rowMatches("overhead", row("a", "Top-down view", "camera overhead"))).toBe(true);
    expect(rowMatches("top sideways", row("a", "Top-down view"))).toBe(false);
    expect(rowMatches("", row("a", "Anything"))).toBe(true);
  });

  it("lists every action with nothing typed, and every group then the agent with words", () => {
    const candidates = {
      places: [row("place-0", "Yosemite Valley")],
      sites: [row("site-1", "North orchard")],
      zones: [row("zone-Z-21", "Z-21 North fence")],
      layers: [row("quick-zones", "Hide zones")],
      actions: [row("action-settings", "Settings"), row("action-north", "Reset north")],
    };
    const empty = buildCommandGroups("  ", candidates, () => undefined);
    expect(empty.map((g) => g.id)).toEqual(["actions"]);
    expect(empty[0]?.rows).toHaveLength(2);

    const north = buildCommandGroups("north", candidates, () => undefined);
    // Places come from the geocoder already matched; the rest are filtered here. What the
    // console knows is listed above what the geocoder found, and the agent is last.
    expect(north.map((g) => g.id)).toEqual(["sites", "zones", "actions", "places", "agent"]);
    expect(north.at(-1)?.rows[0]).toMatchObject({
      id: AGENT_ROW_ID,
      label: "Ask the agent: “north”",
    });
  });

  it("hands the words to the agent exactly as typed", () => {
    const ask = vi.fn();
    const groups = buildCommandGroups(" mow Z-21 by Friday ", {}, ask);
    groups.at(-1)?.rows[0]?.run();
    expect(ask).toHaveBeenCalledWith("mow Z-21 by Friday");
  });

  it("asks the agent for instructions, questions and sentences, and finds names", () => {
    const none = { draftOpen: false, localMatches: 0 };
    expect(prefersAgent("where is TR-07", none)).toBe(true);
    expect(prefersAgent("hide zones", none)).toBe(true);
    expect(prefersAgent("Mow Z-21 weekly with one mower", none)).toBe(true);
    expect(prefersAgent("3D scan this field into a splat", none)).toBe(true);
    expect(prefersAgent("is the north fence done?", none)).toBe(true);
    expect(prefersAgent("settings", none)).toBe(false);
    expect(prefersAgent("Yosemite", none)).toBe(false);
    expect(prefersAgent("plans", none)).toBe(false);
    // With a draft open, plain words change it — unless they name something the box found.
    expect(prefersAgent("two drones", { draftOpen: true, localMatches: 0 })).toBe(true);
    expect(prefersAgent("Z-21", { draftOpen: true, localMatches: 1 })).toBe(false);
  });

  it("highlights a local match before a place, so Enter does not change under a fast typist", () => {
    const groups = buildCommandGroups(
      "north",
      {
        places: [row("place-0", "North Pole")],
        zones: [row("zone-Z-21", "Z-21 North fence")],
      },
      () => undefined,
    );
    expect(defaultActiveId("north", groups, { draftOpen: false })).toBe("zone-Z-21");
    // The highlighted best match is the top row, not one below a list of places.
    expect(groups[0]?.rows[0]?.id).toBe("zone-Z-21");
    expect(groups.map((g) => g.id)).toEqual(["zones", "places", "agent"]);
    const placesOnly = buildCommandGroups(
      "yosemite",
      { places: [row("place-0", "Yosemite")] },
      () => undefined,
    );
    expect(defaultActiveId("yosemite", placesOnly, { draftOpen: false })).toBe("place-0");
    const instruction = buildCommandGroups(
      "show zones",
      { layers: [row("quick-zones", "Show zones")] },
      () => undefined,
    );
    expect(defaultActiveId("show zones", instruction, { draftOpen: false })).toBe(AGENT_ROW_ID);
    // Nothing matches yet (places still on their way): Enter asks the agent, which can geocode.
    const nothing = buildCommandGroups("Lake Tahoe", {}, () => undefined);
    expect(defaultActiveId("Lake Tahoe", nothing, { draftOpen: false })).toBe(AGENT_ROW_ID);
  });

  it("moves the highlight with wrap-around", () => {
    const rows = [row("a", "A"), row("b", "B"), row("c", "C")];
    expect(moveActive(rows, "a", 1)).toBe("b");
    expect(moveActive(rows, "c", 1)).toBe("a");
    expect(moveActive(rows, "a", -1)).toBe("c");
    expect(moveActive(rows, null, -1)).toBe("c");
    expect(moveActive([], null, 1)).toBeNull();
  });
});

describe("the hotkey registry", () => {
  it("prints each key as its key caps", () => {
    expect(hotkeyKeys(HOTKEYS.command)).toEqual([MOD_LABEL, "K"]);
    expect(hotkeyKeys(HOTKEYS.shortcuts)).toEqual(["?"]);
    expect(hotkeyKeys(HOTKEYS.escape)).toEqual(["Esc"]);
    expect(hotkeyKeys(HOTKEYS.layers)).toEqual(["L"]);
    expect(hotkeyKeys(HOTKEYS.zoom)).toEqual(["+", "−"]);
  });

  it("binds no key twice, and lists developer keys only when they exist", () => {
    const bound = (Object.values(HOTKEYS) as Hotkey[]).filter((h) => !h.scene).map((h) => h.combo);
    expect(new Set(bound).size).toBe(bound.length);
    const labels = (dev: boolean) =>
      hotkeySheet(dev).flatMap((section) => section.hotkeys.map((h) => h.label));
    expect(labels(true)).toContain("Developer tools");
    expect(labels(false)).not.toContain("Developer tools");
    expect(hotkeySheet(false).map((s) => s.group)).toEqual([
      "General",
      "Views",
      "Tools",
      "Camera",
      "Explore mode",
    ]);
  });
});

describe("CommandBox", () => {
  beforeEach(() => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useUi.setState({ settingsOpen: false, shortcutsOpen: false, activityOpen: false });
    useMission.setState({ project, composer: null, log: [], selection: null, view: "map" });
    useSettings.getState().reset();
  });
  afterEach(() => vi.restoreAllMocks());

  it("is a combobox whose highlight moves with the arrows and runs with Enter", async () => {
    const user = userEvent.setup();
    render(wrap(<CommandBox />));
    const input = screen.getByRole("combobox", { name: "Search, run an action or ask the agent" });
    await user.click(input);
    // Empty, it is a menu of every action with its key.
    const list = screen.getByRole("listbox", { name: "Results" });
    expect(within(list).getByRole("option", { name: /Settings/ })).toBeInTheDocument();
    expect(input).toHaveAttribute("aria-expanded", "true");

    await user.type(input, "settings");
    const active = () => document.getElementById(input.getAttribute("aria-activedescendant") ?? "");
    expect(active()).toHaveTextContent("Settings");
    expect(active()).toHaveAttribute("aria-selected", "true");
    // Down moves to the next row (the agent, last), Up comes back.
    await user.keyboard("{ArrowDown}");
    expect(active()).toHaveTextContent("Ask the agent: “settings”");
    await user.keyboard("{ArrowUp}");
    expect(active()).toHaveTextContent("Settings");
    await user.keyboard("{Enter}");
    expect(useUi.getState().settingsOpen).toBe(true);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("finds the project's zones and selects one", async () => {
    const user = userEvent.setup();
    render(wrap(<CommandBox />));
    await user.type(screen.getByRole("combobox"), "z-21");
    expect(screen.getByRole("group", { name: "Zones" })).toHaveTextContent("Z-21 North fence");
    await user.keyboard("{Enter}");
    expect(useMission.getState().selection).toEqual({ kind: "zone", id: "Z-21" });
  });

  it("sends a sentence of work to the agent, and Escape closes without running", async () => {
    const user = userEvent.setup();
    render(wrap(<CommandBox />));
    const input = screen.getByRole("combobox");
    await user.type(input, "where is the north fence");
    const agent = screen.getByTestId(`command-row-${AGENT_ROW_ID}`);
    expect(agent).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(useMission.getState().log).toEqual([]);

    await user.click(input);
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(useMission.getState().log.map((e) => e.role)).toEqual(["you", "agent"]),
    );
    expect(useMission.getState().log[0]?.text).toBe("where is the north fence");
  });

  it("takes focus and the words when a card puts them in the box", async () => {
    render(wrap(<CommandBox />));
    window.dispatchEvent(new CustomEvent("twin:bar", { detail: "Mow the orchard" }));
    const input = screen.getByRole("combobox");
    await waitFor(() => expect(input).toHaveFocus());
    expect(input).toHaveValue("Mow the orchard");
  });
});

describe("places in the command box", () => {
  /** A scene whose geocoder finds one place named after the words, and a camera that flies. */
  function fakeScene() {
    const flights: { longitude: number; latitude: number }[] = [];
    const scene = {
      geocoder: {
        attribution: "Test geocoder",
        search: vi.fn((words: string) =>
          Promise.resolve([
            {
              label: `${words[0]?.toUpperCase() ?? ""}${words.slice(1)} (place)`,
              destination: {
                kind: "point",
                longitude: words.startsWith("yose") ? -119.5 : -110.5,
                latitude: 44,
              },
            },
          ]),
        ),
      },
      camera: {
        flyTo: (longitude: number, latitude: number) => flights.push({ longitude, latitude }),
        flyToRectangle: vi.fn(),
      },
      sites: { flyTo: vi.fn() },
      layers: { setVisible: vi.fn() },
      mission: { setLayer: vi.fn() },
    };
    const registry = new SceneRegistry();
    registry.set(scene as unknown as CesiumSceneManager);
    return { registry, flights, scene };
  }

  beforeEach(() => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useMission.setState({ project, composer: null, log: [], selection: null, view: "map" });
  });
  afterEach(() => vi.restoreAllMocks());

  it("never runs the last words' place for new words whose search is still pending", async () => {
    const { registry, flights } = fakeScene();
    const user = userEvent.setup();
    render(wrap(<SceneContext.Provider value={registry}>{<CommandBox />}</SceneContext.Provider>));
    const input = screen.getByRole("combobox");
    await user.type(input, "yosemite");
    await screen.findByText("Yosemite (place)");
    // New words: Yosemite's row goes at once, and the box says it is searching.
    await user.clear(input);
    await user.type(input, "yellowstone");
    expect(screen.queryByText("Yosemite (place)")).not.toBeInTheDocument();
    expect(screen.getByText("Searching places…")).toBeInTheDocument();
    // Enter before the new places arrive asks the agent with the words as typed, which finds
    // and flies to them -- never to the place the old words found.
    await user.keyboard("{Enter}");
    await waitFor(() => expect(flights).toHaveLength(1));
    expect(flights[0]?.longitude).toBe(-110.5);
    expect(useMission.getState().log[0]).toMatchObject({ role: "you", text: "yellowstone" });
  });
});

describe("ShortcutSheet", () => {
  it("prints every group of the registry", () => {
    useUi.setState({ shortcutsOpen: true });
    render(wrap(<ShortcutSheet />));
    const sheet = screen.getByTestId("shortcut-sheet");
    for (const group of ["General", "Views", "Tools", "Camera", "Explore mode"])
      expect(within(sheet).getByRole("region", { name: group })).toBeInTheDocument();
    expect(sheet).toHaveTextContent("Walk / explore mode");
    useUi.setState({ shortcutsOpen: false });
  });
});
