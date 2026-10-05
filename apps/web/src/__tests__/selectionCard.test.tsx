/**
 * One card for what is selected, one step back per Escape (features/mission/SelectionCard,
 * features/sites/ObjectCard, state/oneSelection.ts, features/shell/stepBack.ts), and the keys
 * a selected scan object answers to (cesium/sceneSelect/SceneSelectController.ts): `B` the
 * brush, `V` saved views, Tab only from the map or the card.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import { BoundingSphere, Cartesian3 } from "cesium";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { HOTKEYS, hotkeySheet } from "@/app/hotkeys";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type * as SplatInstances from "@/cesium/splatInstances";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import { registerPickSource } from "@/cesium/sceneSelect/pickSources";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { SelectionCard } from "@/features/mission/SelectionCard";
import { ObjectCard } from "@/features/sites/ObjectCard";
import { stepBack } from "@/features/shell/stepBack";
import { useHotkey } from "@/lib/hotkeys";
import { parseInstances } from "@/lib/instances";
import { TOUCH_MEDIA } from "@/lib/media";
import type { Machine, Project } from "@/missions/types";
import { useInstances } from "@/state/instances";
import { bindDockRules, rightDockBusy, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { bindOneSelection } from "@/state/oneSelection";
import { selectedId, selectedIds, useSceneSelect } from "@/state/sceneSelect";
import { useSelection } from "@/state/selection";
import { useUi } from "@/state/ui";

const ASSET = "yard";

const box = { min: [0, 0, 0], max: [1, 1, 1] };
/**
 * A conifer (2) whose branch is 3 and trunk 4, in a forest floor (1): a click's chain, leaf
 * first; a shrub (5) beside it. One tile of six splats: two of the branch, three of the trunk,
 * one of the shrub.
 */
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    { id: 1, bounds: box, splats: 900, tags: [{ label: "forest floor", score: 0.6 }] },
    { id: 2, parent: 1, bounds: box, splats: 400, tags: [{ label: "conifer", score: 0.6 }] },
    { id: 3, parent: 2, bounds: box, splats: 100, tags: [{ label: "branch", score: 0.6 }] },
    { id: 4, parent: 2, bounds: box, splats: 60, tags: [{ label: "trunk", score: 0.6 }] },
    { id: 5, bounds: box, splats: 30, tags: [{ label: "shrub", score: 0.6 }] },
  ],
  tiles: { "fnv1a32:6:0000abcd": [3, 2, 4, 3, 5, 1] },
})!;

// The scan as if its tileset were in the scene: its document, and a sphere to fly to.
vi.mock("@/cesium/splatInstances", async (original) => ({
  ...(await original<typeof SplatInstances>()),
  instancesDocOf: (assetId: string) => (assetId === "yard" ? DOC : undefined),
  instanceSphere: (assetId: string) =>
    assetId === "yard" ? new BoundingSphere(Cartesian3.fromDegrees(0, 0, 0), 2) : undefined,
}));

function viewer() {
  return {
    scene: {
      screenSpaceCameraController: { enableInputs: true },
      requestRender: () => undefined,
    },
    camera: {},
    canvas: document.body.appendChild(document.createElement("canvas")),
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
  machines: [machine("TR-04")],
  zones: [],
  plans: [],
  agent: { headline: "Agent idle", summary: "", footer: "", actions: [] },
  fleetStats: [],
  fleetNote: "",
  workLog: [],
  feeds: [],
};

/** The scan's objects in the objects store, as when its instances loaded. */
function stageScan(): void {
  useInstances.getState().setTable(ASSET, DOC);
}

/** A scene manager with just what the card and Escape touch. */
function fakeScene(controller: SceneSelectController) {
  return {
    sceneSelect: controller,
    mission: { setSelectedZone: vi.fn() },
    selection: { clear: vi.fn() },
  };
}

function wrap(children: ReactNode, registry?: SceneRegistry) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const tree = (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
  return registry ? <SceneContext.Provider value={registry}>{tree}</SceneContext.Provider> : tree;
}

/** Answers `query` as a phone or a tablet would, for the touch screen's card. */
function touchScreen(touch: boolean) {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: touch && query === TOUCH_MEDIA,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

let controller: SceneSelectController;
const offs: (() => void)[] = [];

beforeEach(() => {
  useSceneSelect.getState().clear();
  useSceneSelect.getState().setMode("pick");
  useInstances.setState({ assets: {}, gaps: {} });
  useMission.setState({ project, selection: null, view: "map", feedsOpen: false, composer: null });
  useUi.setState({
    activePanel: null,
    inspectorOpen: false,
    shortcutsOpen: false,
    moreOpen: false,
    activityOpen: false,
    writeTokenPrompt: false,
    measureMode: null,
  });
  useSelection.setState({ selection: null });
  useLayout.setState({ focus: null });
  controller = new SceneSelectController(viewer() as never, { ownKeys: false });
});

afterEach(() => {
  for (const off of offs.splice(0)) off();
  controller.destroy();
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("the hotkey registry's objects", () => {
  it("keeps B for the brush and moves saved views to V, with nothing else on V", () => {
    expect(HOTKEYS.brush.combo).toBe("b");
    expect(HOTKEYS.bookmarks.combo).toBe("v");
    const bound = Object.values(HOTKEYS)
      .map((hotkey) => hotkey.combo)
      .filter((combo) => combo !== "");
    expect(bound.filter((combo) => combo === "v")).toHaveLength(1);
    expect(bound.filter((combo) => combo === "b")).toHaveLength(1);
    const objects = hotkeySheet(false).find((section) => section.group === "Objects");
    expect(objects?.hotkeys.map((hotkey) => hotkey.label)).toContain(HOTKEYS.brush.label);
    expect(objects?.hotkeys.map((hotkey) => hotkey.label)).toContain(HOTKEYS.cycleObject.label);
  });
});

describe("useHotkey", () => {
  it("leaves a key something else already acted on alone", () => {
    const handler = vi.fn();
    renderHook(() => useHotkey("escape", handler));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }));
    expect(handler).toHaveBeenCalledTimes(1);
    // A dialog or popover closing on Escape marks the key as handled first (Radix does).
    const handled = new KeyboardEvent("keydown", { key: "Escape", cancelable: true });
    handled.preventDefault();
    window.dispatchEvent(handled);
    expect(handler).toHaveBeenCalledTimes(1);
  });
});

describe("the object card", () => {
  it("names the object and its category, cycles the candidates, and offers its actions", async () => {
    const user = userEvent.setup();
    touchScreen(false);
    stageScan();
    act(() => useSceneSelect.getState().select(ASSET, [3, 2, 1], 3, 1, null));
    render(wrap(<ObjectCard controller={controller} />));
    const card = screen.getByTestId("selection-card");
    expect(card).toHaveAttribute("data-kind", "object");
    expect(within(card).getByTestId("object-label")).toHaveTextContent("Conifer");
    expect(within(card).getByTestId("object-candidates")).toHaveTextContent("2 of 3");
    await user.click(within(card).getByRole("button", { name: "Next candidate" }));
    expect(useSceneSelect.getState().index).toBe(2);
    await user.click(within(card).getByRole("button", { name: "Previous candidate" }));
    expect(useSceneSelect.getState().index).toBe(1);
    // The keyboard's way to cycle is said beside the arrows.
    expect(card).toHaveTextContent("Tab");
    for (const name of ["Hide", "Show only", "Fly to"])
      expect(within(card).getByRole("button", { name })).toBeInTheDocument();
    const hide = vi.spyOn(controller, "hide");
    await user.click(within(card).getByRole("button", { name: "Hide" }));
    expect(hide).toHaveBeenCalled();
  });

  it("names a combination the brush chose, acts on all of it, and keeps it as an object", async () => {
    const user = userEvent.setup();
    touchScreen(false);
    stageScan();
    act(() => useSceneSelect.getState().selectSet(ASSET, { ids: [3, 4], iou: 0.87 }, [2, 1], null));
    const fly = vi.fn();
    const flying = new SceneSelectController(viewer() as never, { ownKeys: false, fly });
    try {
      render(wrap(<ObjectCard controller={flying} />));
      const card = screen.getByTestId("selection-card");
      const label = within(card).getByTestId("object-label");
      expect(label).toHaveTextContent("Branch + Trunk (of Conifer)");
      expect(label).toHaveAttribute("data-ids", "3 4");
      expect(card).toHaveTextContent("87% overlap with the painted area");
      // The combination is the finest level; the conifer and the floor are above it.
      expect(within(card).getByTestId("object-candidates")).toHaveTextContent("3 of 3");
      // Both members lit, Hide hides both.
      expect([...(useInstances.getState().assets[ASSET]?.highlighted ?? [])].sort()).toEqual([
        3, 4,
      ]);
      await user.click(within(card).getByRole("button", { name: "Show only" }));
      expect([...(useInstances.getState().assets[ASSET]?.hidden ?? [])]).toEqual([5]);
      await user.click(within(card).getByRole("button", { name: "Fly to" }));
      expect(fly).toHaveBeenCalledTimes(1);
      await user.click(within(card).getByRole("button", { name: /Save as object/ }));
      // Kept as an object of its own, by its members' splats, and selected.
      const kept = useSceneSelect.getState().customOf(ASSET);
      expect(kept.map((set) => [set.name, set.tiles, set.splats])).toEqual([
        ["Branch + Trunk (of Conifer)", { "fnv1a32:6:0000abcd": [0, 5] }, 5],
      ]);
      expect(selectedIds(useSceneSelect.getState())).toEqual([6]);
      await waitFor(() =>
        expect(within(card).getByTestId("object-label")).toHaveTextContent(
          "Branch + Trunk (of Conifer)",
        ),
      );
      expect(within(card).queryByRole("button", { name: /Save as object/ })).toBeNull();
      act(() => useSceneSelect.getState().selectSet(ASSET, { ids: [3, 5], iou: 0.6 }, [], null));
      await user.click(within(card).getByRole("button", { name: "Hide" }));
      expect([...(useInstances.getState().assets[ASSET]?.hidden ?? [])].sort()).toEqual([3, 5]);
      expect(selectedIds(useSceneSelect.getState())).toEqual([]);
    } finally {
      flying.destroy();
      for (const set of useSceneSelect.getState().customOf(ASSET))
        act(() => useSceneSelect.getState().removeCustom(ASSET, set.key));
    }
  });

  it("paints with Shift, Alt and the wheel at a desk, and with buttons on a touch screen", async () => {
    const user = userEvent.setup();
    touchScreen(false);
    act(() => useSceneSelect.getState().setMode("paint"));
    const { unmount } = render(wrap(<ObjectCard controller={controller} />));
    expect(screen.getByTestId("object-paint-hint")).toHaveTextContent(/Shift adds, Alt removes/);
    expect(screen.queryByTestId("object-touch-brush")).not.toBeInTheDocument();
    unmount();

    touchScreen(true);
    render(wrap(<ObjectCard controller={controller} />));
    const hint = screen.getByTestId("object-paint-hint");
    expect(hint).not.toHaveTextContent(/Shift|Alt|wheel/);
    const modes = screen.getByRole("radiogroup", { name: "What a stroke does" });
    await user.click(within(modes).getByRole("radio", { name: "Add" }));
    expect(useSceneSelect.getState().strokeMode).toBe("add");
    const before = useSceneSelect.getState().brush;
    await user.click(screen.getByRole("button", { name: "Larger brush" }));
    expect(useSceneSelect.getState().brush).toBeGreaterThan(before);
    // No key caps for a keyboard that is not there.
    expect(screen.getByTestId("selection-card").querySelector("kbd")).toBeNull();
    // Putting the brush away starts the next session from "New".
    await user.click(screen.getByRole("button", { name: "Stop painting" }));
    expect(useSceneSelect.getState()).toMatchObject({ mode: "pick", strokeMode: "replace" });
  });
});

describe("one selection at a time", () => {
  it("shows one card: an object replaces a machine, and a machine an object", async () => {
    touchScreen(false);
    stageScan();
    const registry = new SceneRegistry();
    registry.set(fakeScene(controller) as unknown as CesiumSceneManager);
    const scene = fakeScene(controller);
    offs.push(
      bindOneSelection({
        clearZone: () => {
          scene.mission.setSelectedZone(null);
        },
        clearObject: () => {
          controller.setPainting(false);
          controller.clear();
        },
        clearInspector: () => {
          scene.selection.clear();
        },
      }),
    );
    act(() => useMission.getState().select({ kind: "machine", id: "TR-04" }));
    render(wrap(<SelectionCard />, registry));
    expect(screen.getByTestId("selection-card")).toHaveTextContent("TR-04 Kestrel");

    act(() => useSceneSelect.getState().select(ASSET, [3, 2], 2, 0, null));
    expect(useMission.getState().selection).toBeNull();
    expect(scene.mission.setSelectedZone).toHaveBeenCalledWith(null);
    // The machine's card leaves before the object's comes in: never two at once.
    await waitFor(() => {
      const cards = screen.getAllByTestId("selection-card");
      expect(cards).toHaveLength(1);
      expect(cards[0]).toHaveAttribute("data-kind", "object");
    });

    act(() => useMission.getState().select({ kind: "machine", id: "TR-04" }));
    expect(selectedId(useSceneSelect.getState())).toBeNull();
    await waitFor(() => {
      const cards = screen.getAllByTestId("selection-card");
      expect(cards).toHaveLength(1);
      expect(cards[0]).toHaveTextContent("TR-04 Kestrel");
    });
  });

  it("closes the inspector for an object, and the brush counts as one", () => {
    const clearInspector = vi.fn();
    offs.push(
      bindOneSelection({
        clearZone: () => undefined,
        clearObject: () => controller.clear(),
        clearInspector,
      }),
    );
    useUi.setState({ inspectorOpen: true });
    useSelection.setState({
      selection: {
        kind: "ground",
        title: "Location",
        longitude: 0,
        latitude: 0,
        height: 0,
        terrainHeight: null,
        at: 1,
      },
    });
    useMission.getState().select({ kind: "zone", id: "Z-1" });
    useSceneSelect.getState().setMode("paint");
    expect(useUi.getState().inspectorOpen).toBe(false);
    expect(useSelection.getState().selection).toBeNull();
    expect(useMission.getState().selection).toBeNull();
    expect(clearInspector).toHaveBeenCalled();
  });

  it("brings the card to the front of a phone's one sheet", () => {
    offs.push(bindDockRules());
    useUi.getState().openLayers("browse");
    expect(useLayout.getState().focus).toBe("left");
    useSceneSelect.getState().select(ASSET, [3], 1, 0, null);
    expect(rightDockBusy()).toBe(true);
    expect(useLayout.getState().focus).toBe("right");
  });
});

describe("Escape, once", () => {
  it("puts the brush away, then clears the object, then closes the panel", () => {
    const scene = fakeScene(controller);
    useUi.getState().openLayers("browse");
    useSceneSelect.getState().select(ASSET, [3, 2], 2, 0, null);
    useSceneSelect.getState().setMode("paint");
    expect(stepBack(scene)).toBe("brush");
    expect(useSceneSelect.getState().mode).toBe("pick");
    expect(selectedId(useSceneSelect.getState())).toBe(3);
    expect(useUi.getState().activePanel).toBe("layers");
    expect(stepBack(scene)).toBe("object");
    expect(selectedId(useSceneSelect.getState())).toBeNull();
    expect(useUi.getState().activePanel).toBe("layers");
    expect(stepBack(scene)).toBe("panel");
    expect(useUi.getState().activePanel).toBeNull();
    expect(stepBack(scene)).toBeNull();
  });

  it("closes what floats over the HUD before the selection", () => {
    const scene = fakeScene(controller);
    useSceneSelect.getState().select(ASSET, [3], 1, 0, null);
    useUi.setState({ activityOpen: true });
    expect(stepBack(scene)).toBe("activity");
    expect(selectedId(useSceneSelect.getState())).toBe(3);
    expect(stepBack(scene)).toBe("object");
  });
});

describe("the scene's keys", () => {
  function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
    const event = new KeyboardEvent("keydown", { key, cancelable: true, bubbles: true, ...init });
    (document.activeElement ?? document.body).dispatchEvent(event);
    return event;
  }

  it("cycles on Tab only from the map or the card; on the body Tab moves focus", () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) throw new Error("no canvas");
    useSceneSelect.getState().select(ASSET, [3, 2, 1], 3, 0, null);
    (document.activeElement as HTMLElement | null)?.blur();
    const fromBody = press("Tab");
    expect(fromBody.defaultPrevented).toBe(false);
    expect(useSceneSelect.getState().index).toBe(0);
    // `[` and `]` cycle from anywhere but a field.
    expect(press("]").defaultPrevented).toBe(true);
    expect(useSceneSelect.getState().index).toBe(1);
    // The map has the keyboard after a hit: Tab cycles there.
    expect(canvas.tabIndex).toBe(-1);
    canvas.focus();
    expect(press("Tab").defaultPrevented).toBe(true);
    expect(useSceneSelect.getState().index).toBe(2);
    expect(press("Tab", { shiftKey: true }).defaultPrevented).toBe(true);
    expect(useSceneSelect.getState().index).toBe(1);
    // A button inside the card keeps Tab's meaning.
    const button = document.body.appendChild(document.createElement("button"));
    button.focus();
    expect(press("Tab").defaultPrevented).toBe(false);
    expect(useSceneSelect.getState().index).toBe(1);
  });

  it("cycles on Tab while the card itself has focus", () => {
    touchScreen(false);
    stageScan();
    act(() => useSceneSelect.getState().select(ASSET, [3, 2, 1], 3, 0, null));
    render(wrap(<ObjectCard controller={controller} />));
    act(() => screen.getByTestId("selection-card").focus());
    expect(press("Tab").defaultPrevented).toBe(true);
    expect(useSceneSelect.getState().index).toBe(1);
  });

  it("leaves B and Escape to the app's keys in the app, and answers them standalone", () => {
    useSceneSelect.getState().select(ASSET, [3], 1, 0, null);
    expect(press("Escape").defaultPrevented).toBe(false);
    expect(selectedId(useSceneSelect.getState())).toBe(3);

    const own = new SceneSelectController(viewer() as never);
    stageScan();
    const off = registerPickSource(
      ASSET,
      { renderer: "test", tiles: () => [], toWorld: () => undefined },
      0,
    );
    try {
      expect(press("b").defaultPrevented).toBe(true);
      expect(useSceneSelect.getState().mode).toBe("paint");
      // Escape steps back once: the brush away, the selection kept.
      expect(press("Escape").defaultPrevented).toBe(true);
      expect(useSceneSelect.getState().mode).toBe("pick");
      expect(selectedId(useSceneSelect.getState())).toBe(3);
      expect(press("Escape").defaultPrevented).toBe(true);
      expect(selectedId(useSceneSelect.getState())).toBeNull();
    } finally {
      off();
      own.destroy();
    }
  });

  it("flies through the app's camera controller, not CesiumJS's own flight", () => {
    const fly = vi.fn();
    const own = vi.fn();
    const camera = { flyToBoundingSphere: own, heading: 0 };
    const routed = new SceneSelectController({ ...viewer(), camera } as never, { fly });
    useSceneSelect.getState().select(ASSET, [2], 1, 0, null);
    routed.flyTo();
    expect(fly).toHaveBeenCalledTimes(1);
    expect(fly.mock.calls[0]?.[0]).toBeInstanceOf(BoundingSphere);
    expect(own).not.toHaveBeenCalled();
    routed.destroy();
    // Standalone (the harness), CesiumJS flies.
    const standalone = new SceneSelectController({ ...viewer(), camera } as never);
    standalone.flyTo();
    expect(own).toHaveBeenCalledTimes(1);
    standalone.destroy();
  });
});
