/**
 * The map's menu (features/map/MapContextMenu.tsx): a right-click or long press on the map
 * opens it at the point, since a plain click on empty ground does nothing. Its items, its
 * keyboard, what closes it, its place on the screen, its step in Escape's order
 * (features/shell/stepBack.ts), and the long press that opens it on touch
 * (cesium/longPress.ts).
 */

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { LONG_PRESS_MS, LongPress } from "@/cesium/longPress";
import { sceneRegistry } from "@/cesium/SceneContext";
import type { SceneEvents } from "@/cesium/types";
import { MapContextMenu } from "@/features/map/MapContextMenu";
import { placeMenu } from "@/features/map/placeMenu";
import { stepBack } from "@/features/shell/stepBack";
import { Emitter } from "@/lib/emitter";
import { useMission } from "@/state/mission";
import { useSceneSelect } from "@/state/sceneSelect";
import { useUi, type MapMenuAt } from "@/state/ui";

const AT: MapMenuAt = { x: 200, y: 150, longitude: -122.1385, latitude: 47.645, height: 41.5 };

function fakeScene() {
  const events = new Emitter<SceneEvents>();
  const scene = {
    events,
    viewer: { canvas: document.createElement("canvas") },
    selection: { inspectAt: vi.fn(), flyTowards: vi.fn(), clear: vi.fn() },
    measurement: { seed: vi.fn() },
    sceneSelect: { escape: () => false },
    mission: { setSelectedZone: vi.fn() },
  };
  return scene;
}

let scene: ReturnType<typeof fakeScene>;

beforeEach(() => {
  scene = fakeScene();
  act(() => sceneRegistry.set(scene as unknown as CesiumSceneManager));
  useUi.setState({
    mapMenu: null,
    measureMode: null,
    shortcutsOpen: false,
    moreOpen: false,
    activityOpen: false,
    writeTokenPrompt: false,
    inspectorOpen: false,
    activePanel: null,
  });
});

afterEach(() => {
  act(() => sceneRegistry.set(null));
});

function open(at: MapMenuAt = AT): void {
  act(() => useUi.getState().setMapMenu(at));
}

describe("MapContextMenu", () => {
  it("offers what to do at the point, as a menu with the keyboard on its first item", async () => {
    render(<MapContextMenu />);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    open();
    const menu = await screen.findByRole("menu", { name: "Map" });
    const items = screen.getAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "What's here",
      "Measure from here",
      "Plan here",
      "Fly here",
    ]);
    expect(menu).toContainElement(document.activeElement as HTMLElement);
    expect(document.activeElement).toBe(items[0]);
  });

  it("moves with the arrows, Home and End, and closes on Escape without a second step", async () => {
    const user = userEvent.setup();
    render(<MapContextMenu />);
    open();
    const items = await screen.findAllByRole("menuitem");
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(items[1]);
    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(document.activeElement).toBe(items[3]);
    await user.keyboard("{Home}");
    expect(document.activeElement).toBe(items[0]);
    await user.keyboard("{End}");
    expect(document.activeElement).toBe(items[3]);
    // The app's own Escape (stepBack) must not see the key the menu took.
    const seen = vi.fn();
    window.addEventListener("keydown", seen);
    await user.keyboard("{Escape}");
    window.removeEventListener("keydown", seen);
    expect(seen).not.toHaveBeenCalled();
    expect(useUi.getState().mapMenu).toBeNull();
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument());
  });

  it("asks the scene what is at the point, and closes", async () => {
    const user = userEvent.setup();
    render(<MapContextMenu />);
    open();
    await user.click(await screen.findByRole("menuitem", { name: "What's here" }));
    expect(scene.selection.inspectAt).toHaveBeenCalledWith(AT);
    expect(useUi.getState().mapMenu).toBeNull();
  });

  it("measures from the point: the first point is seeded, then the distance tool starts", async () => {
    const user = userEvent.setup();
    render(<MapContextMenu />);
    open();
    await user.click(await screen.findByRole("menuitem", { name: "Measure from here" }));
    expect(scene.measurement.seed).toHaveBeenCalledWith(AT);
    expect(useUi.getState().measureMode).toBe("distance");
    expect(useUi.getState().activePanel).toBe("measure");
  });

  it("flies towards the point", async () => {
    const user = userEvent.setup();
    render(<MapContextMenu />);
    open();
    await user.click(await screen.findByRole("menuitem", { name: "Fly here" }));
    expect(scene.selection.flyTowards).toHaveBeenCalledWith(AT);
  });

  it("closes when the camera moves or on a press anywhere else", async () => {
    render(<MapContextMenu />);
    open();
    await screen.findByRole("menu");
    act(() => scene.events.emit("motion", false));
    expect(useUi.getState().mapMenu).toEqual(AT);
    act(() => scene.events.emit("motion", true));
    expect(useUi.getState().mapMenu).toBeNull();

    open({ ...AT, x: 300 });
    const menu = await screen.findByRole("menu");
    act(() => {
      menu.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(useUi.getState().mapMenu).not.toBeNull();
    act(() => {
      document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(useUi.getState().mapMenu).toBeNull();
  });
});

describe("placeMenu", () => {
  const size = { width: 200, height: 160 };
  const screenSize = { width: 1000, height: 700 };

  it("opens below and right of the point", () => {
    expect(placeMenu({ x: 100, y: 100 }, size, screenSize)).toEqual({ left: 100, top: 100 });
  });

  it("turns to the other side of the point at the right and bottom edges", () => {
    expect(placeMenu({ x: 950, y: 650 }, size, screenSize)).toEqual({ left: 750, top: 490 });
  });

  it("stays on a screen too small to turn on", () => {
    expect(placeMenu({ x: 100, y: 100 }, size, { width: 210, height: 170 })).toEqual({
      left: 8,
      top: 8,
    });
  });
});

describe("stepBack: the map menu", () => {
  it("closes before anything else that floats, the shortcut sheet included", () => {
    useUi.setState({ mapMenu: AT, shortcutsOpen: true, inspectorOpen: true });
    useMission.setState({
      projectsOpen: false,
      feedsOpen: false,
      composer: null,
      selection: null,
      view: "map",
    });
    useSceneSelect.getState().setMode("pick");
    useSceneSelect.getState().clear();
    const stepScene = scene as unknown as Parameters<typeof stepBack>[0];
    expect(stepBack(stepScene)).toBe("map-menu");
    expect(useUi.getState().mapMenu).toBeNull();
    expect(useUi.getState().shortcutsOpen).toBe(true);
    expect(stepBack(stepScene)).toBe("shortcuts");
    expect(stepBack(stepScene)).toBe("inspector");
    expect(stepBack(stepScene)).toBeNull();
  });
});

describe("LongPress", () => {
  let element: HTMLElement;
  let press: ReturnType<typeof vi.fn<(x: number, y: number) => boolean>>;
  let longPress: LongPress;

  const pointer = (
    type: string,
    init: { id?: number; x?: number; y?: number; kind?: string } = {},
  ): void => {
    const event = new PointerEvent(type, {
      bubbles: true,
      pointerId: init.id ?? 1,
      pointerType: init.kind ?? "touch",
      clientX: init.x ?? 40,
      clientY: init.y ?? 60,
    });
    (type === "pointerdown" ? element : window).dispatchEvent(event);
  };

  beforeEach(() => {
    vi.useFakeTimers();
    element = document.body.appendChild(document.createElement("div"));
    press = vi.fn<(x: number, y: number) => boolean>(() => true);
    longPress = new LongPress(element, press);
  });

  afterEach(() => {
    longPress.destroy();
    element.remove();
    vi.useRealTimers();
  });

  it("opens after a finger rests, where it rests, and swallows the click that follows", () => {
    pointer("pointerdown");
    expect(longPress.touching).toBe(true);
    vi.advanceTimersByTime(LONG_PRESS_MS - 1);
    expect(press).not.toHaveBeenCalled();
    pointer("pointermove", { x: 44, y: 63 });
    vi.advanceTimersByTime(1);
    expect(press).toHaveBeenCalledWith(40, 60);
    pointer("pointerup");
    expect(longPress.touching).toBe(false);
    expect(longPress.takeClick()).toBe(true);
    expect(longPress.takeClick()).toBe(false);
  });

  it("is a pan, not a press, once the finger wanders", () => {
    pointer("pointerdown");
    pointer("pointermove", { x: 52, y: 60 });
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(press).not.toHaveBeenCalled();
    expect(longPress.takeClick()).toBe(false);
  });

  it("is a pinch, not a press, with a second finger", () => {
    pointer("pointerdown", { id: 1 });
    pointer("pointerdown", { id: 2, x: 140 });
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(press).not.toHaveBeenCalled();
  });

  it("ends when the finger lifts, and leaves the mouse to its right button", () => {
    pointer("pointerdown");
    pointer("pointerup");
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    pointer("pointerdown", { kind: "mouse" });
    expect(longPress.touching).toBe(false);
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(press).not.toHaveBeenCalled();
  });

  it("swallows no click when the menu could not open (a tool owns the pointer)", () => {
    press.mockReturnValue(false);
    pointer("pointerdown");
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(press).toHaveBeenCalled();
    pointer("pointerup");
    expect(longPress.takeClick()).toBe(false);
  });

  it("forgets a swallowed click at the next touch", () => {
    pointer("pointerdown");
    vi.advanceTimersByTime(LONG_PRESS_MS);
    pointer("pointerup");
    pointer("pointerdown");
    expect(longPress.takeClick()).toBe(false);
  });
});
