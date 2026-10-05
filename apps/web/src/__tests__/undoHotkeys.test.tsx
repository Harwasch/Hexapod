/**
 * Ctrl+Z and Ctrl+Shift+Z / Ctrl+Y (features/shell/undoHotkeys.ts, app/hotkeys.ts): bound from
 * the registry and printed on the shortcut sheet, left to the browser in a text field, said in
 * a toast with the opposite step beside it; and the selection card's Hide and Show only taken
 * back and done again from the keyboard.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BoundingSphere, Cartesian3 } from "cesium";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { HOTKEYS, hotkeyChoices, hotkeySheet, type Hotkey } from "@/app/hotkeys";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import type * as SplatInstances from "@/cesium/splatInstances";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { ShortcutSheet } from "@/features/command-palette/ShortcutSheet";
import { MeasurePanel } from "@/features/measure/MeasurePanel";
import { Toasts } from "@/features/notices/Toasts";
import { useUndoHotkeys } from "@/features/shell/undoHotkeys";
import { ObjectCard } from "@/features/sites/ObjectCard";
import { isTextEntry, MOD_LABEL } from "@/lib/hotkeys";
import { parseInstances } from "@/lib/instances";
import { clearHistory, recordAction, useHistory } from "@/state/history";
import { useInstances } from "@/state/instances";
import { useMeasurements, type Measurement } from "@/state/measurements";
import { objectSelected, useSceneSelect } from "@/state/sceneSelect";
import { useToasts } from "@/state/toasts";
import { useUi } from "@/state/ui";

const ASSET = "yard";
const box = { min: [0, 0, 0], max: [1, 1, 1] };
/** A forest floor (1) under a conifer (2) and its branch (3); two pumpkins (4, 5); a bench (6). */
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    { id: 1, bounds: box, splats: 900, tags: [{ label: "forest floor", score: 0.6 }] },
    { id: 2, parent: 1, bounds: box, splats: 400, tags: [{ label: "conifer", score: 0.6 }] },
    { id: 3, parent: 2, bounds: box, splats: 100 },
    { id: 4, bounds: box, splats: 300, tags: [{ label: "pumpkin", score: 0.7 }] },
    { id: 5, bounds: box, splats: 200, tags: [{ label: "pumpkin", score: 0.7 }] },
    { id: 6, bounds: box, splats: 150, tags: [{ label: "bench", score: 0.7 }] },
  ],
  tiles: {},
});

// The scan as if its tileset were in the scene: its document, and a sphere to fly to.
vi.mock("@/cesium/splatInstances", async (original) => ({
  ...(await original<typeof SplatInstances>()),
  instancesDocOf: (assetId: string) => (assetId === "yard" ? DOC : undefined),
  instanceSphere: (assetId: string) =>
    assetId === "yard" ? new BoundingSphere(Cartesian3.fromDegrees(0, 0, 0), 2) : undefined,
}));

function UndoKeys({ children }: { children?: ReactNode }) {
  useUndoHotkeys();
  return (
    <>
      {children}
      <Toasts />
    </>
  );
}

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

/** A number the steps set, as a store would hold it. */
let value = 0;
function step(to: number): void {
  const from = value;
  recordAction(
    `Set ${String(to)}`,
    () => (value = to),
    () => (value = from),
  );
}

const said = (): string[] => useToasts.getState().items.map((t) => t.title);

beforeEach(() => {
  clearHistory();
  value = 0;
  useToasts.setState({ items: [] });
});

afterEach(() => {
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("the keys", () => {
  it("undo with Ctrl+Z or ⌘Z and redo with Ctrl+Shift+Z or Ctrl+Y, and say so", async () => {
    const user = userEvent.setup();
    render(wrap(<UndoKeys />));
    await user.keyboard("{Control>}z{/Control}");
    expect(said()).toEqual(["Nothing to undo"]);
    await user.keyboard("{Control>}{Shift>}z{/Shift}{/Control}");
    expect(said()).toEqual(["Nothing to redo"]);

    act(() => {
      step(1);
      step(2);
    });
    await user.keyboard("{Control>}z{/Control}");
    expect(value).toBe(1);
    // One line at a time: the newest replaces the last.
    expect(said()).toEqual(["Undid: Set 2"]);
    await user.keyboard("{Meta>}z{/Meta}");
    expect(value).toBe(0);
    expect(said()).toEqual(["Undid: Set 1"]);
    await user.keyboard("{Control>}{Shift>}Z{/Shift}{/Control}");
    expect(value).toBe(1);
    expect(said()).toEqual(["Redid: Set 1"]);
    await user.keyboard("{Control>}y{/Control}");
    expect(value).toBe(2);
    expect(said()).toEqual(["Redid: Set 2"]);

    // The line offers the step back: Undo after a redo, Redo after an undo.
    const toast = () => screen.getByRole("status");
    await user.click(within(toast()).getByRole("button", { name: "Undo" }));
    expect(value).toBe(1);
    expect(said()).toEqual(["Undid: Set 2"]);
    await user.click(within(toast()).getByRole("button", { name: "Redo" }));
    expect(value).toBe(2);
    expect(said()).toEqual(["Redid: Set 2"]);
  });

  it("leave a text field's own undo alone, and work from a button, a switch or a select", () => {
    render(
      wrap(
        <UndoKeys>
          <input aria-label="Search objects" type="search" />
          <textarea aria-label="Goal" />
          <div aria-label="Note" contentEditable suppressContentEditableWarning>
            <span>words</span>
          </div>
          <input aria-label="Show layer" type="checkbox" />
          <select aria-label="Left">
            <option>A</option>
          </select>
          <button type="button">Hide</button>
        </UndoKeys>,
      ),
    );
    act(() => {
      for (let k = 1; k <= 6; k += 1) step(k);
    });
    /** Ctrl+Z on `target`; whether the app took it (`defaultPrevented`). */
    const press = (target: HTMLElement): boolean => {
      target.focus();
      const event = new KeyboardEvent("keydown", {
        key: "z",
        ctrlKey: true,
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        target.dispatchEvent(event);
      });
      return event.defaultPrevented;
    };
    expect(press(screen.getByRole("searchbox", { name: "Search objects" }))).toBe(false);
    expect(press(screen.getByRole("textbox", { name: "Goal" }))).toBe(false);
    const words = screen.getByText("words");
    expect(isTextEntry(words)).toBe(true);
    expect(press(words)).toBe(false);
    expect(value).toBe(6);
    expect(useHistory.getState().past).toHaveLength(6);

    expect(press(screen.getByRole("checkbox", { name: "Show layer" }))).toBe(true);
    expect(press(screen.getByRole("combobox", { name: "Left" }))).toBe(true);
    expect(press(screen.getByRole("button", { name: "Hide" }))).toBe(true);
    expect(value).toBe(3);
  });

  it("are in the registry, bound once each, and on the shortcut sheet", () => {
    expect(hotkeyChoices(HOTKEYS.undo)).toEqual([[MOD_LABEL, "Z"]]);
    expect(hotkeyChoices(HOTKEYS.redo)).toEqual([
      [MOD_LABEL, "Shift", "Z"],
      [MOD_LABEL, "Y"],
    ]);
    const bound = (Object.values(HOTKEYS) as Hotkey[])
      .filter((h) => !h.scene)
      .flatMap((h) => [h.combo, ...(h.also ?? [])]);
    expect(new Set(bound).size).toBe(bound.length);
    const general = hotkeySheet(false).find((s) => s.group === "General");
    expect(general?.hotkeys.map((h) => h.label)).toEqual(expect.arrayContaining(["Undo", "Redo"]));

    useUi.setState({ shortcutsOpen: true });
    render(wrap(<ShortcutSheet />));
    const section = within(screen.getByTestId("shortcut-sheet")).getByRole("region", {
      name: "General",
    });
    expect(section).toHaveTextContent(`Undo${MOD_LABEL}Z`);
    expect(section).toHaveTextContent(`Redo${MOD_LABEL}ShiftZor${MOD_LABEL}Y`);
    useUi.setState({ shortcutsOpen: false });
  });
});

/** A scene with just what the card's actions touch. */
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

describe("the selection card's Hide and Show only", () => {
  let controller: SceneSelectController;
  beforeEach(() => {
    useSceneSelect.getState().clear();
    useSceneSelect.getState().setMode("pick");
    useInstances.setState({ assets: {}, gaps: {} });
    if (DOC) useInstances.getState().setTable(ASSET, DOC);
    controller = new SceneSelectController(viewer() as never, { ownKeys: false });
  });
  afterEach(() => controller.destroy());

  function Card() {
    return useSceneSelect(objectSelected) ? <ObjectCard controller={controller} /> : null;
  }
  const hidden = (): number[] =>
    [...(useInstances.getState().assets[ASSET]?.hidden ?? [])].sort((a, b) => a - b);
  const select = (id: number) =>
    act(() => useSceneSelect.getState().select(ASSET, [id], 1, 0, null));

  it("come back with Ctrl+Z, one press each, and go again with Ctrl+Shift+Z", async () => {
    const user = userEvent.setup();
    render(
      wrap(
        <UndoKeys>
          <Card />
        </UndoKeys>,
      ),
    );
    select(4);
    await user.click(screen.getByRole("button", { name: "Hide" }));
    expect(hidden()).toEqual([4]);
    select(5);
    await user.click(screen.getByRole("button", { name: "Show only" }));
    const only = hidden();
    expect(only).toEqual([1, 2, 3, 4, 6]);

    await user.keyboard("{Control>}z{/Control}");
    expect(hidden()).toEqual([4]);
    expect(said()[0]).toMatch(/^Undid: Show only Pumpkin \d$/);
    await user.keyboard("{Control>}z{/Control}");
    expect(hidden()).toEqual([]);
    expect(said()[0]).toMatch(/^Undid: Hide Pumpkin \d$/);

    await user.keyboard("{Control>}{Shift>}z{/Shift}{/Control}");
    expect(hidden()).toEqual([4]);
    await user.keyboard("{Control>}{Shift>}z{/Shift}{/Control}");
    expect(hidden()).toEqual(only);
    await user.keyboard("{Control>}{Shift>}z{/Shift}{/Control}");
    expect(said()).toEqual(["Nothing to redo"]);
  });
});

describe("a measurement removed in the Measure panel", () => {
  function measurement(id: string, createdAt: number): Measurement {
    return {
      id,
      mode: "distance",
      points: [],
      distance2dM: createdAt,
      distance3dM: createdAt,
      complete: true,
      createdAt,
    };
  }

  it("comes back to the list and the map with Ctrl+Z, and Clear all too", async () => {
    const user = userEvent.setup();
    const restored: string[] = [];
    const scene = {
      measurement: {
        remove: vi.fn((id: string) => () => restored.push(id)),
      },
    };
    const registry = new SceneRegistry();
    registry.set(scene as unknown as CesiumSceneManager);
    useUi.setState({ activePanel: "measure", measureMode: null });
    useMeasurements.setState({
      items: [measurement("a", 1), measurement("b", 2), measurement("c", 3)],
    });
    render(
      <SceneContext.Provider value={registry}>
        {wrap(
          <UndoKeys>
            <MeasurePanel />
          </UndoKeys>,
        )}
      </SceneContext.Provider>,
    );
    const ids = () => useMeasurements.getState().items.map((m) => m.id);
    const [, second] = screen.getAllByRole("button", { name: "Remove measurement" });
    if (!second) throw new Error("three measurements listed");
    await user.click(second);
    expect(ids()).toEqual(["a", "c"]);
    expect(scene.measurement.remove).toHaveBeenCalledWith("b");
    await user.keyboard("{Control>}z{/Control}");
    expect(said()).toEqual(["Undid: Remove distance measurement"]);
    expect(ids()).toEqual(["a", "b", "c"]);
    expect(restored).toEqual(["b"]);

    await user.click(screen.getByRole("button", { name: "Clear all" }));
    expect(ids()).toEqual([]);
    await user.keyboard("{Control>}z{/Control}");
    expect(said()).toEqual(["Undid: Clear 3 measurements"]);
    expect(ids()).toEqual(["a", "b", "c"]);
    expect(restored).toEqual(["b", "a", "b", "c"]);
    useUi.setState({ activePanel: null });
    useMeasurements.setState({ items: [] });
  });
});
