/**
 * Undo and redo (state/history.ts): the history itself, and every change recorded through it
 * -- objects hidden and shown (state/instances.ts), painted objects made and deleted
 * (state/sceneSelect.ts), measurements removed (MeasurementManager, the measurements store),
 * layers switched (features/layers/layerVisibility.ts) and an area reshaped
 * (state/mission.ts `reshapeArea`). Each undo restores exactly what was there before.
 */

import { EntityCollection, type Entity } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Layer } from "@twin/contracts";

import { MeasurementManager } from "@/cesium/MeasurementManager";
import { setLayerVisible } from "@/features/layers/layerVisibility";
import { loadCustomSets, type CustomSet } from "@/lib/customSets";
import { Emitter } from "@/lib/emitter";
import { parseInstances } from "@/lib/instances";
import type { Project, Zone } from "@/missions/types";
import {
  bindHistoryScope,
  clearHistory,
  HISTORY_LIMIT,
  isReplaying,
  record,
  recordAction,
  redo,
  undo,
  useHistory,
  type UndoStep,
} from "@/state/history";
import { TOGGLE_COALESCE_MS, useInstances } from "@/state/instances";
import { useLayers } from "@/state/layers";
import { useMeasurements, type Measurement } from "@/state/measurements";
import { reshapeArea, useMission } from "@/state/mission";
import { useSceneSelect } from "@/state/sceneSelect";
import { useSites } from "@/state/sites";

/** Lets the task end: what follows is another gesture (`currentTask`). */
const nextTask = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

const labels = (): string[] => useHistory.getState().past.map((s) => s.label);

/** A step that moves `value` between `from` and `to`. */
function counterStep(box: { value: number }, from: number, to: number, scope?: "site"): UndoStep {
  box.value = to;
  return record({
    label: `${String(from)} → ${String(to)}`,
    undo: () => (box.value = from),
    redo: () => (box.value = to),
    ...(scope ? { scope } : {}),
  });
}

beforeEach(() => {
  clearHistory();
});

describe("the history", () => {
  it("undoes and redoes in order; a new step empties what there was to redo", () => {
    const box = { value: 0 };
    counterStep(box, 0, 1);
    counterStep(box, 1, 2);
    counterStep(box, 2, 3);
    expect(labels()).toEqual(["0 → 1", "1 → 2", "2 → 3"]);
    expect(undo()?.label).toBe("2 → 3");
    expect(box.value).toBe(2);
    expect(undo()?.label).toBe("1 → 2");
    expect(box.value).toBe(1);
    expect(redo()?.label).toBe("1 → 2");
    expect(box.value).toBe(2);
    expect(useHistory.getState().future.map((s) => s.label)).toEqual(["2 → 3"]);
    // Something new: "2 → 3" can no longer be redone.
    counterStep(box, 2, 7);
    expect(useHistory.getState().future).toEqual([]);
    expect(redo()).toBeNull();
    expect(box.value).toBe(7);
    expect(undo()?.label).toBe("2 → 7");
    expect(undo()?.label).toBe("1 → 2");
    expect(undo()?.label).toBe("0 → 1");
    expect(box.value).toBe(0);
    expect(undo()).toBeNull();
  });

  it(`keeps the last ${String(HISTORY_LIMIT)} steps`, () => {
    const box = { value: 0 };
    for (let k = 0; k < HISTORY_LIMIT + 10; k += 1) counterStep(box, k, k + 1);
    expect(useHistory.getState().past).toHaveLength(HISTORY_LIMIT);
    let undone = 0;
    while (undo()) undone += 1;
    expect(undone).toBe(HISTORY_LIMIT);
    // The oldest ten fell off: undoing all that is kept stops at 10.
    expect(box.value).toBe(10);
  });

  it("applies and records in one call", () => {
    const box = { value: 0 };
    recordAction(
      "Set 5",
      () => (box.value = 5),
      () => (box.value = 0),
    );
    expect(box.value).toBe(5);
    undo();
    expect(box.value).toBe(0);
    redo();
    expect(box.value).toBe(5);
  });

  it("records nothing while a step is undone or redone", () => {
    const box = { value: 0 };
    const seen: boolean[] = [];
    record({
      label: "noisy",
      undo: () => {
        seen.push(isReplaying());
        counterStep(box, 9, 10);
      },
      redo: () => undefined,
    });
    undo();
    expect(seen).toEqual([true]);
    expect(isReplaying()).toBe(false);
    expect(useHistory.getState().past).toEqual([]);
    expect(useHistory.getState().future.map((s) => s.label)).toEqual(["noisy"]);
  });

  it("drops a step that no longer applies, or that fails, and goes on to the one before", () => {
    const box = { value: 0 };
    counterStep(box, 0, 1);
    let alive = true;
    record({ label: "gone", undo: () => undefined, redo: () => undefined, alive: () => alive });
    record({
      label: "broken",
      undo: () => {
        throw new Error("no scene");
      },
      redo: () => undefined,
    });
    alive = false;
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    // Both are dropped on the way, and the step before them is undone.
    expect(undo()?.label).toBe("0 → 1");
    expect(box.value).toBe(0);
    expect(labels()).toEqual([]);
    vi.restoreAllMocks();
  });

  it("drops the site's steps when the active site changes, and keeps the globe's", () => {
    const off = bindHistoryScope();
    useSites.getState().setActiveSite("a");
    const box = { value: 0 };
    counterStep(box, 0, 1, "site");
    counterStep(box, 1, 2);
    counterStep(box, 2, 3, "site");
    undo();
    expect(useHistory.getState().future).toHaveLength(1);
    useSites.getState().setActiveSite("b");
    expect(labels()).toEqual(["1 → 2"]);
    expect(useHistory.getState().future).toEqual([]);
    // The same site again changes nothing.
    useSites.getState().setActiveSite("b");
    expect(labels()).toEqual(["1 → 2"]);
    off();
    useSites.getState().setActiveSite(null);
  });
});

// ---- Objects -----------------------------------------------------------------------------

const box = { min: [0, 0, 0], max: [1, 1, 1] };
/** Dirt (1) under conifer 1 (2) and its branch (3); conifer 2 (4); a picnic table (5). */
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    { id: 1, bounds: box, splats: 100, tags: [{ label: "dirt", score: 0.6 }] },
    { id: 2, parent: 1, bounds: box, splats: 50, tags: [{ label: "conifer", score: 0.5 }] },
    { id: 3, parent: 2, bounds: box, splats: 20 },
    { id: 4, bounds: box, splats: 40, tags: [{ label: "conifer", score: 0.4 }] },
    { id: 5, bounds: box, splats: 30, tags: [{ label: "picnic table", score: 0.4 }] },
  ],
  tiles: {},
});
if (!DOC) throw new Error("fixture");

const objects = useInstances.getState;
const hidden = (): number[] => [...(objects().assets.a?.hidden ?? [])].sort((x, y) => x - y);

describe("objects hidden and shown", () => {
  let now = 1_000_000;
  beforeEach(() => {
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    objects().setTable("a", DOC);
    now = 1_000_000;
    vi.spyOn(Date, "now").mockImplementation(() => now);
  });
  afterEach(() => vi.restoreAllMocks());

  /** Another gesture, later than a quick repeat. */
  const later = async (): Promise<void> => {
    await nextTask();
    now += TOGGLE_COALESCE_MS + 1;
  };

  it("undoes each change exactly, a part of a category hidden included, and redoes it", async () => {
    const states: number[][] = [hidden()];
    objects().setCategoryHidden("a", "trees", true);
    states.push(hidden());
    await later();
    // One conifer shown again: the category is partly hidden.
    objects().setObjectsHidden("a", [4], false);
    states.push(hidden());
    await later();
    objects().setQuery("a", "picnic");
    objects().hideMatches("a");
    states.push(hidden());
    await later();
    objects().setQuery("a", "conifer");
    objects().showOnlyMatches("a");
    states.push(hidden());
    await later();
    objects().setHidden("a", [4], true);
    states.push(hidden());
    await later();
    objects().showAll("a");
    states.push(hidden());
    await later();
    objects().setCategoryHidden("a", "ground", true);
    states.push(hidden());
    await later();
    objects().toggleFocus("a", { kind: "category", id: "trees" });
    objects().reset("a");
    states.push(hidden());
    expect(states).toEqual([[], [2, 3, 4], [2, 3], [2, 3, 5], [1, 5], [1, 4, 5], [], [1], []]);
    expect(labels()).toEqual([
      "Hide Trees",
      "Show Conifer 2",
      "Hide 1 match for “picnic”",
      "Show only 2 matches for “conifer”",
      "Hide Conifer 2",
      "Show all objects",
      "Hide Ground & soil",
      "Reset objects",
    ]);
    for (let k = states.length - 2; k >= 0; k -= 1) {
      undo();
      expect(hidden(), `after undoing step ${String(k + 1)}`).toEqual(states[k]);
    }
    expect(undo()).toBeNull();
    for (let k = 1; k < states.length; k += 1) {
      redo();
      expect(hidden(), `after redoing step ${String(k)}`).toEqual(states[k]);
    }
    // Undo brings back what was hidden, not the highlight: a highlight is a selection.
    undo();
    expect(objects().assets.a?.highlighted.size).toBe(0);
  });

  it("names a change by ids from what it did: the card's Hide, and Show only as one step", async () => {
    // The card's Hide: the selection with what it contains.
    objects().setHidden("a", [2], true);
    expect(hidden()).toEqual([2, 3]);
    await later();
    // The card's Show only: everything shown, then the rest hidden, in one gesture.
    objects().showAll("a");
    objects().setHidden("a", [1, 2, 3, 5], true);
    expect(hidden()).toEqual([1, 2, 3, 5]);
    expect(labels()).toEqual(["Hide Conifer 1", "Show only Conifer 2"]);
    undo();
    expect(hidden()).toEqual([2, 3]);
    undo();
    expect(hidden()).toEqual([]);
    redo();
    redo();
    expect(hidden()).toEqual([1, 2, 3, 5]);
  });

  it("folds a quick toggle back into nothing, and a slower one into a step of its own", async () => {
    objects().setObjectsHidden("a", [5], true);
    await nextTask();
    now += TOGGLE_COALESCE_MS / 2;
    objects().setObjectsHidden("a", [5], false);
    expect(labels()).toEqual([]);
    await nextTask();
    now += TOGGLE_COALESCE_MS / 2;
    // Hide, show, hide: one step, the net change.
    objects().setObjectsHidden("a", [5], true);
    await nextTask();
    now += 100;
    objects().setObjectsHidden("a", [5], false);
    await nextTask();
    now += 100;
    objects().setObjectsHidden("a", [5], true);
    expect(labels()).toEqual(["Hide Picnic table"]);
    await later();
    objects().setObjectsHidden("a", [5], false);
    expect(labels()).toEqual(["Hide Picnic table", "Show Picnic table"]);
    // Another thing toggled at once is its own step.
    await nextTask();
    objects().setObjectsHidden("a", [4], true);
    expect(labels()).toHaveLength(3);
    // After an undo nothing folds into the step undone or the one before it.
    undo();
    await nextTask();
    objects().setObjectsHidden("a", [5], true);
    expect(labels()).toEqual(["Hide Picnic table", "Show Picnic table", "Hide Picnic table"]);
    expect(useHistory.getState().future).toEqual([]);
  });

  it("records nothing for a change that changes nothing, or a scan with no table", () => {
    objects().showAll("a");
    objects().setCategoryHidden("a", "nope", true);
    objects().hideMatches("a");
    objects().setHidden("b", [1], true);
    expect(labels()).toEqual([]);
  });

  it("does not apply a step to a scan that was loaded again", () => {
    objects().setHidden("a", [5], true);
    // The scan reloads: a new table, nothing hidden; the old step's ids are not these.
    objects().setTable("a", { instances: [...DOC.instances] });
    objects().setHidden("a", [4], true);
    expect(undo()?.label).toBe("Hide Conifer 2");
    expect(hidden()).toEqual([]);
    expect(undo()).toBeNull();
    expect(hidden()).toEqual([]);
  });

  it("is a site's: another site active, Ctrl+Z does not reach this scan", () => {
    const off = bindHistoryScope();
    useSites.getState().setActiveSite("pumpkin");
    objects().setHidden("a", [5], true);
    useSites.getState().setActiveSite("camp");
    expect(undo()).toBeNull();
    expect(hidden()).toEqual([5]);
    off();
    useSites.getState().setActiveSite(null);
  });
});

// ---- Painted objects ---------------------------------------------------------------------

function painted(key: string, name: string): CustomSet {
  return {
    key,
    name,
    tiles: { abc: [0, 4] },
    splats: 4,
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    created: 1,
  };
}

describe("painted objects", () => {
  const picker = useSceneSelect.getState;
  const keys = (): string[] => (picker().custom.a ?? []).map((c) => c.key);
  beforeEach(() => {
    localStorage.clear();
    useSceneSelect.setState({ custom: {} });
    picker().clear();
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    objects().setTable("a", DOC);
  });

  it("takes back one made, and its selection with it", () => {
    picker().addCustom("a", painted("p1", "Painted area 1"));
    // "Use painted area" selects what it made: an id past the file's (maxId 5).
    picker().select("a", [6], 1, 0, null);
    expect(labels()).toEqual(["Create Painted area 1"]);
    undo();
    expect(keys()).toEqual([]);
    expect(loadCustomSets("a")).toEqual([]);
    expect(picker().candidates).toEqual([]);
    redo();
    expect(keys()).toEqual(["p1"]);
    expect(loadCustomSets("a").map((c) => c.key)).toEqual(["p1"]);
  });

  it("puts one deleted back where it was, so the later ones keep their ids", () => {
    for (const k of [1, 2, 3])
      picker().addCustom("a", painted(`p${String(k)}`, `Area ${String(k)}`));
    clearHistory();
    picker().removeCustom("a", "p2");
    expect(keys()).toEqual(["p1", "p3"]);
    expect(labels()).toEqual(["Delete Area 2"]);
    undo();
    expect(keys()).toEqual(["p1", "p2", "p3"]);
    expect(loadCustomSets("a").map((c) => c.key)).toEqual(["p1", "p2", "p3"]);
    redo();
    expect(keys()).toEqual(["p1", "p3"]);
    // Deleting what is not there is nothing to undo.
    picker().removeCustom("a", "nope");
    expect(labels()).toEqual(["Delete Area 2"]);
  });
});

// ---- Measurements ------------------------------------------------------------------------

function measurement(id: string, createdAt: number): Measurement {
  return { id, mode: "distance", points: [], complete: true, createdAt };
}

describe("measurements removed", () => {
  it("come back on the map as the same entities, and in the list in the order they were made", () => {
    const entities = new EntityCollection();
    const manager = new MeasurementManager(
      { scene: { requestRender: () => undefined }, entities } as never,
      new Emitter(),
    );
    const drawn = [entities.add({ id: "m1-line" }), entities.add({ id: "m1-label" })];
    (manager as unknown as { finished: Map<string, Entity[]> }).finished.set("m1", drawn);
    const restore = manager.remove("m1");
    expect(entities.values).toHaveLength(0);
    expect(manager.remove("m1")).toBeNull();
    restore?.();
    expect(entities.values).toEqual(drawn);
    // Once back it can be removed again.
    expect(manager.remove("m1")).not.toBeNull();

    const store = useMeasurements.getState;
    useMeasurements.setState({
      items: [measurement("a", 1), measurement("b", 2), measurement("c", 3)],
    });
    const gone = store().items.filter((m) => m.id !== "c");
    for (const m of gone) store().remove(m.id);
    store().upsert(measurement("d", 4));
    store().restore(gone);
    expect(store().items.map((m) => m.id)).toEqual(["a", "b", "c", "d"]);
    // What is already there is not doubled.
    store().restore(gone);
    expect(store().items).toHaveLength(4);
  });
});

// ---- Layers ------------------------------------------------------------------------------

function layer(id: string, exclusiveGroup?: string): Pick<Layer, "id" | "name" | "render"> {
  return { id, name: id.toUpperCase(), render: { exclusiveGroup } };
}

describe("layers switched", () => {
  const catalog = [layer("aerial", "basemap"), layer("streets", "basemap"), layer("cover")];
  /** The scene's layers as `LayerManager.setVisible` keeps them: a basemap at a time. */
  const scene = {
    layers: {
      setVisible: vi.fn((id: string, visible: boolean) => {
        const runtime = useLayers.getState();
        const group = catalog.find((l) => l.id === id)?.render.exclusiveGroup;
        if (visible && group)
          for (const other of catalog)
            if (other.id !== id && other.render.exclusiveGroup === group)
              runtime.update(other.id, { visible: false });
        runtime.update(id, { visible });
        return Promise.resolve();
      }),
    },
  };
  const visible = (): string[] =>
    Object.entries(useLayers.getState().runtime)
      .filter(([, r]) => r.visible)
      .map(([id]) => id)
      .sort();

  beforeEach(() => {
    useLayers.setState({ runtime: {} });
    useLayers.getState().update("aerial", { visible: true });
  });

  it("undo turns back on the basemap that a new one switched off", () => {
    setLayerVisible(scene as never, layer("streets", "basemap"), true, catalog);
    expect(visible()).toEqual(["streets"]);
    setLayerVisible(scene as never, layer("cover"), true, catalog);
    expect(visible()).toEqual(["cover", "streets"]);
    expect(labels()).toEqual(["Show STREETS", "Show COVER"]);
    undo();
    expect(visible()).toEqual(["streets"]);
    undo();
    expect(visible()).toEqual(["aerial"]);
    redo();
    expect(visible()).toEqual(["streets"]);
    // Asking for what already is records nothing.
    setLayerVisible(scene as never, layer("streets", "basemap"), true, catalog);
    expect(labels()).toEqual(["Show STREETS"]);
  });
});

// ---- Areas -------------------------------------------------------------------------------

describe("an area reshaped on the map", () => {
  const square = (size: number): Zone["footprint"] => ({
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [size, 0],
        [size, size],
        [0, size],
        [0, 0],
      ],
    ],
  });

  it("goes back to its outline, and the plan hears of it", () => {
    const zone = {
      id: "A-1",
      name: "Drawn area 1",
      footprint: square(0.001),
    } as unknown as Zone;
    const project = { id: "p", zones: [], plans: [] } as unknown as Project;
    useMission.setState({ project: null, areas: {} });
    useMission.getState().setProject(project);
    useMission.getState().addArea("p", zone);
    const announced: Zone["footprint"][] = [];
    const announce = (_id: string, footprint: Zone["footprint"]) => announced.push(footprint);
    const outline = () => useMission.getState().project?.zones.find((z) => z.id === "A-1");

    reshapeArea("A-1", square(0.002), announce);
    expect(outline()?.footprint).toEqual(square(0.002));
    expect(labels()).toEqual(["Reshape Drawn area 1"]);
    expect(announced).toEqual([]);
    undo();
    // Exactly the area it was, not one rebuilt from its outline.
    expect(outline()).toBe(zone);
    expect(announced).toEqual([square(0.001)]);
    redo();
    expect(outline()?.footprint).toEqual(square(0.002));
    expect(announced).toEqual([square(0.001), square(0.002)]);
    // Another project: the step no longer applies.
    useMission.getState().setProject({ ...project, id: "q" });
    expect(undo()).toBeNull();
    useMission.setState({ project: null, areas: {} });
  });
});
