import { type CesiumWidget, Color, EntityCollection, JulianDate } from "cesium";
import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { MissionManager } from "@/cesium/MissionManager";
import { rehearsalPick, RehearsalManager, type RehearsalFrame } from "@/cesium/RehearsalManager";
import { planRehearsal } from "@/missions/rehearsal";

const origin = { longitude: -120.04, latitude: 34.698 };
const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
const ky = DEG_TO_RAD * EARTH_RADIUS_M;
const at = (x: number, y: number): LonLat => ({
  longitude: origin.longitude + x / kx,
  latitude: origin.latitude + y / ky,
});

/** Two 600 m lanes for one mower; a creek 10 m from the first raises one hand on it. */
function rehearsal(machineIds = ["TR-04"]) {
  return planRehearsal({
    zones: [
      {
        zoneId: "A-01",
        label: "A-01",
        lanes: [
          { points: [at(-200, -300), at(-200, 300)], pass: 0 },
          { points: [at(-180, 300), at(-180, -300)], pass: 1 },
        ],
        machineIds,
        classId: "mower",
      },
    ],
    machines: [{ id: "TR-04", name: "TR-04 Kestrel", model: "Ranger 2 · mower" }],
    elevation: null,
    creeks: {
      label: "USGS NHD flowlines, 1:24,000",
      lines: [
        {
          id: "nhd-1",
          name: "Figueroa Creek",
          kind: "intermittent",
          coordinates: [at(-210, -50), at(-210, 50)],
        },
      ],
    },
  });
}

let frames: FrameRequestCallback[] = [];

function setup() {
  const entities = new EntityCollection();
  const requestRender = vi.fn();
  const viewer = { entities, scene: { requestRender } } as unknown as CesiumWidget;
  const mission = { setPreviewPassesShown: vi.fn(), flyToPoint: vi.fn(), hideAgents: vi.fn() };
  const manager = new RehearsalManager(viewer, mission as unknown as MissionManager);
  const heard: (RehearsalFrame | null)[] = [];
  manager.onFrame((frame) => heard.push(frame));
  const ids = () => entities.values.map((e) => e.id);
  const shown = (id: string) => entities.getById(id)?.show;
  return { entities, requestRender, mission, manager, heard, ids, shown };
}

const look = { colorOf: () => Color.CYAN, heightAt: () => 300 };

beforeEach(() => {
  frames = [];
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames.push(callback);
    return frames.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {
    frames = [];
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("RehearsalManager", () => {
  it("draws none of its simulated run while As captured hides it, and the run as it is after", () => {
    const { manager, entities, requestRender, shown, heard } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const ghost = entities.getById("rehearsal:ghost:TR-04");
    const now = JulianDate.now();
    expect(ghost?.position?.getValue(now)).toBeDefined();
    const first = run.lanes[0];
    const hand = run.hands[0];
    if (!first || !hand) throw new Error("expected a lane and a hand");
    manager.seek(first.t1 + 1);
    expect(shown("rehearsal:done:A-01:0")).toBe(true);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(true);
    requestRender.mockClear();
    manager.hideAgents(true);
    expect(requestRender).toHaveBeenCalled();
    // No ghost (no position: the billboard and its label go with it), no lane done, no hand
    // raised: every lane is the plan's, pending. The mapped creek stays.
    expect(ghost?.position?.getValue(now)).toBeUndefined();
    expect(shown("rehearsal:done:A-01:0")).toBe(false);
    expect(shown("rehearsal:lane:A-01:0")).toBe(true);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(false);
    expect(shown("rehearsal:creek:0")).not.toBe(false);
    // Its own state is untouched: the clock and the hands raised are where they were.
    expect(heard.at(-1)).toMatchObject({ raised: 1 });
    expect(manager.frame).toMatchObject({ t: first.t1 + 1, raised: 1 });
    // The clock moved meanwhile draws nothing of it either.
    manager.seek(run.durationS);
    expect(shown("rehearsal:done:A-01:1")).toBe(false);
    expect(shown("rehearsal:lane:A-01:1")).toBe(true);
    // A rehearsal loaded meanwhile is hidden too, until it is shown again, as it stands.
    manager.load(rehearsal(), look, first.t1 + 1);
    const again = entities.getById("rehearsal:ghost:TR-04");
    expect(again?.position?.getValue(now)).toBeUndefined();
    expect(shown("rehearsal:done:A-01:0")).toBe(false);
    manager.hideAgents(false);
    expect(again?.position?.getValue(now)).toBeDefined();
    expect(shown("rehearsal:done:A-01:0")).toBe(true);
    expect(shown("rehearsal:lane:A-01:0")).toBe(false);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(true);
  });

  it("hides the hand open in the card with the rest of the run while As captured is on", () => {
    const { manager, shown, ids } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const hand = run.hands[0];
    if (!hand) throw new Error("expected a hand");
    manager.seek(hand.t + 1);
    manager.focus(hand.id, false);
    const focus = ids().find((id) => id.startsWith("rehearsal:focus:"));
    if (!focus) throw new Error("no focus marker");
    expect(shown(focus)).not.toBe(false);
    manager.hideAgents(true);
    expect(shown(focus)).toBe(false);
    manager.hideAgents(false);
    expect(shown(focus)).toBe(true);
  });

  it("draws pending lanes, ghosts and the checked creek, hides the plan's passes", () => {
    const { manager, mission, ids, shown, heard } = setup();
    const run = rehearsal();
    manager.load(run, look);
    expect(mission.setPreviewPassesShown).toHaveBeenCalledWith(false);
    expect(ids()).toEqual(
      expect.arrayContaining([
        "rehearsal:lane:A-01:0",
        "rehearsal:lane:A-01:1",
        "rehearsal:ghost:TR-04",
        "rehearsal:creek:0",
        `rehearsal:hand:${run.hands[0]?.id}`,
      ]),
    );
    // Nothing is done and no hand is up at the start; each lane's done line waits, hidden.
    const done = ids().filter((id) => id.startsWith("rehearsal:done:"));
    expect(done).toEqual(["rehearsal:done:A-01:0", "rehearsal:done:A-01:1"]);
    expect(done.every((id) => shown(id) === false)).toBe(true);
    expect(shown(`rehearsal:hand:${run.hands[0]?.id}`)).toBe(false);
    expect(heard.at(-1)).toMatchObject({ t: 0, playing: false, raised: 0, ended: false });
  });

  it("marks lanes done and raises hands as the clock passes them, and undoes it going back", () => {
    const { manager, ids, shown, heard, entities } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const first = run.lanes[0];
    const hand = run.hands[0];
    if (!first || !hand) throw new Error("expected a lane and a hand");
    const drawn = ids();
    manager.seek(first.t1 + 1);
    expect(shown("rehearsal:done:A-01:0")).toBe(true);
    expect(shown("rehearsal:done:A-01:1")).toBe(false);
    expect(shown("rehearsal:lane:A-01:0")).toBe(false);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(true);
    expect(heard.at(-1)?.raised).toBe(1);
    manager.seek(0);
    expect(shown("rehearsal:done:A-01:0")).toBe(false);
    expect(shown("rehearsal:lane:A-01:0")).toBe(true);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(false);
    manager.seek(Number.POSITIVE_INFINITY);
    expect(shown("rehearsal:done:A-01:1")).toBe(true);
    expect(heard.at(-1)).toMatchObject({ ended: true, playing: false, t: run.durationS });
    // Scrubbing shows and hides; nothing is added or removed (a ground polyline batch rebuild).
    expect(entities.values.map((e) => e.id)).toEqual(drawn);
  });

  it("plays on its own frames at the chosen speed, asking the scene to render only then", () => {
    const { manager, requestRender, heard } = setup();
    const run = rehearsal();
    manager.load(run, look);
    manager.play(600);
    expect(frames).toHaveLength(1);
    frames.shift()?.(1000);
    frames.shift()?.(1500);
    // Half a second at 600x is 300 simulated seconds (listeners hear it at most every 100 ms).
    expect(manager.frame?.t).toBeCloseTo(300, 6);
    expect(manager.frame).toMatchObject({ playing: true, speed: 600 });
    const renders = requestRender.mock.calls.length;
    manager.pause();
    expect(frames).toHaveLength(0);
    expect(heard.at(-1)?.playing).toBe(false);
    expect(requestRender.mock.calls.length).toBe(renders);
    // A long gap between frames (a tab in the background) moves the clock one second at most.
    manager.play(300);
    frames.shift()?.(10_000);
    frames.shift()?.(70_000);
    expect(manager.frame?.t).toBeCloseTo(600, 6);
  });

  it("names ghosts so a click selects a project machine and looks through the rest", () => {
    const { manager, ids } = setup();
    manager.load(rehearsal(), look);
    expect(ids()).toContain("rehearsal:ghost:TR-04");
    // A zone nobody is assigned to: an assumed generic machine, which is not a machine to select.
    manager.load(rehearsal([]), look);
    expect(ids()).toContain("rehearsal:sim:SIM-1");
    expect(rehearsalPick("rehearsal:hand:A-01:0:creek-buffer")).toEqual({
      kind: "hand",
      handId: "A-01:0:creek-buffer",
    });
    expect(rehearsalPick("rehearsal:ghost:TR-04")).toEqual({ kind: "machine", machineId: "TR-04" });
    for (const id of [
      "rehearsal:sim:SIM-1",
      "rehearsal:lane:A-01:0",
      "rehearsal:done:A-01:0",
      "rehearsal:creek:0",
    ])
      expect(rehearsalPick(id)).toBe("through");
    expect(rehearsalPick("mission:zone:Z-14")).toBeNull();
  });

  it("stands one counting marker per cluster of hands, never one per hand", () => {
    // Eight lanes side by side, 10 m apart, each crossing the same creek: eight hands, one
    // band of neighbouring lanes, so one marker that counts them as the ghost reaches them.
    const lanes = Array.from({ length: 8 }, (_, i) => ({
      points:
        i % 2 === 0
          ? [at(-200 + i * 10, -300), at(-200 + i * 10, 300)]
          : [at(-200 + i * 10, 300), at(-200 + i * 10, -300)],
      pass: i,
    }));
    const run = planRehearsal({
      zones: [{ zoneId: "A-01", label: "A-01", lanes, machineIds: ["TR-04"], classId: "mower" }],
      machines: [{ id: "TR-04", name: "TR-04 Kestrel", model: "Ranger 2 · mower" }],
      elevation: null,
      creeks: {
        label: "USGS NHD flowlines, 1:24,000",
        lines: [
          {
            id: "nhd-1",
            name: "Figueroa Creek",
            kind: "intermittent",
            coordinates: [at(-300, 0), at(0, 0)],
          },
        ],
      },
    });
    expect(run.hands).toHaveLength(8);
    const { manager, ids, entities } = setup();
    manager.load(run, look);
    const markers = ids().filter((id) => id.startsWith("rehearsal:hand:"));
    expect(markers).toEqual([`rehearsal:hand:${run.hands[0]?.id}`]);
    const marker = entities.getById(markers[0] ?? "");
    const image = () => String(marker?.billboard?.image?.getValue());
    manager.seek((run.hands[2]?.t ?? 0) + 1);
    expect(marker?.show).toBe(true);
    expect(decodeURIComponent(image())).toContain(">3</text>");
    expect(marker?.label?.text?.getValue()).toBe("3 lanes inside a creek buffer · 15 m (assumed)");
    manager.seek(Number.POSITIVE_INFINITY);
    expect(decodeURIComponent(image())).toContain(">8</text>");
    // The frame still counts every hand, not the markers.
    expect(manager.frame?.raised).toBe(8);
  });

  it("marks the hand open in the card on its own, and a click on it keeps it open", () => {
    const { manager, ids, entities } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const hand = run.hands[0];
    manager.focus(hand?.id ?? null, false);
    const id = `rehearsal:focus:${hand?.id}`;
    expect(ids()).toContain(id);
    expect(entities.getById(id)?.label?.text?.getValue()).toBe(hand?.label);
    expect(rehearsalPick(id)).toEqual({ kind: "hand", handId: hand?.id });
    manager.focus(null);
    expect(ids()).not.toContain(id);
  });

  it("hatches left-out ground through a stop, until it is cleared", () => {
    const { manager, ids, entities } = setup();
    const square = (x: number, y: number, size: number) => ({
      type: "MultiPolygon" as const,
      coordinates: [
        [
          [
            [at(x, y).longitude, at(x, y).latitude],
            [at(x + size, y).longitude, at(x + size, y).latitude],
            [at(x + size, y + size).longitude, at(x + size, y + size).latitude],
            [at(x, y + size).longitude, at(x, y + size).latitude],
            [at(x, y).longitude, at(x, y).latitude],
          ],
        ],
      ],
    });
    manager.load(rehearsal(), look);
    manager.showLeftOut(
      [
        { reason: "steep", label: "Left out: too steep", footprint: square(0, 0, 100) },
        { reason: "steep", label: "Left out: too steep", footprint: square(300, 0, 20) },
        { reason: "creek-buffer", label: "Left out: creek buffer", footprint: square(0, 200, 30) },
        { reason: "sliver", label: "Left out: too small to work", footprint: square(500, 0, 5) },
      ],
      () => 250,
    );
    const hatched = () => ids().filter((id) => id.startsWith("rehearsal:left-out:"));
    expect(hatched()).toHaveLength(6);
    const steep = entities.getById("rehearsal:left-out:label:steep");
    expect(steep?.label?.text?.getValue()).toBe("Left out: too steep · derived");
    expect(entities.getById("rehearsal:left-out:label:creek-buffer")).toBeDefined();
    expect(entities.getById("rehearsal:left-out:label:sliver")).toBeUndefined();
    expect(entities.getById("rehearsal:left-out:0:0")?.name).toBe("Left out: too steep (derived)");
    // Looked through on a click, like the rest of the rehearsal's drawing.
    expect(rehearsalPick("rehearsal:left-out:0:0")).toBe("through");
    manager.stop();
    expect(hatched()).toHaveLength(6);
    manager.showLeftOut([]);
    expect(entities.values).toHaveLength(0);
  });

  it("flies to a focused hand and clears everything on stop", () => {
    const { manager, mission, entities, heard } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const hand = run.hands[0];
    manager.focus(hand?.id ?? null);
    expect(mission.flyToPoint).toHaveBeenCalledWith(hand?.at, expect.any(Number));
    manager.stop();
    expect(entities.values).toHaveLength(0);
    expect(mission.setPreviewPassesShown).toHaveBeenLastCalledWith(true);
    expect(heard.at(-1)).toBeNull();
    expect(manager.loaded).toBeNull();
  });
});
