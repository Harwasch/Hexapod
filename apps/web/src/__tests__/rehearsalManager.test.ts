import { type CesiumWidget, Color, EntityCollection } from "cesium";
import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { MissionManager } from "@/cesium/MissionManager";
import { RehearsalManager, type RehearsalFrame } from "@/cesium/RehearsalManager";
import { planRehearsal } from "@/missions/rehearsal";

const origin = { longitude: -120.04, latitude: 34.698 };
const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
const ky = DEG_TO_RAD * EARTH_RADIUS_M;
const at = (x: number, y: number): LonLat => ({
  longitude: origin.longitude + x / kx,
  latitude: origin.latitude + y / ky,
});

/** Two 600 m lanes for one mower; a creek 10 m from the first raises one hand on it. */
function rehearsal() {
  return planRehearsal({
    zones: [
      {
        zoneId: "A-01",
        label: "A-01",
        lanes: [
          { points: [at(-200, -300), at(-200, 300)], pass: 0 },
          { points: [at(-180, 300), at(-180, -300)], pass: 1 },
        ],
        machineIds: ["TR-04"],
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
  const mission = { setPreviewPassesShown: vi.fn(), flyToPoint: vi.fn() };
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
    // Nothing is done and no hand is up at the start.
    expect(ids().some((id) => id.startsWith("rehearsal:done:"))).toBe(false);
    expect(shown(`rehearsal:hand:${run.hands[0]?.id}`)).toBe(false);
    expect(heard.at(-1)).toMatchObject({ t: 0, playing: false, raised: 0, ended: false });
  });

  it("marks lanes done and raises hands as the clock passes them, and undoes it going back", () => {
    const { manager, ids, shown, heard } = setup();
    const run = rehearsal();
    manager.load(run, look);
    const first = run.lanes[0];
    const hand = run.hands[0];
    if (!first || !hand) throw new Error("expected a lane and a hand");
    manager.seek(first.t1 + 1);
    expect(ids()).toContain("rehearsal:done:A-01:0");
    expect(shown("rehearsal:lane:A-01:0")).toBe(false);
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(true);
    expect(heard.at(-1)?.raised).toBe(1);
    manager.seek(0);
    expect(ids()).not.toContain("rehearsal:done:A-01:0");
    expect(shown(`rehearsal:hand:${hand.id}`)).toBe(false);
    manager.seek(Number.POSITIVE_INFINITY);
    expect(ids()).toContain("rehearsal:done:A-01:1");
    expect(heard.at(-1)).toMatchObject({ ended: true, playing: false, t: run.durationS });
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
