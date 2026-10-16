/**
 * Ways into the Land tool from elsewhere on the map (features/land/landEntry.ts). The rules
 * that matter: a new outline never leaves a saved land active (Save would revise it), and an
 * unsaved outline is never replaced.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import type { LandArea, LandCreate } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { openLand, outlineLandAt, proposeOutline } from "@/features/land/landEntry";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

const SAVED: LandArea = {
  id: "land-1",
  name: "Oak Bench",
  description: "",
  revision: 2,
  areaM2: 1,
  perimeterM: 4,
  createdAt: "2026-10-16",
  updatedAt: "2026-10-16",
  source: { method: "drawn", label: "Fixture" },
  boundary: { type: "Polygon", coordinates: [] },
};
const DRAFT: LandCreate = {
  name: "Creek field",
  description: "",
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.72, 35.53],
        [-120.71, 35.53],
        [-120.71, 35.54],
        [-120.72, 35.53],
      ],
    ],
  },
  source: { method: "mapped-feature", label: "OpenStreetMap feature" },
};

function fakeScene() {
  return { areas: { cancelPick: vi.fn(), edit: vi.fn() } };
}

afterEach(() => {
  useLand.getState().clear();
  useLandContext.getState().clear();
  useUi.setState({ activePanel: null, measureMode: null });
});

describe("outlineLandAt", () => {
  it("frees the pointer, clears the active land and finds candidates at the point", () => {
    const scene = fakeScene();
    useLand.getState().select(SAVED);
    useUi.getState().setMeasureMode("distance");
    expect(
      outlineLandAt(
        { longitude: -120.72, latitude: 35.53 },
        scene as unknown as CesiumSceneManager,
      ),
    ).toBe("started");
    expect(scene.areas.cancelPick).toHaveBeenCalled();
    expect(scene.areas.edit).toHaveBeenCalledWith(null);
    expect(useUi.getState().measureMode).toBeNull();
    expect(useUi.getState().activePanel).toBe("land");
    expect(useLand.getState()).toMatchObject({
      active: null,
      mode: "candidates",
      points: [[-120.72, 35.53]],
    });
  });

  it("opens the tool on an unsaved outline instead of replacing it", () => {
    useLand.getState().propose(DRAFT);
    expect(outlineLandAt({ longitude: 0, latitude: 0 }, null)).toBe("has-draft");
    expect(useLand.getState().draft).toEqual(DRAFT);
    expect(useLand.getState().points).toEqual([]);
    expect(useUi.getState().activePanel).toBe("land");
  });
});

describe("proposeOutline", () => {
  it("proposes a new land, never a revision of the active one", () => {
    useLand.getState().select(SAVED);
    expect(proposeOutline(DRAFT, null)).toBe("started");
    expect(useLand.getState().active).toBeNull();
    expect(useLand.getState().draft).toEqual(DRAFT);
    expect(useUi.getState().activePanel).toBe("land");
  });

  it("keeps an unsaved outline", () => {
    const first = { ...DRAFT, name: "First" };
    useLand.getState().propose(first);
    expect(proposeOutline(DRAFT, null)).toBe("has-draft");
    expect(useLand.getState().draft).toEqual(first);
  });
});

describe("openLand", () => {
  it("selects the land on Discover", () => {
    useLandContext.getState().setSection("inventory");
    expect(openLand(SAVED)).toBe("opened");
    expect(useLand.getState().active).toEqual(SAVED);
    expect(useLandContext.getState().section).toBe("discover");
    expect(useUi.getState().activePanel).toBe("land");
  });

  it("does not drop an unsaved outline to select another land", () => {
    useLand.getState().propose(DRAFT);
    expect(openLand(SAVED, "actions")).toBe("has-draft");
    expect(useLand.getState().draft).toEqual(DRAFT);
    expect(useLand.getState().active).toBeNull();
  });
});
