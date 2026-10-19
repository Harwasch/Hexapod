/**
 * Ways into the Land tool from elsewhere on the map (features/land/landEntry.ts). The rules
 * that matter: a new outline never leaves a saved land active (Save would revise it), and an
 * unsaved outline, proposed or still being drawn, is never replaced.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import type { LandArea, LandCreate } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import {
  drawOutline,
  hasUnsavedOutline,
  openLand,
  outlineLandAt,
  proposeOutline,
} from "@/features/land/landEntry";
import { useLens } from "@/features/land/surface/lenses";
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

const SKETCH: [number, number][] = [
  [-120.72, 35.53],
  [-120.71, 35.53],
  [-120.71, 35.54],
];

function fakeScene() {
  return { areas: { cancelPick: vi.fn(), edit: vi.fn() } };
}

/** An outline still being drawn: points placed, no draft yet. */
function sketch(mode: "draw" | "corridor" = "draw"): void {
  useLand.getState().begin(mode);
  for (const point of SKETCH) useLand.getState().addPoint(point);
}

afterEach(() => {
  useLens.getState().setLens("overview");
  useLand.getState().clear();
  useLandContext.getState().clear();
  useUi.setState({ activePanel: null, measureMode: null });
});

describe("hasUnsavedOutline", () => {
  it("is a draft, or a sketch with points placed", () => {
    expect(hasUnsavedOutline(useLand.getState())).toBe(false);
    useLand.getState().begin("draw");
    expect(hasUnsavedOutline(useLand.getState())).toBe(false);
    useLand.getState().addPoint([-120.72, 35.53]);
    expect(hasUnsavedOutline(useLand.getState())).toBe(true);
    useLand.getState().cancel();
    // A lookup point in candidates or pick mode is not an outline.
    useLand.getState().begin("candidates");
    useLand.getState().addPoint([-120.72, 35.53]);
    expect(hasUnsavedOutline(useLand.getState())).toBe(false);
    useLand.getState().propose(DRAFT);
    expect(hasUnsavedOutline(useLand.getState())).toBe(true);
  });
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

  it("keeps an outline still being drawn, with its points", () => {
    for (const mode of ["draw", "corridor"] as const) {
      sketch(mode);
      expect(outlineLandAt({ longitude: 0, latitude: 0 }, null)).toBe("has-draft");
      expect(useLand.getState()).toMatchObject({ mode, points: SKETCH, draft: null });
      expect(useUi.getState().activePanel).toBe("land");
      useLand.getState().cancel();
    }
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

  it("keeps an outline still being drawn", () => {
    sketch();
    expect(proposeOutline(DRAFT, null)).toBe("has-draft");
    expect(useLand.getState()).toMatchObject({ mode: "draw", points: SKETCH, draft: null });
  });
});

describe("drawOutline", () => {
  it("starts a new drawing, never a revision of the active land, with the drawing guides", () => {
    const scene = fakeScene();
    useLand.getState().select(SAVED);
    useLandContext.getState().setLayer({
      id: "drawing-guides",
      title: "Drawing guides",
      features: [],
      drawingGuideSources: [{ method: "mapped-feature", label: "Fixture fence" }],
    });
    expect(drawOutline(scene as unknown as CesiumSceneManager)).toBe("started");
    expect(scene.areas.cancelPick).toHaveBeenCalled();
    expect(useLand.getState()).toMatchObject({ active: null, mode: "draw", points: [] });
    expect(useLand.getState().traceSources[0]?.label).toBe("Fixture fence");
    expect(useUi.getState().activePanel).toBe("land");
  });

  it("keeps an unsaved outline", () => {
    useLand.getState().propose(DRAFT);
    expect(drawOutline(null)).toBe("has-draft");
    expect(useLand.getState().draft).toEqual(DRAFT);
  });
});

describe("openLand", () => {
  it("selects the land on its Overview lens, whichever lens was open before", () => {
    useLens.getState().setLens("assets");
    useLandContext.getState().setSection("inventory");
    expect(openLand(SAVED)).toBe("opened");
    expect(useLand.getState().active).toEqual(SAVED);
    expect(useLens.getState().lens).toBe("overview");
    // Views saved before lenses, and Show Once, still read the section.
    expect(useLandContext.getState().section).toBe("discover");
    expect(useUi.getState().activePanel).toBe("land");
  });

  it("opens on another lens when asked", () => {
    expect(openLand(SAVED, "plans")).toBe("opened");
    expect(useLens.getState().lens).toBe("plans");
    expect(useLandContext.getState().section).toBe("actions");
  });

  it("does not drop an unsaved outline to select another land", () => {
    useLand.getState().propose(DRAFT);
    useLens.getState().setLens("water");
    expect(openLand(SAVED, "plans")).toBe("has-draft");
    expect(useLens.getState().lens).toBe("water");
    expect(useLand.getState().draft).toEqual(DRAFT);
    expect(useLand.getState().active).toBeNull();
  });

  it("does not drop an outline still being drawn to select another land", () => {
    sketch();
    expect(openLand(SAVED)).toBe("has-draft");
    expect(useLand.getState()).toMatchObject({ mode: "draw", points: SKETCH, active: null });
  });
});
