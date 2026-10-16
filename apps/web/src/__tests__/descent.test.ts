/**
 * The Descent (features/descent/descent.ts): typed coordinates, and the one flight from orbit
 * down to a place that two better answers re-point on the way -- the terrain at full detail and
 * the mapped OpenStreetMap area -- without a second take-off, and without ever waiting on
 * either. The camera is faked; what it is asked to do is checked.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as Osm from "@/missions/osm";

import type { Footprint } from "@twin/contracts";

import { env } from "@/app/env";
import type { GlideOptions } from "@/cesium/CameraController";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { GlidePose } from "@/cesium/glide";
import {
  DESCENT_LAYER_ID,
  descend,
  isLargeArea,
  looksLikeCoordinates,
  parseCoordinates,
  stopDescent,
  targetFromPlace,
} from "@/features/descent/descent";
import type { OsmCandidate } from "@/missions/osm";
import { useDescent } from "@/state/descent";
import { useLandContext } from "@/state/landContext";
import { useSettings } from "@/state/settings";

const osm = vi.hoisted(() => ({
  answer: null as null | (() => Promise<unknown>),
  calls: [] as { point: unknown; maxAcres: unknown; signal: AbortSignal | undefined }[],
}));
vi.mock("@/missions/osm", async (importOriginal) => ({
  ...(await importOriginal<typeof Osm>()),
  fetchOsmContaining: (point: unknown, maxAcres: unknown, signal?: AbortSignal) => {
    osm.calls.push({ point, maxAcres, signal });
    return osm.answer ? osm.answer() : Promise.resolve(null);
  },
}));

const FIELD: Footprint = {
  type: "Polygon",
  coordinates: [
    [
      [-120.61, 35.61],
      [-120.59, 35.61],
      [-120.59, 35.63],
      [-120.61, 35.63],
      [-120.61, 35.61],
    ],
  ],
};
const MAPPED: OsmCandidate = { id: "osm-w42", name: "Creek pasture", footprint: FIELD, acres: 960 };

interface Flight {
  pose: GlidePose;
  options: GlideOptions;
  retargets: GlidePose[];
  active: boolean;
}

/** A deferred value, for answers that arrive mid-flight. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function fakeScene(ground: Promise<number | undefined> = Promise.resolve(undefined)) {
  const flights: Flight[] = [];
  const camera = {
    glide: vi.fn((pose: GlidePose, options: GlideOptions) => {
      const flight: Flight = { pose, options, retargets: [], active: true };
      flights.push(flight);
      return {
        get active() {
          return flight.active;
        },
        retarget: (next: GlidePose) => {
          if (flight.active) flight.retargets.push(next);
        },
        cancel: () => {
          flight.active = false;
          options.onCancel?.();
        },
      };
    }),
    groundHeightAt: vi.fn(() => ground),
    rectangleArrival: vi.fn(
      (
        west: number,
        south: number,
        east: number,
        north: number,
        options: { groundHeight?: number } = {},
      ): GlidePose => ({
        longitude: (west + east) / 2,
        latitude: south - 0.01,
        height: 800,
        heading: 0,
        pitch: -38,
        ground:
          options.groundHeight === undefined
            ? { height: 0, measured: false }
            : { height: options.groundHeight, measured: true },
      }),
    ),
    footprintArrival: vi.fn(
      (_footprint: Footprint, options: { groundHeight?: number } = {}): GlidePose => ({
        longitude: -120.6,
        latitude: 35.6,
        height: 2400,
        heading: 0,
        pitch: -38,
        ground:
          options.groundHeight === undefined
            ? { height: 0, measured: false }
            : { height: options.groundHeight, measured: true },
      }),
    ),
    flyToRectangle: vi.fn(),
  };
  return { scene: { camera } as unknown as CesiumSceneManager, camera, flights };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  osm.answer = null;
  osm.calls = [];
  useDescent.getState().clear();
  useLandContext.getState().clear();
  useSettings.getState().set({ onboardingDismissed: false });
});
afterEach(() => {
  stopDescent();
  vi.restoreAllMocks();
});

describe("coordinates", () => {
  it("reads 'lat, lon' and 'lat lon' in decimal degrees", () => {
    expect(parseCoordinates("35.62, -120.60")).toMatchObject({
      latitude: 35.62,
      longitude: -120.6,
    });
    expect(parseCoordinates("35.62 -120.60")).toMatchObject({ latitude: 35.62, longitude: -120.6 });
    expect(parseCoordinates("34.698° N, 120.04° W")).toMatchObject({
      latitude: 34.698,
      longitude: -120.04,
    });
    const target = parseCoordinates("35.62, -120.60");
    expect(target?.label).toBe("35.62000° N, 120.60000° W");
    expect(target?.attribution).toBe("Coordinates you entered");
  });

  it("refuses what is out of range or not coordinates at all", () => {
    expect(parseCoordinates("95, -120.60")).toBeNull();
    expect(looksLikeCoordinates("95, -120.60")).toBe(true);
    expect(parseCoordinates("35.62, -190")).toBeNull();
    expect(parseCoordinates("Los Olivos, CA")).toBeNull();
    expect(looksLikeCoordinates("Los Olivos, CA")).toBe(false);
  });

  it("takes a geocoder's box with its centre, and tells a town from land", () => {
    const town = targetFromPlace(
      {
        label: "Santa Ynez, CA",
        destination: {
          kind: "rectangle",
          west: -120.15,
          south: 34.58,
          east: -120.03,
          north: 34.66,
        },
      },
      "Geocoding by Cesium ion",
    );
    expect(town.longitude).toBeCloseTo(-120.09);
    expect(town.attribution).toBe("Geocoding by Cesium ion");
    expect(isLargeArea(town)).toBe(true);
    const farm = targetFromPlace(
      {
        label: "1 Ranch Rd",
        destination: {
          kind: "rectangle",
          west: -120.041,
          south: 34.697,
          east: -120.039,
          north: 34.699,
        },
      },
      "x",
    );
    expect(isLargeArea(farm)).toBe(false);
  });
});

describe("the flight", () => {
  const target = parseCoordinates("35.62, -120.60");
  if (!target) throw new Error("fixture");

  it("flies a town to its box with a plain flight, and looks nothing up", () => {
    const { scene, camera } = fakeScene();
    descend(scene, {
      label: "Santa Ynez",
      longitude: -120.09,
      latitude: 34.62,
      rectangle: { west: -120.15, south: 34.58, east: -120.03, north: 34.66 },
      attribution: "x",
    });
    expect(camera.flyToRectangle).toHaveBeenCalledTimes(1);
    expect(camera.glide).not.toHaveBeenCalled();
    expect(osm.calls).toHaveLength(0);
    expect(useDescent.getState().mappedStatus).toBe("skipped");
  });

  it("leaves at once as one glide, and dismisses the welcome card", () => {
    const { scene, camera } = fakeScene();
    descend(scene, target);
    expect(camera.glide).toHaveBeenCalledTimes(1);
    expect(useDescent.getState().phase).toBe("flying");
    expect(useSettings.getState().onboardingDismissed).toBe(true);
    expect(osm.calls[0]?.maxAcres).toBe(5000);
  });

  it("re-aims at the measured ground when the terrain answers", async () => {
    const { scene, flights } = fakeScene(Promise.resolve(412));
    descend(scene, target);
    await settle();
    expect(flights).toHaveLength(1);
    expect(flights[0]?.retargets.at(-1)?.ground).toEqual({ height: 412, measured: true });
  });

  it("frames the mapped area once found, and draws it as not a legal parcel", async () => {
    const enabled = env.landExplorationEnabled;
    env.landExplorationEnabled = true;
    try {
      const ground = deferred<number | undefined>();
      const { scene, camera, flights } = fakeScene(ground.promise);
      osm.answer = () => Promise.resolve(MAPPED);
      descend(scene, target);
      await settle();
      expect(camera.footprintArrival).toHaveBeenCalledWith(FIELD, {});
      expect(flights[0]?.retargets.at(-1)?.height).toBe(2400);
      const layer = useLandContext.getState().layers[DESCENT_LAYER_ID];
      expect(layer?.title).toContain("not a legal parcel");
      expect(layer?.features[0]?.label).toBe("Creek pasture");
      expect(useDescent.getState()).toMatchObject({ mappedStatus: "found", mapped: MAPPED });
      // The terrain answers after the mapped area: the area stays the frame, over measured ground.
      ground.resolve(300);
      await settle();
      expect(camera.footprintArrival).toHaveBeenLastCalledWith(FIELD, { groundHeight: 300 });
      expect(flights).toHaveLength(1);
    } finally {
      env.landExplorationEnabled = enabled;
    }
  });

  it("does not draw the outline without the Land tool", async () => {
    const { scene } = fakeScene();
    osm.answer = () => Promise.resolve(MAPPED);
    descend(scene, target);
    await settle();
    expect(useLandContext.getState().layers[DESCENT_LAYER_ID]).toBeUndefined();
    expect(useDescent.getState().mappedStatus).toBe("found");
  });

  it("stops correcting a flight that a hand on the map has taken", async () => {
    const ground = deferred<number | undefined>();
    const mapped = deferred<OsmCandidate | null>();
    const { scene, camera, flights } = fakeScene(ground.promise);
    osm.answer = () => mapped.promise;
    descend(scene, target);
    const flight = flights[0];
    if (!flight) throw new Error("no flight");
    flight.active = false;
    flight.options.onCancel?.();
    expect(useDescent.getState().phase).toBe("cancelled");
    ground.resolve(300);
    mapped.resolve(MAPPED);
    await settle();
    expect(flight.retargets).toHaveLength(0);
    // The card still learns what is mapped there.
    expect(useDescent.getState().mappedStatus).toBe("found");
    expect(camera.glide).toHaveBeenCalledTimes(1);
  });

  it("stops the first descent's corrections when a second one starts", async () => {
    const ground = deferred<number | undefined>();
    const { scene, flights } = fakeScene(ground.promise);
    descend(scene, target);
    const first = flights[0];
    expect(osm.calls[0]?.signal?.aborted).toBe(false);
    descend(scene, { ...target, latitude: 36 });
    expect(osm.calls[0]?.signal?.aborted).toBe(true);
    // The second glide replacing the first does not mark the second as cancelled.
    first?.options.onCancel?.();
    expect(useDescent.getState().phase).toBe("flying");
    ground.resolve(300);
    await settle();
    expect(first?.retargets).toHaveLength(0);
  });

  it("arrives with reduced motion at once", () => {
    const { scene, flights } = fakeScene();
    descend(scene, target, { reducedMotion: true });
    expect(flights[0]?.options.durationS).toBeLessThanOrEqual(0.01);
    flights[0]?.options.onComplete?.();
    expect(useDescent.getState().phase).toBe("arrived");
  });

  it("says when OpenStreetMap could not be reached, and never throws", async () => {
    const { scene } = fakeScene();
    osm.answer = () => Promise.reject(new Error("OpenStreetMap lookup failed (504)"));
    descend(scene, target);
    await settle();
    expect(useDescent.getState().mappedStatus).toBe("error");
    expect(useDescent.getState().error).toContain("504");
  });
});
