/**
 * The Descent: from the whole planet down to someone's land in one continuous flight.
 *
 * The onboarding card asks "Where is your land?" and geocodes the answer once, on submit
 * (Nominatim's usage policy forbids search-as-you-type; places.ts's usePlaceSearch is for the
 * command box). The flight then leaves at once for the best pose known -- a 700 m square around
 * the point, over whatever ground the globe has loaded -- as one glide (CameraController.glide,
 * glide.ts), and two better answers move its destination on the way, never a second take-off:
 *
 * 1. the terrain under the point at full detail (CameraController.groundHeightAt), so the
 *    arrival is framed over the real ground rather than a coarse tile or the ellipsoid;
 * 2. the OpenStreetMap area containing the point (fetchOsmContaining), whose outline is drawn
 *    as "a mapped area, not a legal parcel" and framed whole.
 *
 * Neither is waited for: a slow or failing Overpass never holds the camera, and its failure is
 * a line on the arrival card, never an exception. A town or a region (a geocoder box more than
 * 8 km across) is not one piece of land: it gets a plain flight to its box and no lookup.
 *
 * Pure helpers first (parseCoordinates, targetFromPlace), tested directly; `descend` runs the
 * flight against the scene manager's camera, so React never touches Cesium (ADR 0002).
 */
import { env } from "@/app/env";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { GlideHandle } from "@/cesium/CameraController";
import type { GeocodeResult } from "@/cesium/types";
import { describeError } from "@/lib/log";
import { fetchOsmContaining, type OsmCandidate } from "@/missions/osm";
import { useDescent, type DescentTarget } from "@/state/descent";
import { useLandContext } from "@/state/landContext";
import { useSettings } from "@/state/settings";

/** The land context layer the mapped outline is drawn in until it is adopted or closed. */
export const DESCENT_LAYER_ID = "descent-mapped-feature";
/** What the mapped outline is called on the map: what it is, and what it is not. */
export const DESCENT_LAYER_TITLE = "Mapped area from OpenStreetMap, not a legal parcel";
export const COORDINATES_ATTRIBUTION = "Coordinates you entered";
/** A geocoder box wider than this corner to corner (m) is a town or a region, not land. */
export const LARGE_AREA_M = 8_000;
/** The square framed around a point before anything better is known (m, side). */
const FIRST_BOX_M = 700;
/** Mapped areas larger than this (acres) are a county or a forest, not the land asked about. */
const MAX_MAPPED_ACRES = 5_000;

const METRES_PER_DEGREE = 111_320;

const COORDINATES =
  /^\s*([+-]?\d{1,3}(?:\.\d+)?)\s*°?\s*([NS])?\s*(?:,\s*|\s+)([+-]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])?\s*$/i;

/** Whether text is written as a latitude and a longitude, in range or not. */
export function looksLikeCoordinates(text: string): boolean {
  return COORDINATES.test(text);
}

/**
 * "lat, lon" or "lat lon" in decimal degrees (an optional ° and N/S, E/W), as a target; null
 * when the text is not coordinates or is out of range (latitude ±90, longitude ±180).
 */
export function parseCoordinates(text: string): DescentTarget | null {
  const match = COORDINATES.exec(text);
  if (!match) return null;
  const [, latText = "", ns, lonText = "", ew] = match;
  let latitude = Number(latText);
  let longitude = Number(lonText);
  if (ns && /s/i.test(ns)) latitude = -Math.abs(latitude);
  if (ew && /w/i.test(ew)) longitude = -Math.abs(longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return {
    label: formatDegrees(longitude, latitude),
    longitude,
    latitude,
    attribution: COORDINATES_ATTRIBUTION,
  };
}

/** "34.69800° N, 120.04000° W": five decimals is about a metre. */
export function formatDegrees(longitude: number, latitude: number): string {
  const lat = `${Math.abs(latitude).toFixed(5)}° ${latitude < 0 ? "S" : "N"}`;
  const lon = `${Math.abs(longitude).toFixed(5)}° ${longitude < 0 ? "W" : "E"}`;
  return `${lat}, ${lon}`;
}

/** A geocoder's answer as a target: its point, or its box and the box's centre. */
export function targetFromPlace(place: GeocodeResult, attribution: string): DescentTarget {
  const d = place.destination;
  if (d.kind === "point")
    return {
      label: place.label,
      longitude: d.longitude,
      latitude: d.latitude,
      attribution: place.attribution ?? attribution,
    };
  return {
    label: place.label,
    longitude: (d.west + d.east) / 2,
    latitude: (d.south + d.north) / 2,
    rectangle: { west: d.west, south: d.south, east: d.east, north: d.north },
    attribution: place.attribution ?? attribution,
  };
}

/** Distance between two points on the sphere (m): enough to tell a field from a town. */
function distanceM(lon1: number, lat1: number, lon2: number, lat2: number): number {
  const toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad;
  const dLon = (lon2 - lon1) * toRad;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6_371_008 * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Whether a target is a town or a region rather than a piece of land. */
export function isLargeArea(target: DescentTarget): boolean {
  const r = target.rectangle;
  return Boolean(r && distanceM(r.west, r.south, r.east, r.north) > LARGE_AREA_M);
}

/** A square `sideM` across, centred on a point (degrees). */
export function boxAround(
  longitude: number,
  latitude: number,
  sideM: number,
): { west: number; south: number; east: number; north: number } {
  const dLat = sideM / 2 / METRES_PER_DEGREE;
  const dLon =
    sideM / 2 / (METRES_PER_DEGREE * Math.max(Math.cos((latitude * Math.PI) / 180), 0.01));
  return {
    west: Math.max(-180, longitude - dLon),
    south: Math.max(-90, latitude - dLat),
    east: Math.min(180, longitude + dLon),
    north: Math.min(90, latitude + dLat),
  };
}

/** The descent under way: a new one, or the app closing, stops its lookups and corrections. */
let current: AbortController | null = null;

/** Stops the descent under way, if any: its lookups, and any further move of the camera. */
export function stopDescent(): void {
  current?.abort();
  current = null;
}

export interface DescendOptions {
  /** The person asked for less motion: arrive at once rather than fly. */
  reducedMotion?: boolean;
}

/** Takes the mapped outline off the map (adopted as a land, or the card closed). */
export function removeMappedPreview(): void {
  if (useLandContext.getState().layers[DESCENT_LAYER_ID])
    useLandContext.getState().removeLayer(DESCENT_LAYER_ID);
}

/**
 * Flies from wherever the camera is down to `target`, and looks up the mapped area there on the
 * way (see the module comment). Returns once the flight has started and its lookups are under
 * way; the arrival card follows `useDescent`. A second call replaces the first.
 */
export function descend(
  scene: CesiumSceneManager,
  target: DescentTarget,
  options: DescendOptions = {},
): void {
  stopDescent();
  removeMappedPreview();
  const controller = new AbortController();
  current = controller;
  const { signal } = controller;
  const store = useDescent.getState();
  store.start(target);
  // The welcome card has done its job; the map is the place now.
  useSettings.getState().set({ onboardingDismissed: true });
  const arrive = () => {
    if (!signal.aborted) useDescent.getState().arrive();
  };
  const cancel = () => {
    if (!signal.aborted) useDescent.getState().cancel();
  };
  const camera = scene.camera;

  if (isLargeArea(target) && target.rectangle) {
    const { west, south, east, north } = target.rectangle;
    useDescent.getState().setMapped(null, "skipped");
    camera.flyToRectangle(west, south, east, north, {
      durationS: options.reducedMotion ? 0.01 : undefined,
      onComplete: arrive,
      onCancel: cancel,
    });
    return;
  }

  const { longitude, latitude } = target;
  const box = boxAround(longitude, latitude, FIRST_BOX_M);
  const handle: GlideHandle = camera.glide(
    camera.rectangleArrival(box.west, box.south, box.east, box.north),
    { durationS: options.reducedMotion ? 0.01 : undefined, onComplete: arrive, onCancel: cancel },
  );
  let ground: number | undefined;
  let mapped: OsmCandidate | null = null;
  // The better of what is known: the mapped area once found, else the first square; over the
  // measured ground once sampled.
  const steer = () => {
    if (signal.aborted || !handle.active) return;
    const options = ground === undefined ? {} : { groundHeight: ground };
    handle.retarget(
      mapped
        ? camera.footprintArrival(mapped.footprint, options)
        : camera.rectangleArrival(box.west, box.south, box.east, box.north, options),
    );
  };

  void camera.groundHeightAt(longitude, latitude).then((height) => {
    if (height === undefined || signal.aborted) return;
    ground = height;
    steer();
  });

  fetchOsmContaining({ longitude, latitude }, MAX_MAPPED_ACRES, signal)
    .then((found) => {
      if (signal.aborted) return;
      if (!found) {
        useDescent.getState().setMapped(null, "none");
        return;
      }
      mapped = found;
      useDescent.getState().setMapped(found, "found");
      if (env.landExplorationEnabled)
        useLandContext.getState().setLayer({
          id: DESCENT_LAYER_ID,
          title: DESCENT_LAYER_TITLE,
          features: [{ id: found.id, label: found.name, geometry: found.footprint }],
        });
      steer();
    })
    .catch((error: unknown) => {
      if (signal.aborted) return;
      useDescent.getState().setMapped(null, "error", describeError(error));
    });
}
