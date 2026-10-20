import { useQuery } from "@tanstack/react-query";
import { Cartographic, sampleTerrainMostDetailed } from "cesium";
import { useState } from "react";

import type { Footprint } from "@twin/contracts";
import type { LonLat } from "@twin/geo";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { useScene } from "@/cesium/SceneContext";
import {
  kmzRefusal,
  reliefFrom,
  reliefOutlines,
  reliefPoints,
  SCOUT_ALTITUDES_M,
  SCOUT_CAMERAS,
  scoutGeometry,
  scoutOf,
  scoutWaylines,
  sharedScout,
  USABLE_BATTERY_MIN,
  waylineStats,
  type Relief,
} from "@/missions/scout";
import type { Zone } from "@/missions/types";
import { useLand } from "@/state/land";

/** How long each source may take before the next is asked, ms. */
const RELIEF_WAIT_MS = 10_000;

function within<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return Promise.race([
    promise,
    new Promise<undefined>((resolve) => window.setTimeout(() => resolve(undefined), ms)),
  ]);
}

interface Sampled {
  points: LonLat[];
  heights: (number | undefined)[];
  source: Relief["source"];
}

function heightsOf(
  samples: readonly ({ height?: number } | undefined)[] | undefined,
): (number | undefined)[] | null {
  if (!samples) return null;
  const heights = samples.map((sample) => sample?.height);
  return heights.filter((h) => typeof h === "number" && Number.isFinite(h)).length >= 2
    ? heights
    : null;
}

/**
 * Heights under `points`: the drawn surface (Google's 3D tiles, trees and roofs included) when
 * it answers in time, else the terrain model (bare ground); "unknown" when neither answers.
 */
async function sampleRelief(
  scene: CesiumSceneManager,
  points: LonLat[],
): Promise<Sampled | "unknown"> {
  const cartographics = points.map((p) => Cartographic.fromDegrees(p.longitude, p.latitude));
  if (scene.scene.sampleHeightSupported) {
    const surface = await within(
      scene.scene
        .sampleHeightMostDetailed(cartographics.map((p) => p.clone()))
        .catch(() => undefined),
      RELIEF_WAIT_MS,
    );
    const heights = heightsOf(surface);
    if (heights) return { points, heights, source: "surface" };
  }
  const terrain = await within(
    sampleTerrainMostDetailed(
      scene.viewer.terrainProvider,
      cartographics.map((p) => p.clone()),
    ).catch(() => undefined),
    RELIEF_WAIT_MS,
  );
  const heights = heightsOf(terrain);
  return heights ? { points, heights, source: "terrain" } : "unknown";
}

/**
 * The ground under the scout areas, sampled once per set of outlines (`reliefPoints`: a fixed,
 * bounded set of points) and kept in the query cache, so a new height or line spacing, the
 * card redrawing while the plan redrafts, or reopening the plan reuse it: the most detailed
 * surface pulls billed photorealistic tiles, and its loading cannot be cancelled. A sample
 * that found nothing is asked again next time. Null while sampling; "unknown" when neither
 * source answers.
 */
function useReliefSamples(footprints: readonly Footprint[]): Sampled | null | "unknown" {
  const scene = useScene();
  const key = JSON.stringify(footprints);
  const ready = Boolean(scene) && footprints.length > 0;
  const query = useQuery({
    queryKey: ["scout-relief", key],
    queryFn: () =>
      scene ? sampleRelief(scene, reliefPoints(footprints)) : Promise.resolve("unknown" as const),
    enabled: ready,
    staleTime: (cached) => (cached.state.data === "unknown" ? 0 : Infinity),
    gcTime: 30 * 60_000,
    retry: false,
  });
  if (!ready) return "unknown";
  return query.data ?? null;
}

function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 40) || "land"
  );
}

function km(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}

/**
 * A scout route for the fog patches in a draft or plan, as a DJI Pilot 2 / FlightHub 2 KMZ:
 * the aircraft and camera it assumes, the height above take-off, what that gives (ground
 * detail, line and photo spacing), the route's length, time and batteries, the relief under
 * it from the 3D surface, and the warnings that go with flying it. `zones` come in the order
 * the map routes them (`scoutZonesInOrder`), and each is flown as its own capture's pattern
 * asks (`scoutWaylines`: nadir lanes, a re-scan's crossing obliques or orbit, exactly the
 * passes `MissionManager.showPlan` draws, and the route's length, time and photos with them);
 * the route is one capture, so areas whose
 * heights or overlaps differ are not exported together. The plan card's machine-hours come
 * from the planner flying the same capture. `onAltitude` (a draft) changes the patches'
 * height; an approved plan exports what it stored.
 *
 * Loaded only with drone missions on (`ScoutExport`, `VITE_ENABLE_DRONE_EXPORT`): see
 * docs/DRONES.md before turning that on.
 */
export function DroneScoutExport({
  zones,
  onAltitude,
}: {
  zones: readonly Zone[];
  onAltitude?: (altitudeM: number) => void;
}) {
  const landName = useLand((s) => s.active?.name ?? null);
  const scouts = zones.filter((z) => z.scout);
  const first = scouts[0];
  const shared = sharedScout(scouts);
  const scout = shared ?? scoutOf(first?.scout);
  const geometry = scoutGeometry(scout);
  // What the KMZ flies, area by area as its pattern asks: a nadir area the lanes the map
  // draws, an oblique re-scan those and a crossing set, an orbit its rings (a few hundred
  // waypoints at most: cheap per render).
  const areas = scouts.map((z) => ({ footprint: z.footprint, scout: scoutOf(z.scout) }));
  const waylines = scoutWaylines(areas);
  const stats = waylineStats(waylines, scout);
  // The ground under the areas and under any orbit, which flies outside its area.
  const samples = useReliefSamples(reliefOutlines(areas));
  const firstWaypoint = waylines[0]?.waypoints[0] ?? null;
  const relief =
    samples === null || samples === "unknown"
      ? samples
      : firstWaypoint
        ? (reliefFrom(samples.points, samples.heights, firstWaypoint, samples.source) ?? "unknown")
        : "unknown";
  const [error, setError] = useState<string | null>(null);
  if (!first) return null;
  const refusal = kmzRefusal(scouts);
  const camera = SCOUT_CAMERAS[scout.camera];
  const download = () => {
    setError(null);
    if (refusal) {
      setError(refusal);
      return;
    }
    // The writer is needed only for this click: it loads then, not with the globe.
    import("@/missions/wpml")
      .then(({ buildScoutKmz }) => {
        const bytes = buildScoutKmz(waylines, scout);
        const blob = new Blob([bytes as BlobPart], { type: "application/vnd.google-earth.kmz" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `scout-${slug(landName ?? first.name)}-${new Date().toISOString().slice(0, 10)}.kmz`;
        anchor.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      })
      .catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "The route could not be written.");
      });
  };
  const reliefLine =
    relief === null
      ? "Sampling the ground under the route…"
      : relief === "unknown"
        ? "Relief unknown: the ground under the route could not be sampled. Check how it rises from the take-off point before flying."
        : (() => {
            const rise = relief.max - relief.start;
            const fall = relief.start - relief.min;
            const clearance = geometry.altitudeM - rise;
            const finest = (geometry.gsdM * Math.max(clearance, 1)) / geometry.altitudeM;
            const coarsest = (geometry.gsdM * (geometry.altitudeM + fall)) / geometry.altitudeM;
            const what =
              relief.source === "surface"
                ? "The surface under the patches (Google 3D tiles, trees and roofs included)"
                : "The terrain under the patches (bare ground; trees and roofs not included)";
            return `${what} rises ${Math.round(rise)} m and falls ${Math.round(fall)} m from the ground at the first waypoint. Taking off there: ${(finest * 100).toFixed(1)} to ${(coarsest * 100).toFixed(1)} cm per pixel, lowest clearance ${Math.round(clearance)} m.`;
          })();
  const tooLow =
    relief && relief !== "unknown" && geometry.altitudeM - (relief.max - relief.start) < 30;
  return (
    <section className="mc-scout" data-testid="scout-export" aria-label="Scout route">
      <span className="mc-eyebrow">SCOUT ROUTE</span>
      <p className="mc-muted">
        {camera.label},{" "}
        {waylines.every((w) => !w.poi && w.gimbalPitchDeg === -90)
          ? "flying lines with the camera straight down."
          : "flying each area as it asks: lines straight down, crossing obliques or an orbit."}
      </p>
      {onAltitude && (
        <div className="mc-tags" role="group" aria-label="Height above take-off">
          {SCOUT_ALTITUDES_M.map((height) => (
            <button
              key={height}
              type="button"
              className={`mc-tag ${shared && height === geometry.altitudeM ? "is-on" : ""}`}
              aria-pressed={Boolean(shared) && height === geometry.altitudeM}
              onClick={() => onAltitude(height)}
            >
              {height} m
            </button>
          ))}
        </div>
      )}
      {!shared && (
        <p className="mc-note mc-note--warn" role="status" data-testid="scout-mixed">
          These patches are set to different heights or overlaps, and one route flies one.{" "}
          {onAltitude
            ? "Pick a height above to export them together."
            : "Revise the plan to give them one height, then export it."}
        </p>
      )}
      {refusal && (
        <p className="mc-note mc-note--warn" role="status" data-testid="scout-unsupported">
          {refusal}
        </p>
      )}
      {shared && !refusal && (
        <>
          <dl className="mc-scout__facts mc-mono" data-testid="scout-facts">
            <dt>Height</dt>
            <dd>{Math.round(geometry.altitudeM)} m above take-off</dd>
            <dt>Return</dt>
            <dd>
              flies home at {Math.round(geometry.returnHeightM)} m above take-off, at the end or if
              the controller link is lost
            </dd>
            <dt>Detail</dt>
            <dd>{(geometry.gsdM * 100).toFixed(1)} cm per pixel over flat ground</dd>
            <dt>Lines</dt>
            <dd>
              {Math.round(geometry.lineSpacingM)} m apart ({Math.round(scout.sideOverlap * 100)} %
              side overlap)
            </dd>
            <dt>Photos</dt>
            <dd>
              every {geometry.photoSpacingM.toFixed(1)} m ({Math.round(scout.forwardOverlap * 100)}{" "}
              % forward overlap), about {stats.photos.toLocaleString()}
            </dd>
            <dt>Route</dt>
            <dd>
              {km(stats.workingM + stats.transitM)} at {geometry.speedMs} m/s ·{" "}
              {Math.round(stats.minutes)} min · {stats.batteries} batter
              {stats.batteries === 1 ? "y" : "ies"} ({USABLE_BATTERY_MIN} min each, assumed)
            </dd>
          </dl>
          <p className="mc-muted" data-testid="scout-relief">
            {reliefLine}
          </p>
          <ul className="mc-list">
            {[
              ...geometry.warnings,
              ...(tooLow
                ? [
                    "The ground rises close to the flight height: fly higher or take off from higher ground.",
                  ]
                : []),
              "Heights are relative to the take-off point; the route does not follow terrain.",
              "Draft, not flight-tested: review it in DJI Pilot 2 or FlightHub 2 before flying.",
              "Check airspace (B4UFLY or LAANC) and fly under Part 107.",
            ].map((warning) => (
              <li key={warning} className="mc-note mc-note--warn">
                {warning}
              </li>
            ))}
          </ul>
          <p className="mc-muted">
            Computed here from the lanes on the map; the speed, overlaps and battery time are the
            assumed drone class&apos;s.
          </p>
          {error && <p role="alert">{error}</p>}
          <div className="mc-actions">
            <button
              type="button"
              className="mc-btn"
              onClick={download}
              data-testid="scout-download"
            >
              Download DJI route (KMZ)
            </button>
          </div>
        </>
      )}
    </section>
  );
}
