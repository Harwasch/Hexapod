import { Cartographic, sampleTerrainMostDetailed } from "cesium";
import { useEffect, useState } from "react";

import { destination, bearing, type CoverageLane } from "@twin/geo";

import { useScene } from "@/cesium/SceneContext";
import {
  routeStats,
  SCOUT_ALTITUDES_M,
  SCOUT_CAMERAS,
  scoutGeometry,
  scoutLanes,
  scoutOf,
  USABLE_BATTERY_MIN,
  type Scout,
} from "@/missions/scout";
import type { Zone } from "@/missions/types";
import { buildScoutKmz } from "@/missions/wpml";
import { useLand } from "@/state/land";

/** Ground samples along the route for the relief check: lane ends and every 60 m between. */
const RELIEF_STEP_M = 60;
const RELIEF_MAX_SAMPLES = 150;
/** How long each source may take before the next is asked, ms. */
const RELIEF_WAIT_MS = 10_000;

function reliefPoints(lanes: readonly CoverageLane[]): { longitude: number; latitude: number }[] {
  const points: { longitude: number; latitude: number }[] = [];
  for (const lane of lanes) {
    points.push(lane.a);
    const heading = bearing(lane.a, lane.b);
    for (let d = RELIEF_STEP_M; d < lane.lengthM; d += RELIEF_STEP_M)
      points.push(destination(lane.a, heading, d));
    points.push(lane.b);
  }
  if (points.length <= RELIEF_MAX_SAMPLES) return points;
  const stride = points.length / RELIEF_MAX_SAMPLES;
  return points.filter((_, i) => Math.floor(i / stride) !== Math.floor((i - 1) / stride));
}

interface Relief {
  /** Ground under the first waypoint, where the take-off is assumed. */
  start: number;
  min: number;
  max: number;
  /** What was sampled: the drawn 3D surface (trees and roofs) or the bare terrain model. */
  source: "surface" | "terrain";
}

function within<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return Promise.race([
    promise,
    new Promise<undefined>((resolve) => window.setTimeout(() => resolve(undefined), ms)),
  ]);
}

function reliefOf(
  samples: readonly ({ height?: number } | undefined)[] | undefined,
  source: Relief["source"],
): Relief | null {
  if (!samples) return null;
  const heights = samples
    .map((sample) => sample?.height)
    .filter((h): h is number => typeof h === "number" && Number.isFinite(h));
  const start = samples[0]?.height;
  if (heights.length < 2 || typeof start !== "number" || !Number.isFinite(start)) return null;
  return { start, min: Math.min(...heights), max: Math.max(...heights), source };
}

/**
 * The ground under the route: the drawn surface (Google's 3D tiles, trees and roofs
 * included) when it answers in time, else the terrain model (bare ground). Null while
 * sampling; "unknown" when neither answers.
 */
function useRelief(lanes: readonly CoverageLane[]): Relief | null | "unknown" {
  const scene = useScene();
  const key = lanes.map((l) => `${l.a.longitude.toFixed(6)},${l.a.latitude.toFixed(6)}`).join(";");
  const [sampled, setSampled] = useState<{ key: string; relief: Relief | "unknown" } | null>(null);
  const ready = Boolean(scene) && lanes.length > 0;
  useEffect(() => {
    if (!scene || !ready) return;
    let live = true;
    const points = reliefPoints(lanes).map((p) =>
      Cartographic.fromDegrees(p.longitude, p.latitude),
    );
    const run = async () => {
      let relief: Relief | null = null;
      if (scene.scene.sampleHeightSupported) {
        const surface = await within(
          scene.scene.sampleHeightMostDetailed(points.map((p) => p.clone())).catch(() => undefined),
          RELIEF_WAIT_MS,
        );
        relief = reliefOf(surface, "surface");
      }
      if (!relief) {
        const terrain = await within(
          sampleTerrainMostDetailed(
            scene.viewer.terrainProvider,
            points.map((p) => p.clone()),
          ).catch(() => undefined),
          RELIEF_WAIT_MS,
        );
        relief = reliefOf(terrain, "terrain");
      }
      if (live) setSampled({ key, relief: relief ?? "unknown" });
    };
    void run();
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the route's own key
  }, [scene, ready, key]);
  if (!ready) return "unknown";
  return sampled?.key === key ? sampled.relief : null;
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
 * it from the 3D surface, and the warnings that go with flying it. The numbers are computed
 * here from the same lanes the map draws (`missions/scout.ts`); the plan card's machine-hours
 * come from the planner flying the same capture. `onAltitude` (a draft) changes the patches'
 * height; an approved plan exports what it stored.
 */
export function ScoutExport({
  zones,
  onAltitude,
}: {
  zones: readonly Zone[];
  onAltitude?: (altitudeM: number) => void;
}) {
  const landName = useLand((s) => s.active?.name ?? null);
  const scouts = zones.filter((z) => z.scout);
  const first = scouts[0];
  const scout: Scout = scoutOf(first?.scout);
  const geometry = scoutGeometry(scout);
  // The same lanes the map draws for these areas (a few hundred at most: cheap per render).
  const routes = scouts.map((z) => scoutLanes(z.footprint, scout));
  const stats = routeStats(routes, scout);
  const relief = useRelief(routes.flat());
  const [error, setError] = useState<string | null>(null);
  if (!first) return null;
  const camera = SCOUT_CAMERAS[scout.camera];
  const download = () => {
    setError(null);
    try {
      const bytes = buildScoutKmz(routes, scout);
      const blob = new Blob([bytes as BlobPart], { type: "application/vnd.google-earth.kmz" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `scout-${slug(landName ?? first.name)}-${new Date().toISOString().slice(0, 10)}.kmz`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The route could not be written.");
    }
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
                ? "The surface under the route (Google 3D tiles, trees and roofs included)"
                : "The terrain under the route (bare ground; trees and roofs not included)";
            return `${what} rises ${Math.round(rise)} m and falls ${Math.round(fall)} m from the first waypoint. Taking off there: ${(finest * 100).toFixed(1)} to ${(coarsest * 100).toFixed(1)} cm per pixel, lowest clearance ${Math.round(clearance)} m.`;
          })();
  const tooLow =
    relief && relief !== "unknown" && geometry.altitudeM - (relief.max - relief.start) < 30;
  return (
    <section className="mc-scout" data-testid="scout-export" aria-label="Scout route">
      <span className="mc-eyebrow">SCOUT ROUTE</span>
      <p className="mc-muted">{camera.label}, flying lines with the camera straight down.</p>
      {onAltitude && (
        <div className="mc-tags" role="group" aria-label="Height above take-off">
          {SCOUT_ALTITUDES_M.map((height) => (
            <button
              key={height}
              type="button"
              className={`mc-tag ${height === geometry.altitudeM ? "is-on" : ""}`}
              aria-pressed={height === geometry.altitudeM}
              onClick={() => onAltitude(height)}
            >
              {height} m
            </button>
          ))}
        </div>
      )}
      <dl className="mc-scout__facts mc-mono" data-testid="scout-facts">
        <dt>Height</dt>
        <dd>{Math.round(geometry.altitudeM)} m above take-off</dd>
        <dt>Detail</dt>
        <dd>{(geometry.gsdM * 100).toFixed(1)} cm per pixel over flat ground</dd>
        <dt>Lines</dt>
        <dd>
          {Math.round(geometry.lineSpacingM)} m apart ({Math.round(scout.sideOverlap * 100)} % side
          overlap)
        </dd>
        <dt>Photos</dt>
        <dd>
          every {geometry.photoSpacingM.toFixed(1)} m ({Math.round(scout.forwardOverlap * 100)} %
          forward overlap), about {stats.photos.toLocaleString()}
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
        <button type="button" className="mc-btn" onClick={download} data-testid="scout-download">
          Download DJI route (KMZ)
        </button>
      </div>
    </section>
  );
}
