import { useMemo, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import { X } from "lucide-react";

import type { LandArea, LandMapGeometry, ResearchRun } from "@twin/contracts";
import { boundsOf, type UnitSystem } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";
import { useSettings } from "@/state/settings";

import { coversUs, signaturePeriods } from "../../rasterRuns";
import { analysisBlocker } from "../../researchAvailability";
import { cancelResearch } from "../../researchApi";
import { ResearchWorkerIdle, ResearchWorkerNotice } from "../../ResearchWorkerNotice";
import { vegetationPeriods } from "../../vegetationPeriods";
import { LensCard, LensEmpty } from "../LensCard";
import { MapLink, type MapTarget } from "../mapLinks";
import { useStartAnalysis } from "../useLandAnalysis";
import { useLensPins } from "../useLensMap";
import { OverviewAnalysis } from "./overview/OverviewAnalysis";
import { OverviewDetails } from "./overview/OverviewDetails";
import { OverviewLegend } from "./overview/OverviewLegend";
import {
  ALL_OVERVIEW_KEYS,
  bandOf,
  capitalise,
  className,
  defaultMonths,
  elevationPhrase,
  greenness,
  heightLabel,
  isStale,
  mainClasses,
  OVERVIEW_KEYS,
  OUTSIDE_US,
  OVERVIEW_SPECS,
  overviewHeadline,
  percentLabel,
  presentClasses,
  rasterRequest,
  shareLabel,
  slopePercent,
  slopeSummary,
  slopeValue,
  terrainKey,
  terrainNote,
  terrainOf,
  type OverviewKey,
  type OverviewSpec,
  type Period,
} from "./overview/overviewData";
import {
  resolveView,
  useOverview,
  useOverviewMap,
  useOverviewRasterLayer,
  useSpotReading,
  type OverviewLayer,
  type SpotReading,
} from "./overview/useOverview";
import "./overview/overview.css";

type Analysis = ReturnType<typeof useOverview>["analyses"][OverviewKey];

/**
 * Overview: what the land is like at a glance. Its terrain (height and slope), what covers it
 * and how green it is, read from saved rasters and drawn on the map one at a time. Analyses
 * start only when the person asks, one step at a time: terrain first, then land cover, then
 * greenness. Details offers bare-earth terrain (USGS 3DEP, which the terrain facts then read)
 * and the seasonal signatures Show once compares.
 */
export function OverviewLens({ land }: { land: LandArea }) {
  const units = useSettings((s) => s.units);
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const { analyses, data, loading, settling, error, retry } = useOverview(land);
  const starter = useStartAnalysis(land);
  const chosen = useOverviewMap((s) => s.views[land.id]);
  const setView = useOverviewMap((s) => s.setView);
  const shown = resolveView(chosen, data);
  const rasterLayerId = useOverviewRasterLayer(shown);
  const tileError = useLandContext((s) => (rasterLayerId ? s.rasterErrors[rasterLayerId] : null));
  const spot = useSpotReading(land, data);
  const [dismissed, setDismissed] = useState<number | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);
  const visibleSpot = spot && spot.at !== dismissed ? spot : null;
  const spotText = visibleSpot ? spotLabel(visibleSpot, units) : null;
  const landTarget = useMemo<MapTarget>(
    () => ({ key: "overview-land", geometry: land.boundary }),
    [land.boundary],
  );
  const [spotLon, spotLat] = visibleSpot?.position ?? [];
  const pins = useMemo<MapPin[] | null>(
    () =>
      spotLon != null && spotLat != null && spotText && !visibleSpot?.loading
        ? [{ id: "overview-spot", position: [spotLon, spotLat], label: spotText, tone: "accent" }]
        : null,
    [spotLon, spotLat, spotText, visibleSpot?.loading],
  );
  useLensPins("overview", pins);

  const run = (key: OverviewKey, periods?: Period[]) => {
    const spec = OVERVIEW_SPECS[key];
    const months =
      key === "vegetation"
        ? (periods ?? vegetationPeriods(defaultMonths()) ?? undefined)
        : key === "signature"
          ? (periods ?? signaturePeriods())
          : undefined;
    return starter.start({
      title: spec.title,
      kind: "raster",
      question: spec.question,
      analysis: rasterRequest(spec, months),
      // At least the processor's own limit and the save, or the worker stops it early.
      budget: { maxSeconds: spec.maxSeconds },
    });
  };
  const stop = async (runId: string) => {
    setStopError(null);
    try {
      await cancelResearch(runId);
    } catch (cause) {
      setStopError(`It could not be stopped. ${describeError(cause)}`);
    }
    await cache.invalidateQueries({ queryKey: ["land-research", scope] });
  };
  const choose = (layer: OverviewLayer) =>
    setView(land.id, shown?.view.layer === layer ? null : { layer });

  const running = ALL_OVERVIEW_KEYS.filter((key) => analyses[key].running).map((key) => (
    <RunningLine
      key={key}
      spec={OVERVIEW_SPECS[key]}
      run={analyses[key].run}
      canStop={starter.canEdit}
      onStop={() => {
        const id = analyses[key].run?.id;
        if (id) void stop(id);
      }}
    />
  ));
  // While one of its runs waits, that run's notice says what for; the idle note would repeat it.
  const anyQueued = ALL_OVERVIEW_KEYS.some((key) => analyses[key].run?.status === "queued");
  const idle = starter.canEdit && !anyQueued ? <ResearchWorkerIdle /> : null;
  const terrainBusy = analyses.terrain.running || analyses.bareEarth.running;
  const { west, south, east, north } = boundsOf(land.boundary);
  const inUs = coversUs((west + east) / 2, (south + north) / 2);
  const bareEarth = (
    <OverviewAnalysis
      spec="bareEarth"
      analysis={analyses.bareEarth}
      raster={data.bareEarth}
      revision={land.revision}
      ready={ready}
      canEdit={canEdit}
      busy={starter.busy}
      unavailable={inUs ? null : OUTSIDE_US}
      onStart={() => void run("bareEarth")}
      testId="overview-bare-earth"
    >
      <p className="land-muted">
        Ground height and slope at 10 m from the USGS 3DEP model, with trees and buildings left out.
        The terrain facts use it once it is ready. United States only.
      </p>
    </OverviewAnalysis>
  );
  const signatures = (
    <OverviewAnalysis
      spec="signature"
      analysis={analyses.signature}
      raster={data.signature}
      revision={land.revision}
      ready={ready}
      canEdit={canEdit}
      busy={starter.busy}
      onStart={() => void run("signature")}
      testId="overview-signatures"
      titled={false}
    >
      <p className="land-muted">
        Four dated Sentinel-2 scenes, one per season: green, red, near-infrared, shortwave-infrared
        and NDVI for each 10 m cell clear that season. Show once in Ecology compares places by them.
        A similar signature means similar reflectance, not the same species. Takes a minute or two.
      </p>
      <p className="land-muted">
        {signaturePeriods()
          .map((period) => period.startDate.slice(0, 7))
          .join(" · ")}{" "}
        · 10 m grid
      </p>
    </OverviewAnalysis>
  );
  const details = (cardIdle: boolean) => (
    <OverviewDetails
      data={data}
      shown={shown}
      rasterLayerId={rasterLayerId}
      units={units}
      canEdit={starter.canEdit}
      busy={starter.busy}
      onView={(view) => setView(land.id, view)}
      onCompare={(periods) => void run("vegetation", periods)}
      compareBlocked={analysisBlocker({
        ready,
        canEdit,
        run: analyses.vegetation.run ?? undefined,
      })}
      spotNote={visibleSpot?.interpretation ?? null}
      bareEarth={bareEarth}
      signatures={signatures}
      idle={cardIdle ? null : idle}
    />
  );
  const problems = (
    <>
      {starter.error && (
        <p className="land-alert" role="alert">
          The analysis could not start. {starter.error}
        </p>
      )}
      {stopError && (
        <p className="land-alert" role="alert">
          {stopError}
        </p>
      )}
    </>
  );

  const ground = terrainOf(data);
  const hasData = [ground, data.cover, data.vegetation].some(Boolean);
  if (loading || (settling && !hasData))
    return (
      <LensCard lens="overview">
        <p className="land-muted" role="status">
          Loading the overview…
        </p>
      </LensCard>
    );

  if (!hasData) {
    if (error)
      return (
        <LensCard lens="overview">
          <p className="land-alert" role="alert">
            The overview could not be loaded. {describeError(error)}
          </p>
          <div className="land-row">
            <GlassButton size="sm" onClick={retry}>
              Try again
            </GlassButton>
          </div>
        </LensCard>
      );
    const terrain = analyses.terrain;
    return (
      <LensCard lens="overview">
        {running}
        {!terrainBusy &&
          (terrain.failed && terrain.run ? (
            <FailedLine
              analysis={terrain}
              spec="terrain"
              primary
              canRetry={starter.canEdit}
              busy={starter.busy}
              onRetry={() => void run("terrain")}
            />
          ) : (
            <LensEmpty
              action={
                starter.canEdit ? (
                  <GlassButton
                    variant="primary"
                    loading={starter.busy}
                    onClick={() => void run("terrain")}
                  >
                    {OVERVIEW_SPECS.terrain.start}
                  </GlassButton>
                ) : undefined
              }
            >
              {starter.canEdit
                ? "See how high and how steep this land is, drawn on the map."
                : "No overview yet. Someone who can edit this land can read its terrain."}
            </LensEmpty>
          ))}
        {!terrainBusy && idle}
        {problems}
        {details(!terrainBusy)}
      </LensCard>
    );
  }

  // Facts: one per layer this land has; each is also that layer's switch on the map. The
  // land's size is on the land chip above, so it is not repeated here. Height and slope name
  // their model: bare earth where this land has it, else the surface model.
  const elevation = bandOf(ground, 1);
  const slope = slopeSummary(bandOf(ground, 2));
  const groundNote = ground ? terrainNote(ground) : undefined;
  const classes = presentClasses(bandOf(data.cover, 1));
  const green = greenness(data.vegetation?.metadata.vegetation);
  const pressed = (layer: OverviewLayer) => shown?.view.layer === layer;
  const legend = (layer: OverviewLayer) =>
    shown?.view.layer === layer ? <OverviewLegend shown={shown} units={units} /> : null;

  // One next step at a time: a failure to retry, else the first analysis not yet run. Bare
  // earth, where this land has it, stands for the terrain step.
  const shownOf = (key: OverviewKey) => (key === "terrain" ? ground : data[key]);
  const current = (key: OverviewKey) => {
    const raster = shownOf(key);
    return raster ? !isStale(raster, land.revision) : false;
  };
  const failedKey = OVERVIEW_KEYS.find(
    (key) => analyses[key].failed && analyses[key].run && !current(key),
  );
  const nextKey = OVERVIEW_KEYS.find(
    (key) => !shownOf(key) && !(key === "terrain" ? terrainBusy : analyses[key].running),
  );
  const staleKeys = [terrainKey(data), "cover" as const, "vegetation" as const].filter((key) => {
    const raster = data[key];
    return raster && isStale(raster, land.revision) && !analyses[key].running;
  });
  const cardStarts = starter.canEdit && (Boolean(failedKey ?? nextKey) || staleKeys.length > 0);
  // A re-run compares the same months the earlier one did.
  const monthsOf = (key: OverviewKey) => {
    const request = analyses[key].run?.analysis;
    const asked = request && "periods" in request ? request.periods : undefined;
    return asked?.length ? asked : (data[key]?.request?.periods ?? undefined);
  };
  const rerun = async () => {
    for (const key of staleKeys) await run(key, monthsOf(key));
  };

  return (
    <LensCard lens="overview" headline={overviewHeadline(data, units)}>
      <ul className="land-facts">
        {ground && (
          <LayerFact
            label="Height"
            target={landTarget}
            pressed={pressed("elevation")}
            disabled={!elevation?.validCells}
            onSelect={() => choose("elevation")}
            value={
              elevation?.validCells && elevation.minimum != null && elevation.maximum != null
                ? capitalise(elevationPhrase(elevation.minimum, elevation.maximum, units))
                : "No samples at this grid size"
            }
            note={groundNote}
          >
            {legend("elevation")}
          </LayerFact>
        )}
        {ground && (
          <LayerFact
            label="Slope"
            target={landTarget}
            pressed={pressed("slope")}
            disabled={!slope}
            onSelect={() => choose("slope")}
            value={slope ? slopeValue(slope) : "No samples at this grid size"}
            note={groundNote}
          >
            {legend("slope")}
          </LayerFact>
        )}
        {data.cover && (
          <LayerFact
            label="Land cover"
            target={landTarget}
            pressed={pressed("cover")}
            disabled={!classes.length}
            onSelect={() => choose("cover")}
            value={
              classes.length ? (
                // On the map, every class found is listed: the fact is the legend.
                <span className="overview-classes">
                  {(pressed("cover") ? classes : mainClasses(classes)).map((item) => (
                    <span key={item.code} className="overview-class">
                      <span
                        className="overview-chip"
                        style={{ background: item.color }}
                        aria-hidden="true"
                      />
                      {className(item)} {shareLabel(item.fraction)}
                    </span>
                  ))}
                </span>
              ) : (
                "No classes sampled"
              )
            }
          />
        )}
        {data.vegetation && (
          <LayerFact
            label="Greenness"
            target={landTarget}
            pressed={pressed("vegetation")}
            onSelect={() => choose("vegetation")}
            value={green?.value ?? "No clear satellite view"}
            note={green?.note ?? undefined}
          >
            {legend("vegetation")}
          </LayerFact>
        )}
      </ul>
      {tileError && (
        <p className="land-alert" role="alert">
          {tileError}
        </p>
      )}
      {visibleSpot ? (
        <MapLink
          as="div"
          fly={false}
          className="overview-spot"
          target={{
            key: "overview-spot",
            geometry: {
              type: "Point",
              coordinates: visibleSpot.position,
            } as LandMapGeometry,
          }}
        >
          <span className="overview-spot__label">This spot</span>
          <span className="overview-spot__value" aria-live="polite">
            {visibleSpot.loading
              ? "Reading…"
              : visibleSpot.error
                ? `It could not be read. ${visibleSpot.error}`
                : spotText}
          </span>
          <button
            type="button"
            className="land-icon-button"
            aria-label="Clear this spot"
            onClick={() => setDismissed(visibleSpot.at)}
          >
            <X size={14} aria-hidden="true" />
          </button>
        </MapLink>
      ) : (
        (ground ?? data.cover) && (
          <p className="land-muted">
            Right-click the land and choose What&apos;s here to read a spot.
          </p>
        )
      )}
      {running}
      {starter.canEdit && !failedKey && !settling && nextKey && (
        <div className="land-row">
          <GlassButton size="sm" loading={starter.busy} onClick={() => void run(nextKey)}>
            {OVERVIEW_SPECS[nextKey].start}
          </GlassButton>
        </div>
      )}
      {failedKey && (
        <FailedLine
          analysis={analyses[failedKey]}
          spec={failedKey}
          canRetry={starter.canEdit}
          busy={starter.busy}
          onRetry={() => void run(failedKey, monthsOf(failedKey))}
        />
      )}
      {staleKeys.length > 0 && (
        <p className="land-note overview-stale">
          Made for an earlier outline of this land.{" "}
          {starter.canEdit && (
            <button
              type="button"
              className="land-link"
              disabled={starter.busy}
              onClick={() => void rerun()}
            >
              Run again
            </button>
          )}
        </p>
      )}
      {cardStarts && idle}
      {problems}
      {details(cardStarts)}
    </LensCard>
  );
}

function spotLabel(spot: SpotReading, units: UnitSystem): string {
  const parts: string[] = [];
  if (spot.elevation != null) parts.push(heightLabel(spot.elevation, units));
  if (spot.slope != null) parts.push(`${percentLabel(slopePercent(spot.slope))} slope`);
  if (spot.cover) parts.push(className(spot.cover));
  return parts.length ? parts.join(" · ") : "No data here";
}

/** A fact that is also the switch for its layer on the map; hovering it points at the land. */
function LayerFact({
  label,
  value,
  note,
  target,
  pressed,
  disabled,
  onSelect,
  children,
}: {
  label: string;
  value: ReactNode;
  note?: string;
  target: MapTarget;
  pressed: boolean;
  disabled?: boolean;
  onSelect: () => void;
  /** The legend, while this fact's layer is on the map. */
  children?: ReactNode;
}) {
  return (
    <MapLink
      as="li"
      target={target}
      fly={false}
      className={clsx("land-fact overview-fact", pressed && "is-on")}
    >
      <button
        type="button"
        className="overview-fact__button"
        aria-pressed={pressed}
        disabled={disabled}
        onClick={onSelect}
      >
        <span className="land-fact__label">{label}</span>
        {!disabled && (
          <span className="overview-fact__state" aria-hidden="true">
            {pressed ? "On map" : "Show"}
          </span>
        )}
        <span className="land-fact__value">{value}</span>
        {note && <span className="land-fact__note">{note}</span>}
      </button>
      {children}
    </MapLink>
  );
}

/**
 * A run on one line, with its way out: "Stop" while it runs, "Cancel queued request" while it
 * waits, and under it what it waits for (ResearchWorkerNotice, when the worker is known).
 */
function RunningLine({
  spec,
  run,
  canStop,
  onStop,
}: {
  spec: OverviewSpec;
  run: ResearchRun | null;
  canStop: boolean;
  onStop: () => void;
}) {
  const queued = run?.status === "queued";
  return (
    <>
      <p className="overview-status" role="status" aria-live="polite">
        <span className="overview-spinner" aria-hidden="true" />
        <span>{queued ? spec.queued : spec.running}</span>
        {canStop && (
          <button type="button" className="land-link" onClick={onStop}>
            {queued ? "Cancel queued request" : "Stop"}
          </button>
        )}
      </p>
      <ResearchWorkerNotice run={run ?? undefined} className="land-muted overview-worker" />
    </>
  );
}

function FailedLine({
  analysis,
  spec,
  primary,
  canRetry,
  busy,
  onRetry,
}: {
  analysis: Analysis;
  spec: OverviewKey;
  primary?: boolean;
  canRetry: boolean;
  busy: boolean;
  onRetry: () => void;
}) {
  const stopped = analysis.run?.status === "cancelled";
  const name = OVERVIEW_SPECS[spec].name;
  return (
    <div className="overview-failed">
      <p className="land-alert" role="alert">
        {stopped
          ? `${name} was stopped.`
          : `${OVERVIEW_SPECS[spec].failed} ${analysis.run?.error ?? "The source did not answer."}`}
      </p>
      {canRetry && (
        <div className="land-row">
          <GlassButton
            size={primary ? "md" : "sm"}
            variant={primary ? "primary" : "default"}
            loading={busy}
            onClick={onRetry}
          >
            {stopped ? "Start again" : "Try again"}
          </GlassButton>
        </div>
      )}
    </div>
  );
}
