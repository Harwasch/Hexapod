import { useMemo, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import { X } from "lucide-react";

import type { LandArea, LandMapGeometry } from "@twin/contracts";
import type { UnitSystem } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { describeError } from "@/lib/log";
import { useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";
import { useSettings } from "@/state/settings";

import { cancelResearch } from "../../researchApi";
import { vegetationPeriods } from "../../vegetationPeriods";
import { LensCard, LensEmpty } from "../LensCard";
import { MapLink, type MapTarget } from "../mapLinks";
import { useStartAnalysis } from "../useLandAnalysis";
import { useLensPins } from "../useLensMap";
import { OverviewDetails } from "./overview/OverviewDetails";
import { OverviewLegend } from "./overview/OverviewLegend";
import {
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
  OVERVIEW_SPECS,
  overviewHeadline,
  percentLabel,
  presentClasses,
  rasterRequest,
  shareLabel,
  slopePercent,
  slopeSummary,
  slopeValue,
  type OverviewKey,
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
 * greenness.
 */
export function OverviewLens({ land }: { land: LandArea }) {
  const units = useSettings((s) => s.units);
  const scope = useLandScope();
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
        : undefined;
    return starter.start({
      title: spec.title,
      kind: "raster",
      question: spec.question,
      analysis: rasterRequest(spec, months),
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

  const running = OVERVIEW_KEYS.filter((key) => analyses[key].running).map((key) => (
    <RunningLine
      key={key}
      text={OVERVIEW_SPECS[key].running}
      canStop={starter.canEdit}
      onStop={() => {
        const id = analyses[key].run?.id;
        if (id) void stop(id);
      }}
    />
  ));
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

  const hasData = [data.terrain, data.cover, data.vegetation].some(Boolean);
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
        {!terrain.running &&
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
        {problems}
      </LensCard>
    );
  }

  // Facts: one per layer this land has; each is also that layer's switch on the map. The
  // land's size is on the land chip above, so it is not repeated here.
  const elevation = bandOf(data.terrain, 1);
  const slope = slopeSummary(bandOf(data.terrain, 2));
  const classes = presentClasses(bandOf(data.cover, 1));
  const green = greenness(data.vegetation?.metadata.vegetation);
  const pressed = (layer: OverviewLayer) => shown?.view.layer === layer;
  const legend = (layer: OverviewLayer) =>
    shown?.view.layer === layer ? <OverviewLegend shown={shown} units={units} /> : null;

  // One next step at a time: a failure to retry, else the first analysis not yet run.
  const current = (key: OverviewKey) => {
    const raster = data[key];
    return raster ? !isStale(raster, land.revision) : false;
  };
  const failedKey = OVERVIEW_KEYS.find(
    (key) => analyses[key].failed && analyses[key].run && !current(key),
  );
  const nextKey = OVERVIEW_KEYS.find((key) => !data[key] && !analyses[key].running);
  const staleKeys = OVERVIEW_KEYS.filter((key) => {
    const raster = data[key];
    return raster && isStale(raster, land.revision) && !analyses[key].running;
  });
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
        {data.terrain && (
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
          >
            {legend("elevation")}
          </LayerFact>
        )}
        {data.terrain && (
          <LayerFact
            label="Slope"
            target={landTarget}
            pressed={pressed("slope")}
            disabled={!slope}
            onSelect={() => choose("slope")}
            value={slope ? slopeValue(slope) : "No samples at this grid size"}
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
        (data.terrain ?? data.cover) && (
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
      {problems}
      <OverviewDetails
        data={data}
        shown={shown}
        rasterLayerId={rasterLayerId}
        units={units}
        canEdit={starter.canEdit}
        busy={starter.busy || analyses.vegetation.running}
        onView={(view) => setView(land.id, view)}
        onCompare={(periods) => void run("vegetation", periods)}
        spotNote={visibleSpot?.interpretation ?? null}
      />
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

function RunningLine({
  text,
  canStop,
  onStop,
}: {
  text: string;
  canStop: boolean;
  onStop: () => void;
}) {
  return (
    <p className="overview-status" role="status" aria-live="polite">
      <span className="overview-spinner" aria-hidden="true" />
      <span>{text}</span>
      {canStop && (
        <button type="button" className="land-link" onClick={onStop}>
          Stop
        </button>
      )}
    </p>
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
