import { useEffect, useMemo, useRef, type ReactNode } from "react";

import type { LandArea } from "@twin/contracts";

import type { LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";
import { useSettings } from "@/state/settings";

import { LensCard, LensDetails } from "../LensCard";
import { Fact, MapLink } from "../mapLinks";
import { useFlyTo } from "../mapHooks";
import { useLens } from "../lenses";
import { useLensLayer, useLensPins } from "../useLensMap";
import { SOURCE_SPECS, useOverviewBundle, type OverviewProvider } from "./overviewBundle";
import { firstKnown, drainageFact } from "./soil/soilWords";
import {
  AskButton,
  AskLinks,
  DataTable,
  OverviewEmpty,
  OverviewLoading,
  OverviewNotices,
  SourceList,
} from "./water/bundleStates";
import {
  areaShort,
  areaWords,
  floodHeadline,
  floodRisk,
  RISK_COLORS,
  RISK_LABELS,
  riskMeaning,
  SHOWN_RISKS,
  subtypeWords,
  summarizeFlood,
  type FloodRisk,
  type FloodSummary,
  type ShownRisk,
} from "./water/flood";
import { MONTH_NAMES, rainAmount, rainSummary, rainWords, temperature } from "./water/rain";
import { RainBars } from "./water/RainBars";
import "./water/water.css";

const SOURCES: readonly OverviewProvider[] = ["fema-flood-zones", "nasa-power", "usda-soils"];

const RISK_WORDS: Record<FloodRisk, string> = {
  coastal: "High, coastal",
  high: "High",
  moderate: "Moderate",
  minimal: "Minimal",
  undetermined: "Not studied",
  other: "Other",
};

const LEGEND: Record<ShownRisk, string> = {
  coastal: "Coastal flood zone",
  high: "Mapped flood zone",
  moderate: "Moderate risk",
  undetermined: "Not studied",
};

/** The flood layers: mapped (special hazard) zones drawn strongest, moderate risk lighter. */
function floodLayers(summary: FloodSummary | null) {
  const layer = (
    id: string,
    title: string,
    risks: ShownRisk[],
    fill: number,
    width: number,
  ): LandContextLayer | null => {
    const features = risks.flatMap((risk) =>
      (summary?.groups[risk]?.areas ?? []).map((area) => ({
        id: area.id,
        label: `Zone ${area.zone}`,
        geometry: area.geometry,
        value: area.areaM2,
        color: RISK_COLORS[risk],
      })),
    );
    const first = risks[0] ?? "high";
    return features.length
      ? { id, title, unit: "m²", features, style: { color: RISK_COLORS[first], fill, width } }
      : null;
  };
  return {
    hazard: layer("water-flood-zones", "Mapped flood zones", ["high", "coastal"], 0.42, 2.5),
    moderate: layer("water-flood-moderate", "Moderate flood risk", ["moderate"], 0.2, 1.5),
    unstudied: layer(
      "water-flood-unstudied",
      "Flood risk not studied",
      ["undetermined"],
      0.14,
      1.5,
    ),
  };
}

/**
 * Water: where the mapped flood zones are, how much rain falls and when, and how the soil
 * drains. All of it comes from the overview bundle (FEMA, NASA POWER, USDA soils), which runs
 * only when the person asks.
 */
export function WaterLens({ land }: { land: LandArea }) {
  const state = useOverviewBundle(land);
  const units = useSettings((s) => s.units);
  const setLens = useLens((s) => s.setLens);
  const fly = useFlyTo();
  const flyRef = useRef(fly);
  useEffect(() => {
    flyRef.current = fly;
  });

  const bundle = state.bundle;
  const flood = bundle?.flood ?? null;
  const { boundary, areaM2 } = land;
  const summary = useMemo(
    () => (flood ? summarizeFlood(flood.areas, { boundary, areaM2 }) : null),
    [flood, boundary, areaM2],
  );
  const layers = useMemo(() => floodLayers(summary), [summary]);
  useLensLayer(layers.hazard);
  useLensLayer(layers.moderate);
  useLensLayer(layers.unstudied);
  const pins = useMemo<MapPin[] | null>(
    () =>
      summary?.zones.map((zone) => ({
        id: `water-${zone.key}`,
        position: zone.anchor,
        label: `Zone ${zone.zone} · ${areaShort(zone.areaM2)}`,
        tone: zone.risk === "high" || zone.risk === "coastal" ? "warn" : "neutral",
        onSelect: () => flyRef.current(zone.geometry),
      })) ?? null,
    [summary],
  );
  useLensPins("water", pins);
  const climate = bundle?.climate ?? null;
  const rain = useMemo(() => rainSummary(climate?.precipitation ?? null), [climate]);

  if (state.loading || state.loadError || (state.investigation && !state.shown))
    return (
      <LensCard lens="water">
        <OverviewLoading state={state} />
      </LensCard>
    );
  if (!state.investigation || !bundle)
    return (
      <LensCard lens="water">
        <OverviewEmpty state={state} />
      </LensCard>
    );

  const fema = bundle.sources["fema-flood-zones"];
  const finished = state.shown?.status !== "queued" && state.shown?.status !== "running";
  const headline = floodHeadline(fema, summary, finished);
  const unreachable = SOURCES.filter((p) => bundle.sources[p].status === "unavailable").map((p) =>
    SOURCE_SPECS[p].name.toLowerCase(),
  );

  const facts: ReactNode[] = [];
  for (const risk of SHOWN_RISKS) {
    const group = summary?.groups[risk];
    if (!group) continue;
    facts.push(
      <Fact
        key={risk}
        label={RISK_LABELS[risk]}
        value={`${areaWords(group.areaM2)} · ${riskMeaning(risk, group)}`}
        note={
          risk === "high" && summary?.floodway
            ? "Includes a floodway, where building is restricted."
            : undefined
        }
        target={{ key: `water-flood-${risk}`, geometry: group.geometry, anchor: group.anchor }}
        tone={risk === "high" || risk === "coastal" ? "warn" : "accent"}
      />,
    );
  }
  if (!facts.length && (fema.status === "empty" || summary?.minimalOnly))
    facts.push(
      <Fact
        key="flood-map"
        label="Flood map"
        value={fema.status === "empty" ? "No flood zone mapped here" : "Minimal flood hazard"}
        note="Local flooding can still happen."
      />,
    );
  if (rain)
    facts.push(
      <Fact
        key="rain"
        label="Rain"
        value={
          <span className="water-rain">
            <span>{rainWords(rain, units)}</span>
            <RainBars summary={rain} units={units} />
          </span>
        }
        note={`Regional average${climate?.period ? `, ${climate.period}` : ""}`}
      />,
    );
  const drains = drainageFact(firstKnown(bundle.soils?.components ?? [], "drainage"));
  if (drains)
    facts.push(
      <Fact
        key="soil"
        label="Soil drainage"
        value={drains}
        note={
          <>
            Sampled at one point.{" "}
            <button type="button" className="land-link" onClick={() => setLens("soil")}>
              See Soil
            </button>
          </>
        }
      />,
    );

  const drawn = SHOWN_RISKS.filter((risk) => summary?.groups[risk]);
  const temperatures = climate?.temperature?.values ?? [];
  const questions = [
    ...(fema.finding?.suggestedQuestions ?? []),
    "What do the flood zones and rainfall mean for how I use this land?",
  ];

  return (
    <LensCard
      lens="water"
      headline={headline ?? undefined}
      action={
        <AskButton
          question="What do the flood zones, rainfall and soil drainage mean for this land?"
          label="Ask the agent about water on this land"
        />
      }
    >
      <OverviewNotices
        state={state}
        unreachable={
          unreachable.length ? `Some sources didn't respond: ${unreachable.join(", ")}.` : null
        }
      />
      {facts.length > 0 && <ul className="land-facts">{facts.slice(0, 4)}</ul>}
      {drawn.length > 0 && (
        <ul className="water-legend" aria-label="Map key">
          {drawn.map((risk) => (
            <li key={risk}>
              <i style={{ background: RISK_COLORS[risk] }} aria-hidden="true" />
              {LEGEND[risk]}
            </li>
          ))}
        </ul>
      )}
      <LensDetails>
        {flood && flood.areas.length > 0 && (
          <section className="bundle-section" aria-label="Flood zones">
            <h4 className="bundle-heading">Flood zones</h4>
            <DataTable
              label="Flood zones on this land"
              rows={flood.areas}
              rowKey={(area) => area.id}
              columns={[
                {
                  key: "zone",
                  label: "Zone",
                  cell: (area) => (
                    <MapLink
                      as="span"
                      className="water-zone-link"
                      target={{ key: `water-zone-${area.id}`, geometry: area.geometry }}
                      label={`Zone ${area.zone}: show on map`}
                    >
                      {area.zone}
                    </MapLink>
                  ),
                },
                { key: "risk", label: "Risk", cell: (area) => RISK_WORDS[floodRisk(area)] },
                {
                  key: "area",
                  label: "Area here",
                  numeric: true,
                  cell: (area) => areaWords(area.areaM2),
                },
                {
                  key: "subtype",
                  label: "Description",
                  cell: (area) => subtypeWords(area.subtype) || "None given",
                },
                {
                  key: "sfha",
                  label: "Special flood hazard area",
                  cell: (area) =>
                    area.specialFloodHazard === null
                      ? "Not stated"
                      : area.specialFloodHazard
                        ? "Yes"
                        : "No",
                },
                { key: "study", label: "Study", cell: (area) => area.studyType || "None given" },
                { key: "map", label: "Flood map", cell: (area) => area.mapId || "None given" },
                { key: "record", label: "Record", cell: (area) => area.recordId || "None given" },
              ]}
            />
            <p className="land-muted">
              Areas can overlap, so they are not added up. Totals above count overlaps once.
            </p>
            {flood.truncated && (
              <p className="land-muted">
                FEMA returned its limit of areas for this land; some may be missing.
              </p>
            )}
            {flood.methods.map((method) => (
              <p key={method} className="land-muted">
                {method}
              </p>
            ))}
            {flood.legend && <p className="land-muted">{flood.legend}</p>}
          </section>
        )}
        {rain && (
          <section className="bundle-section" aria-label="Rain and temperature">
            <h4 className="bundle-heading">Rain and temperature by month</h4>
            <DataTable
              label="Rain and temperature by month"
              pageSize={12}
              rows={MONTH_NAMES.map((month, index) => ({ month, index }))}
              rowKey={(row) => row.month}
              columns={[
                { key: "month", label: "Month", cell: (row) => row.month },
                {
                  key: "rain",
                  label: "Rain",
                  numeric: true,
                  cell: (row) => {
                    const value = rain.monthlyMm[row.index];
                    return value === null || value === undefined
                      ? "No data"
                      : rainAmount(value, units);
                  },
                },
                {
                  key: "temperature",
                  label: "Average temperature",
                  numeric: true,
                  cell: (row) => {
                    const value = temperatures[row.index];
                    return value === null || value === undefined
                      ? "No data"
                      : temperature(value, units);
                  },
                },
              ]}
            />
            <p className="land-muted">
              {rain.annualMm !== null && `About ${rainAmount(rain.annualMm, units)} a year. `}
              Regional estimates from satellite and weather models
              {climate?.period ? ` averaged over ${climate.period}` : ""}, not measured on this
              land.
            </p>
            {climate?.method && <p className="land-muted">{climate.method}</p>}
          </section>
        )}
        <SourceList
          bundle={bundle}
          providers={SOURCES}
          running={state.running}
          note={state.eventsMessage}
        />
        <AskLinks questions={questions} />
        {state.shown && (
          <p className="land-muted">
            Checked {new Date(state.shown.finishedAt ?? state.shown.createdAt).toLocaleDateString()}
            .
          </p>
        )}
      </LensDetails>
    </LensCard>
  );
}
