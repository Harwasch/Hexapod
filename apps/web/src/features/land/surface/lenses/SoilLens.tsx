import { useMemo, type ReactNode } from "react";

import type { LandArea, LandMapGeometry } from "@twin/contracts";
import { toFeet, type UnitSystem } from "@twin/geo";

import type { MapPin } from "@/state/landMap";
import { useSettings } from "@/state/settings";

import { anchorOf } from "../geometryAnchor";
import { LensCard, LensDetails } from "../LensCard";
import { Fact } from "../mapLinks";
import { useLensPins } from "../useLensMap";
import {
  sideOfLand,
  SOURCE_SPECS,
  useOverviewBundle,
  type OverviewProvider,
  type SoilComponent,
  type SourceState,
} from "./overviewBundle";
import {
  drainageFact,
  firstKnown,
  HYDROLOGIC_GROUPS,
  mainComponent,
  rainSoak,
  slopeWords,
  soilHeadline,
  soilName,
} from "./soil/soilWords";
import {
  AskButton,
  AskLinks,
  DataTable,
  OverviewEmpty,
  OverviewLoading,
  OverviewNotices,
  SourceList,
} from "./water/bundleStates";
import "./soil/soil.css";

const SOURCES: readonly OverviewProvider[] = ["usda-soils", "usgs-elevation"];

/** An elevation in the person's units, rounded as a map would: "123 m", "405 ft". */
function elevationWords(metres: number, units: UnitSystem): string {
  return units === "imperial"
    ? `${Math.round(toFeet(metres)).toLocaleString()} ft`
    : `${Math.round(metres).toLocaleString()} m`;
}

/** The headline when the soil survey gave nothing to describe. */
function missingSoil(source: SourceState, finished: boolean): string | null {
  if (source.status === "uncovered") return "Soil maps don't cover this location.";
  if (source.status === "unavailable") return "Soil couldn't be checked.";
  if (source.status === "empty") return "No mapped soil at the sample point.";
  return finished ? "Soil couldn't be checked." : null;
}

/**
 * Soil: what the USDA soil survey maps at one point near the middle of the land, in plain
 * words: its name, how it drains, what rain does on it and its usual slope. The survey is read
 * at a single point, so the lens pins that point and says so.
 */
export function SoilLens({ land }: { land: LandArea }) {
  const state = useOverviewBundle(land);
  const units = useSettings((s) => s.units);
  const bundle = state.bundle;
  const soils = bundle?.soils ?? null;
  const elevation = bundle?.elevation ?? null;
  const boundary = land.boundary;
  // The survey samples the boundary's representative point; the evidence usually names it.
  const sample = useMemo(
    () =>
      soils?.samplePoint ??
      elevation?.point ??
      (soils?.components.length || elevation ? anchorOf(boundary) : null),
    [soils, elevation, boundary],
  );
  const pins = useMemo<MapPin[] | null>(
    () =>
      sample
        ? [{ id: "soil-sample", position: sample, label: "Sampled near here", tone: "accent" }]
        : null,
    [sample],
  );
  useLensPins("soil", pins);

  if (state.loading || state.loadError || (state.investigation && !state.shown))
    return (
      <LensCard lens="soil">
        <OverviewLoading state={state} />
      </LensCard>
    );
  if (!state.investigation || !bundle)
    return (
      <LensCard lens="soil">
        <OverviewEmpty state={state} />
      </LensCard>
    );

  const source = bundle.sources["usda-soils"];
  const finished = state.shown?.status !== "queued" && state.shown?.status !== "running";
  const components = soils?.components ?? [];
  const main = mainComponent(components);
  const mapUnits = [...new Set(components.map((row) => row.mapUnit).filter(Boolean))];
  const where = sample ? sideOfLand(sample, boundary) : "the middle";
  const headline =
    (source.status === "available" || !source.status) && main
      ? soilHeadline(components, where)
      : missingSoil(source, finished);
  const target = sample
    ? {
        key: "soil-sample",
        geometry: { type: "Point", coordinates: sample } satisfies LandMapGeometry,
      }
    : null;
  const unreachable = SOURCES.filter((p) => bundle.sources[p].status === "unavailable").map((p) =>
    SOURCE_SPECS[p].name.toLowerCase(),
  );

  const facts: ReactNode[] = [];
  if (main) {
    const name = soilName(main.mapUnit) || main.component;
    if (name)
      facts.push(
        <Fact
          key="soil"
          label="Soil"
          value={name}
          note={
            main.component && main.percent !== null
              ? `Mostly ${main.component} soil (${Math.round(main.percent)}% of this map unit)`
              : undefined
          }
          target={target}
        />,
      );
    const drains = drainageFact(firstKnown(components, "drainage"));
    if (drains) facts.push(<Fact key="drainage" label="Drainage" value={drains} target={target} />);
    const soak = rainSoak(firstKnown(components, "hydrologicGroup"));
    if (soak) facts.push(<Fact key="rain" label="When it rains" value={soak} target={target} />);
    const slope = slopeWords(firstKnown(components, "slopePercent"));
    if (slope) facts.push(<Fact key="slope" label="Typical slope" value={slope} target={target} />);
  }

  const questions = [
    ...(source.finding?.suggestedQuestions ?? []),
    "What could this soil grow well, and what should I test before planting?",
  ];

  return (
    <LensCard
      lens="soil"
      headline={headline ?? undefined}
      action={
        <AskButton
          question="What does the mapped soil mean for how I could use this land?"
          label="Ask the agent about the soil"
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
      {sample && (main !== null || elevation !== null) && (
        <p className="bundle-sample">
          Sampled at one point
          {elevation ? `, ${elevationWords(elevation.valueM, units)} above sea level` : ""}. Soil
          can differ elsewhere on the land.
        </p>
      )}
      <LensDetails>
        {components.length > 0 && (
          <section className="bundle-section" aria-label="Soil components">
            <h4 className="bundle-heading">Soil at the sample point</h4>
            {mapUnits.length === 1 && <p>Map unit: {mapUnits[0]}</p>}
            <DataTable
              label="Soil components at the sample point"
              rows={components}
              rowKey={(row, index) => `${row.component}:${index}`}
              columns={[
                ...(mapUnits.length > 1
                  ? [
                      {
                        key: "unit",
                        label: "Map unit",
                        cell: (row: SoilComponent) => row.mapUnit || "Unnamed",
                      },
                    ]
                  : []),
                { key: "component", label: "Component", cell: (row) => row.component || "Unnamed" },
                {
                  key: "percent",
                  label: "Share",
                  numeric: true,
                  cell: (row) => (row.percent === null ? "No data" : `${row.percent}%`),
                },
                {
                  key: "drainage",
                  label: "Drainage class",
                  cell: (row) => row.drainage ?? "No data",
                },
                {
                  key: "group",
                  label: "Hydrologic group",
                  cell: (row) => row.hydrologicGroup ?? "No data",
                },
                {
                  key: "slope",
                  label: "Slope",
                  numeric: true,
                  cell: (row) => (row.slopePercent === null ? "No data" : `${row.slopePercent}%`),
                },
              ]}
            />
            <dl className="soil-groups">
              {HYDROLOGIC_GROUPS.map(([group, meaning]) => (
                <div key={group}>
                  <dt>Group {group}</dt>
                  <dd>{meaning}</dd>
                </div>
              ))}
            </dl>
            {soils?.method && <p className="land-muted">{soils.method}</p>}
            <p className="land-muted">
              What the soil suits is a rule of thumb from its drainage and slope. It is not a land
              capability rating or a soil test.
            </p>
          </section>
        )}
        {elevation && (
          <section className="bundle-section" aria-label="Elevation">
            <h4 className="bundle-heading">Elevation</h4>
            <p>
              {elevationWords(elevation.valueM, units)} above sea level at the sample point (
              {elevation.valueM.toLocaleString(undefined, { maximumFractionDigits: 1 })} m).
            </p>
            <p className="land-muted">
              One interpolated point, not a surveyed height or the land's average.
            </p>
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
