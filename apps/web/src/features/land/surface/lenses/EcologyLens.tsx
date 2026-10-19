import { useMemo } from "react";
import { ClipboardList, RefreshCw, Sparkles, Sprout, Square } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassProgress } from "@twin/ui";

import { useLandContext } from "@/state/landContext";
import { useLandScope } from "@/state/landIdentity";

import { askAgent } from "../agentBus";
import { areaLabel } from "../landFlow";
import { LensCard, LensDetails, LensEmpty } from "../LensCard";
import { Fact, MapLink } from "../mapLinks";
import { useFlyTo } from "../mapHooks";
import { ECOLOGY_QUESTION } from "./ecology/ecologyAgent";
import {
  completedSources,
  ecologyHeadline,
  formatDay,
  missingSources,
  percent,
  plannedSources,
  readEcology,
  shortList,
  SOURCE_LABELS,
  surveyTaxa,
  topSpecies,
  yearRange,
  type EcologyContext,
} from "./ecology/ecologyData";
import { EcologyDetails } from "./ecology/EcologyDetails";
import { EcologySurveyRecorder } from "./ecology/EcologySurveyRecorder";
import { SpeciesBars } from "./ecology/EcologySurveyView";
import { ECOLOGY_COLORS, useEcologyMap } from "./ecology/useEcologyMap";
import { useEcologyResearch, type EcologyResearch } from "./ecology/useEcologyResearch";
import {
  plotsShape,
  useSurveyCatalog,
  useSurveyDraft,
  type SurveyCatalog,
  type SurveyRead,
} from "./ecology/useEcologySurveys";
import "./ecology/ecology.css";

/**
 * Ecology: what kind of country the land sits in, what plant community its soil points to and
 * which species have been recorded here, in plain words; and the field surveys recorded on it,
 * drawn on the map as plots. Lookups run only when asked; tables and sources sit in Details.
 */
export function EcologyLens({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <EcologyCard key={`${scope}:${land.id}`} land={land} scope={scope} />;
}

function EcologyCard({ land, scope }: { land: LandArea; scope: string }) {
  const research = useEcologyResearch(land, scope);
  const catalog = useSurveyCatalog(land, scope);
  const survey = useSurveyDraft(land, scope);
  const { detail, run, events } = research;
  const context = useMemo(
    () => (detail && run ? readEcology(detail, run.id, events) : null),
    [detail, run, events],
  );
  useEcologyMap({
    context,
    surveys: catalog.surveys,
    selected: catalog.selected,
    draft: survey.draft,
    corners: survey.corners,
    onSelectSurvey: catalog.select,
  });
  const surveyNames = surveyTaxa(survey.draft?.plots ?? catalog.selected?.plots ?? []);
  const canEdit = research.canEdit;

  if (survey.draft)
    return (
      <LensCard
        lens="ecology"
        headline={survey.draft.supersedesId ? "Correct a field survey" : "Record a field survey"}
      >
        <EcologySurveyRecorder land={land} survey={survey} />
      </LensCard>
    );

  const hasContext = Boolean(context && run && !research.running && !research.failed);
  const surveys = catalog.surveys;
  const headline =
    context && hasContext
      ? ecologyHeadline(context, research.request ?? undefined)
      : surveys.length
        ? surveyHeadline(surveys)
        : undefined;
  const ask =
    hasContext || surveys.length ? (
      <GlassButton
        size="sm"
        variant="ghost"
        leadingIcon={<Sparkles size={14} aria-hidden="true" />}
        onClick={() => {
          if (research.selectedId)
            useLandContext.getState().selectInvestigation(research.selectedId);
          askAgent(ECOLOGY_QUESTION);
        }}
      >
        Ask
      </GlassButton>
    ) : undefined;

  if (research.loading && catalog.loading)
    return (
      <LensCard lens="ecology">
        <p className="land-muted" aria-live="polite">
          Loading ecology…
        </p>
      </LensCard>
    );

  return (
    <LensCard lens="ecology" headline={headline} action={ask}>
      {research.loadError && (
        <p role="alert" className="land-alert">
          Ecology could not be loaded.{" "}
          <button type="button" className="land-link" onClick={research.retry}>
            Try again
          </button>
        </p>
      )}
      {survey.recovery && (
        <div className="land-note land-eco-recovery">
          <p>An unfinished survey is saved in this browser.</p>
          <div className="land-row">
            <GlassButton size="sm" variant="primary" disabled={survey.busy} onClick={survey.resume}>
              Resume survey
            </GlassButton>
            <GlassButton size="sm" variant="ghost" onClick={survey.exportDraft}>
              Export draft
            </GlassButton>
            <GlassButton size="sm" variant="ghost" onClick={survey.discard}>
              Discard draft
            </GlassButton>
          </div>
          {survey.error && <p role="alert">{survey.error}</p>}
        </div>
      )}

      <ContextState land={land} research={research} context={context} />

      {surveys.length > 0 && <SurveyList catalog={catalog} />}
      {catalog.error && (
        <p role="alert" className="land-alert">
          Surveys could not be loaded.{" "}
          <button type="button" className="land-link" onClick={catalog.retry}>
            Retry
          </button>
        </p>
      )}

      {canEdit && !survey.recovery && (
        <div className="land-row">
          <GlassButton
            variant={hasContext ? "primary" : "ghost"}
            size={hasContext ? "md" : "sm"}
            leadingIcon={<ClipboardList size={hasContext ? 16 : 14} aria-hidden="true" />}
            onClick={survey.begin}
          >
            Record a field survey
          </GlassButton>
        </div>
      )}

      <LensDetails>
        <EcologyDetails
          land={land}
          research={research}
          context={context}
          catalog={catalog}
          survey={survey}
          surveyNames={surveyNames}
        />
      </LensDetails>
    </LensCard>
  );
}

function surveyHeadline(surveys: SurveyRead[]): string {
  const latest = surveys[0];
  const top = latest ? topSpecies(latest.summary)[0] : undefined;
  const count = `${surveys.length}${surveys.length >= 20 ? "+" : ""} field ${surveys.length === 1 ? "survey" : "surveys"} recorded here.`;
  return top ? `${count} The latest found mostly ${top.taxon}.` : count;
}

/** The lookup's part of the card: start it, watch it, or read what it found. */
function ContextState({
  land,
  research,
  context,
}: {
  land: LandArea;
  research: EcologyResearch;
  context: EcologyContext | null;
}) {
  const { run, canEdit } = research;
  if (research.loading) return <p className="land-muted">Loading what lives here…</p>;
  // The card already says the saved lookups could not be loaded; offer nothing to start.
  if (research.loadError && !run) return null;
  if (!run)
    return (
      <>
        <LensEmpty
          action={
            canEdit ? (
              <GlassButton
                variant="primary"
                disabled={research.busy}
                leadingIcon={<Sprout size={16} aria-hidden="true" />}
                onClick={() => research.start()}
              >
                {research.busy ? "Starting…" : "Learn what lives here"}
              </GlassButton>
            ) : undefined
          }
        >
          See the ecoregion, the plant community the soil points to, and the species recorded here.
          {!canEdit && " An editor can look this up."}
        </LensEmpty>
        {research.error && (
          <p role="alert" className="land-alert">
            Could not start the lookup. {research.error}
          </p>
        )}
      </>
    );

  if (research.running) {
    const total = Math.max(1, plannedSources(research.request));
    const done = Math.min(total, completedSources(research.events));
    return (
      <div className="land-eco-running" aria-live="polite">
        <p className="land-lede">
          Looking up what lives here… {done} of {total} {total === 1 ? "source" : "sources"} done
        </p>
        <GlassProgress value={done} max={total} label="Ecology lookup progress" />
        {canEdit && (
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="ghost"
              disabled={research.busy}
              leadingIcon={<Square size={13} aria-hidden="true" />}
              onClick={research.cancel}
            >
              Stop
            </GlassButton>
          </div>
        )}
      </div>
    );
  }

  const again = canEdit ? (
    <GlassButton
      size="sm"
      disabled={research.busy}
      leadingIcon={<RefreshCw size={14} aria-hidden="true" />}
      onClick={() => research.start(research.request ?? undefined)}
    >
      {research.busy ? "Starting…" : research.failed ? "Try again" : "Run again"}
    </GlassButton>
  ) : null;

  if (research.failed)
    return (
      <div className="land-alert" role="alert">
        <p>
          {run.status === "cancelled" ? "The lookup was stopped." : "The lookup failed."}{" "}
          {run.error ?? ""}
        </p>
        {again && <div className="land-row">{again}</div>}
        {research.error && <p>{research.error}</p>}
      </div>
    );

  return (
    <>
      {research.stale && (
        <div className="land-note land-row">
          <span>Made for an earlier boundary.</span>
          {again}
        </div>
      )}
      {context && <ContextFacts land={land} context={context} />}
      {research.error && (
        <p role="alert" className="land-alert">
          Could not start the lookup. {research.error}
        </p>
      )}
    </>
  );
}

/** At most four facts, each pointing at where it is true. */
function ContextFacts({ land, context }: { land: LandArea; context: EcologyContext }) {
  const boundary = land.boundary;
  const { regions, sites, species, names } = context;
  const missing = missingSources(context);
  const top = species?.ranked.slice(0, 3) ?? [];
  const years = yearRange(species?.years ?? null);
  const any =
    (regions?.level4.length ?? 0) > 0 ||
    Boolean(sites) ||
    (species?.ranked.length ?? 0) > 0 ||
    (names?.rows.length ?? 0) > 0;
  return (
    <>
      {any && (
        <ul className="land-facts" aria-label="What lives here">
          {regions && regions.level4.length > 0 && (
            <Fact
              label="Ecoregion"
              value={shortList(regions.level4)}
              note={regions.level3.length === 1 ? `Part of the ${regions.level3[0]}` : undefined}
              target={{ key: "ecology-region", geometry: boundary }}
            />
          )}
          {sites && (
            <Fact
              label="Reference plant community"
              value={sites.names.length ? shortList(sites.names) : "None listed for this soil"}
              note="Matched to the soil at one sample point"
              target={
                sites.samplePoint
                  ? {
                      key: "ecology-sample",
                      geometry: { type: "Point", coordinates: sites.samplePoint },
                    }
                  : null
              }
            />
          )}
          {species && species.ranked.length > 0 && (
            <Fact
              label="Species recorded"
              value={`${species.ranked.length.toLocaleString()} species in ${species.records.toLocaleString()} ${species.records === 1 ? "record" : "records"}${years ? `, ${years}` : ""}`}
              note={
                <>
                  Most often:{" "}
                  {top.map((item, index) => (
                    <span key={item.name}>
                      {index > 0 && ", "}
                      <i>{item.name}</i>
                    </span>
                  ))}
                </>
              }
              target={{ key: "ecology-species", geometry: boundary }}
            />
          )}
          {names && names.rows.length > 0 && (
            <Fact
              label="Names checked"
              value={`${names.exact} of ${names.rows.length} matched exactly`}
              note={names.unsure ? "Check the others in Details" : undefined}
              tone={names.unsure ? "warn" : "accent"}
            />
          )}
        </ul>
      )}
      {missing.length > 0 && (
        <p className="land-muted">
          {missing
            .map(
              (source) => `${SOURCE_LABELS[source.provider] ?? source.provider}: ${source.message}`,
            )
            .join(" ")}
        </p>
      )}
    </>
  );
}

/** Saved surveys, each pointing at its plots; the selected one shows its species as bars. */
function SurveyList({ catalog }: { catalog: SurveyCatalog }) {
  const fly = useFlyTo();
  const surveys = catalog.surveys;
  const shown = surveys.slice(0, 4);
  const selected = catalog.selected;
  const listedSelected = selected && !shown.some((item) => item.id === selected.id);
  const rows = listedSelected ? [selected, ...shown.slice(0, 3)] : shown;
  return (
    <section className="land-eco-surveys" aria-label="Field surveys">
      <div className="land-eco-surveys__head">
        <span className="land-eco-label">Field surveys</span>
        {surveys.length > 1 && (
          <span className="land-eco-legend" aria-hidden="true">
            <i style={{ background: ECOLOGY_COLORS.selected }} /> selected
            <i style={{ background: ECOLOGY_COLORS.surveys, opacity: 0.6 }} /> others
          </span>
        )}
      </div>
      <ul className="land-facts">
        {rows.map((item) => {
          const shape = plotsShape(item.plots);
          const active = item.id === catalog.selectedId;
          const top = topSpecies(item.summary)[0];
          return (
            <MapLink
              key={item.id}
              as="li"
              fly={false}
              tone="accent"
              target={shape ? { key: `ecology-survey-${item.id}`, geometry: shape } : null}
              className={`land-fact land-eco-survey${active ? " is-selected" : ""}`}
            >
              <button
                type="button"
                className="land-eco-survey__main"
                aria-pressed={active}
                onClick={() => {
                  catalog.select(active ? null : item.id);
                  if (!active && shape) fly(shape);
                }}
              >
                <span className="land-fact__label">
                  {item.name} · {formatDay(item.observedOn)}
                  {item.stale ? " · older boundary" : ""}
                  {item.supersedesId ? " · correction" : ""}
                </span>
                <span className="land-fact__value">
                  {item.plots.length} {item.plots.length === 1 ? "plot" : "plots"} ·{" "}
                  {areaLabel(item.summary.sampledAreaM2)}
                  {top && (
                    <>
                      {" · "}
                      <i>{top.taxon}</i> {percent(top.meanPercent)}
                    </>
                  )}
                </span>
              </button>
              {active && (
                <div className="land-eco-survey__detail">
                  <SpeciesBars summary={item.summary} limit={5} label={`Species in ${item.name}`} />
                  {item.summary.species.length > 5 && (
                    <span className="land-muted">
                      {item.summary.species.length - 5} more in Details
                    </span>
                  )}
                </div>
              )}
            </MapLink>
          );
        })}
      </ul>
      {surveys.length > rows.length && (
        <p className="land-muted">{surveys.length - rows.length} more in Details</p>
      )}
    </section>
  );
}
