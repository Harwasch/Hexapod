import { useMemo, useState } from "react";
import { Plus, Sun } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import type { MapPin } from "@/state/landMap";

import { cancelResearch } from "../../researchApi";
import { ResearchWorkerIdle, ResearchWorkerNotice } from "../../ResearchWorkerNotice";
import { askAgent } from "../agentBus";
import { anchorOf } from "../geometryAnchor";
import { landError } from "../landFlow";
import { LensCard, LensDetails, LensEmpty } from "../LensCard";
import { Fact, MapLink } from "../mapLinks";
import { useFlyTo } from "../mapHooks";
import { useLandAnalysis } from "../useLandAnalysis";
import { useLensLayer, useLensPins } from "../useLensMap";
import { SolarAssessmentAbout, SolarAssessmentFacts } from "./solar/SolarAssessmentInfo";
import { PAGE, useLandScenarios, useSolarAssessment, useSolarAssessments } from "./solar/solarData";
import { runRequest, zoneTarget, type SolarAssessment } from "./solar/solarDraft";
import { scenarioNumbers, type Scenario } from "./solar/solarEconomics";
import { SolarEconomicsEditor, SolarScenarios } from "./solar/SolarEconomics";
import { SolarMonthly } from "./solar/SolarMonthly";
import { SolarSettings } from "./solar/SolarSettings";
import { SolarSketch, SolarZoneSources } from "./solar/SolarSketch";
import {
  assessmentHeadline,
  assessmentPin,
  dateLabel,
  draftHeadline,
  energyLabel,
  moneyLabel,
  paybackWords,
  powerLabel,
  zoneArea,
  zoneSize,
} from "./solar/solarWords";
import { SOLAR_RUN_TITLE, useSolarDraft } from "./solar/useSolarDraft";
import { useSolarEconomics } from "./solar/useSolarEconomics";
import "./solar/solarLens.css";

/** Saved arrays in a quiet warm outline; the chosen one bright; a plan in progress amber. */
const STYLE_SAVED = { color: "#d9a441", fill: 0.1, width: 1.5 };
const STYLE_CHOSEN = { color: "#ffcf5a", fill: 0.32, width: 2.5 };
const STYLE_DRAFT = { color: "#ffb84d", fill: 0.26, width: 2.5 };
const STYLE_RUNNING = { color: "#ffcf5a", fill: 0.16, width: 2 };
const MAX_PINS = 6;

/**
 * Solar: where panels could go and what they would make. Draw an array on the map, see its
 * size at once, run the hourly weather model for a year of energy, then work out costs and
 * payback. Everything technical waits under Details.
 */
export function SolarLens({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Solar key={`${scope}:${land.id}`} land={land} />;
}

function Solar({ land }: { land: LandArea }) {
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit() && ready;
  const fly = useFlyTo();
  const draft = useSolarDraft(land);
  const econ = useSolarEconomics(land);
  const analysis = useLandAnalysis(land, SOLAR_RUN_TITLE);
  const [offset, setOffset] = useState(0);
  const [dismissedRun, setDismissedRun] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const refresh = `${analysis.run?.id ?? ""}:${analysis.run?.status ?? ""}`;
  const newest = useSolarAssessments(land, 0, refresh);
  const page = useSolarAssessments(land, offset, refresh);
  const selectedId = useLandContext((s) => s.selectedSolarId);
  const listed = newest.data?.find((item) => item.id === selectedId);
  const single = useSolarAssessment(
    selectedId && !listed && newest.isSuccess ? selectedId : null,
    land.revision,
  );
  const latest = newest.data?.[0] ?? null;
  // A chosen assessment off the first page loads on its own; a missing one falls back.
  const selected: SolarAssessment | null =
    (selectedId ? (listed ?? single.data) : null) ??
    (single.isError || !selectedId ? latest : null);
  const selectionLoading = Boolean(selectedId && !listed && !single.data && !single.isError);
  const scenarioQuery = useLandScenarios(land);
  const scenarios = useMemo(
    () =>
      (scenarioQuery.data ?? [])
        .filter((scenario) => scenario.inputs.kind === "solar")
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    [scenarioQuery.data],
  );
  const linkedScenario = selected
    ? (scenarios.find((scenario) => scenario.solarAssessmentId === selected.id) ?? null)
    : null;
  const manualScenario = scenarios.find((scenario) => !scenario.solarAssessmentId) ?? null;

  const run = analysis.run;
  const runZone = run && analysis.running ? runRequest(run.analysis) : null;
  const failedRun =
    run &&
    analysis.failed &&
    run.id !== dismissedRun &&
    (!latest || run.createdAt > latest.createdAt)
      ? run
      : null;

  const plan = draft.draft;
  const mode = econ.draft
    ? "economics"
    : plan?.sketching
      ? "sketch"
      : plan
        ? "draft"
        : selected
          ? "result"
          : analysis.running
            ? "running"
            : newest.isPending || analysis.loading || selectionLoading
              ? "loading"
              : newest.isError
                ? "error"
                : "empty";

  /* ---- On the map -------------------------------------------------------------- */
  const highlight = mode === "result" || mode === "economics" ? selected : null;
  const savedLayer = useMemo<LandContextLayer | null>(() => {
    const items = (newest.data ?? []).filter((item) => item.id !== highlight?.id);
    if (!items.length) return null;
    return {
      id: "solar-arrays",
      title: "Saved solar arrays",
      style: STYLE_SAVED,
      features: items.map((item) => ({
        id: item.id,
        label: assessmentPin(item),
        geometry: item.request.arrayZone,
      })),
    };
  }, [newest.data, highlight?.id]);
  const chosenLayer = useMemo<LandContextLayer | null>(
    () =>
      highlight
        ? {
            id: "solar-array",
            title: "Chosen solar array",
            style: STYLE_CHOSEN,
            features: [
              {
                id: highlight.id,
                label: assessmentPin(highlight),
                geometry: highlight.request.arrayZone,
              },
            ],
          }
        : null,
    [highlight],
  );
  // Only the outline and corners redraw the map, not every settings keystroke.
  const corners = plan?.points;
  const sketching = Boolean(plan?.sketching);
  const plannedZone = plan?.hasZone ? plan.request.arrayZone : null;
  const draftLayer = useMemo<LandContextLayer | null>(() => {
    if (!corners) return null;
    const first = corners[0];
    const features: LandContextLayer["features"] = [];
    if (sketching) {
      if (corners.length >= 3 && first)
        features.push({
          id: "zone",
          label: "Array being drawn",
          geometry: { type: "Polygon", coordinates: [[...corners, first]] },
        });
      else if (corners.length === 2)
        features.push({
          id: "edge",
          label: "Array being drawn",
          geometry: { type: "LineString", coordinates: corners },
        });
      corners.forEach((coordinates, index) =>
        features.push({
          id: `corner-${index}`,
          label: `Corner ${index + 1}`,
          geometry: { type: "Point", coordinates },
        }),
      );
    } else if (plannedZone)
      features.push({ id: "zone", label: "Planned array", geometry: plannedZone });
    return features.length
      ? { id: "solar-draft", title: "Planned solar array", style: STYLE_DRAFT, features }
      : null;
  }, [corners, sketching, plannedZone]);
  const runningLayer = useMemo<LandContextLayer | null>(
    () =>
      runZone
        ? {
            id: "solar-running",
            title: "Array being modeled",
            style: STYLE_RUNNING,
            features: [{ id: "zone", label: "Being modeled", geometry: runZone.arrayZone }],
          }
        : null,
    [runZone],
  );
  useLensLayer(savedLayer);
  useLensLayer(chosenLayer);
  useLensLayer(draftLayer);
  useLensLayer(runningLayer);

  const draftCapacity = draft.preview?.capacityKwDc ?? draft.capacityKw;
  const draftZone = sketching ? null : plannedZone;
  const planning = Boolean(plan);
  const pins = useMemo<MapPin[]>(() => {
    const list: MapPin[] = [];
    if (draftZone)
      list.push({
        id: "solar-draft",
        position: anchorOf(draftZone),
        label: Number.isFinite(draftCapacity)
          ? `Planned · ≈ ${powerLabel(draftCapacity)}`
          : "Planned array",
        tone: "accent",
      });
    if (runZone)
      list.push({
        id: "solar-running",
        position: anchorOf(runZone.arrayZone),
        label: "Modeling…",
        tone: "neutral",
      });
    if (highlight)
      list.push({
        id: `solar-${highlight.id}`,
        position: anchorOf(highlight.request.arrayZone),
        label: assessmentPin(highlight),
        tone: "accent",
      });
    if (!planning)
      for (const item of (newest.data ?? []).filter((entry) => entry.id !== highlight?.id)) {
        if (list.length >= MAX_PINS) break;
        list.push({
          id: `solar-${item.id}`,
          position: anchorOf(item.request.arrayZone),
          label: `${powerLabel(item.metadata.capacityKwDc)} · ${item.request.year}`,
          tone: "neutral",
          onSelect: () => useLandContext.getState().selectSolar(item.id),
        });
      }
    return list;
  }, [draftZone, draftCapacity, runZone, highlight, planning, newest.data]);
  useLensPins("solar", pins);

  /* ---- Words ------------------------------------------------------------------- */
  const headline =
    mode === "economics"
      ? selected && econ.payload?.solarAssessmentId === selected.id
        ? `Costs and payback for the ${powerLabel(selected.metadata.capacityKwDc)} array`
        : "Costs and payback"
      : mode === "sketch"
        ? "Draw where the panels could go."
        : mode === "draft" && plan
          ? draftHeadline(plan.request.arrayZone, draftCapacity, land)
          : mode === "result" && selected
            ? assessmentHeadline(selected, land)
            : mode === "running"
              ? runZone
                ? `Modeling a ${zoneSize(footprintAreaM2(runZone.arrayZone))} array with ${runZone.year} weather…`
                : "Running the hourly solar model…"
              : mode === "error"
                ? "Solar assessments could not load."
                : mode === "empty"
                  ? "See what the sun could make here."
                  : undefined;

  const action =
    canEdit && (mode === "result" || mode === "running") ? (
      <GlassButton
        size="sm"
        variant="ghost"
        leadingIcon={<Plus size={15} aria-hidden="true" />}
        onClick={draft.begin}
      >
        New array
      </GlassButton>
    ) : undefined;

  const paybackFact = (scenario: Scenario | null, linked: boolean) => {
    if (!scenario) return null;
    const numbers = scenarioNumbers(scenario.result);
    const years = Math.max(0, scenario.result.rows.length - 1);
    const value = `${paybackWords(numbers.payback, years)}${numbers.netPresentValue === null ? "" : ` · worth ${moneyLabel(numbers.netPresentValue, numbers.currency)} today`}`;
    return (
      <Fact
        label={linked ? "Payback" : "Payback, entered values"}
        value={value}
        target={linked && selected ? zoneTarget(selected) : null}
      />
    );
  };

  const running =
    analysis.running && run ? (
      <>
        <div className="land-solar-running" role="status" aria-live="polite">
          <span className="land-solar-running__pulse" aria-hidden="true" />
          <span>
            {mode === "running"
              ? run.status === "queued"
                ? "Waiting to start. It takes about a minute once it runs."
                : "This takes about a minute."
              : runZone
                ? `Modeling a ${zoneSize(footprintAreaM2(runZone.arrayZone))} array with ${runZone.year} weather…`
                : "Running the hourly solar model…"}
          </span>
          {canEdit && (
            <button
              type="button"
              className="land-link land-solar-push"
              onClick={() => {
                setCancelError(null);
                void cancelResearch(run.id).catch((cause: unknown) =>
                  setCancelError(`The model could not be stopped. ${landError(cause)}`),
                );
              }}
            >
              {run.status === "queued" ? "Cancel queued request" : "Cancel"}
            </button>
          )}
        </div>
        <ResearchWorkerNotice run={run} className="land-muted" />
      </>
    ) : null;

  const failedRequest = failedRun ? runRequest(failedRun.analysis) : null;
  const failed =
    failedRun && mode !== "draft" && mode !== "sketch" ? (
      <div className="land-alert" role="alert">
        <p>
          {failedRun.status === "cancelled"
            ? "The hourly model was stopped before it finished."
            : `The hourly model did not finish. ${failedRun.error ?? ""}`}
        </p>
        <div className="land-row">
          {canEdit && failedRequest && (
            <GlassButton
              size="sm"
              onClick={() =>
                draft.load(
                  { ...failedRequest },
                  "Settings from the run that did not finish. Run it again when ready.",
                )
              }
            >
              Try again
            </GlassButton>
          )}
          <GlassButton size="sm" variant="ghost" onClick={() => setDismissedRun(failedRun.id)}>
            Dismiss
          </GlassButton>
        </div>
      </div>
    ) : null;

  const recovery = (
    <>
      {draft.unreadable && (
        <div className="land-note" role="region" aria-label="Unreadable array sketch">
          <p>A saved array sketch in this browser could not be read.</p>
          <GlassButton size="sm" variant="ghost" onClick={draft.discard}>
            Discard it
          </GlassButton>
        </div>
      )}
      {econ.unreadable && (
        <div className="land-note" role="region" aria-label="Unreadable saved costs">
          <p>Saved costs in this browser could not be read.</p>
          <div className="land-row">
            <button
              type="button"
              className="land-link"
              onClick={() => {
                const url = URL.createObjectURL(
                  new Blob([econ.unreadable ?? ""], { type: "application/json" }),
                );
                const anchor = document.createElement("a");
                anchor.href = url;
                anchor.download = "scenario-draft.json";
                anchor.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
              }}
            >
              Download them
            </button>
            <GlassButton size="sm" variant="ghost" onClick={econ.discardUnreadable}>
              Discard them
            </GlassButton>
          </div>
        </div>
      )}
      {draft.notice && (
        <p className="land-note" role="status">
          {draft.notice}{" "}
          <button type="button" className="land-link" onClick={draft.dismissNotice}>
            OK
          </button>
        </p>
      )}
      {econ.notice && (
        <p className="land-note" role="status">
          {econ.notice}{" "}
          <button type="button" className="land-link" onClick={econ.dismissNotice}>
            OK
          </button>
        </p>
      )}
      {draft.storageError && (
        <p className="land-alert" role="alert">
          {draft.storageError}
        </p>
      )}
    </>
  );

  /* ---- The card ---------------------------------------------------------------- */
  return (
    <LensCard lens="solar" headline={headline} action={action}>
      <div className="land-solar">
        {recovery}
        {running}
        {cancelError && (
          <p className="land-alert" role="alert">
            {cancelError}
          </p>
        )}
        {failed}
        {analysis.error && mode !== "loading" && (
          <p className="land-muted">
            The model&apos;s status could not load. Reopen the lens to try again.
          </p>
        )}

        {mode === "loading" && (
          <p className="land-muted" role="status">
            Loading solar assessments…
          </p>
        )}

        {mode === "error" && (
          <p className="land-alert" role="alert">
            Check the connection and{" "}
            <button type="button" className="land-link" onClick={() => void newest.refetch()}>
              try again
            </button>
            .
          </p>
        )}

        {mode === "empty" && (
          <LensEmpty
            action={
              canEdit ? (
                <GlassButton
                  variant="primary"
                  leadingIcon={<Sun size={16} aria-hidden="true" />}
                  onClick={draft.begin}
                >
                  Sketch an array
                </GlassButton>
              ) : undefined
            }
          >
            {canEdit
              ? "Draw where panels could go. You see their size at once, then a year of hourly weather gives the energy."
              : "No one has sketched a solar array on this land yet."}
          </LensEmpty>
        )}

        {mode === "sketch" && <SolarSketch draft={draft} land={land} />}

        {mode === "draft" && plan && (
          <DraftBody
            draft={draft}
            land={land}
            capacityKw={draftCapacity}
            // A run already waiting says what it waits for; the idle note would repeat it.
            idle={run?.status !== "queued"}
          />
        )}

        {mode === "result" && selected && (
          <>
            {selected.stale && (
              <p className="land-note">
                Made for an earlier outline of the land.{" "}
                {canEdit && (
                  <button
                    type="button"
                    className="land-link"
                    onClick={() =>
                      draft.load(
                        { ...selected.request },
                        "Check the array still fits the land's new outline.",
                      )
                    }
                  >
                    Run again
                  </button>
                )}
              </p>
            )}
            <SolarAssessmentFacts
              assessment={selected}
              extra={paybackFact(linkedScenario ?? manualScenario, Boolean(linkedScenario))}
            />
            <SolarMonthly assessment={selected} />
            {canEdit && !linkedScenario && (
              <div className="land-row">
                <GlassButton
                  variant="primary"
                  disabled={
                    !selected.metadata.completeYear || selected.stale || Boolean(econ.draft)
                  }
                  onClick={() => econ.fromAssessment(selected)}
                >
                  Work out costs and payback
                </GlassButton>
                {!selected.metadata.completeYear && (
                  <span className="land-muted">Needs a full year of weather.</span>
                )}
              </div>
            )}
          </>
        )}

        {mode === "economics" && <SolarEconomicsEditor econ={econ} land={land} />}

        <LensDetails>
          {mode === "draft" && plan && (
            <section className="land-solar-section" aria-label="Array outline">
              <h4 className="land-solar-heading">Array outline</h4>
              <p className="land-muted">{plan.request.zoneBasis}</p>
              <SolarZoneSources draft={draft} land={land} />
            </section>
          )}
          {mode === "result" && selected && (
            <section className="land-solar-section" aria-label="This assessment">
              <h4 className="land-solar-heading">
                This assessment · {selected.request.year} weather
              </h4>
              <SolarAssessmentAbout assessment={selected} />
              {canEdit && (
                <p>
                  <button
                    type="button"
                    className="land-link"
                    onClick={() => draft.load({ ...selected.request })}
                  >
                    Change its settings and run again
                  </button>
                </p>
              )}
            </section>
          )}
          <SolarScenarios scenarios={scenarios} econ={econ} land={land} />
          {scenarioQuery.isError && (
            <p className="land-alert" role="alert">
              Saved costs could not load.{" "}
              <button
                type="button"
                className="land-link"
                onClick={() => void scenarioQuery.refetch()}
              >
                Try again
              </button>
            </p>
          )}
          <SavedAssessments
            items={page.data ?? []}
            selectedId={mode === "result" || mode === "economics" ? (selected?.id ?? null) : null}
            offset={offset}
            onPage={setOffset}
            onSelect={(item) => {
              useLandContext.getState().selectSolar(item.id);
              fly(item.request.arrayZone);
            }}
          />
          {page.isError && offset > 0 && (
            <p className="land-alert" role="alert">
              Older assessments could not load.{" "}
              <button type="button" className="land-link" onClick={() => void page.refetch()}>
                Try again
              </button>
            </p>
          )}
          <div className="land-row">
            {canEdit && !econ.draft && (
              <button type="button" className="land-link" onClick={econ.manual}>
                Work out costs without the hourly model
              </button>
            )}
            <button
              type="button"
              className="land-link"
              onClick={() =>
                askAgent(
                  selected
                    ? `What should I check before building a ${powerLabel(selected.metadata.capacityKwDc)} solar array on this land?`
                    : "Where on this land would solar panels work best, and what should I check first?",
                )
              }
            >
              Ask the agent about solar here
            </button>
          </div>
          <p className="land-muted">
            Hourly weather from NASA POWER, modeled with pvlib. Planning estimates, not a design.
          </p>
        </LensDetails>
      </div>
    </LensCard>
  );
}

/** A planned array: its size now, the model one press away, the settings folded below. */
function DraftBody({
  draft,
  land,
  capacityKw,
  idle,
}: {
  draft: ReturnType<typeof useSolarDraft>;
  land: LandArea;
  capacityKw: number;
  /** Say, before it is asked, when no research worker would pick the model up. */
  idle: boolean;
}) {
  const request = draft.request;
  if (!request) return null;
  const area = draft.preview?.mappedZoneAreaM2 ?? footprintAreaM2(request.arrayZone);
  const target = { key: "solar-draft-zone", geometry: request.arrayZone };
  const acPower =
    draft.preview?.inverterKwAc ??
    (Number.isFinite(request.dcAcRatio) && request.dcAcRatio > 0
      ? capacityKw / request.dcAcRatio
      : Number.NaN);
  const status =
    draft.problem ??
    (draft.checking
      ? "Checking the array fits the land…"
      : draft.preview
        ? `Ready to model with ${request.year} hourly weather. It takes about a minute.`
        : "");
  return (
    <>
      <ul className="land-facts">
        <Fact
          label="Array"
          value={`${zoneArea(area)} of ${zoneArea(land.areaM2)}`}
          target={target}
        />
        <Fact
          label="Panels"
          value={
            Number.isFinite(capacityKw)
              ? `About ${powerLabel(capacityKw)} of panels${Number.isFinite(acPower) ? `, ${powerLabel(acPower)} of inverters` : ""}`
              : "Fill in the panel settings"
          }
          target={target}
        />
      </ul>
      <p className="land-muted" aria-live="polite">
        {status}
      </p>
      {draft.previewError && (
        <p className="land-alert" role="alert">
          {draft.previewError}
        </p>
      )}
      {draft.error && (
        <p className="land-alert" role="alert">
          The model could not start. {draft.error}
        </p>
      )}
      <div className="land-row">
        <GlassButton
          variant="primary"
          disabled={!draft.preview || draft.busy}
          onClick={() => void draft.run()}
        >
          {draft.busy ? "Starting…" : "Run the hourly model"}
        </GlassButton>
        <GlassButton size="sm" variant="ghost" disabled={!draft.canDraw} onClick={draft.redraw}>
          Redraw
        </GlassButton>
        <GlassButton size="sm" variant="ghost" onClick={draft.discard}>
          Discard
        </GlassButton>
      </div>
      {idle && <ResearchWorkerIdle />}
      <SolarSettings request={request} onChange={draft.updateRequest} disabled={draft.busy} />
    </>
  );
}

/** Earlier assessments, newest first: each points at its array and opens it. */
function SavedAssessments({
  items,
  selectedId,
  offset,
  onPage,
  onSelect,
}: {
  items: SolarAssessment[];
  selectedId: string | null;
  offset: number;
  onPage: (offset: number) => void;
  onSelect: (item: SolarAssessment) => void;
}) {
  if (!items.length && offset === 0) return null;
  return (
    <section className="land-solar-section" aria-label="Saved assessments">
      <h4 className="land-solar-heading">Saved assessments</h4>
      <ul className="land-solar-saved">
        {items.map((item) => (
          <MapLink key={item.id} as="li" target={zoneTarget(item)} fly={false}>
            <button
              type="button"
              className="land-solar-saved__row"
              aria-pressed={item.id === selectedId}
              onClick={() => onSelect(item)}
            >
              <span>
                {powerLabel(item.metadata.capacityKwDc)} ·{" "}
                {item.metadata.completeYear && item.metadata.annualGenerationKwh != null
                  ? `${energyLabel(item.metadata.annualGenerationKwh)} a year`
                  : "part year"}
              </span>
              <small>
                {item.request.year} weather · {dateLabel(item.createdAt)}
                {item.stale ? " · earlier outline" : ""}
              </small>
            </button>
          </MapLink>
        ))}
      </ul>
      {(offset > 0 || items.length === PAGE) && (
        <div className="land-row">
          <GlassButton
            size="sm"
            variant="ghost"
            disabled={!offset}
            onClick={() => onPage(Math.max(0, offset - PAGE))}
          >
            Newer
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            disabled={items.length !== PAGE}
            onClick={() => onPage(offset + PAGE)}
          >
            Older
          </GlassButton>
        </div>
      )}
    </section>
  );
}
