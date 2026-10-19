import { useId, useState, type CSSProperties, type ReactNode } from "react";
import { clsx } from "clsx";
import { ClipboardList, MessageCircleQuestion, Plus, Scale } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassSelect } from "@twin/ui";

import { planOnLand } from "@/features/mission/landPlanning";

import { askAgent } from "../../agentBus";
import { areaLabel } from "../../landFlow";
import { LensCard, LensDetails, LensEmpty } from "../../LensCard";
import { Fact, MapLink } from "../../mapLinks";
import { PlanDetails } from "./PlanDetails";
import { PlanLegend } from "./PlanLegend";
import { PlanNextStep } from "./PlanNextStep";
import { PLANS_PAGE, useActionRecord } from "./plansApi";
import { stepColor, type Action, type Scenario } from "./planModel";
import { usePlanOnMap } from "./usePlanOnMap";
import {
  costSummary,
  money,
  planState,
  plansHeadline,
  STATE_LABEL,
  stepArea,
  stepTiming,
} from "./plansFormat";

const SHOWN_STEPS = 6;

function PlanSwitcher({
  actions,
  selected,
  onSelect,
}: {
  actions: Action[];
  selected: Action;
  onSelect: (id: string) => void;
}) {
  if (actions.length > 5)
    return (
      <GlassSelect
        aria-label="Plan"
        className="plans-switcher__select"
        value={selected.id}
        onChange={(event) => onSelect(event.target.value)}
      >
        {actions.map((action) => (
          <option key={action.id} value={action.id}>
            {action.title} · {STATE_LABEL[planState(action)]}
          </option>
        ))}
      </GlassSelect>
    );
  return (
    <div className="plans-switcher" role="group" aria-label="Choose a plan">
      {actions.map((action) => (
        <button
          key={action.id}
          type="button"
          aria-pressed={action.id === selected.id}
          className="plans-switcher__chip"
          onClick={() => onSelect(action.id)}
          title={`${action.title} · ${STATE_LABEL[planState(action)]}`}
        >
          <i className={`plans-dot plans-dot--${planState(action)}`} aria-hidden="true" />
          <span>{action.title}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * "Plan work on this land": the Plan drawer opens on a new mission plan with this land as its
 * ground, and the command bar takes the sentence of work. Viewers see why they cannot.
 */
function PlanWork({ land, canEdit }: { land: LandArea; canEdit: boolean }) {
  const [opening, setOpening] = useState(false);
  const noteId = useId();
  return (
    <div className="plans-work">
      <GlassButton
        variant="primary"
        loading={opening}
        disabled={!canEdit || opening}
        leadingIcon={<ClipboardList size={15} aria-hidden="true" />}
        aria-describedby={canEdit ? undefined : noteId}
        data-testid="land-plan-work"
        onClick={() => {
          setOpening(true);
          void planOnLand(land).finally(() => setOpening(false));
        }}
      >
        Plan work on this land
      </GlassButton>
      {!canEdit && (
        <p className="land-muted" id={noteId} data-testid="land-plan-work-note">
          Viewers can see this land&apos;s plans. A workspace editor can plan work here.
        </p>
      )}
    </div>
  );
}

/**
 * The plans card: one sentence about every plan, the way to plan robot work on this land, then
 * one plan at a time with its steps (each tied to its area on the map), its known cost and
 * work area, and the one next step.
 */
export function PlanCard({
  land,
  canEdit,
  actions,
  loading,
  failed,
  retry,
  offset,
  setOffset,
  scenarios,
  selectedId,
  onSelect,
  onMake,
  onRevise,
  onCompare,
  banner,
  makeBlocked,
  compareNote,
}: {
  land: LandArea;
  canEdit: boolean;
  actions: Action[];
  loading: boolean;
  failed: boolean;
  retry: () => void;
  offset: number;
  setOffset: (offset: number) => void;
  scenarios: Scenario[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onMake: () => void;
  onRevise: (action: Action) => void;
  onCompare: () => void;
  banner?: ReactNode;
  /** An unfinished plan waits to be recovered; a new one would overwrite it. */
  makeBlocked: boolean;
  compareNote?: string | null;
}) {
  const record = useActionRecord(land, selectedId);
  const selected =
    record.data ?? actions.find((item) => item.id === selectedId) ?? actions[0] ?? null;
  const [mapVersion, setMapVersion] = useState<Action | null>(null);
  const [allSteps, setAllSteps] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const shown = mapVersion && mapVersion.id === selected?.id ? mapVersion : selected;
  usePlanOnMap(shown, (id) => {
    setAllSteps(true);
    setFlash(id);
    requestAnimationFrame(() => {
      const row = document.getElementById(`plans-step-${id}`)?.closest("li");
      row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      row?.focus({ preventScroll: true });
    });
  });

  const select = (id: string) => {
    setMapVersion(null);
    setAllSteps(false);
    setFlash(null);
    onSelect(id);
  };
  const state = selected ? planState(selected) : null;
  const headline = loading
    ? undefined
    : failed && !actions.length
      ? "Plans could not be loaded."
      : actions.length
        ? plansHeadline(actions, scenarios, selected)
        : scenarios.length
          ? `No plans yet. ${scenarios.length === 1 ? "One possibility" : `${scenarios.length} possibilities`} to compare.`
          : "No plans yet.";
  const steps = selected
    ? allSteps || selected.steps.length <= SHOWN_STEPS
      ? selected.steps
      : selected.steps.slice(0, SHOWN_STEPS - 1)
    : [];
  const exclusions = selected?.exclusions?.length ?? 0;

  return (
    <LensCard
      lens="plans"
      headline={headline}
      action={
        canEdit && actions.length > 0 ? (
          <GlassButton
            size="sm"
            variant="ghost"
            disabled={makeBlocked}
            leadingIcon={<Plus size={14} aria-hidden="true" />}
            onClick={onMake}
          >
            New plan
          </GlassButton>
        ) : undefined
      }
    >
      <PlanWork land={land} canEdit={canEdit} />
      {banner}
      {loading && (
        <p className="land-muted" aria-live="polite">
          Loading plans…
        </p>
      )}
      {failed && (
        <p className="land-alert" role="alert">
          Plans could not be loaded.{" "}
          <button type="button" className="land-link" onClick={retry}>
            Try again
          </button>
        </p>
      )}
      {!loading && !failed && !actions.length && (
        <LensEmpty
          action={
            <>
              {canEdit && !makeBlocked && (
                <GlassButton leadingIcon={<Plus size={15} aria-hidden="true" />} onClick={onMake}>
                  Make a plan
                </GlassButton>
              )}
              <GlassButton leadingIcon={<Scale size={15} aria-hidden="true" />} onClick={onCompare}>
                Compare possibilities
              </GlassButton>
            </>
          }
        >
          Lay out the work step by step and draw where each step happens. Then review it and
          schedule it as a robot mission.
        </LensEmpty>
      )}
      {selected && state && (
        <>
          {actions.length > 1 && (
            <PlanSwitcher actions={actions} selected={selected} onSelect={select} />
          )}
          <section className="plans-plan" aria-label={`Plan: ${selected.title}`}>
            <div className="plans-plan__head">
              <h4>{selected.title}</h4>
              <span className={`plans-state plans-state--${state}`}>{STATE_LABEL[state]}</span>
            </div>
            {mapVersion?.id === selected.id && (
              <p className="land-note">
                The map shows revision {mapVersion.revision}.{" "}
                <button type="button" className="land-link" onClick={() => setMapVersion(null)}>
                  Show the current plan
                </button>
              </p>
            )}
            <ol className="plans-steps" aria-label="Steps">
              {steps.map((step, index) => (
                <MapLink
                  key={step.id}
                  as="li"
                  className={clsx("plans-step", flash === step.id && "is-flash")}
                  target={{
                    key: `plans-step:${step.id}`,
                    geometry: step.footprint ?? selected.effectiveBoundary,
                  }}
                  label={`Step ${index + 1}, ${step.title}: show on map`}
                >
                  <span
                    className="plans-step-badge"
                    style={{ "--step": stepColor(index) } as CSSProperties}
                    aria-hidden="true"
                  >
                    {index + 1}
                  </span>
                  <span className="plans-step__title" id={`plans-step-${step.id}`}>
                    {step.title}
                    <small>{stepTiming(step, selected)}</small>
                  </span>
                  <span className="plans-step__area">{stepArea(step)}</span>
                  <span className="plans-step__cost">
                    {step.estimatedCost != null
                      ? money(step.estimatedCost, selected.currency)
                      : "—"}
                  </span>
                </MapLink>
              ))}
            </ol>
            {steps.length < selected.steps.length && (
              <button type="button" className="land-link" onClick={() => setAllSteps(true)}>
                Show all {selected.steps.length} steps
              </button>
            )}
            <ul className="land-facts">
              <Fact label="Known cost" value={costSummary(selected)} tone="neutral" />
              <Fact
                label="Work area"
                value={`${areaLabel(footprintAreaM2(selected.effectiveBoundary))}${
                  exclusions
                    ? `, avoiding ${exclusions === 1 ? "one area" : `${exclusions} areas`}`
                    : ""
                }`}
                target={{ key: "plans-work-area", geometry: selected.effectiveBoundary }}
              />
            </ul>
            <PlanLegend
              steps={selected.steps.some((step) => step.footprint)}
              avoid={exclusions > 0}
            />
            <PlanNextStep
              key={`${selected.id}:${selected.revision}`}
              land={land}
              action={selected}
              canEdit={canEdit}
              onRevise={() => onRevise(selected)}
            />
          </section>
          {record.isError && (
            <p className="land-alert" role="alert">
              This plan could not be refreshed.{" "}
              <button type="button" className="land-link" onClick={() => void record.refetch()}>
                Try again
              </button>
            </p>
          )}
        </>
      )}
      {(actions.length > 0 || loading || failed) && (
        <div className="plans-links">
          <button type="button" className="land-link" onClick={onCompare}>
            <Scale size={14} aria-hidden="true" /> Compare possibilities
            {scenarios.length ? ` · ${scenarios.length}` : ""}
          </button>
          {compareNote && <span className="land-muted">{compareNote}</span>}
          {selected && (
            <button
              type="button"
              className="land-link"
              onClick={() =>
                askAgent(
                  `What could go wrong with the plan "${selected.title}" on this land, and what should I check before approving it?`,
                )
              }
            >
              <MessageCircleQuestion size={14} aria-hidden="true" /> Ask about this plan
            </button>
          )}
        </div>
      )}
      {selected && (
        <LensDetails>
          <PlanDetails
            land={land}
            action={selected}
            mapRevision={mapVersion?.id === selected.id ? mapVersion.revision : null}
            onMapRevision={setMapVersion}
          />
          {(offset > 0 || actions.length === PLANS_PAGE) && (
            <div className="land-row">
              <GlassButton
                size="sm"
                disabled={!offset}
                onClick={() => setOffset(Math.max(0, offset - PLANS_PAGE))}
              >
                Previous plans
              </GlassButton>
              <GlassButton
                size="sm"
                disabled={actions.length !== PLANS_PAGE}
                onClick={() => setOffset(offset + PLANS_PAGE)}
              >
                More plans
              </GlassButton>
            </div>
          )}
        </LensDetails>
      )}
    </LensCard>
  );
}
