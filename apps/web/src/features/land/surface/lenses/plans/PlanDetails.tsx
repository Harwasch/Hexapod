import type { CSSProperties } from "react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { useActionHistory } from "./plansApi";
import { PlanEvidence } from "./PlanEvidence";
import { downloadJson, stepColor, type Action } from "./planModel";
import { dayLabel, money, stepArea, stepTiming } from "./plansFormat";

/** Everything a plan says, written out: goal, every step in full, questions, assumptions. */
export function PlanSummary({ action }: { action: Action }) {
  const days = Math.max(0, ...action.steps.map((step) => step.startDay + step.days));
  return (
    <div className="plans-summary">
      <p>
        <span className="land-muted">Goal</span>
        <br />
        {action.objective}
      </p>
      <p className="land-muted">
        Starts {dayLabel(action.startDate)} · {days} {days === 1 ? "day" : "days"} in all
      </p>
      <ol className="plans-summary__steps">
        {action.steps.map((step, index) => (
          <li key={step.id} style={{ "--step": stepColor(index) } as CSSProperties}>
            <span className="plans-step-badge" aria-hidden="true">
              {index + 1}
            </span>
            <div>
              <strong>{step.title}</strong>
              <p className="land-muted">
                {stepTiming(step, action)} · {step.days} {step.days === 1 ? "day" : "days"} ·{" "}
                {stepArea(step)}
              </p>
              {step.detail && <p>{step.detail}</p>}
              <p>Done when: {step.successMeasure}</p>
              {step.estimatedCost != null && (
                <p>
                  {money(step.estimatedCost, action.currency)}
                  {step.costBasis ? ` · ${step.costBasis}` : ""}
                </p>
              )}
              {!!step.resources?.length && <p>Needs: {step.resources.join(", ")}</p>}
              {!!step.dependsOn?.length && (
                <p>
                  After:{" "}
                  {step.dependsOn
                    .map((id) => action.steps.find((row) => row.id === id)?.title ?? id)
                    .join(", ")}
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>
      {!!action.constraints?.length && (
        <div>
          <span className="land-muted">Questions to settle</span>
          <ul className="plans-summary__list">
            {action.constraints.map((item, index) => (
              <li key={index} className={item.resolved ? undefined : "plans-open"}>
                {item.resolved ? "Settled" : "Open"}: {item.text}
                {item.resolution ? `. ${item.resolution}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
      {!!action.assumptions?.length && (
        <div>
          <span className="land-muted">Assumptions</span>
          <ul className="plans-summary__list">
            {action.assumptions.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </div>
      )}
      <p className="land-muted">
        {action.evidenceIds?.length ?? 0} sources · {action.features?.length ?? 0} mapped assets ·{" "}
        {action.exclusions?.length ?? 0} areas to avoid · boundary revision{" "}
        {action.boundaryRevision}
        {action.scenario ? ` · scenario revision ${action.scenario.revision}` : ""}
      </p>
    </div>
  );
}

/**
 * The plan's details and sources: the full plan, who approved it, its sources and scenario,
 * export, and earlier revisions (each can be shown on the map).
 */
export function PlanDetails({
  land,
  action,
  mapRevision,
  onMapRevision,
}: {
  land: LandArea;
  action: Action;
  mapRevision: number | null;
  onMapRevision: (version: Action | null) => void;
}) {
  const history = useActionHistory(land, action.id, action.revision);
  return (
    <>
      <PlanSummary action={action} />
      {action.approvedAt && (
        <p className="land-muted">
          Revision {action.revision} approved {new Date(action.approvedAt).toLocaleString()}
          {action.approvalNote ? `: ${action.approvalNote}` : ""}
        </p>
      )}
      <PlanEvidence key={`${action.id}:${action.revision}`} action={action} />
      <div className="land-row">
        <GlassButton size="sm" onClick={() => downloadJson(action, "land-action.json")}>
          Export plan
        </GlassButton>
      </div>
      <div className="plans-history" role="group" aria-label="Plan revision history">
        <span className="land-muted">Revisions</span>
        {history.isError && <p role="alert">Earlier revisions could not be loaded.</p>}
        {history.isPending && <p className="land-muted">Loading revisions…</p>}
        <ul>
          {history.data?.map((version) => (
            <li key={version.revision}>
              <span>
                Revision {version.revision} · {version.status}
                {version.title !== action.title ? ` · ${version.title}` : ""}
              </span>
              <span className="land-row">
                <button
                  type="button"
                  className="land-link"
                  aria-pressed={mapRevision === version.revision}
                  onClick={() => onMapRevision(mapRevision === version.revision ? null : version)}
                >
                  {mapRevision === version.revision
                    ? "Showing"
                    : `Map revision ${version.revision}`}
                </button>
                <button
                  type="button"
                  className="land-link"
                  onClick={() => downloadJson(version, "land-action.json")}
                >
                  Export revision {version.revision}
                </button>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}
