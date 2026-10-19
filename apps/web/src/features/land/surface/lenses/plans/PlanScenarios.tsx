import { useState, type ReactNode } from "react";
import { ArrowLeft, MessageCircleQuestion, Plus, Sun } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";

import { ScenarioResultView } from "../../../ScenarioResultView";
import { askAgent } from "../../agentBus";
import { LensCard, LensDetails, LensEmpty } from "../../LensCard";
import { useLens } from "../../lenses";
import { useScenarioSurveys } from "./plansApi";
import { downloadJson, type Scenario } from "./planModel";
import { scenarioKindLabel, scenarioNumbers, scenariosHeadline } from "./plansFormat";

const ROWS = ["Area", "Cost", "Outcome"] as const;

function openSurvey(id: string) {
  useLandContext.getState().selectSurvey(id);
  useLens.getState().setLens("ecology");
}

/** One scenario's full record: assumptions, results, sources and revisions. */
function ScenarioRecord({ land, scenario }: { land: LandArea; scenario: Scenario }) {
  const surveys = useScenarioSurveys(land);
  const [history, setHistory] = useState<Scenario[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const surveyName = (id: string) => surveys.data?.find((item) => item.id === id)?.name ?? id;
  return (
    <details className="plans-more">
      <summary>
        {scenario.name} · revision {scenario.revision}
        {scenario.stale ? " · earlier outline" : ""}
      </summary>
      <p className="land-muted">{scenario.inputs.assumptions}</p>
      <ScenarioResultView result={scenario.result} />
      {scenario.solarAssessmentId && (
        <button
          type="button"
          className="land-link"
          onClick={() => {
            useLandContext.getState().selectSolar(scenario.solarAssessmentId ?? null);
            useLens.getState().setLens("solar");
          }}
        >
          Open its hourly solar assessment in Solar
        </button>
      )}
      {!!scenario.fieldSurveyIds?.length && (
        <div className="land-row">
          {scenario.fieldSurveyIds.map((id) => (
            <GlassButton key={id} size="sm" onClick={() => openSurvey(id)}>
              Open survey {surveyName(id)}
            </GlassButton>
          ))}
        </div>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          onClick={() => {
            setError(null);
            void unwrap(
              api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}/revisions", {
                params: { path: { land_id: land.id, scenario_id: scenario.id } },
              }),
            )
              .then(setHistory)
              .catch((cause: unknown) => setError(describeError(cause)));
          }}
        >
          Revision history
        </GlassButton>
        <GlassButton size="sm" onClick={() => downloadJson(scenario, "land-scenario.json")}>
          Export assumptions and results
        </GlassButton>
      </div>
      {error && <p role="alert">{error}</p>}
      {history && (
        <div className="plans-history" role="group" aria-label={`${scenario.name} revisions`}>
          {history.map((version) => (
            <details key={version.revision} className="plans-more">
              <summary>
                Revision {version.revision} · {new Date(version.updatedAt).toLocaleDateString()} ·
                boundary revision {version.boundaryRevision}
              </summary>
              <ScenarioResultView result={version.result} />
              {!!version.fieldSurveyIds?.length && (
                <div className="land-row">
                  {version.fieldSurveyIds.map((id) => (
                    <GlassButton key={id} size="sm" onClick={() => openSurvey(id)}>
                      Open survey {surveyName(id)}
                    </GlassButton>
                  ))}
                </div>
              )}
            </details>
          ))}
        </div>
      )}
    </details>
  );
}

/**
 * Possibilities side by side: up to three scenarios, each by its area, cost and outcome, with a
 * bar so costs compare at a glance. A scenario can start a plan. Restoration scenarios are made
 * and edited here; solar ones come from the Solar lens and only open there.
 */
export function PlanScenarios({
  land,
  canEdit,
  scenarios,
  loading,
  failed,
  retry,
  highlight,
  banner,
  onBack,
  onNew,
  onEdit,
  onStartPlan,
}: {
  land: LandArea;
  canEdit: boolean;
  scenarios: Scenario[];
  loading: boolean;
  failed: boolean;
  retry: () => void;
  highlight: string | null;
  banner?: ReactNode;
  onBack: () => void;
  onNew: () => void;
  onEdit: (scenario: Scenario) => void;
  onStartPlan: (scenario: Scenario) => void;
}) {
  const [picked, setPicked] = useState<string[] | null>(null);
  const chosenIds = (
    picked ?? [
      ...(highlight && scenarios.some((item) => item.id === highlight) ? [highlight] : []),
      ...scenarios.map((item) => item.id),
    ]
  )
    .filter((id, index, all) => all.indexOf(id) === index)
    .filter((id) => scenarios.some((item) => item.id === id))
    .slice(0, 3);
  const chosen = chosenIds.flatMap((id) => scenarios.filter((item) => item.id === id));
  const numbers = chosen.map((scenario) => scenarioNumbers(scenario));
  const sameKind = chosen.every((item) => item.inputs.kind === chosen[0]?.inputs.kind);
  const currencies = chosen.map((item) =>
    String(item.result.summary.currency ?? item.inputs.currency),
  );
  const maxCost = (currency: string) =>
    Math.max(
      0,
      ...numbers.flatMap((row, index) =>
        currencies[index] === currency ? [row[1]?.amount ?? 0] : [],
      ),
    );
  const toggle = (id: string) => {
    const current = chosenIds;
    if (current.includes(id)) setPicked(current.filter((item) => item !== id));
    else if (current.length < 3) setPicked([...current, id]);
  };

  return (
    <LensCard
      lens="plans"
      headline={loading ? undefined : scenariosHeadline(scenarios)}
      action={
        <GlassButton
          size="sm"
          variant="ghost"
          aria-label="Back to plans"
          leadingIcon={<ArrowLeft size={14} aria-hidden="true" />}
          onClick={onBack}
        >
          Plans
        </GlassButton>
      }
    >
      {banner}
      {loading && (
        <p className="land-muted" aria-live="polite">
          Loading scenarios…
        </p>
      )}
      {failed && (
        <p className="land-alert" role="alert">
          Scenarios could not be loaded.{" "}
          <button type="button" className="land-link" onClick={retry}>
            Try again
          </button>
        </p>
      )}
      {!loading && !failed && !scenarios.length && (
        <LensEmpty
          action={
            canEdit ? (
              <GlassButton
                variant="primary"
                leadingIcon={<Plus size={15} aria-hidden="true" />}
                onClick={onNew}
              >
                New restoration scenario
              </GlassButton>
            ) : undefined
          }
        >
          A scenario tests an idea before you commit: what restoring cover would cost, or what solar
          panels would earn. Compare them here, then start a plan from the best one.
        </LensEmpty>
      )}
      {scenarios.length > 3 && (
        <div className="plans-switcher" role="group" aria-label="Choose up to three to compare">
          {scenarios.map((scenario) => (
            <button
              key={scenario.id}
              type="button"
              className="plans-switcher__chip"
              aria-pressed={chosenIds.includes(scenario.id)}
              disabled={!chosenIds.includes(scenario.id) && chosenIds.length >= 3}
              onClick={() => toggle(scenario.id)}
            >
              <span>{scenario.name}</span>
            </button>
          ))}
        </div>
      )}
      {chosen.length > 0 && (
        <div className="plans-compare-scroll">
          <table className="plans-compare">
            <caption className="land-visually-hidden">Scenario comparison</caption>
            <thead>
              <tr>
                <th scope="col">
                  <span className="land-visually-hidden">Measure</span>
                </th>
                {chosen.map((scenario) => (
                  <th
                    key={scenario.id}
                    scope="col"
                    className={scenario.id === highlight ? "is-new" : undefined}
                  >
                    <span className="plans-compare__name">{scenario.name}</span>
                    <span className="plans-tag">{scenarioKindLabel(scenario)}</span>
                    {scenario.stale && (
                      <span className="plans-tag plans-tag--warn">Earlier outline</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row, rowIndex) => (
                <tr key={row}>
                  <th scope="row">{sameKind ? (numbers[0]?.[rowIndex]?.label ?? row) : row}</th>
                  {chosen.map((scenario, column) => {
                    const cell = numbers[column]?.[rowIndex];
                    const top = maxCost(currencies[column] ?? "");
                    return (
                      <td key={scenario.id}>
                        {cell?.value ?? "—"}
                        {rowIndex === 1 && cell?.amount != null && top > 0 && (
                          <i
                            className="plans-bar"
                            aria-hidden="true"
                            style={{ width: `${Math.max(4, (cell.amount / top) * 100)}%` }}
                          />
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td />
                {chosen.map((scenario) => (
                  <td key={scenario.id}>
                    <div className="plans-compare__actions">
                      {canEdit && (
                        <GlassButton
                          size="sm"
                          aria-label={`Start a plan from ${scenario.name}`}
                          onClick={() => onStartPlan(scenario)}
                        >
                          Start a plan
                        </GlassButton>
                      )}
                      {scenario.inputs.kind === "restoration" && canEdit && (
                        <button
                          type="button"
                          className="land-link"
                          aria-label={`Edit ${scenario.name}`}
                          onClick={() => onEdit(scenario)}
                        >
                          Edit
                        </button>
                      )}
                      {scenario.inputs.kind === "solar" && (
                        <button
                          type="button"
                          className="land-link"
                          aria-label={`Open ${scenario.name} in Solar`}
                          onClick={() => useLens.getState().setLens("solar")}
                        >
                          Open in Solar
                        </button>
                      )}
                    </div>
                  </td>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>
      )}
      {scenarios.length > 0 && (
        <div className="land-row">
          {canEdit && (
            <GlassButton
              size="sm"
              leadingIcon={<Plus size={14} aria-hidden="true" />}
              onClick={onNew}
            >
              New restoration scenario
            </GlassButton>
          )}
        </div>
      )}
      <div className="plans-links">
        <button
          type="button"
          className="land-link"
          onClick={() =>
            askAgent(
              "Help me explore possible changes to this land. Start with my goals, the constraints, and the evidence needed to compare options.",
            )
          }
        >
          <MessageCircleQuestion size={14} aria-hidden="true" /> Ask the agent about options
        </button>
        <button
          type="button"
          className="land-link"
          onClick={() => useLens.getState().setLens("solar")}
        >
          <Sun size={14} aria-hidden="true" /> Solar scenarios are made in Solar
        </button>
      </div>
      {scenarios.length > 0 && (
        <LensDetails>
          {scenarios.map((scenario) => (
            <ScenarioRecord key={scenario.id} land={land} scenario={scenario} />
          ))}
        </LensDetails>
      )}
    </LensCard>
  );
}
