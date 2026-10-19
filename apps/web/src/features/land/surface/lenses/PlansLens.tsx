import { useState } from "react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { actionEdit } from "../../actionDraft";
import { restorationDefaults } from "../../scenarioDefaults";
import { PlanCard } from "./plans/PlanCard";
import { PlanEditor } from "./plans/PlanEditor";
import {
  usePlanRecovery,
  useScenarioRecovery,
  type PlanSeed,
  type ScenarioSeed,
} from "./plans/planRecovery";
import { useActions, useScenarios } from "./plans/plansApi";
import { PlanScenarioEditor } from "./plans/PlanScenarioEditor";
import { PlanScenarios } from "./plans/PlanScenarios";
import {
  editablePlan,
  freshPlan,
  planFromScenario,
  type ActionDraft,
  type Scenario,
} from "./plans/planModel";
import "./plans/plans.css";

type View =
  | { name: "plans" }
  | { name: "compare"; highlight: string | null }
  | { name: "plan-edit"; seed: PlanSeed }
  | { name: "scenario-edit"; seed: ScenarioSeed };

/**
 * Plans: deciding and doing. The card shows one plan at a time, drawn on the map step by step,
 * with its single next step (review, approve, schedule as a mission). Making or revising a plan,
 * comparing possibilities, and editing a restoration scenario are focused views of the same
 * card. Unfinished plans and scenarios are kept in this browser and offered back.
 */
export function PlansLens({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Plans key={`${scope}:${land.id}`} land={land} scope={scope} />;
}

function Plans({ land, scope }: { land: LandArea; scope: string }) {
  const canEdit = useLandCanEdit();
  const [view, setView] = useState<View>({ name: "plans" });
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const actions = useActions(land, offset);
  const scenarios = useScenarios(land);
  const planRecovery = usePlanRecovery(
    land,
    scope,
    (seed) => setView({ name: "plan-edit", seed }),
    (id) => {
      setSelectedId(id);
      setView({ name: "plans" });
    },
  );
  const scenarioRecovery = useScenarioRecovery(
    land,
    scope,
    (seed) => setView({ name: "scenario-edit", seed }),
    (id) => setView({ name: "compare", highlight: id }),
  );
  const planPending = Boolean(planRecovery.recovery);
  const scenarioPending = Boolean(scenarioRecovery.recovery) && scenarioRecovery.kind !== "solar";

  const openPlan = (draft: ActionDraft, editing: PlanSeed["editing"]) => {
    if (planPending) {
      setNotice("Recover or discard the unfinished plan first.");
      setView({ name: "plans" });
      return;
    }
    setNotice(null);
    planRecovery.clearNotice();
    setView({
      name: "plan-edit",
      seed: { draft, editing, conflict: null, notice: null, recovered: false },
    });
  };
  const openScenario = (scenario: Scenario | null) => {
    if (scenarioPending) {
      setNotice("Recover or discard the unfinished scenario first.");
      return;
    }
    setNotice(null);
    scenarioRecovery.clearNotice();
    setView({
      name: "scenario-edit",
      seed: scenario
        ? {
            name: scenario.name,
            inputs: scenario.inputs,
            monitoringText:
              scenario.inputs.kind === "restoration"
                ? scenario.inputs.monitoringYears.join(", ")
                : "1, 3, 5",
            boundaryRevision: scenario.boundaryRevision,
            evidenceIds: scenario.evidenceIds ?? [],
            fieldSurveyIds: scenario.fieldSurveyIds ?? [],
            solarAssessmentId: scenario.solarAssessmentId ?? null,
            requestKey: crypto.randomUUID(),
            editing: { id: scenario.id, revision: scenario.revision, landId: scenario.landId },
            conflict: null,
            notice: null,
            recovered: false,
          }
        : {
            name: "Restoration option",
            inputs: restorationDefaults(),
            monitoringText: "1, 3, 5",
            boundaryRevision: land.revision,
            evidenceIds: [],
            fieldSurveyIds: [],
            solarAssessmentId: null,
            requestKey: crypto.randomUUID(),
            editing: null,
            conflict: null,
            notice: null,
            recovered: false,
          },
    });
  };

  const status = [notice, planRecovery.notice, scenarioRecovery.notice].filter(Boolean);
  const statusLine = status.length ? (
    <p className="land-muted" role="status">
      {status.join(" ")}
    </p>
  ) : null;

  if (view.name === "plan-edit")
    return (
      <PlanEditor
        land={land}
        scope={scope}
        seed={view.seed}
        onSaved={(id) => {
          setOffset(0);
          setSelectedId(id);
          setView({ name: "plans" });
        }}
        onClose={(discarded) => {
          if (!discarded) planRecovery.reload();
          setView({ name: "plans" });
        }}
      />
    );

  if (view.name === "scenario-edit")
    return (
      <PlanScenarioEditor
        land={land}
        scope={scope}
        seed={view.seed}
        onSaved={(id) => setView({ name: "compare", highlight: id })}
        onClose={(discarded) => {
          if (!discarded) scenarioRecovery.reload();
          setView({ name: "compare", highlight: null });
        }}
      />
    );

  if (view.name === "compare") {
    const banner = (
      <>
        {statusLine}
        {canEdit && scenarioRecovery.recovery && (
          <div
            className="land-note plans-recovery"
            role="region"
            aria-label="Recover scenario draft"
          >
            {scenarioRecovery.kind === "solar" ? (
              <>
                <p>
                  An unfinished solar scenario from an earlier version of this app is saved in this
                  browser. Download it to keep its numbers; Solar makes new ones.
                </p>
                <div className="land-row">
                  <GlassButton size="sm" onClick={scenarioRecovery.download}>
                    Download scenario draft
                  </GlassButton>
                  <GlassButton size="sm" variant="ghost" onClick={scenarioRecovery.discard}>
                    Discard scenario draft
                  </GlassButton>
                </div>
              </>
            ) : (
              <>
                <p>An unfinished scenario is saved in this browser.</p>
                <div className="land-row">
                  <GlassButton
                    size="sm"
                    variant="primary"
                    disabled={scenarioRecovery.busy}
                    onClick={() => void scenarioRecovery.recover()}
                  >
                    Recover scenario draft
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    disabled={scenarioRecovery.busy}
                    onClick={() => void scenarioRecovery.recover(true)}
                  >
                    Recover as a new scenario
                  </GlassButton>
                  <GlassButton size="sm" variant="ghost" onClick={scenarioRecovery.download}>
                    Download scenario draft
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="ghost"
                    disabled={scenarioRecovery.busy}
                    onClick={scenarioRecovery.discard}
                  >
                    Discard scenario draft
                  </GlassButton>
                </div>
              </>
            )}
            {scenarioRecovery.error && (
              <p className="land-alert" role="alert">
                {scenarioRecovery.error}
              </p>
            )}
          </div>
        )}
      </>
    );
    return (
      <PlanScenarios
        land={land}
        canEdit={canEdit}
        scenarios={scenarios.data ?? []}
        loading={scenarios.isPending}
        failed={scenarios.isError}
        retry={() => void scenarios.refetch()}
        highlight={view.highlight}
        banner={banner}
        onBack={() => {
          setNotice(null);
          setView({ name: "plans" });
        }}
        onNew={() => openScenario(null)}
        onEdit={openScenario}
        onStartPlan={(scenario) => openPlan(planFromScenario(land, scenario), null)}
      />
    );
  }

  const banner = (
    <>
      {statusLine}
      {canEdit && planRecovery.recovery && (
        <div className="land-note plans-recovery" role="region" aria-label="Recover plan draft">
          <p>An unfinished plan is saved in this browser. Recovering it approves nothing.</p>
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="primary"
              disabled={planRecovery.busy}
              onClick={() => void planRecovery.recover()}
            >
              Recover plan draft
            </GlassButton>
            <GlassButton
              size="sm"
              disabled={planRecovery.busy}
              onClick={() => void planRecovery.recover(true)}
            >
              Recover as a new plan
            </GlassButton>
            <GlassButton size="sm" variant="ghost" onClick={planRecovery.download}>
              Download draft
            </GlassButton>
            <GlassButton
              size="sm"
              variant="ghost"
              disabled={planRecovery.busy}
              onClick={() => {
                planRecovery.discard();
                setNotice(null);
              }}
            >
              Discard draft
            </GlassButton>
          </div>
          {planRecovery.error && (
            <p className="land-alert" role="alert">
              {planRecovery.error}
            </p>
          )}
        </div>
      )}
    </>
  );
  return (
    <PlanCard
      land={land}
      canEdit={canEdit}
      actions={actions.data ?? []}
      loading={actions.isPending}
      failed={actions.isError}
      retry={() => void actions.refetch()}
      offset={offset}
      setOffset={(next) => {
        setOffset(next);
        setSelectedId(null);
      }}
      scenarios={scenarios.data ?? []}
      selectedId={selectedId}
      onSelect={setSelectedId}
      onMake={() => openPlan(freshPlan(land), null)}
      onRevise={(action) => openPlan(editablePlan(action), actionEdit(action))}
      onCompare={() => {
        setNotice(null);
        setView({ name: "compare", highlight: null });
      }}
      banner={banner}
      makeBlocked={planPending}
      compareNote={scenarioPending ? "An unfinished scenario is waiting" : null}
    />
  );
}
