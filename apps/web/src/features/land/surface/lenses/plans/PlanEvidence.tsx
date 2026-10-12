import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

import { EvidenceView } from "../../../EvidenceView";
import { ScenarioResultView } from "../../../ScenarioResultView";
import type { Action } from "./planModel";

/** The sources a plan cites and the scenario it came from, each opened on request. */
export function PlanEvidence({ action }: { action: Action }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const [id, setId] = useState<string | null>(null);
  const [showScenario, setShowScenario] = useState(false);
  const evidence = useQuery({
    queryKey: ["land-action-evidence", scope, id],
    enabled: ready && Boolean(id),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: id ?? "" } },
        }),
      ),
    retry: false,
  });
  const scenario = useQuery({
    queryKey: ["land-action-scenario", scope, action.scenario?.id, action.scenario?.revision],
    enabled: ready && showScenario && Boolean(action.scenario),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
          params: {
            path: { land_id: action.landId, scenario_id: action.scenario?.id ?? "" },
            query: { revision: action.scenario?.revision },
          },
        }),
      ),
    retry: false,
  });
  const ids = [
    ...new Set([
      ...(action.evidenceIds ?? []),
      ...(action.constraints ?? []).flatMap((constraint) => constraint.evidenceIds ?? []),
    ]),
  ];
  if (!ids.length && !action.scenario) return null;
  return (
    <div className="plans-evidence">
      <div className="land-row">
        {ids.map((identifier, index) => (
          <GlassButton
            key={identifier}
            size="sm"
            active={id === identifier}
            onClick={() => setId(identifier)}
          >
            Source {index + 1}
          </GlassButton>
        ))}
        {action.scenario && (
          <GlassButton
            size="sm"
            active={showScenario}
            onClick={() => setShowScenario(!showScenario)}
          >
            {showScenario ? "Hide its scenario" : "Show its scenario"}
          </GlassButton>
        )}
      </div>
      {evidence.isError && <p role="alert">That source could not be loaded.</p>}
      {evidence.data && id && <EvidenceView evidence={evidence.data} onClose={() => setId(null)} />}
      {showScenario && scenario.isError && <p role="alert">The scenario could not be loaded.</p>}
      {showScenario && scenario.data && (
        <div className="plans-evidence__scenario">
          <strong>
            {scenario.data.name} · revision {scenario.data.revision}
          </strong>
          <ScenarioResultView result={scenario.data.result} />
          <p className="land-muted">{scenario.data.inputs.assumptions}</p>
        </div>
      )}
    </div>
  );
}
