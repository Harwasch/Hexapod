import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useLandScope, useLandAccessReady } from "@/state/landIdentity";
import { EvidenceView } from "./LandResearch";
import { ScenarioResultView } from "./ScenarioResultView";

export function ActionEvidence({ action }: { action: components["schemas"]["LandActionRead"] }) {
  const scope = useLandScope(),
    ready = useLandAccessReady();
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
  return (
    <div className="land-action-evidence">
      <div className="land-actions">
        {[
          ...new Set([
            ...(action.evidenceIds ?? []),
            ...(action.constraints ?? []).flatMap((constraint) => constraint.evidenceIds ?? []),
          ]),
        ].map((identifier, index) => (
          <button key={identifier} type="button" onClick={() => setId(identifier)}>
            Inspect evidence {index + 1}
          </button>
        ))}
        {action.scenario && (
          <button type="button" onClick={() => setShowScenario(!showScenario)}>
            {showScenario ? "Hide" : "Inspect"} supporting scenario
          </button>
        )}
      </div>
      {evidence.isError && <p role="alert">The supporting evidence could not be loaded.</p>}
      {evidence.data && id && <EvidenceView evidence={evidence.data} onClose={() => setId(null)} />}
      {showScenario && scenario.isError && (
        <p role="alert">The supporting scenario could not be loaded.</p>
      )}
      {showScenario && scenario.data && (
        <div>
          <h4>
            {scenario.data.name} · revision {scenario.data.revision}
          </h4>
          <ScenarioResultView result={scenario.data.result} />
          <p>{scenario.data.inputs.assumptions}</p>
        </div>
      )}
    </div>
  );
}
