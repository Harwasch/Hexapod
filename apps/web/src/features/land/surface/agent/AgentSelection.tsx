import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";
import { useLandScope } from "@/state/landIdentity";

import { EvidenceView } from "../../EvidenceView";
import { mapValue } from "../../researchMap";
import { MapLink } from "../mapLinks";
import { useAgentUi } from "./agentStore";

/**
 * A research map feature picked on the map (or in a map's feature list): what it is, a way to
 * ask the agent about it, and the source it came from.
 */
export function AgentSelection({ land, canAsk }: { land: LandArea; canAsk: boolean }) {
  const selected = useLandContext((state) => state.selectedMapFeature);
  const layer = useLandContext((state) => (selected ? state.layers[selected.layerId] : null));
  const scope = useLandScope();
  const panel = useRef<HTMLElement>(null);
  const feature = layer?.features.find((item) => item.id === selected?.featureId);
  const [inspect, setInspect] = useState(false);
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const artifactId = layer?.researchArtifactId;
  const artifact = useQuery({
    queryKey: ["land-map-source", scope, land.id, artifactId],
    queryFn: async () => {
      const result = await unwrap(
        api.GET("/api/v1/research/artifacts/{artifact_id}", {
          params: { path: { artifact_id: artifactId ?? "" } },
        }),
      );
      if (result.landId !== land.id)
        throw new Error("This research output belongs to different land.");
      return result;
    },
    enabled: inspect && Boolean(artifactId && feature),
    retry: false,
  });
  const shownEvidence =
    evidenceId && artifact.data?.evidenceIds.includes(evidenceId) ? evidenceId : null;
  const evidence = useQuery({
    queryKey: ["land-map-evidence", scope, artifactId, shownEvidence],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: shownEvidence ?? "" } },
        }),
      ),
    enabled: Boolean(shownEvidence),
    retry: false,
  });
  useEffect(() => {
    if (selected?.origin !== "map" || !feature) return;
    panel.current?.scrollIntoView?.({ block: "nearest" });
  }, [selected, feature]);
  if (!layer || !feature || !artifactId) return null;
  const askable = /^\d+$/.test(feature.id) && Number(feature.id) <= 1999;
  return (
    <section ref={panel} className="land-agent-selection" aria-label="Selected map feature">
      <MapLink
        className="land-agent-selection__what"
        tone="agent"
        target={{ key: `agent-selection:${layer.id}:${feature.id}`, geometry: feature.geometry }}
        label={`${feature.label}: show on map`}
      >
        <span className="land-fact__label">{layer.title}</span>
        <span className="land-fact__value">{feature.label}</span>
        {feature.value != null && (
          <span className="land-fact__note">{mapValue(feature.value, layer.unit)}</span>
        )}
      </MapLink>
      <div className="land-row">
        {canAsk && (
          <GlassButton
            size="sm"
            variant="primary"
            disabled={!askable}
            onClick={() => {
              const context = useLandContext.getState();
              context.setResearchFocus({
                artifactId,
                featureIndex: Number(feature.id),
                label: feature.label,
              });
              if (!context.researchQuestion.trim())
                context.setResearchQuestion("What can you tell me about this feature?");
              useAgentUi.getState().focusInput();
            }}
          >
            Ask about this feature
          </GlassButton>
        )}
        <button
          type="button"
          className="land-link"
          aria-expanded={inspect}
          onClick={() => {
            setInspect((value) => !value);
            setEvidenceId(null);
          }}
        >
          {inspect ? "Hide its source" : "Its source"}
        </button>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Clear map selection"
          onClick={() => useLandContext.getState().clearMapSelection()}
        >
          <X size={14} aria-hidden="true" />
        </GlassButton>
      </div>
      {layer.legend && <p className="land-muted">{layer.legend}</p>}
      {inspect &&
        (artifact.isPending ? (
          <p className="land-muted">Loading its source…</p>
        ) : artifact.isError ? (
          <p className="land-alert" role="alert">
            {describeError(artifact.error)}{" "}
            <button type="button" className="land-link" onClick={() => void artifact.refetch()}>
              Try again
            </button>
          </p>
        ) : (
          artifact.data && (
            <>
              <p className="land-muted">{artifact.data.method}</p>
              {artifact.data.stale && (
                <p className="land-note">Made for an earlier outline of this land.</p>
              )}
              <div className="land-row">
                {artifact.data.evidenceIds.map((id, index) => (
                  <button
                    type="button"
                    key={id}
                    className="land-link"
                    aria-pressed={evidenceId === id}
                    onClick={() => setEvidenceId(id)}
                  >
                    Source {index + 1}
                  </button>
                ))}
              </div>
              {shownEvidence && evidence.isPending && (
                <p className="land-muted">Loading the source…</p>
              )}
              {shownEvidence && evidence.isError && (
                <p className="land-alert" role="alert">
                  {describeError(evidence.error)}
                </p>
              )}
              {shownEvidence && evidence.data && (
                <EvidenceView evidence={evidence.data} onClose={() => setEvidenceId(null)} />
              )}
            </>
          )
        ))}
    </section>
  );
}
