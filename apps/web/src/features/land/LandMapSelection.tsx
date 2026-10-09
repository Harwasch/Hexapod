import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { LandArea } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLandScope } from "@/state/landIdentity";
import { describeError } from "@/lib/log";
import { EvidenceView } from "./LandResearch";
import { mapValue, researchMapBounds } from "./researchMap";

export function LandMapSelection({ land }: { land: LandArea }) {
  const selected = useLandContext((state) => state.selectedMapFeature);
  const layer = useLandContext((state) => selected && state.layers[selected.layerId]);
  const scope = useLandScope();
  const scene = useScene();
  const panel = useRef<HTMLElement>(null);
  const feature = layer?.features.find((item) => item.id === selected?.featureId);
  const [inspect, setInspect] = useState(false),
    [evidenceId, setEvidenceId] = useState<string | null>(null);
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
  const selectedEvidenceId =
    evidenceId && artifact.data?.evidenceIds.includes(evidenceId) ? evidenceId : null;
  const evidence = useQuery({
    queryKey: ["land-map-evidence", scope, artifactId, selectedEvidenceId],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: selectedEvidenceId ?? "" } },
        }),
      ),
    enabled: Boolean(selectedEvidenceId),
    retry: false,
  });
  useEffect(() => {
    if (selected?.origin !== "map" || !feature) return;
    panel.current?.scrollIntoView({ block: "nearest" });
  }, [selected, feature]);
  if (!layer || !feature) return null;
  return (
    <section
      ref={panel}
      className="land-selected-map-feature"
      aria-label="Selected research map feature"
    >
      <p className="land-eyebrow">{layer.title}</p>
      <h4>{feature.label}</h4>
      <p>
        {feature.geometry.type} · {mapValue(feature.value, layer.unit)}
      </p>
      {layer.legend && <p>{layer.legend}</p>}
      <div className="land-actions">
        <button
          type="button"
          disabled={!scene}
          onClick={() => {
            const bounds = researchMapBounds([feature]);
            if (bounds)
              scene?.camera.flyToRectangle(bounds.west, bounds.south, bounds.east, bounds.north);
          }}
        >
          Focus selected feature
        </button>
        <button
          type="button"
          onClick={() => {
            setInspect((value) => !value);
            setEvidenceId(null);
          }}
        >
          {inspect ? "Hide research source" : "Inspect research source"}
        </button>
        <button type="button" onClick={() => useLandContext.getState().clearMapSelection()}>
          Clear map selection
        </button>
      </div>
      {inspect &&
        (artifact.isPending ? (
          <p>Loading research source…</p>
        ) : artifact.isError ? (
          <p role="alert">
            {describeError(artifact.error)}{" "}
            <button type="button" onClick={() => void artifact.refetch()}>
              Retry source
            </button>
          </p>
        ) : (
          artifact.data && (
            <>
              <p>{artifact.data.method}</p>
              {artifact.data.stale && (
                <p>
                  This output uses boundary revision {artifact.data.boundaryRevision}; the current
                  boundary has changed.
                </p>
              )}
              <div className="land-actions">
                {artifact.data.evidenceIds.map((id, index) => (
                  <button
                    type="button"
                    key={id}
                    aria-pressed={evidenceId === id}
                    onClick={() => setEvidenceId(id)}
                  >
                    Source {index + 1}
                  </button>
                ))}
              </div>
              {selectedEvidenceId && evidence.isPending && <p>Loading evidence…</p>}
              {selectedEvidenceId && evidence.isError && (
                <p role="alert">{describeError(evidence.error)}</p>
              )}
              {selectedEvidenceId && evidence.data && (
                <EvidenceView evidence={evidence.data} onClose={() => setEvidenceId(null)} />
              )}
            </>
          )
        ))}
    </section>
  );
}
