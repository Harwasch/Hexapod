import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { EvidenceView } from "./LandResearch";

export function LandEvidencePicker({
  landId,
  value,
  onChange,
  label,
  limit = 20,
}: {
  landId: string;
  value: string[];
  onChange: (ids: string[]) => void;
  label: string;
  limit?: number;
}) {
  const scope = useLandScope(),
    ready = useLandAccessReady();
  const [query, setQuery] = useState(""),
    [offset, setOffset] = useState(0);
  const [open, setOpen] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null);
  const sources = useQuery({
    queryKey: ["land-evidence-options", scope, landId, query, offset],
    enabled: ready && open,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/evidence", {
          params: { path: { land_id: landId }, query: { query, offset, limit: 20 } },
        }),
      ),
    retry: false,
  });
  const inspectedSource = useQuery({
    queryKey: ["land-source", scope, inspected],
    enabled: ready && !!inspected,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: inspected ?? "" } },
        }),
      ),
    retry: false,
  });
  return (
    <details
      className="land-evidence-picker"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        {label} · {value.length} linked
      </summary>
      <label className="land-name">
        Find saved source evidence
        <input
          aria-label={`${label}: find sources`}
          maxLength={100}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOffset(0);
          }}
        />
      </label>
      {sources.isError && (
        <p role="alert">
          Sources could not be loaded.{" "}
          <button type="button" onClick={() => void sources.refetch()}>
            Retry sources
          </button>
        </p>
      )}
      {sources.data?.map((item) => (
        <div key={item.id} className="land-source-choice">
          <label className="land-check">
            <input
              type="checkbox"
              checked={value.includes(item.id)}
              disabled={!value.includes(item.id) && value.length >= limit}
              onChange={(e) =>
                onChange(
                  e.target.checked ? [...value, item.id] : value.filter((id) => id !== item.id),
                )
              }
            />
            {item.title} · boundary {item.boundaryRevision}
          </label>
          <button
            type="button"
            onClick={() => setInspected(item.id)}
            aria-label={`Inspect ${item.title}`}
          >
            Inspect source
          </button>
        </div>
      ))}
      {sources.data?.length === 0 && (
        <p>No saved sources match. Research this land in Discover or Ecology to gather evidence.</p>
      )}
      <div className="land-actions">
        <button
          type="button"
          disabled={offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 20))}
        >
          Previous sources
        </button>
        <button
          type="button"
          disabled={(sources.data?.length ?? 0) < 20}
          onClick={() => setOffset(offset + 20)}
        >
          More sources
        </button>
      </div>
      {value
        .filter((id) => !sources.data?.some((item) => item.id === id))
        .map((id, i) => (
          <div key={id} className="land-actions">
            <button type="button" onClick={() => setInspected(id)}>
              Inspect linked source {i + 1}
            </button>
            <button type="button" onClick={() => onChange(value.filter((item) => item !== id))}>
              Unlink source {i + 1}
            </button>
          </div>
        ))}
      {inspectedSource.isError && <p role="alert">The selected evidence could not be read.</p>}
      {inspected && inspectedSource.data && (
        <EvidenceView evidence={inspectedSource.data} onClose={() => setInspected(null)} />
      )}
    </details>
  );
}
