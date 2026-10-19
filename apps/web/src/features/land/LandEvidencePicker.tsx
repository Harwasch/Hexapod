import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

import { EvidenceView } from "./EvidenceView";
import { formatDate, providerLabel } from "./surface/lenses/records/recordsModel";
import "./surface/lenses/records/records.css";

const PAGE = 20;

/**
 * Picks saved sources (evidence from any analysis or document) to back a plan, a target or an
 * action: find them by name, tick up to `limit`, and inspect one before linking it.
 */
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
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null);
  const sources = useQuery({
    queryKey: ["land-evidence-options", scope, landId, query, offset],
    enabled: ready && open,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/evidence", {
          params: { path: { land_id: landId }, query: { query, offset, limit: PAGE } },
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
  const listed = new Set(sources.data?.map((item) => item.id));
  const unlisted = value.filter((id) => !listed.has(id));
  return (
    <details
      className="records-more land-evidence-picker"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        {label} · {value.length} linked
      </summary>
      <div className="records-search">
        <Search size={14} aria-hidden="true" />
        <input
          className="glass-input"
          aria-label={`${label}: find sources`}
          placeholder="Find saved sources by name"
          maxLength={100}
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setOffset(0);
          }}
        />
      </div>
      {sources.isPending && open && ready && (
        <p className="land-muted" role="status">
          Loading sources…
        </p>
      )}
      {sources.isError && (
        <p className="land-alert" role="alert">
          Sources did not load.{" "}
          <button type="button" className="land-link" onClick={() => void sources.refetch()}>
            Retry sources
          </button>
        </p>
      )}
      <ul className="records-list">
        {sources.data?.map((item) => {
          const checked = value.includes(item.id);
          return (
            <li key={item.id} className="records-row records-row--choice">
              <label className="records-row__main">
                <span>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!checked && value.length >= limit}
                    onChange={(event) =>
                      onChange(
                        event.target.checked
                          ? [...value, item.id]
                          : value.filter((id) => id !== item.id),
                      )
                    }
                  />{" "}
                  <strong>{item.title}</strong>
                </span>
                <small>
                  {providerLabel(item.provider)} · {formatDate(item.retrievedAt)} · outline v
                  {item.boundaryRevision}
                </small>
              </label>
              <button
                type="button"
                className="land-link"
                onClick={() => setInspected(item.id)}
                aria-label={`Inspect ${item.title}`}
              >
                Inspect source
              </button>
            </li>
          );
        })}
      </ul>
      {sources.data?.length === 0 && (
        <p className="land-muted">
          No saved sources match. Run an analysis from any lens, or add a document in Records, to
          gather sources.
        </p>
      )}
      {(offset > 0 || sources.data?.length === PAGE) && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE))}
          >
            Previous sources
          </button>
          <button
            type="button"
            className="land-link"
            disabled={(sources.data?.length ?? 0) < PAGE}
            onClick={() => setOffset(offset + PAGE)}
          >
            More sources
          </button>
        </div>
      )}
      {unlisted.map((id, index) => (
        <div key={id} className="land-row">
          <span className="land-muted">Linked source {index + 1}</span>
          <button type="button" className="land-link" onClick={() => setInspected(id)}>
            Inspect linked source {index + 1}
          </button>
          <button
            type="button"
            className="land-link"
            onClick={() => onChange(value.filter((item) => item !== id))}
          >
            Unlink source {index + 1}
          </button>
        </div>
      ))}
      {inspectedSource.isError && (
        <p className="land-alert" role="alert">
          That source could not be read.
        </p>
      )}
      {inspected && inspectedSource.data && (
        <EvidenceView evidence={inspectedSource.data} onClose={() => setInspected(null)} />
      )}
    </details>
  );
}
