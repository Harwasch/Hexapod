/**
 * Outputs: every artifact, what produced it, and what still points at it.
 *
 * The unreferenced filter is the cleanup list. It is a filter rather than a delete
 * button on purpose: "nothing references this" and "this is safe to delete" are not the
 * same sentence — a `manifest.json` is read by people and a `canonical.ply` is what a
 * re-run resumes from — so the console shows the list and leaves the decision to whoever
 * is reading it.
 */
import { useMemo, useState } from "react";

import type { ArtifactKind } from "@twin/contracts";

import { EmptyState, GlassBadge, GlassSelect, Spinner, useFieldId } from "@twin/ui";

import { formatBytes, formatDate, shortId } from "./format";
import { useArtifacts } from "./queries";

type Filter = "all" | "unreferenced" | "referenced";

export function OutputsView() {
  const artifacts = useArtifacts();
  const [kind, setKind] = useState<ArtifactKind | "all">("all");
  const [filter, setFilter] = useState<Filter>("all");
  const kindId = useFieldId("outputs-kind");
  const filterId = useFieldId("outputs-filter");

  const kinds = useMemo(
    () => [...new Set((artifacts.data ?? []).map((row) => row.kind))].sort(),
    [artifacts.data],
  );

  const rows = (artifacts.data ?? []).filter((row) => {
    if (kind !== "all" && row.kind !== kind) return false;
    if (filter === "unreferenced") return row.references.length === 0;
    if (filter === "referenced") return row.references.length > 0;
    return true;
  });

  const total = rows.reduce((sum, row) => sum + (row.bytes ?? 0), 0);

  if (artifacts.isPending) return <Spinner />;
  if (artifacts.isError) {
    return (
      <p className="admin-error" role="alert">
        The API did not answer, so there are no outputs to show.
      </p>
    );
  }

  return (
    <section className="admin-card" aria-labelledby="outputs-heading">
      <div className="admin-card__head">
        <h2 id="outputs-heading">Outputs</h2>
        <div className="admin-card__actions">
          <label className="admin-inline-label" htmlFor={kindId}>
            Kind
          </label>
          <GlassSelect
            id={kindId}
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as ArtifactKind | "all");
            }}
          >
            <option value="all">All kinds</option>
            {kinds.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </GlassSelect>

          <label className="admin-inline-label" htmlFor={filterId}>
            References
          </label>
          <GlassSelect
            id={filterId}
            value={filter}
            onChange={(event) => {
              setFilter(event.target.value as Filter);
            }}
            data-testid="outputs-filter"
          >
            <option value="all">Everything</option>
            <option value="unreferenced">Unreferenced — the cleanup list</option>
            <option value="referenced">Referenced by a site</option>
          </GlassSelect>

          <span className="admin-note">
            {rows.length} artifacts, {formatBytes(total)}
          </span>
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState title="Nothing here" body="No artifact matches this filter." />
      ) : (
        <table className="admin-table" data-testid="outputs-table">
          <caption className="sr-only">Every artifact the pipeline has written</caption>
          <thead>
            <tr>
              <th scope="col">Kind</th>
              <th scope="col">Storage key</th>
              <th scope="col">Bytes</th>
              <th scope="col">Checksum</th>
              <th scope="col">Produced by</th>
              <th scope="col">Capture</th>
              <th scope="col">Written</th>
              <th scope="col">Referenced by</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} data-testid="output-row">
                <td>
                  <GlassBadge>{row.kind}</GlassBadge>
                </td>
                <td className="admin-mono admin-ellipsis">{row.storageKey}</td>
                <td className="admin-num">{formatBytes(row.bytes)}</td>
                <td className="admin-mono admin-ellipsis">{row.checksum ?? "—"}</td>
                <td>
                  {row.stageId} <span className="admin-dim">({row.impl})</span>
                  <br />
                  <span className="admin-mono admin-dim">
                    {row.recipe} · {shortId(row.jobId)}
                  </span>
                </td>
                <td>{row.captureName}</td>
                <td className="admin-num">{formatDate(row.createdAt)}</td>
                <td>
                  {row.references.length === 0 ? (
                    <GlassBadge tone="warning" data-testid="unreferenced">
                      nothing
                    </GlassBadge>
                  ) : (
                    row.references.map((reference) => (
                      <GlassBadge key={`${reference.kind}:${reference.siteId}`} tone="success">
                        {reference.label}
                      </GlassBadge>
                    ))
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
