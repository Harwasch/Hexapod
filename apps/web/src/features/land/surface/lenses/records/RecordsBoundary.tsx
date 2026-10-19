import { ArrowUpRight } from "lucide-react";

import type { LandArea } from "@twin/contracts";

import { sourceMeaning } from "../../sourceLabels";
import { formatDate } from "./recordsModel";

const METHOD: Record<LandArea["source"]["method"], string> = {
  parcel: "Parcel record",
  drawn: "Drawn on the map",
  corridor: "Corridor along a line",
  imported: "Imported file",
  "mapped-feature": "Mapped feature",
  imagery: "Traced from imagery",
};

const safeUrl = (url: string | null | undefined) => (url && /^https?:\/\//i.test(url) ? url : null);

/** Where the outline came from, what it means, and the records it was traced from. */
export function RecordsBoundary({ land }: { land: LandArea }) {
  const source = land.source;
  const url = safeUrl(source.url);
  const references = source.references ?? [];
  return (
    <section className="records-section" aria-label="Where the boundary came from">
      <h5 className="records-subhead">Where the boundary came from</h5>
      <dl className="records-facts">
        <div>
          <dt>Source</dt>
          <dd>{source.label}</dd>
        </div>
        <div>
          <dt>How</dt>
          <dd>{METHOD[source.method]}</dd>
        </div>
        {source.recordId && (
          <div>
            <dt>Record number</dt>
            <dd>{source.recordId}</dd>
          </div>
        )}
        {source.observedAt && (
          <div>
            <dt>Observed</dt>
            <dd>{formatDate(source.observedAt)}</dd>
          </div>
        )}
        <div>
          <dt>Revision</dt>
          <dd>
            {land.revision} · saved {formatDate(land.updatedAt)}
          </dd>
        </div>
      </dl>
      <p className="land-muted">{sourceMeaning(source.meaning)}</p>
      {source.attribution && <p className="land-muted">{source.attribution}</p>}
      {url && (
        <a className="records-out" href={url} target="_blank" rel="noopener noreferrer">
          Open the source <ArrowUpRight size={13} aria-hidden="true" />
        </a>
      )}
      {references.length > 0 && (
        <details className="records-more">
          <summary>
            Traced from {references.length === 1 ? "1 record" : `${references.length} records`}
          </summary>
          <ul className="records-plain-list">
            {references.map((reference, index) => {
              const link = safeUrl(reference.url);
              return (
                <li key={`${reference.label}:${reference.recordId ?? index}`}>
                  <strong>{reference.label}</strong>
                  {reference.recordId ? ` · record ${reference.recordId}` : ""}
                  {reference.observedAt ? ` · observed ${formatDate(reference.observedAt)}` : ""}
                  {reference.attribution && (
                    <span className="land-muted"> · {reference.attribution}</span>
                  )}
                  {link && (
                    <>
                      {" "}
                      <a href={link} target="_blank" rel="noopener noreferrer">
                        Open
                      </a>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </section>
  );
}
