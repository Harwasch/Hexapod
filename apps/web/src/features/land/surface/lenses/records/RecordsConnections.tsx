import { useState } from "react";

import { LINK_PAGE, useDocumentLinks } from "./recordsApi";
import { formatDate, relationLabel } from "./recordsModel";

/**
 * Connections people recorded between documents, a hundred at a time; with `documentId`,
 * only those that touch that document.
 */
export function RecordsConnections({
  landId,
  documentId,
  onOpen,
}: {
  landId: string;
  documentId?: string;
  onOpen: (documentId: string, page: number) => void;
}) {
  const [offset, setOffset] = useState(0);
  const links = useDocumentLinks(landId, offset);
  const rows = (links.data ?? []).filter(
    (link) => !documentId || link.fromDocumentId === documentId || link.toDocumentId === documentId,
  );
  if (documentId && !links.isError && !rows.length && !offset) return null;
  return (
    <section className="records-section" aria-label="Connections between records">
      <h5 className="records-subhead">Connections between records</h5>
      {links.isError && (
        <p className="land-alert" role="alert">
          Connections did not load.{" "}
          <button type="button" className="land-link" onClick={() => void links.refetch()}>
            Try again
          </button>
        </p>
      )}
      {links.data && !rows.length && <p className="land-muted">No connections recorded yet.</p>}
      <ul className="records-list">
        {rows.map((link) => (
          <li key={link.id} className="records-connection">
            <p>
              <button
                type="button"
                className="land-link"
                onClick={() => onOpen(link.fromDocumentId, link.fromPage)}
              >
                {link.fromTitle}, page {link.fromPage}
              </button>{" "}
              {relationLabel(link.relation)}{" "}
              <button
                type="button"
                className="land-link"
                onClick={() => onOpen(link.toDocumentId, link.toPage)}
              >
                {link.toTitle}, page {link.toPage}
              </button>
            </p>
            <p className="land-muted">
              {link.basis}
              {link.effectiveDate ? ` Took effect ${formatDate(link.effectiveDate)}.` : ""}
            </p>
          </li>
        ))}
      </ul>
      {(offset > 0 || links.data?.length === LINK_PAGE) && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - LINK_PAGE))}
          >
            Previous connections
          </button>
          <button
            type="button"
            className="land-link"
            disabled={links.data?.length !== LINK_PAGE}
            onClick={() => setOffset(offset + LINK_PAGE)}
          >
            More connections
          </button>
        </div>
      )}
    </section>
  );
}
