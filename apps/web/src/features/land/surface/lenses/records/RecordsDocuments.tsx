import { useState } from "react";
import { FileText, Search, X } from "lucide-react";

import { DOCUMENT_PAGE, SEARCH_PAGE, useDocumentSearch, useLandDocuments } from "./recordsApi";
import { documentYear, kindLabel, namesParcel, STATUS_LABEL } from "./recordsModel";
import { RecordsConnections } from "./RecordsConnections";

/**
 * Every document on the land: search their text, browse them fifty at a time, and follow the
 * connections between them. Opening one hands it to `onOpen` (a page, maybe a machine reading).
 */
export function RecordsDocuments({
  landId,
  parcelIds = [],
  selectedId,
  onOpen,
}: {
  landId: string;
  parcelIds?: string[];
  selectedId?: string | null;
  onOpen: (documentId: string, page: number, ocrId?: string) => void;
}) {
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [searchOffset, setSearchOffset] = useState(0);
  const documents = useLandDocuments(landId, offset);
  const results = useDocumentSearch(landId, search, searchOffset);
  return (
    <section className="records-section" aria-label="Documents">
      <h5 className="records-subhead">Documents</h5>
      <form
        className="records-search"
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          setSearch(query.trim());
          setSearchOffset(0);
        }}
      >
        <Search size={14} aria-hidden="true" />
        <input
          className="glass-input"
          aria-label="Search record text"
          minLength={3}
          required
          maxLength={200}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search the text: right of way, mineral, well…"
        />
        <button type="submit" className="land-visually-hidden">
          Search records
        </button>
        {search && (
          <button
            type="button"
            className="land-icon-button"
            aria-label="Clear search"
            onClick={() => {
              setSearch("");
              setQuery("");
            }}
          >
            <X size={14} />
          </button>
        )}
      </form>
      {results.isError && (
        <p className="land-alert" role="alert">
          The search did not finish. Try again.
        </p>
      )}
      {search && (
        <ul className="records-list" aria-label="Matching pages">
          {results.isPending && (
            <li className="land-muted" role="status">
              Searching…
            </li>
          )}
          {results.data?.map((hit) => (
            <li key={`${hit.documentId}:${hit.page}:${hit.ocrId ?? "text"}`}>
              <button
                type="button"
                className="records-row"
                onClick={() => onOpen(hit.documentId, hit.page, hit.ocrId ?? undefined)}
              >
                <span className="records-row__main">
                  <strong>
                    {hit.title} · page {hit.page}
                    {hit.ocrId ? " · machine reading" : ""}
                  </strong>
                  <small>{hit.excerpt}</small>
                </span>
              </button>
            </li>
          ))}
          {results.data?.length === 0 && (
            <li className="land-muted">
              No page text matches. Scans that were never read are not searched.
            </li>
          )}
        </ul>
      )}
      {search && (searchOffset > 0 || results.data?.length === SEARCH_PAGE) && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={!searchOffset}
            onClick={() => setSearchOffset(Math.max(0, searchOffset - SEARCH_PAGE))}
          >
            Previous matches
          </button>
          <button
            type="button"
            className="land-link"
            disabled={results.data?.length !== SEARCH_PAGE}
            onClick={() => setSearchOffset(searchOffset + SEARCH_PAGE)}
          >
            More matches
          </button>
        </div>
      )}
      {documents.isError && (
        <p className="land-alert" role="alert">
          Documents did not load.{" "}
          <button type="button" className="land-link" onClick={() => void documents.refetch()}>
            Try again
          </button>
        </p>
      )}
      {documents.data?.length === 0 && !offset && <p className="land-muted">No documents yet.</p>}
      <ul className="records-list" aria-label="All documents">
        {documents.data?.map((document) => {
          const year = documentYear(document);
          return (
            <li key={document.id}>
              <button
                type="button"
                className="records-row"
                aria-pressed={selectedId === document.id}
                onClick={() => onOpen(document.id, 1)}
              >
                <FileText size={15} aria-hidden="true" className="records-row__icon" />
                <span className="records-row__main">
                  <strong>{document.title}</strong>
                  <small>
                    {[
                      kindLabel(document.kind),
                      year ? String(year) : "",
                      STATUS_LABEL[document.status],
                      namesParcel(document, parcelIds) ? "names this parcel" : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </small>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {(offset > 0 || documents.data?.length === DOCUMENT_PAGE) && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - DOCUMENT_PAGE))}
          >
            Previous documents
          </button>
          <button
            type="button"
            className="land-link"
            disabled={documents.data?.length !== DOCUMENT_PAGE}
            onClick={() => setOffset(offset + DOCUMENT_PAGE)}
          >
            More documents
          </button>
        </div>
      )}
      <RecordsConnections landId={landId} onOpen={(id, page) => onOpen(id, page)} />
    </section>
  );
}
