import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
import { api, unwrap, ApiError } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandScope, useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { LandDocumentViewer } from "./LandDocumentViewer";

type Document = components["schemas"]["LandDocumentRead"];
type Upload = components["schemas"]["LandDocumentCreate"];
type Link = components["schemas"]["DocumentLinkCreate"];
const statusLabel: Record<Document["status"], string> = {
  "awaiting-upload": "Waiting for file",
  ready: "Text ready to research",
  "needs-ocr": "Needs OCR or visual review",
  unreadable: "Original saved; text could not be read",
};

export function LandDocuments({ land }: { land: LandArea }) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const [offset, setOffset] = useState(0),
    [linkOffset, setLinkOffset] = useState(0);
  const [selected, setSelected] = useState<{ id: string; page: number; ocrId?: string } | null>(
    null,
  );
  const [query, setQuery] = useState(""),
    [search, setSearch] = useState("");
  const [searchOffset, setSearchOffset] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [kind, setKind] = useState<Upload["kind"]>("other");
  const [sourceNote, setSourceNote] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [documentDate, setDocumentDate] = useState(""),
    [recordedDate, setRecordedDate] = useState("");
  const [recordingNumber, setRecordingNumber] = useState(""),
    [jurisdiction, setJurisdiction] = useState("");
  const [parcels, setParcels] = useState("");
  const [relevance, setRelevance] = useState(
    "Applicability to the selected land has not been established.",
  );
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const [link, setLink] = useState<Link | null>(null);
  const controller = useRef<AbortController | null>(null);
  const viewer = useRef<HTMLDivElement | null>(null);
  const pending = useRef<{ fingerprint: string; requestKey: string; documentId?: string }>(null);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    viewer.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }, [selected?.id, selected?.page]);
  const records = useQuery({
    queryKey: ["land-documents", scope, land.id, offset],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents", {
          params: { path: { land_id: land.id }, query: { limit: 50, offset } },
        }),
      ),
    retry: false,
  });
  const relations = useQuery({
    queryKey: ["land-document-links", scope, land.id, linkOffset],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/links", {
          params: { path: { land_id: land.id }, query: { limit: 100, offset: linkOffset } },
        }),
      ),
    retry: false,
  });
  const results = useQuery({
    queryKey: ["land-document-search", scope, land.id, search, searchOffset],
    enabled: ready && search.length >= 3,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/search", {
          params: {
            path: { land_id: land.id },
            query: { q: search, limit: 20, offset: searchOffset },
          },
        }),
      ),
    retry: false,
  });
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-documents", scope, land.id] });
  const upload = async () => {
    if (!file) return;
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(null);
    const body: Upload = {
      title,
      filename: file.name,
      sizeBytes: file.size,
      mediaType: file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : "text/plain",
      kind,
      sourceNote,
      sourceUrl: sourceUrl.trim() || null,
      documentDate: documentDate || null,
      recordedDate: recordedDate || null,
      recordingNumber,
      jurisdiction,
      parcelReferences: parcels
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean),
      relevanceNote: relevance,
    };
    const fingerprint = JSON.stringify([body, file.lastModified]);
    if (pending.current?.fingerprint !== fingerprint)
      pending.current = { fingerprint, requestKey: crypto.randomUUID() };
    const operation = pending.current;
    try {
      let id = operation.documentId;
      if (!id) {
        const record = await unwrap(
          api.POST("/api/v1/land/{land_id}/documents", {
            params: { path: { land_id: land.id } },
            body: { ...body, requestKey: operation.requestKey },
            signal: request.signal,
          }),
        );
        operation.documentId = record.id;
        id = record.id;
      }
      const record = await unwrap(
        api.PUT("/api/v1/land/{land_id}/documents/{document_id}/content", {
          params: { path: { land_id: land.id, document_id: id } },
          body: file.name,
          bodySerializer: () => file,
          headers: { "Content-Type": "application/octet-stream" },
          signal: request.signal,
        }),
      );
      if (!request.signal.aborted) {
        setSelected({ id: record.id, page: 1 });
        setFile(null);
        setTitle("");
        setKind("other");
        setSourceNote("");
        setSourceUrl("");
        setDocumentDate("");
        setRecordedDate("");
        setRecordingNumber("");
        setJurisdiction("");
        setParcels("");
        setRelevance("Applicability to the selected land has not been established.");
        pending.current = null;
      }
      await refresh();
      await cache.invalidateQueries({ queryKey: ["land-document-search", scope, land.id] });
    } catch (cause) {
      if (!request.signal.aborted)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      void refresh();
      if (!request.signal.aborted) setBusy(false);
    }
  };
  const saveLink = async () => {
    if (!link) return;
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(null);
    try {
      await unwrap(
        api.POST("/api/v1/land/{land_id}/documents/links", {
          params: { path: { land_id: land.id } },
          body: link,
          signal: request.signal,
        }),
      );
      if (!request.signal.aborted) setLink(null);
      await cache.invalidateQueries({ queryKey: ["land-document-links", scope, land.id] });
    } catch (cause) {
      if (!request.signal.aborted) setError(describeError(cause));
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  return (
    <section className="land-records" aria-label="Land records">
      <span className="land-eyebrow">The evidence beneath the story</span>
      <h3>Records, rights and history.</h3>
      <p>
        Bring deeds, easements, surveys and historical records into the investigation. Trace an
        observation to its page, and compare related instruments over time.
      </p>
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
      {canEdit && (
        <details>
          <summary>Add a land record</summary>
          <form
            className="land-record-form"
            onSubmit={(event) => {
              event.preventDefault();
              void upload();
            }}
          >
            <fieldset disabled={busy}>
              <legend>Private workspace record</legend>
              <label className="land-name">
                Original PDF or text file
                <input
                  type="file"
                  accept=".pdf,.txt,.md"
                  onChange={(event) => {
                    const chosen = event.target.files?.[0];
                    event.target.value = "";
                    if (!chosen) return;
                    if (
                      !/\.(pdf|txt|md)$/i.test(chosen.name) ||
                      chosen.size > 20 * 1024 * 1024 ||
                      chosen.size === 0
                    ) {
                      setError("Choose a PDF or UTF-8 text file between 1 byte and 20 MiB.");
                      return;
                    }
                    setFile(chosen);
                    setTitle(chosen.name.replace(/\.[^.]+$/, ""));
                    setError(null);
                    pending.current = null;
                  }}
                />
              </label>
              {file && (
                <p>
                  {file.name} ·{" "}
                  {(file.size / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} KB
                </p>
              )}
              <label className="land-name">
                Record title
                <input
                  required
                  maxLength={300}
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                />
              </label>
              <label className="land-name">
                Record type
                <select
                  aria-label="Record type"
                  value={kind}
                  onChange={(event) => setKind(event.target.value as Upload["kind"])}
                >
                  {[
                    "deed",
                    "easement",
                    "mineral",
                    "water",
                    "survey",
                    "historical",
                    "report",
                    "other",
                  ].map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label className="land-name">
                Where did this record come from?
                <textarea
                  required
                  maxLength={2000}
                  value={sourceNote}
                  onChange={(event) => setSourceNote(event.target.value)}
                  placeholder="Archive, recording office, report author, or how you obtained it"
                />
              </label>
              <label className="land-name">
                Why might it apply to this land?
                <textarea
                  required
                  maxLength={2000}
                  value={relevance}
                  onChange={(event) => setRelevance(event.target.value)}
                />
              </label>
              <details>
                <summary>Dates, recording details and parcel references</summary>
                <div className="land-action-grid">
                  <label className="land-name">
                    Document date
                    <input
                      type="date"
                      value={documentDate}
                      onChange={(event) => setDocumentDate(event.target.value)}
                    />
                  </label>
                  <label className="land-name">
                    Recorded date
                    <input
                      type="date"
                      value={recordedDate}
                      onChange={(event) => setRecordedDate(event.target.value)}
                    />
                  </label>
                </div>
                <label className="land-name">
                  Recording number
                  <input
                    maxLength={200}
                    value={recordingNumber}
                    onChange={(event) => setRecordingNumber(event.target.value)}
                  />
                </label>
                <label className="land-name">
                  Jurisdiction
                  <input
                    maxLength={300}
                    value={jurisdiction}
                    onChange={(event) => setJurisdiction(event.target.value)}
                  />
                </label>
                <label className="land-name">
                  Parcel references (one per line)
                  <textarea value={parcels} onChange={(event) => setParcels(event.target.value)} />
                </label>
                <label className="land-name">
                  Original source website
                  <input
                    type="url"
                    value={sourceUrl}
                    onChange={(event) => setSourceUrl(event.target.value)}
                  />
                </label>
              </details>
              <div className="land-actions">
                <button disabled={!file || !title.trim() || !sourceNote.trim()}>
                  {busy ? "Saving and extracting pages…" : "Save and read record"}
                </button>
              </div>
              <p className="land-footnote">
                The original remains private to this workspace. Extraction preserves page numbers;
                unreadable scans and handwriting are flagged for further review.
              </p>
            </fieldset>
          </form>
        </details>
      )}
      <form
        className="land-record-search"
        onSubmit={(event) => {
          event.preventDefault();
          setSearch(query.trim());
          setSearchOffset(0);
        }}
      >
        <label className="land-name">
          Search record text
          <input
            minLength={3}
            required
            maxLength={200}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Mineral reservation, right of way, former parcel…"
          />
        </label>
        <div className="land-actions">
          <button>Search records</button>
          {search && (
            <button
              type="button"
              onClick={() => {
                setSearch("");
                setQuery("");
              }}
            >
              Clear search
            </button>
          )}
        </div>
      </form>
      {results.isError && <p role="alert">Record search could not be completed.</p>}
      {search && (
        <div className="land-candidates">
          {results.data?.map((hit) => (
            <button
              key={`${hit.documentId}:${hit.page}:${hit.ocrId ?? "native"}`}
              type="button"
              onClick={() =>
                setSelected({ id: hit.documentId, page: hit.page, ocrId: hit.ocrId ?? undefined })
              }
            >
              <strong>
                {hit.title} · page {hit.page}
                {hit.ocrId ? " · machine OCR" : ""}
              </strong>
              <span>{hit.excerpt}</span>
            </button>
          ))}
          {results.data?.length === 0 && (
            <p>No matching extracted text. Unreadable scans are not included in this search.</p>
          )}
        </div>
      )}
      {search && (searchOffset > 0 || results.data?.length === 20) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!searchOffset}
            onClick={() => setSearchOffset(Math.max(0, searchOffset - 20))}
          >
            Previous matches
          </button>
          <button
            type="button"
            disabled={results.data?.length !== 20}
            onClick={() => setSearchOffset(searchOffset + 20)}
          >
            More matches
          </button>
        </div>
      )}
      {records.isError && (
        <p role="alert">
          Records could not be loaded.{" "}
          <button type="button" onClick={() => void records.refetch()}>
            Retry
          </button>
        </p>
      )}
      <div className="land-candidates">
        {records.data?.map((record) => (
          <button
            type="button"
            key={record.id}
            aria-pressed={selected?.id === record.id}
            onClick={() => setSelected({ id: record.id, page: 1 })}
          >
            <strong>{record.title}</strong>
            <span>
              {record.documentDate ?? "Document date unknown"} · {record.kind} ·{" "}
              {statusLabel[record.status]}
            </span>
          </button>
        ))}
      </div>
      {(offset > 0 || records.data?.length === 50) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - 50))}
          >
            Previous records
          </button>
          <button
            type="button"
            disabled={records.data?.length !== 50}
            onClick={() => setOffset(offset + 50)}
          >
            More records
          </button>
        </div>
      )}
      {selected && (
        <div ref={viewer}>
          <LandDocumentViewer
            key={`${selected.id}:${selected.page}`}
            landId={land.id}
            documentId={selected.id}
            initialPage={selected.page}
            onPageChange={(page) => setSelected({ id: selected.id, page })}
            onUploaded={refresh}
            pinnedOcrId={selected.ocrId}
            onOcrSelected={(ocrId) => setSelected({ ...selected, ocrId })}
          />
        </div>
      )}
      {canEdit && selected && (
        <div className="land-actions">
          <button
            type="button"
            onClick={() => {
              const context = useLandContext.getState();
              const prompt = `Investigate document ${selected.id}, page ${selected.page}${selected.ocrId ? `, OCR extraction ${selected.ocrId} (unverified machine reading)` : ""}. Explain what it says about this land, trace any related records, and distinguish historical statements from rights or conditions that are established today. Cite the exact pages and flag missing evidence.`;
              context.setResearchQuestion(
                [context.researchQuestion, prompt].filter(Boolean).join("\n\n"),
              );
              context.setSection("discover");
              requestAnimationFrame(() => document.getElementById("land-question")?.focus());
            }}
          >
            Ask about this page
          </button>
          <button
            type="button"
            onClick={() =>
              setLink({
                requestKey: crypto.randomUUID(),
                fromDocumentId: selected.id,
                fromPage: selected.page,
                toDocumentId: "",
                toPage: 1,
                relation: "related",
                basis: "",
              })
            }
          >
            Relate this record to another
          </button>
        </div>
      )}
      {link && (
        <form
          className="land-record-form"
          onSubmit={(event) => {
            event.preventDefault();
            void saveLink();
          }}
        >
          <fieldset disabled={busy}>
            <legend>Record relationship</legend>
            <p className="land-footnote">
              Record a connection to investigate. A link alone does not establish current legal
              effect.
            </p>
            <label className="land-name">
              Source page
              <input
                required
                type="number"
                min={1}
                max={500}
                value={link.fromPage}
                onChange={(event) => setLink({ ...link, fromPage: Number(event.target.value) })}
              />
            </label>
            <label className="land-name">
              Related record
              <select
                required
                aria-label="Related record"
                value={link.toDocumentId}
                onChange={(event) => setLink({ ...link, toDocumentId: event.target.value })}
              >
                <option value="">Choose a record</option>
                {records.data
                  ?.filter((record) => record.pageCount > 0)
                  .map((record) => (
                    <option key={record.id} value={record.id}>
                      {record.title}
                    </option>
                  ))}
              </select>
            </label>
            <label className="land-name">
              Related page
              <input
                required
                type="number"
                min={1}
                max={500}
                value={link.toPage}
                onChange={(event) => setLink({ ...link, toPage: Number(event.target.value) })}
              />
            </label>
            <label className="land-name">
              Relationship
              <select
                aria-label="Relationship"
                value={link.relation}
                onChange={(event) =>
                  setLink({ ...link, relation: event.target.value as Link["relation"] })
                }
              >
                {[
                  "amends",
                  "supersedes",
                  "conflicts-with",
                  "mentions",
                  "parcel-lineage",
                  "related",
                ].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label className="land-name">
              What supports this connection?
              <textarea
                required
                maxLength={3000}
                value={link.basis}
                onChange={(event) => setLink({ ...link, basis: event.target.value })}
              />
            </label>
            <label className="land-name">
              Effective date, if established
              <input
                type="date"
                value={link.effectiveDate ?? ""}
                onChange={(event) =>
                  setLink({ ...link, effectiveDate: event.target.value || null })
                }
              />
            </label>
            <div className="land-actions">
              <button>Save relationship</button>
              <button type="button" onClick={() => setLink(null)}>
                Cancel
              </button>
            </div>
          </fieldset>
        </form>
      )}
      <h4>Connections between records</h4>
      {relations.isError && <p role="alert">Relationships could not be loaded.</p>}
      {relations.data?.map((relation) => (
        <article key={relation.id} className="land-record-link">
          <p>
            <strong>{relation.fromTitle}</strong> {relation.relation.replaceAll("-", " ")}{" "}
            <strong>{relation.toTitle}</strong>
          </p>
          <p>{relation.basis}</p>
          {relation.effectiveDate && <p>Recorded effective date: {relation.effectiveDate}</p>}
          <div className="land-actions">
            <button
              type="button"
              onClick={() => setSelected({ id: relation.fromDocumentId, page: relation.fromPage })}
            >
              First record · page {relation.fromPage}
            </button>
            <button
              type="button"
              onClick={() => setSelected({ id: relation.toDocumentId, page: relation.toPage })}
            >
              Related record · page {relation.toPage}
            </button>
          </div>
        </article>
      ))}
      {(linkOffset > 0 || relations.data?.length === 100) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!linkOffset}
            onClick={() => setLinkOffset(Math.max(0, linkOffset - 100))}
          >
            Previous relationships
          </button>
          <button
            type="button"
            disabled={relations.data?.length !== 100}
            onClick={() => setLinkOffset(linkOffset + 100)}
          >
            More relationships
          </button>
        </div>
      )}
    </section>
  );
}
