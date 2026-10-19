import { useEffect, useId, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { FileUp } from "lucide-react";

import type { components } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, ApiError, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandScope } from "@/state/landIdentity";

import { DOCUMENT_KINDS, formatBytes, type DocumentKind, type LandDocument } from "./recordsModel";

type Upload = components["schemas"]["LandDocumentCreate"];

const DEFAULT_RELEVANCE = "Applicability to the selected land has not been established.";
const MAX_BYTES = 20 * 1024 * 1024;

const EMPTY = {
  title: "",
  kind: "other" as DocumentKind,
  sourceNote: "",
  relevance: DEFAULT_RELEVANCE,
  documentDate: "",
  recordedDate: "",
  recordingNumber: "",
  jurisdiction: "",
  parcels: "",
  sourceUrl: "",
};

/**
 * Adds a PDF or text record to the land. The record is created, then its file sent; a retry
 * after a dropped connection reuses the same request key and document, so nothing doubles.
 */
export function RecordsAddDocument({
  landId,
  onSaved,
  onCancel,
}: {
  landId: string;
  onSaved: (document: LandDocument) => void;
  onCancel?: () => void;
}) {
  const scope = useLandScope();
  const cache = useQueryClient();
  const fileId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [fields, setFields] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const pending = useRef<{ fingerprint: string; requestKey: string; documentId?: string } | null>(
    null,
  );
  useEffect(() => () => controller.current?.abort(), []);
  const set = (change: Partial<typeof EMPTY>) =>
    setFields((current) => ({ ...current, ...change }));
  const refresh = async () => {
    await cache.invalidateQueries({ queryKey: ["land-documents", scope, landId] });
    await cache.invalidateQueries({ queryKey: ["land-document-search", scope, landId] });
  };

  const upload = async () => {
    if (!file) return;
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(null);
    const body: Upload = {
      title: fields.title,
      filename: file.name,
      sizeBytes: file.size,
      mediaType: file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : "text/plain",
      kind: fields.kind,
      sourceNote: fields.sourceNote,
      sourceUrl: fields.sourceUrl.trim() || null,
      documentDate: fields.documentDate || null,
      recordedDate: fields.recordedDate || null,
      recordingNumber: fields.recordingNumber,
      jurisdiction: fields.jurisdiction,
      parcelReferences: fields.parcels
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean),
      relevanceNote: fields.relevance,
    };
    const fingerprint = JSON.stringify([body, file.lastModified]);
    if (pending.current?.fingerprint !== fingerprint)
      pending.current = { fingerprint, requestKey: crypto.randomUUID() };
    const operation = pending.current;
    try {
      let id = operation.documentId;
      if (!id) {
        const created = await unwrap(
          api.POST("/api/v1/land/{land_id}/documents", {
            params: { path: { land_id: landId } },
            body: { ...body, requestKey: operation.requestKey },
            signal: request.signal,
          }),
        );
        operation.documentId = created.id;
        id = created.id;
      }
      const saved = await unwrap(
        api.PUT("/api/v1/land/{land_id}/documents/{document_id}/content", {
          params: { path: { land_id: landId, document_id: id } },
          body: file.name,
          bodySerializer: () => file,
          headers: { "Content-Type": "application/octet-stream" },
          signal: request.signal,
        }),
      );
      pending.current = null;
      await refresh();
      if (!request.signal.aborted) onSaved(saved);
    } catch (cause) {
      if (!request.signal.aborted) {
        const detail =
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause);
        setError(`The document was not saved. ${detail} Try again; nothing will be doubled.`);
      }
      void refresh();
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };

  return (
    <form
      className="records-form"
      aria-label="Add a document"
      onSubmit={(event) => {
        event.preventDefault();
        void upload();
      }}
    >
      <fieldset disabled={busy} className="records-form__fields">
        <legend className="land-visually-hidden">Add a document</legend>
        <label className="records-drop" htmlFor={fileId}>
          <FileUp size={18} aria-hidden="true" />
          <span>
            {file ? (
              <>
                <strong>{file.name}</strong>
                <small>{formatBytes(file.size)} · choose another</small>
              </>
            ) : (
              <>
                <strong>Original PDF or text file</strong>
                <small>Deeds, easements, surveys, reports. Up to 20 MB.</small>
              </>
            )}
          </span>
          <input
            id={fileId}
            aria-label="Original PDF or text file"
            className="land-visually-hidden"
            type="file"
            accept=".pdf,.txt,.md"
            onChange={(event) => {
              const chosen = event.target.files?.[0];
              event.target.value = "";
              if (!chosen) return;
              if (
                !/\.(pdf|txt|md)$/i.test(chosen.name) ||
                chosen.size > MAX_BYTES ||
                !chosen.size
              ) {
                setError("Choose a PDF or a plain text file up to 20 MB.");
                return;
              }
              setFile(chosen);
              set({ title: chosen.name.replace(/\.[^.]+$/, "") });
              setError(null);
              pending.current = null;
            }}
          />
        </label>
        <div className="records-grid">
          <label className="land-field">
            Title
            <input
              className="glass-input"
              required
              maxLength={300}
              value={fields.title}
              onChange={(event) => set({ title: event.target.value })}
            />
          </label>
          <label className="land-field">
            Kind of record
            <select
              className="glass-input"
              value={fields.kind}
              onChange={(event) => set({ kind: event.target.value as DocumentKind })}
            >
              {DOCUMENT_KINDS.map((kind) => (
                <option key={kind.value} value={kind.value}>
                  {kind.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="land-field">
          Where did this record come from?
          <textarea
            className="glass-input"
            required
            maxLength={2000}
            rows={2}
            value={fields.sourceNote}
            onChange={(event) => set({ sourceNote: event.target.value })}
            placeholder="County recorder, an archive, the seller, a report author"
          />
        </label>
        <details className="records-more">
          <summary>Dates, recording number and parcels</summary>
          <div className="records-grid">
            <label className="land-field">
              Document date
              <input
                className="glass-input"
                type="date"
                value={fields.documentDate}
                onChange={(event) => set({ documentDate: event.target.value })}
              />
            </label>
            <label className="land-field">
              Recorded date
              <input
                className="glass-input"
                type="date"
                value={fields.recordedDate}
                onChange={(event) => set({ recordedDate: event.target.value })}
              />
            </label>
            <label className="land-field">
              Recording number
              <input
                className="glass-input"
                maxLength={200}
                value={fields.recordingNumber}
                onChange={(event) => set({ recordingNumber: event.target.value })}
              />
            </label>
            <label className="land-field">
              Jurisdiction
              <input
                className="glass-input"
                maxLength={300}
                value={fields.jurisdiction}
                placeholder="County, state"
                onChange={(event) => set({ jurisdiction: event.target.value })}
              />
            </label>
          </div>
          <label className="land-field">
            Parcel numbers it names (one per line)
            <textarea
              className="glass-input"
              rows={2}
              value={fields.parcels}
              onChange={(event) => set({ parcels: event.target.value })}
            />
          </label>
          <label className="land-field">
            Why might it apply to this land?
            <textarea
              className="glass-input"
              required
              maxLength={2000}
              rows={2}
              value={fields.relevance}
              onChange={(event) => set({ relevance: event.target.value })}
            />
          </label>
          <label className="land-field">
            Original source website
            <input
              className="glass-input"
              type="url"
              value={fields.sourceUrl}
              onChange={(event) => set({ sourceUrl: event.target.value })}
            />
          </label>
        </details>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
        <div className="land-row">
          <GlassButton
            type="submit"
            variant="primary"
            size="sm"
            loading={busy}
            disabled={busy || !file || !fields.title.trim() || !fields.sourceNote.trim()}
          >
            {busy ? "Saving and reading pages…" : "Save and read record"}
          </GlassButton>
          {onCancel && (
            <GlassButton size="sm" variant="ghost" onClick={onCancel}>
              Cancel
            </GlassButton>
          )}
        </div>
        <p className="land-muted">
          The file stays private to this workspace. Page numbers are kept; scans and handwriting
          that cannot be read are flagged.
        </p>
      </fieldset>
    </form>
  );
}
