import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import type { components } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { formatDate } from "./recordsModel";

type Language = components["schemas"]["DocumentOcrRequest"]["language"];
type Reading = components["schemas"]["DocumentOcrRead"];

const LANGUAGES: Record<string, string> = {
  eng: "English",
  spa: "Spanish",
  fra: "French",
  deu: "German",
};

/**
 * Machine reading (OCR) of one scanned page, run on request, one page at a time. A reading
 * cited by a source is marked, and a missing cited reading is said plainly.
 */
export function RecordsDocumentOcr({
  landId,
  documentId,
  page,
  pinnedId,
  onSelected,
}: {
  landId: string;
  documentId: string;
  page: number;
  pinnedId?: string;
  onSelected?: (id: string) => void;
}) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const [language, setLanguage] = useState<Language>("eng");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const capabilities = useQuery({
    queryKey: ["land-ocr-capabilities", scope, landId],
    enabled: ready && canEdit,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/ocr-capabilities", {
          params: { path: { land_id: landId } },
        }),
      ),
    staleTime: 60_000,
    retry: false,
  });
  const key = ["land-document-ocr", scope, landId, documentId, page];
  const readings = useQuery({
    queryKey: key,
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}/pages/{page}/ocr", {
          params: { path: { land_id: landId, document_id: documentId, page } },
        }),
      ),
    retry: false,
  });
  const read = async () => {
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(null);
    try {
      const result = await unwrap(
        api.POST("/api/v1/land/{land_id}/documents/{document_id}/pages/{page}/ocr", {
          params: { path: { land_id: landId, document_id: documentId, page } },
          body: { language },
          signal: request.signal,
        }),
      );
      if (!request.signal.aborted) {
        cache.setQueryData<Reading[]>(key, (old) => [
          ...(old ?? []).filter((item) => item.id !== result.id),
          result,
        ]);
        onSelected?.(result.id);
      }
      await cache.invalidateQueries({ queryKey: ["land-document-search", scope, landId] });
    } catch (cause) {
      if (!request.signal.aborted) setError(`The page could not be read. ${describeError(cause)}`);
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  const available = capabilities.data?.available ? capabilities.data : null;
  const alreadyRead = readings.data?.some((item) => item.language === language);
  return (
    <section className="records-section" aria-label="Scanned page text">
      <h5 className="records-subhead">Scanned page text</h5>
      {readings.isError && (
        <p className="land-alert" role="alert">
          Saved readings did not load.{" "}
          <button type="button" className="land-link" onClick={() => void readings.refetch()}>
            Try again
          </button>
        </p>
      )}
      {pinnedId && readings.data && !readings.data.some((item) => item.id === pinnedId) && (
        <p className="land-alert" role="alert">
          The cited OCR extraction is not available for this page.
        </p>
      )}
      {readings.data?.map((reading) => (
        <article
          key={reading.id}
          className="records-reading"
          aria-label={`OCR in ${LANGUAGES[reading.language] ?? reading.language}`}
        >
          <strong className="records-reading__title">
            {pinnedId === reading.id ? "Cited machine reading" : "Machine reading"} ·{" "}
            {LANGUAGES[reading.language] ?? reading.language}
          </strong>
          <pre className="records-text">
            {reading.text || "No text was recognised. Check the original page."}
          </pre>
          {reading.truncated && (
            <p className="land-note">The reading was cut off at 20,000 characters.</p>
          )}
          {reading.warnings.map((warning) => (
            <p className="land-muted" key={warning}>
              {warning}
            </p>
          ))}
          {onSelected && (
            <div className="land-row">
              <button
                type="button"
                className="land-link"
                disabled={pinnedId === reading.id}
                onClick={() => onSelected(reading.id)}
              >
                {pinnedId === reading.id
                  ? "Used in your question"
                  : "Use this reading in my question"}
              </button>
            </div>
          )}
          <details className="records-more">
            <summary>Extraction details</summary>
            <p className="land-muted">
              {reading.engineVersion} · page {reading.page} · {formatDate(reading.createdAt)} · mean
              word score {reading.meanWordConfidence?.toFixed(0) ?? "none"} of 100
            </p>
            <p className="land-muted records-hash">Original SHA-256 {reading.sha256}</p>
            <p className="land-muted records-hash">Text SHA-256 {reading.textSha256}</p>
          </details>
        </article>
      ))}
      {canEdit && (
        <>
          {capabilities.isError && (
            <p className="land-alert" role="alert">
              Could not check whether pages can be read here.{" "}
              <button
                type="button"
                className="land-link"
                onClick={() => void capabilities.refetch()}
              >
                Try again
              </button>
            </p>
          )}
          {capabilities.data && !available && (
            <p className="land-muted">{capabilities.data.reason}</p>
          )}
          {available && (
            <div className="land-row">
              <select
                className="glass-input records-select"
                aria-label="OCR language"
                value={language}
                disabled={busy}
                onChange={(event) => setLanguage(event.target.value as Language)}
              >
                <option value="eng" disabled={!available.languages.includes("eng")}>
                  English
                </option>
                {available.languages
                  .filter((item) => item !== "eng")
                  .map((item) => (
                    <option key={item} value={item}>
                      {LANGUAGES[item] ?? item}
                    </option>
                  ))}
              </select>
              <GlassButton
                size="sm"
                loading={busy}
                disabled={busy || !available.languages.includes(language ?? "eng") || alreadyRead}
                onClick={() => void read()}
              >
                {busy
                  ? "Reading the page…"
                  : alreadyRead
                    ? "Already read"
                    : `Read page ${page} with OCR`}
              </GlassButton>
            </div>
          )}
          {busy && (
            <p className="land-muted" role="status">
              Reading one page. You can keep exploring.
            </p>
          )}
          {error && (
            <p className="land-alert" role="alert">
              {error}
            </p>
          )}
        </>
      )}
    </section>
  );
}
