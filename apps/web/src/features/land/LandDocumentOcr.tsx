import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandScope, useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";

type Language = components["schemas"]["DocumentOcrRequest"]["language"];
const languages: Record<string, string> = {
  eng: "English",
  spa: "Spanish",
  fra: "French",
  deu: "German",
};
export function LandDocumentOcr({
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
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const [language, setLanguage] = useState<Language>("eng");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
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
  const records = useQuery({
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
  const extract = async () => {
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
        cache.setQueryData<components["schemas"]["DocumentOcrRead"][]>(key, (old) => [
          ...(old ?? []).filter((item) => item.id !== result.id),
          result,
        ]);
        onSelected?.(result.id);
      }
      await cache.invalidateQueries({ queryKey: ["land-document-search", scope, landId] });
    } catch (cause) {
      if (!request.signal.aborted) setError(describeError(cause));
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  return (
    <section className="land-ocr" aria-label="Scanned page text">
      <h4>Scanned page text</h4>
      {records.isError && (
        <p role="alert">
          Saved OCR could not be loaded.{" "}
          <button type="button" onClick={() => void records.refetch()}>
            Retry OCR records
          </button>
        </p>
      )}
      {pinnedId && records.data && !records.data.some((item) => item.id === pinnedId) && (
        <p role="alert">The cited OCR extraction is not available for this page.</p>
      )}
      {records.data?.map((record) => (
        <article
          key={record.id}
          className="land-record-link"
          aria-label={`OCR in ${languages[record.language] ?? record.language}`}
        >
          <strong>
            {pinnedId === record.id ? "Cited machine reading" : "Machine reading"} ·{" "}
            {languages[record.language] ?? record.language}
          </strong>
          <pre className="land-page-text">
            {record.text || "No text was recognized. Review the original image."}
          </pre>
          {record.truncated && (
            <p className="land-notice">OCR text was truncated at 20,000 characters.</p>
          )}
          {record.warnings.map((warning) => (
            <p className="land-footnote" key={warning}>
              {warning}
            </p>
          ))}
          <details>
            <summary>Extraction details</summary>
            <p>
              {record.engineVersion} · page {record.page} ·{" "}
              {new Date(record.createdAt).toLocaleString()}
            </p>
            <p>
              Mean word score: {record.meanWordConfidence?.toFixed(1) ?? "No words scored"} / 100
            </p>
            <p className="land-footnote">
              Original SHA-256: {record.sha256}
              <br />
              Text SHA-256: {record.textSha256}
            </p>
          </details>
          {onSelected && (
            <div className="land-actions">
              <button
                type="button"
                disabled={pinnedId === record.id}
                onClick={() => onSelected(record.id)}
              >
                {pinnedId === record.id
                  ? "Reading selected for your question"
                  : "Use this reading in my question"}
              </button>
            </div>
          )}
        </article>
      ))}
      {canEdit && (
        <>
          {capabilities.isError && (
            <p role="alert">
              OCR availability could not be checked.{" "}
              <button type="button" onClick={() => void capabilities.refetch()}>
                Retry availability
              </button>
            </p>
          )}
          {capabilities.data && <p className="land-footnote">{capabilities.data.reason}</p>}
          {capabilities.data?.available && (
            <div className="land-actions">
              <label className="land-name">
                Page language
                <select
                  aria-label="OCR language"
                  value={language}
                  disabled={busy}
                  onChange={(event) => setLanguage(event.target.value as Language)}
                >
                  <option value="eng" disabled={!capabilities.data.languages.includes("eng")}>
                    English
                  </option>
                  {capabilities.data.languages
                    .filter((item) => item !== "eng")
                    .map((item) => (
                      <option key={item} value={item}>
                        {languages[item] ?? item}
                      </option>
                    ))}
                </select>
              </label>
              <button
                type="button"
                disabled={
                  busy ||
                  !capabilities.data.languages.includes(language ?? "eng") ||
                  records.data?.some((item) => item.language === language)
                }
                onClick={() => void extract()}
              >
                {busy
                  ? "Reading scanned page…"
                  : records.data?.some((item) => item.language === language)
                    ? "Reading saved above"
                    : `Read page ${page} with OCR`}
              </button>
            </div>
          )}
          {busy && (
            <p role="status">Reading one page. You can continue exploring while this finishes.</p>
          )}
          {error && <p role="alert">{error}</p>}
        </>
      )}
    </section>
  );
}
