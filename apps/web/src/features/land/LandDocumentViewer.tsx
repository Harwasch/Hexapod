import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandScope, useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";
import { LandDocumentOcr } from "./LandDocumentOcr";
import { LandDocumentImage } from "./LandDocumentImage";

export function LandDocumentViewer({
  landId,
  documentId,
  initialPage = 1,
  pinnedHash,
  onPageChange,
  onUploaded,
  pinnedOcrId,
  onOcrSelected,
}: {
  landId: string;
  documentId: string;
  initialPage?: number;
  pinnedHash?: string;
  onPageChange?: (page: number) => void;
  onUploaded?: () => Promise<unknown>;
  pinnedOcrId?: string;
  onOcrSelected?: (id: string) => void;
}) {
  const scope = useLandScope(),
    ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const [page, setPage] = useState(initialPage);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const uploadController = useRef<AbortController | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      uploadController.current?.abort();
    };
  }, []);
  const document = useQuery({
    queryKey: ["land-document", scope, landId, documentId],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}", {
          params: { path: { land_id: landId, document_id: documentId } },
        }),
      ),
    retry: false,
  });
  const resume = async (file: File) => {
    if (!document.data || uploading) return;
    if (file.name !== document.data.filename || file.size !== document.data.sizeBytes) {
      setError(`Choose the original ${document.data.filename} (${document.data.sizeBytes} bytes).`);
      return;
    }
    const controller = new AbortController();
    uploadController.current = controller;
    setUploading(true);
    setError(null);
    try {
      const record = await unwrap(
        api.PUT("/api/v1/land/{land_id}/documents/{document_id}/content", {
          params: { path: { land_id: landId, document_id: documentId } },
          body: file.name,
          bodySerializer: () => file,
          headers: { "Content-Type": "application/octet-stream" },
          signal: controller.signal,
        }),
      );
      cache.setQueryData(["land-document", scope, landId, documentId], record);
      await onUploaded?.();
    } catch (cause) {
      if (alive.current && !controller.signal.aborted) setError(describeError(cause));
    } finally {
      if (alive.current && !controller.signal.aborted) setUploading(false);
    }
  };
  const content = useQuery({
    queryKey: ["land-document-page", scope, landId, documentId, page],
    enabled: ready && Boolean(document.data?.pageCount),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}/pages/{page}", {
          params: { path: { land_id: landId, document_id: documentId, page } },
        }),
      ),
    retry: false,
  });
  const download = async () => {
    setDownloading(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}/content", {
          params: { path: { land_id: landId, document_id: documentId } },
          parseAs: "blob",
        }),
      );
      if (!alive.current) return;
      const url = URL.createObjectURL(blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = document.data?.filename ?? "land-record";
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      if (alive.current) setError(describeError(cause));
    } finally {
      if (alive.current) setDownloading(false);
    }
  };
  return (
    <section className="land-document-viewer" aria-label="Document page">
      {document.isPending && <p role="status">Loading the land record…</p>}
      {document.isError && (
        <p role="alert">
          This record could not be loaded.{" "}
          <button type="button" onClick={() => void document.refetch()}>
            Retry record
          </button>
        </p>
      )}
      {document.data && (
        <>
          <h4>{document.data.title}</h4>
          <p className="land-footnote">
            {document.data.kind} ·{" "}
            {document.data.recordingNumber?.length
              ? document.data.recordingNumber
              : "No recording number provided"}
          </p>
          <dl className="land-document-dates">
            <div>
              <dt>Document date</dt>
              <dd>{document.data.documentDate ?? "Not established"}</dd>
            </div>
            <div>
              <dt>Recorded date</dt>
              <dd>{document.data.recordedDate ?? "Not established"}</dd>
            </div>
          </dl>
          <p>{document.data.relevanceNote}</p>
          {document.data.status === "awaiting-upload" && (
            <div className="land-notice">
              <p>
                The file upload did not finish. Choose the same original to continue. Pending
                uploads are kept for 24 hours.
              </p>
              {canEdit && (
                <label className="land-name">
                  Resume original upload
                  <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    disabled={uploading}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      event.target.value = "";
                      if (file) void resume(file);
                    }}
                  />
                </label>
              )}
              {uploading && <p role="status">Saving and extracting pages…</p>}
            </div>
          )}
          {pinnedHash && document.data.sha256 !== pinnedHash && (
            <p className="land-error" role="alert">
              This original does not match the hash recorded in the citation.
            </p>
          )}
          {document.data.warnings?.map((warning) => (
            <p className="land-notice" key={warning}>
              {warning}
            </p>
          ))}
          {document.data.pageCount > 0 && (
            <label className="land-name">
              Document page
              <select
                aria-label="Document page number"
                value={page}
                onChange={(event) => {
                  const number = Number(event.target.value);
                  setPage(number);
                  onPageChange?.(number);
                }}
              >
                {Array.from({ length: document.data.pageCount }, (_, index) => (
                  <option key={index + 1} value={index + 1}>
                    Page {index + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          {content.isError && <p role="alert">The extracted page could not be loaded.</p>}
          {document.data.mediaType === "application/pdf" && document.data.pageCount > 0 && (
            <LandDocumentImage
              key={`image:${documentId}:${page}`}
              landId={landId}
              documentId={documentId}
              page={page}
            />
          )}
          {content.data && (
            <>
              <pre className="land-page-text">
                {content.data.text ||
                  "No native text was extracted from this page. Review the original image or machine reading below."}
              </pre>
              {content.data.truncated && (
                <p className="land-notice">
                  This extracted page is truncated. The original retains the complete page.
                </p>
              )}
            </>
          )}
          {error && <p role="alert">{error}</p>}
          {document.data.mediaType === "application/pdf" && document.data.pageCount > 0 && (
            <LandDocumentOcr
              key={`ocr:${documentId}:${page}`}
              landId={landId}
              documentId={documentId}
              page={page}
              pinnedId={page === initialPage ? pinnedOcrId : undefined}
              onSelected={onOcrSelected}
            />
          )}
          <div className="land-actions">
            <button
              type="button"
              disabled={downloading || document.data.status === "awaiting-upload"}
              onClick={() => void download()}
            >
              Download original record
            </button>
            {document.data.sourceUrl && (
              <a href={document.data.sourceUrl} target="_blank" rel="noopener noreferrer">
                Original source website
              </a>
            )}
          </div>
          <p className="land-footnote">
            {document.data.sourceNote}
            <br />
            {document.data.license}
          </p>
        </>
      )}
    </section>
  );
}
