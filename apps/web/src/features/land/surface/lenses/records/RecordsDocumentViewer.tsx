import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, ChevronLeft, ChevronRight, Download } from "lucide-react";

import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { useLandDocument } from "./recordsApi";
import { formatBytes, formatDate, kindLabel, namesParcel, saveBlob } from "./recordsModel";
import { RecordsDocumentOcr } from "./RecordsDocumentOcr";
import { RecordsPageImage } from "./RecordsPageImage";

/**
 * One land document, a page at a time: what it is, its extracted text, the original page,
 * machine reading of scanned pages, and the original file. Used in the Records lens and
 * wherever a source cites a document page (`pinnedHash` flags an original that changed).
 */
export function RecordsDocumentViewer({
  landId,
  documentId,
  initialPage = 1,
  pinnedHash,
  onPageChange,
  onUploaded,
  pinnedOcrId,
  pinnedOcrPage,
  onOcrSelected,
  parcelIds = [],
}: {
  landId: string;
  documentId: string;
  initialPage?: number;
  pinnedHash?: string;
  onPageChange?: (page: number) => void;
  onUploaded?: () => Promise<unknown>;
  pinnedOcrId?: string;
  /** The page the pinned reading belongs to; the first page shown when absent. */
  pinnedOcrPage?: number;
  onOcrSelected?: (id: string) => void;
  /** The land's parcel numbers, to say when the document names one. */
  parcelIds?: string[];
}) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
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
  const record = useLandDocument(landId, documentId);
  const content = useQuery({
    queryKey: ["land-document-page", scope, landId, documentId, page],
    enabled: ready && Boolean(record.data?.pageCount),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}/pages/{page}", {
          params: { path: { land_id: landId, document_id: documentId, page } },
        }),
      ),
    retry: false,
  });
  const resume = async (file: File) => {
    if (!record.data || uploading) return;
    if (file.name !== record.data.filename || file.size !== record.data.sizeBytes) {
      setError(`Choose the original ${record.data.filename} (${record.data.sizeBytes} bytes).`);
      return;
    }
    const controller = new AbortController();
    uploadController.current = controller;
    setUploading(true);
    setError(null);
    try {
      const updated = await unwrap(
        api.PUT("/api/v1/land/{land_id}/documents/{document_id}/content", {
          params: { path: { land_id: landId, document_id: documentId } },
          body: file.name,
          bodySerializer: () => file,
          headers: { "Content-Type": "application/octet-stream" },
          signal: controller.signal,
        }),
      );
      cache.setQueryData(["land-document", scope, landId, documentId], updated);
      await onUploaded?.();
    } catch (cause) {
      if (alive.current && !controller.signal.aborted)
        setError(`The upload did not finish. ${describeError(cause)}`);
    } finally {
      if (alive.current && !controller.signal.aborted) setUploading(false);
    }
  };
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
      if (alive.current) saveBlob(blob, record.data?.filename ?? "land-record");
    } catch (cause) {
      if (alive.current) setError(`The original could not be downloaded. ${describeError(cause)}`);
    } finally {
      if (alive.current) setDownloading(false);
    }
  };
  const turn = (next: number) => {
    setPage(next);
    onPageChange?.(next);
  };

  const data = record.data;
  if (record.isPending)
    return (
      <p className="land-muted" role="status">
        Loading the document…
      </p>
    );
  if (!data)
    return (
      <p className="land-alert" role="alert">
        This document could not be loaded.{" "}
        <button type="button" className="land-link" onClick={() => void record.refetch()}>
          Try again
        </button>
      </p>
    );
  const dated = data.documentDate ?? data.recordedDate;
  const isPdf = data.mediaType === "application/pdf";
  return (
    <section className="records-doc" aria-label="Document page">
      <div className="records-doc__head">
        <h4 className="records-item__title">{data.title}</h4>
        <p className="land-muted">
          {[
            kindLabel(data.kind),
            dated ? formatDate(dated) : "Date unknown",
            data.recordingNumber ? `No. ${data.recordingNumber}` : "",
            data.jurisdiction ?? "",
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {namesParcel(data, parcelIds) && (
          <p className="records-tag">Names this land&apos;s parcel</p>
        )}
      </div>
      {data.status === "awaiting-upload" && (
        <div className="land-note">
          <p>
            The upload did not finish. Choose the same file to continue. Unfinished uploads are kept
            for 24 hours.
          </p>
          {canEdit && (
            <label className="land-field">
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
          {uploading && (
            <p className="land-muted" role="status">
              Saving and reading pages…
            </p>
          )}
        </div>
      )}
      {pinnedHash && data.sha256 !== pinnedHash && (
        <p className="land-alert" role="alert">
          This original does not match the fingerprint recorded where it was cited.
        </p>
      )}
      {data.warnings?.map((warning) => (
        <p className="land-note" key={warning}>
          {warning}
        </p>
      ))}
      {data.pageCount > 0 && (
        <div className="records-pager">
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label="Previous page"
            disabled={page <= 1}
            onClick={() => turn(page - 1)}
          >
            <ChevronLeft size={16} />
          </GlassButton>
          <select
            className="glass-input records-select"
            aria-label="Document page number"
            value={page}
            onChange={(event) => turn(Number(event.target.value))}
          >
            {Array.from({ length: data.pageCount }, (_, index) => (
              <option key={index + 1} value={index + 1}>
                Page {index + 1}
              </option>
            ))}
          </select>
          <span className="land-muted">of {data.pageCount}</span>
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label="Next page"
            disabled={page >= data.pageCount}
            onClick={() => turn(page + 1)}
          >
            <ChevronRight size={16} />
          </GlassButton>
        </div>
      )}
      {isPdf && data.pageCount > 0 && (
        <RecordsPageImage
          key={`image:${page}`}
          landId={landId}
          documentId={documentId}
          page={page}
        />
      )}
      {content.isError && (
        <p className="land-alert" role="alert">
          The page text could not be loaded.
        </p>
      )}
      {content.data && (
        <>
          <pre className="records-text">
            {content.data.text ||
              "No text was found on this page. View the original page or read it with OCR below."}
          </pre>
          {content.data.truncated && (
            <p className="land-note">
              This page&apos;s text is cut short. The original has all of it.
            </p>
          )}
        </>
      )}
      {isPdf && data.pageCount > 0 && (
        <RecordsDocumentOcr
          key={`ocr:${page}`}
          landId={landId}
          documentId={documentId}
          page={page}
          pinnedId={page === (pinnedOcrPage ?? initialPage) ? pinnedOcrId : undefined}
          onSelected={onOcrSelected}
        />
      )}
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<Download size={14} />}
          loading={downloading}
          disabled={downloading || data.status === "awaiting-upload"}
          onClick={() => void download()}
        >
          Download original record
        </GlassButton>
        {data.sourceUrl && (
          <a
            className="records-out"
            href={data.sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Original source website <ArrowUpRight size={13} aria-hidden="true" />
          </a>
        )}
      </div>
      <details className="records-more">
        <summary>Source and licence</summary>
        <p>{data.sourceNote}</p>
        <p>{data.relevanceNote}</p>
        {data.parcelReferences?.length ? (
          <p className="land-muted">Parcel references: {data.parcelReferences.join(", ")}</p>
        ) : null}
        <p className="land-muted">
          {data.filename} · {formatBytes(data.sizeBytes)} · added {formatDate(data.createdAt)}
        </p>
        <p className="land-muted">{data.license}</p>
        {data.sha256 && <p className="land-muted records-hash">SHA-256 {data.sha256}</p>}
      </details>
    </section>
  );
}
