import { useQueries, useQuery } from "@tanstack/react-query";

import type { LandEvidence } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

export const DOCUMENT_PAGE = 50;
export const LINK_PAGE = 100;
export const SEARCH_PAGE = 20;
export const ALIGNMENT_PAGE = 20;

/** One page of the land's documents (newest first, as the server orders them). */
export function useLandDocuments(landId: string, offset = 0) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-documents", scope, landId, offset],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents", {
          params: { path: { land_id: landId }, query: { limit: DOCUMENT_PAGE, offset } },
        }),
      ),
    retry: false,
  });
}

/** One document's record (title, kind, dates, pages, status). */
export function useLandDocument(landId: string, documentId: string) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
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
}

/** Connections people recorded between documents. */
export function useDocumentLinks(landId: string, offset = 0, enabled = true) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-document-links", scope, landId, offset],
    enabled: ready && enabled,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/links", {
          params: { path: { land_id: landId }, query: { limit: LINK_PAGE, offset } },
        }),
      ),
    retry: false,
  });
}

/** Full-text search over extracted and machine-read page text (three characters or more). */
export function useDocumentSearch(landId: string, text: string, offset = 0) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-document-search", scope, landId, text, offset],
    enabled: ready && text.length >= 3,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/documents/search", {
          params: { path: { land_id: landId }, query: { q: text, limit: SEARCH_PAGE, offset } },
        }),
      ),
    retry: false,
  });
}

/** A saved source, from the page already loaded when there is one. */
export function useEvidence(id: string, cached?: LandEvidence) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const query = useQuery({
    queryKey: ["land-research", scope, "evidence", id],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: id } },
        }),
      ),
    enabled: ready && !cached,
    retry: false,
  });
  return { evidence: cached ?? query.data ?? null, query };
}

/** One page of the saved alignments of a historical map, newest first. */
export function useAlignments(evidenceId: string, offset = 0, enabled = true) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-image-alignments", scope, evidenceId, offset],
    enabled: ready && enabled,
    queryFn: ({ signal }) =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}/image/registrations", {
          params: { path: { evidence_id: evidenceId }, query: { limit: ALIGNMENT_PAGE, offset } },
          signal,
        }),
      ),
    retry: false,
  });
}

/**
 * The newest saved alignment of each historical map, so the lens can lay the latest one over
 * the land. Shares the alignment cache prefix, so saving one refreshes these too.
 */
export function useLatestAlignments(evidenceIds: string[]) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQueries({
    queries: evidenceIds.map((id) => ({
      queryKey: ["land-image-alignments", scope, id, "latest"],
      enabled: ready,
      staleTime: 60_000,
      retry: false,
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        unwrap(
          api.GET("/api/v1/research/evidence/{evidence_id}/image/registrations", {
            params: { path: { evidence_id: id }, query: { limit: 1, offset: 0 } },
            signal,
          }),
        ),
    })),
  });
}
