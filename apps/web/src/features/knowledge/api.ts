/**
 * The Fog of Knowing from the API: the grid over a Land, one cell's records, the foggiest
 * patches. All read-only computation over stored evidence (viewers may ask too).
 */

import { useQuery } from "@tanstack/react-query";

import type {
  KnowledgeCell,
  KnowledgeGrid,
  KnowledgeScoutRead,
  KnowledgeScoutRequest,
} from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

/** Cell size the console asks for, metres. */
export const KNOWLEDGE_CELL_M = 20;

export function knowledgeKey(
  scope: string,
  landId: string | null,
  cellM = KNOWLEDGE_CELL_M,
): readonly unknown[] {
  return ["land-knowledge", scope, landId, cellM];
}

export function fetchKnowledgeGrid(
  landId: string,
  cellM = KNOWLEDGE_CELL_M,
): Promise<KnowledgeGrid> {
  return unwrap(
    api.GET("/api/v1/land/{land_id}/knowledge", {
      params: { path: { land_id: landId }, query: { cellM } },
    }),
  );
}

/** The fog over a Land; refetched when evidence changes (`invalidateLandDerived`). */
export function useKnowledgeGrid(landId: string | null, enabled: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: knowledgeKey(scope, landId),
    queryFn: () => fetchKnowledgeGrid(landId ?? ""),
    enabled: enabled && ready && landId !== null,
    staleTime: 30_000,
    retry: false,
  });
}

export function fetchKnowledgeCell(
  landId: string,
  longitude: number,
  latitude: number,
  cellM = KNOWLEDGE_CELL_M,
): Promise<KnowledgeCell> {
  return unwrap(
    api.GET("/api/v1/land/{land_id}/knowledge/cell", {
      params: { path: { land_id: landId }, query: { longitude, latitude, cellM } },
    }),
  );
}

/** One cell's records; keyed by the cell, so reopening a card is instant. */
export function useKnowledgeCell(
  selected: {
    landId: string;
    longitude: number;
    latitude: number;
    col: number;
    row: number;
  } | null,
) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: [
      "land-knowledge",
      scope,
      selected?.landId ?? null,
      KNOWLEDGE_CELL_M,
      "cell",
      selected?.col,
      selected?.row,
    ],
    queryFn: () =>
      fetchKnowledgeCell(selected?.landId ?? "", selected?.longitude ?? 0, selected?.latitude ?? 0),
    enabled: ready && selected !== null,
    staleTime: 30_000,
    retry: false,
  });
}

export function requestScoutAreas(
  landId: string,
  body: KnowledgeScoutRequest,
): Promise<KnowledgeScoutRead> {
  return unwrap(
    api.POST("/api/v1/land/{land_id}/knowledge/scout-areas", {
      params: { path: { land_id: landId } },
      body,
    }),
  );
}
