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
  LandArea,
} from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

/** Cell size the console asks for, metres. */
export const KNOWLEDGE_CELL_M = 20;

/** The Land a fog is asked for: its boundary revision is part of the answer's identity. */
export type KnowledgeLand = Pick<LandArea, "id" | "revision">;

/**
 * Keyed by the boundary revision as well as the Land: a revised boundary is a new grid (new
 * cells, a new inside), never the old one kept for its stale time.
 */
export function knowledgeKey(
  scope: string,
  land: KnowledgeLand | null,
  cellM = KNOWLEDGE_CELL_M,
): readonly unknown[] {
  return ["land-knowledge", scope, land?.id ?? null, land?.revision ?? null, cellM];
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
export function useKnowledgeGrid(land: KnowledgeLand | null, enabled: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: knowledgeKey(scope, land),
    queryFn: () => fetchKnowledgeGrid(land?.id ?? ""),
    enabled: enabled && ready && land !== null,
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

/**
 * One cell's records; keyed by the cell on this boundary revision's grid, so reopening a card
 * is instant and a revised boundary asks again.
 */
export function useKnowledgeCell(
  selected: {
    landId: string;
    longitude: number;
    latitude: number;
    col: number;
    row: number;
  } | null,
  revision: number | null,
) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: [
      "land-knowledge",
      scope,
      selected?.landId ?? null,
      revision,
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
