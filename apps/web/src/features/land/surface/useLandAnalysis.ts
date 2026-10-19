import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { components, Investigation, LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { beginInvestigation, useInvestigation, useInvestigations } from "../researchApi";

type RunCreate = components["schemas"]["RunCreate"];

/** The investigation title every overview bundle gets (`services/research.py ensure_overview`). */
export const OVERVIEW_TITLE = "Land overview";

/**
 * The latest saved result of one analysis on this land: the newest investigation whose title
 * matches, preferring one made for the current boundary. Analyses are deterministic runs
 * (terrain, land cover, the overview bundle, ecology, archives, solar); nothing here starts one.
 */
export function useLandAnalysis(land: LandArea, title: string | RegExp) {
  const list = useInvestigations(land.id);
  const matches = (list.data ?? [])
    .filter((item: Investigation) =>
      typeof title === "string" ? item.title === title : title.test(item.title),
    )
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const investigation =
    matches.find((item) => item.boundaryRevision === land.revision) ?? matches[0] ?? null;
  const detail = useInvestigation(investigation?.id ?? null, 0, 100, 1500);
  const run = detail.data?.runs.at(-1) ?? null;
  const running = run?.status === "queued" || run?.status === "running";
  return {
    /** True until the list (and, when there is a match, its detail) has loaded. */
    loading: list.isPending || (Boolean(investigation) && detail.isPending),
    error: list.error ?? detail.error ?? null,
    investigation,
    detail: detail.data ?? null,
    run,
    running,
    failed: run?.status === "failed" || run?.status === "cancelled",
    /** Made for an earlier boundary: still shown, with a note and a way to run it again. */
    stale: investigation ? investigation.boundaryRevision !== land.revision : false,
  };
}

export interface AnalysisStart {
  title: string;
  kind: Exclude<RunCreate["kind"], "investigation" | "overview" | undefined>;
  question: string;
  analysis?: RunCreate["analysis"];
  budget?: RunCreate["budget"];
}

/**
 * Starts analyses when the person asks for them. A retried start after a network failure
 * reuses the same investigation and request key, so the worker runs it once.
 */
export function useStartAnalysis(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<{ signature: string; investigationId: string; key: string } | null>(null);
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-research", scope] });

  const start = async (spec: AnalysisStart) => {
    if (!ready || !canEdit || busy) return;
    setBusy(true);
    setError(null);
    const signature = JSON.stringify({ revision: land.revision, ...spec });
    try {
      let operation = pending.current;
      if (operation?.signature !== signature) {
        const investigation = await beginInvestigation(land, spec.title);
        operation = { signature, investigationId: investigation.id, key: crypto.randomUUID() };
        pending.current = operation;
      }
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: operation.investigationId } },
          body: {
            kind: spec.kind,
            question: spec.question,
            requestKey: operation.key,
            ...(spec.analysis ? { analysis: spec.analysis } : {}),
            ...(spec.budget ? { budget: spec.budget } : {}),
          },
        }),
      );
      pending.current = null;
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  /** The overview bundle (elevation, climate, species, soils, flood zones), once per boundary. */
  const startOverview = async () => {
    if (!ready || !canEdit || busy) return;
    setBusy(true);
    setError(null);
    try {
      await unwrap(
        api.POST("/api/v1/land/{land_id}/overview", {
          params: { path: { land_id: land.id } },
          body: { boundaryRevision: land.revision },
        }),
      );
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };

  return { start, startOverview, busy, error, canEdit: canEdit && ready };
}
