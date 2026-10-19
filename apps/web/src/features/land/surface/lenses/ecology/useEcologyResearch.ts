import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import type { Investigation, LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";

import {
  beginInvestigation,
  cancelResearch,
  useInvestigation,
  useInvestigations,
} from "../../../researchApi";
import { landError } from "../../landFlow";
import {
  DEFAULT_REQUEST,
  ECOLOGY_TITLE,
  ecologyNames,
  isEcologyRequest,
  type EcologyRequest,
} from "./ecologyData";

const eventPage = (runId: string, after: number) =>
  unwrap(
    api.GET("/api/v1/research/runs/{run_id}/events", {
      params: { path: { run_id: runId }, query: { after } },
    }),
  );

interface Operation {
  signature: string;
  id: string;
  key: string;
}

export interface EcologyChoices {
  names: string;
  kingdom: string;
  regions: boolean;
  sites: boolean;
  occurrences: boolean;
}

const KINGDOMS = ["Plantae", "Animalia", "Fungi"];

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function pendingOperation(key: string): Operation | null {
  try {
    const value: unknown = JSON.parse(read(key) ?? "null");
    if (
      value &&
      typeof value === "object" &&
      "signature" in value &&
      "id" in value &&
      "key" in value &&
      typeof value.signature === "string" &&
      typeof value.id === "string" &&
      typeof value.key === "string"
    )
      return { signature: value.signature, id: value.id, key: value.key };
  } catch {
    /* A damaged browser record cannot change a server request. */
  }
  return null;
}

/**
 * The ecology lookups for one land: the saved runs (newest for this boundary first), the one
 * being shown, its source events, and starting, stopping and choosing runs. Source choices,
 * names, the chosen run and an in-flight start live in this browser under the same keys as
 * before, so a start lost to the network retries with the same request key.
 */
export function useEcologyResearch(land: LandArea, scope: string) {
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const storage = `living-world-land-draft:${encodeURIComponent(scope)}:ecology:${land.id}`;
  const [choices, setChoicesState] = useState<EcologyChoices>(() => {
    const kingdom = read(`${storage}:kingdom`) ?? "";
    return {
      names: read(`${storage}:names`) ?? "",
      kingdom: KINGDOMS.includes(kingdom) ? kingdom : "",
      regions: read(`${storage}:regions`) !== "false",
      sites: read(`${storage}:sites`) !== "false",
      occurrences: read(`${storage}:occurrences`) !== "false",
    };
  });
  const [chosen, setChosen] = useState(() => read(`${storage}:run`));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState(false);
  const pending = useRef<Operation | null>(pendingOperation(`${storage}:operation`));

  const store = (key: string, value: string | null) => {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      setStorageError(true);
    }
  };

  const list = useInvestigations(land.id);
  const saved = (list.data ?? [])
    .filter((item: Investigation) => item.title === ECOLOGY_TITLE)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const fallback = saved.find((item) => item.boundaryRevision === land.revision) ?? saved[0];
  // A remembered choice that is no longer listed (deleted, another land) is ignored.
  const selectedId =
    (chosen && saved.some((item) => item.id === chosen) ? chosen : null) ?? fallback?.id ?? null;
  const detail = useInvestigation(selectedId, 0, 200, 1500);
  const current = detail.data?.investigation.landId === land.id ? detail.data : null;
  const run = current?.runs.filter((item) => item.kind === "ecology").at(-1) ?? null;
  const running = run?.status === "queued" || run?.status === "running";
  const events = useQuery({
    queryKey: ["land-research", scope, "ecology-events", run?.id],
    enabled: ready && Boolean(run),
    queryFn: async () => {
      const values: Awaited<ReturnType<typeof eventPage>> = [];
      for (let page = 0; page < 10; page++) {
        const batch = await eventPage(run?.id ?? "", values.at(-1)?.sequence ?? 0);
        values.push(...batch);
        if (batch.length < 200) break;
      }
      return values;
    },
    retry: false,
    refetchInterval: running ? 1500 : false,
  });

  const setChoices = (change: Partial<EcologyChoices>) => {
    setChoicesState((previous) => ({ ...previous, ...change }));
    for (const [name, value] of Object.entries(change))
      store(`${storage}:${name}`, typeof value === "boolean" ? String(value) : value);
  };

  const taxa = ecologyNames(choices.names, choices.kingdom);
  const chosenRequest: EcologyRequest | null =
    taxa && (taxa.length || choices.regions || choices.sites || choices.occurrences)
      ? {
          dataset: "ecological-context",
          taxa,
          includeEcoregions: choices.regions,
          includeEcologicalSites: choices.sites,
          includeOccurrences: choices.occurrences,
        }
      : null;

  const choose = (id: string) => {
    setChosen(id);
    store(`${storage}:run`, id);
  };

  const start = async (analysis: EcologyRequest) => {
    if (!ready || !canEdit || busy) return;
    setBusy(true);
    setError(null);
    const signature = JSON.stringify({ revision: land.revision, analysis });
    try {
      let operation = pending.current;
      if (operation?.signature !== signature) {
        const investigation = await beginInvestigation(land, ECOLOGY_TITLE);
        operation = { signature, id: investigation.id, key: crypto.randomUUID() };
        pending.current = operation;
      }
      store(`${storage}:operation`, JSON.stringify(operation));
      choose(operation.id);
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: operation.id } },
          body: {
            kind: "ecology",
            requestKey: operation.key,
            question: ECOLOGY_TITLE,
            analysis,
            budget: { maxSteps: 30, maxSeconds: 600, maxWebSearches: 0, maxOutputTokens: 500 },
          },
        }),
      );
      pending.current = null;
      store(`${storage}:operation`, null);
    } catch (cause) {
      setError(landError(cause));
    } finally {
      await cache.invalidateQueries({ queryKey: ["land-research", scope] });
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (!run || busy) return;
    setBusy(true);
    try {
      await cancelResearch(run.id);
      await detail.refetch();
    } catch (cause) {
      setError(landError(cause));
    } finally {
      setBusy(false);
    }
  };

  const investigation = current?.investigation ?? null;
  return {
    canEdit: canEdit && ready,
    loading: list.isPending || (Boolean(selectedId) && detail.isPending),
    loadError: list.error ?? detail.error ?? null,
    retry: () => void (list.error ? list.refetch() : detail.refetch()),
    saved,
    selectedId,
    choose,
    detail: current,
    investigation,
    run,
    running,
    failed: run?.status === "failed" || run?.status === "cancelled",
    stale: investigation
      ? investigation.stale || investigation.boundaryRevision !== land.revision
      : false,
    events: events.data,
    eventsError: events.isError,
    retryEvents: () => void events.refetch(),
    /** What the shown run asked for; "Run again" repeats it on the current boundary. */
    request: run && isEcologyRequest(run.analysis) ? run.analysis : null,
    choices,
    setChoices,
    chosenRequest,
    start: (analysis: EcologyRequest = DEFAULT_REQUEST) => void start(analysis),
    cancel: () => void cancel(),
    busy,
    error,
    storageError,
  };
}

export type EcologyResearch = ReturnType<typeof useEcologyResearch>;
