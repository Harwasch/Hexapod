import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { ResearchEvent } from "@twin/contracts";
import { ApiError, api, unwrap } from "@/api/client";
import {
  landScope,
  landUsesOidc,
  useLandAccessReady,
  useLandIdentity,
  useLandScope,
} from "@/state/landIdentity";
import { useSettings } from "@/state/settings";
import {
  mergeResearchEvents,
  parseResearchEvent,
  readResearchStream,
  ResearchAccessError,
} from "./researchStream";

type Phase = "connecting" | "live" | "reconnecting" | "polling" | "complete" | "unavailable";
const terminal = new Set(["succeeded", "partial", "failed", "cancelled"]);
function pause(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, milliseconds);
    signal.addEventListener("abort", finish, { once: true });
  });
}

/** Authenticated, resumable updates; polling remains available behind buffering proxies. */
export function useResearchProgress(runId: string | undefined, running: boolean) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    cache = useQueryClient();
  const token = useLandIdentity((s) => s.accessToken),
    pilotToken = useSettings((s) => s.writeToken);
  const credentials = landUsesOidc ? token : pilotToken;
  const eventKey = useMemo(() => ["land-research", scope, "events", runId], [scope, runId]);
  const identity = `${scope}:${runId ?? "none"}`;
  const [connection, setConnection] = useState<{
    identity: string;
    phase: Phase;
    message?: string;
  } | null>(null);
  const [attempt, retry] = useState(0);
  const query = useQuery<ResearchEvent[]>({ queryKey: eventKey, enabled: false, initialData: [] });
  useEffect(() => {
    if (!ready || !runId) return;
    const controller = new AbortController(),
      signal = controller.signal;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;
    const valid = () => !signal.aborted && landScope() === scope;
    const phase = (value: Phase, message?: string) => {
      if (valid()) setConnection({ identity, phase: value, message });
    };
    const cursor = () => cache.getQueryData<ResearchEvent[]>(eventKey)?.at(-1)?.sequence ?? 0;
    const refresh = () => {
      if (!valid()) return;
      void cache.invalidateQueries({ queryKey: ["land-research", scope, "investigation"] });
    };
    const append = (incoming: ResearchEvent[]) => {
      if (!valid() || !incoming.length) return;
      const after = cursor(),
        fresh = incoming.filter((item) => item.sequence > after);
      if (!fresh.length) return;
      cache.setQueryData<ResearchEvent[]>(eventKey, (previous) =>
        mergeResearchEvents(previous ?? [], fresh),
      );
      if (fresh.some((item) => terminal.has(item.kind))) {
        refresh();
        void cache.invalidateQueries({ queryKey: ["land-research", scope, "list"] });
      } else if (
        !refreshTimer &&
        fresh.some((item) => ["finding", "artifact", "started"].includes(item.kind))
      ) {
        refreshTimer = setTimeout(() => {
          refreshTimer = null;
          refresh();
        }, 400);
      }
    };
    const catchUp = async () => {
      for (let page = 0; page < 20 && valid(); page++) {
        const batch = await unwrap(
          api.GET("/api/v1/research/runs/{run_id}/events", {
            params: { path: { run_id: runId }, query: { after: cursor() } },
            signal,
          }),
        );
        const parsed = batch.map(parseResearchEvent);
        const last = parsed.at(-1);
        if (last && last.sequence <= cursor()) throw new Error("Research cursor did not advance.");
        append(parsed);
        if (batch.length < 200) return true;
      }
      return false;
    };
    const pump = async () => {
      let failures = 0,
        polling = false;
      while (valid()) {
        try {
          const caughtUp = await catchUp();
          if (!valid()) return;
          if (!caughtUp) {
            await pause(0, signal);
            continue;
          }
          if (!running) {
            phase("complete");
            return;
          }
          if (polling) {
            phase("polling");
            await pause(3000, signal);
            continue;
          }
          const connectionController = new AbortController();
          const abort = () => connectionController.abort();
          signal.addEventListener("abort", abort, { once: true });
          const timeout = setTimeout(abort, 35_000);
          try {
            const result = await api.GET("/api/v1/research/runs/{run_id}/stream", {
              params: { path: { run_id: runId }, query: { after: cursor() } },
              parseAs: "stream",
              signal: connectionController.signal,
            });
            if (!result.response.ok) {
              if ([404, 405, 406, 501].includes(result.response.status)) {
                polling = true;
                continue;
              }
              throw new ApiError(
                result.response.status,
                undefined,
                "Live research updates could not connect.",
              );
            }
            if (
              !result.data ||
              !result.response.headers.get("content-type")?.includes("text/event-stream")
            ) {
              polling = true;
              continue;
            }
            phase("live");
            await readResearchStream(result.data, connectionController.signal, (event) =>
              append([event]),
            );
            if (connectionController.signal.aborted && valid())
              throw new Error("Research stream timed out.");
            failures = 0;
          } finally {
            clearTimeout(timeout);
            signal.removeEventListener("abort", abort);
            connectionController.abort();
          }
          await pause(500, signal);
        } catch (error) {
          if (!valid()) return;
          if (
            error instanceof ResearchAccessError ||
            (error instanceof ApiError && [401, 403, 404].includes(error.status))
          ) {
            const message =
              error instanceof ResearchAccessError
                ? error.message
                : "Sign in or refresh workspace access to resume research updates.";
            phase("unavailable", message);
            if (
              (error instanceof ResearchAccessError && error.reason === "revoked") ||
              (error instanceof ApiError && [403, 404].includes(error.status))
            )
              cache.setQueryData(eventKey, []);
            return;
          }
          phase("reconnecting", "Research continues on the server. Reconnecting to its updates…");
          failures++;
          // A repeatedly blocked stream falls back to incremental event reads.
          if (failures >= 3) polling = true;
          await pause(Math.min(15_000, 1000 * 2 ** failures), signal);
        }
      }
    };
    void pump();
    return () => {
      controller.abort();
      if (refreshTimer) clearTimeout(refreshTimer);
    };
  }, [runId, running, ready, scope, cache, eventKey, identity, attempt, credentials]);
  return {
    data: query.data,
    phase:
      connection?.identity === identity ? connection.phase : running ? "connecting" : "complete",
    message: connection?.identity === identity ? connection.message : undefined,
    retry: () => retry((value) => value + 1),
  };
}
