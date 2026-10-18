import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw, Server, Square, Trash2 } from "lucide-react";

import { createWorldApi, type WorkerHandle } from "./core/api";

interface Worker extends WorkerHandle {
  message?: string;
}
interface Session {
  id: string;
  workerId: string;
  modelId: string;
  status: string;
  leaseExpiresAt?: number;
}
interface WorkerSnapshot {
  serverUrl: string;
  workers: Worker[];
  sessions: Session[];
}
function deadlineLabel(deadline: number) {
  const seconds = Math.max(0, Math.ceil(deadline - Date.now() / 1000));
  return seconds === 0
    ? "due now"
    : seconds < 60
      ? `in ${seconds}s`
      : `in ${Math.ceil(seconds / 60)} min`;
}

/** Recovery stays available after a closed tab or failed inference session. */
export function WorkerManager({ serverUrl }: { serverUrl: string }) {
  const [snapshot, setSnapshot] = useState<WorkerSnapshot>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [confirm, setConfirm] = useState<string | null>(null);
  const connection = useRef(serverUrl);
  const inFlight = useRef(false);
  const current = snapshot?.serverUrl === serverUrl ? snapshot : undefined;
  const workers = current?.workers ?? [];
  const sessions = current?.sessions ?? [];

  const reload = useCallback(
    async (signal?: AbortSignal) => {
      const api = createWorldApi(serverUrl);
      const [workerList, sessionList] = await Promise.all([
        api.request<{ workers: Worker[] }>("/workers", {
          signal: signal ?? AbortSignal.timeout(10_000),
        }),
        api.request<{ sessions: Session[] }>("/sessions", {
          signal: signal ?? AbortSignal.timeout(10_000),
        }),
      ]);
      if (!signal?.aborted && connection.current === serverUrl)
        setSnapshot({ serverUrl, workers: workerList.workers, sessions: sessionList.sessions });
    },
    [serverUrl],
  );

  useEffect(() => {
    connection.current = serverUrl;
    const controller = new AbortController();
    let polling = false;
    const refresh = () => {
      if (document.visibilityState === "hidden" || inFlight.current || polling) return;
      polling = true;
      void reload(AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]))
        .catch((failure: unknown) => {
          if (!controller.signal.aborted)
            setError(
              failure instanceof Error ? failure.message : "Could not refresh worker recovery.",
            );
        })
        .finally(() => {
          polling = false;
        });
    };
    refresh();
    const timer = window.setInterval(refresh, 15_000);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [serverUrl, reload]);

  async function run(action: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await action();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The worker operation failed.");
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  const active = workers.filter((worker) => !["destroyed", "detached"].includes(worker.status));
  return (
    <section className="w-panel">
      <div className="w-settings-title">
        <Server size={19} />
        <h2>Worker recovery</h2>
      </div>
      <p className="w-panel-subtitle">
        Inspect retained workers and interrupted sessions using your saved connection. Status
        refreshes while this page is visible; it never creates compute. Retained or stopped pods can
        still incur provider charges.
      </p>
      <button
        className="w-btn w-btn-quiet"
        disabled={busy}
        onClick={() => void run(() => reload())}
      >
        <RefreshCw size={14} />
        Refresh workers
      </button>
      {error && (
        <p className="w-inline-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="w-muted" role="status">
          {message}
        </p>
      )}
      {!current && !error && (
        <p className="w-muted" role="status" style={{ marginTop: 16 }}>
          Checking retained workers…
        </p>
      )}
      {current && !active.length && (
        <p className="w-muted" style={{ marginTop: 16 }}>
          No retained worker records.
        </p>
      )}
      {active.map((worker) => {
        const terminating = worker.status === "terminating";
        const uncertain = ["unknown", "provisioning"].includes(worker.status);
        return (
          <div
            key={worker.id}
            style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid #ffffff18" }}
          >
            <p>
              <strong>
                {worker.provider} · {worker.status}
              </strong>
            </p>
            <p className="w-muted" style={{ overflowWrap: "anywhere", margin: "8px 0" }}>
              {worker.id}
            </p>
            <p className="w-muted">
              {worker.managed
                ? "Created by Worlds; can be stopped or destroyed here."
                : "Externally managed; disconnecting does not stop compute or billing."}
            </p>
            <div className="w-worker-facts">
              <span>
                {typeof worker.estimatedHourlyCost === "number"
                  ? `GPU rate: $${worker.estimatedHourlyCost.toFixed(2)}/hr · $${(worker.estimatedHourlyCost / 60).toFixed(3)}/min`
                  : "GPU rate not supplied by provider"}
              </span>
              {worker.managed && worker.hardDeadline !== undefined && (
                <span>Maximum lifetime {deadlineLabel(worker.hardDeadline)}</span>
              )}
              {worker.managed && worker.idleDeadline !== undefined && (
                <span>Idle cleanup {deadlineLabel(worker.idleDeadline)}</span>
              )}
            </div>
            {worker.message && <p className="w-muted">{worker.message}</p>}
            {worker.cleanupError && (
              <p className="w-inline-error" role="alert">
                Automatic cleanup could not be confirmed: {worker.cleanupError}. Check the provider
                console; billing may continue.
              </p>
            )}
            {terminating && (
              <p className="w-muted">
                The manager is terminating this worker. Billing may continue until the provider
                confirms deletion.
              </p>
            )}
            {sessions
              .filter(
                (session) =>
                  session.workerId === worker.id &&
                  !["stopped", "ended", "failed"].includes(session.status),
              )
              .map((session) => (
                <div key={session.id} className="w-settings-bottom" style={{ marginTop: 12 }}>
                  <span className="w-muted">
                    {session.modelId} · {session.status}
                    {session.leaseExpiresAt !== undefined
                      ? ` · heartbeat lease expires ${deadlineLabel(session.leaseExpiresAt)}`
                      : ""}
                  </span>
                  <button
                    className="w-btn w-btn-quiet"
                    disabled={busy || terminating}
                    onClick={() =>
                      void run(async () => {
                        await createWorldApi(serverUrl).endSession(session.id);
                        await reload();
                        setMessage("Session ended.");
                      })
                    }
                  >
                    End session
                  </button>
                </div>
              ))}
            <div className="w-settings-bottom" style={{ marginTop: 12 }}>
              <button
                className="w-btn w-btn-quiet"
                disabled={busy}
                onClick={() =>
                  void run(async () => {
                    await createWorldApi(serverUrl).worker(worker.id);
                    await reload();
                  })
                }
              >
                Check status
              </button>
              {worker.managed && worker.status !== "stopped" && (
                <button
                  className="w-btn w-btn-quiet"
                  disabled={busy || terminating || uncertain}
                  onClick={() =>
                    void run(async () => {
                      await createWorldApi(serverUrl).stopWorker(worker.id);
                      await reload();
                      setMessage(
                        "GPU stopped. Storage charges may continue until the pod is destroyed.",
                      );
                    })
                  }
                >
                  <Square size={13} />
                  Stop worker
                </button>
              )}
              <button
                className="w-btn w-btn-danger"
                disabled={busy || uncertain || terminating}
                onClick={() => setConfirm(worker.id)}
              >
                <Trash2 size={13} />
                {worker.managed ? "Destroy worker" : "Remove connection"}
              </button>
            </div>
            {uncertain && (
              <p className="w-muted" style={{ marginTop: 10 }}>
                Creation may have reached the provider. Reconcile the provider console before
                attempting another allocation.
              </p>
            )}
            {confirm === worker.id && (
              <div
                role="group"
                aria-label={`Confirm removal of ${worker.id}`}
                style={{ marginTop: 12 }}
              >
                <p className="w-muted">
                  {worker.managed
                    ? "Destroy this pod and end any remaining inference sessions? Local library files are preserved."
                    : "Remove this connection after ending its sessions? External compute will keep running."}
                </p>
                <div className="w-settings-bottom" style={{ marginTop: 8 }}>
                  <button className="w-btn w-btn-quiet" onClick={() => setConfirm(null)}>
                    Cancel
                  </button>
                  <button
                    className="w-btn w-btn-danger"
                    disabled={busy || terminating}
                    onClick={() =>
                      void run(async () => {
                        await createWorldApi(serverUrl).destroyWorker(worker.id);
                        setConfirm(null);
                        await reload();
                        setMessage(
                          worker.managed
                            ? "Provider confirmed worker destruction."
                            : "Connection removed. Stop external compute with its owner.",
                        );
                      })
                    }
                  >
                    Confirm {worker.managed ? "destroy" : "removal"}
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
