import { useCallback, useEffect, useRef, useState } from "react";
import { Check, FlaskConical, RefreshCw, Square } from "lucide-react";
import { createWorldApi } from "../core/api";
import "./intelligence.css";
interface Profile {
  id: string;
  name: string;
  modelId: string;
  stage: string;
  devices: 8;
  datasetLabel: string;
  activationCompatible: boolean;
}
interface Capabilities {
  configured: boolean;
  trainingEnabled: boolean;
  maxTrainingSeconds: number;
  maxConcurrentJobs: number;
  profiles: Profile[];
  message: string;
}
interface Job {
  id: string;
  profileId: string;
  modelId: string;
  stage: string;
  status: string;
  createdAt: number;
  activationCompatible: boolean;
  gpuValidated: boolean;
  artifacts: { id: string; name: string; size: number }[];
  installed: string[];
  selectedArtifactId?: string | null;
  message: string;
}
export interface CustomizationPanelProps {
  serverUrl: string;
}
export function CustomizationPanel({ serverUrl }: CustomizationPanelProps) {
  return <CustomizationInstance key={serverUrl} serverUrl={serverUrl} />;
}
function validatedCapabilities(value: Capabilities): Capabilities {
  if (
    !value ||
    typeof value.configured !== "boolean" ||
    typeof value.message !== "string" ||
    !Array.isArray(value.profiles) ||
    typeof value.trainingEnabled !== "boolean" ||
    !Number.isFinite(value.maxTrainingSeconds) ||
    !Number.isFinite(value.maxConcurrentJobs) ||
    value.profiles.some(
      (p) =>
        !p ||
        typeof p.id !== "string" ||
        typeof p.name !== "string" ||
        typeof p.modelId !== "string" ||
        typeof p.stage !== "string" ||
        typeof p.datasetLabel !== "string" ||
        p.devices !== 8 ||
        typeof p.activationCompatible !== "boolean",
    )
  )
    throw new Error(
      "The customization worker returned invalid capabilities. Check its compatible API version.",
    );
  return value;
}
function validatedJob(job: Job): Job {
  if (
    !job ||
    typeof job.id !== "string" ||
    typeof job.status !== "string" ||
    typeof job.message !== "string" ||
    !Array.isArray(job.installed) ||
    job.installed.some((id) => typeof id !== "string") ||
    !Array.isArray(job.artifacts) ||
    job.artifacts.some(
      (a) =>
        !a || typeof a.id !== "string" || typeof a.name !== "string" || !Number.isFinite(a.size),
    )
  )
    throw new Error("The customization worker returned an invalid job record.");
  return job;
}
function CustomizationInstance({ serverUrl }: CustomizationPanelProps) {
  const pending = useRef(false);
  const controller = useRef(new AbortController());
  useEffect(() => {
    const current = controller.current;
    return () => current.abort();
  }, []);
  const [capabilities, setCapabilities] = useState<Capabilities>();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [profileId, setProfileId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [message, setMessage] = useState("");
  const load = useCallback(
    async (signal: AbortSignal = controller.current.signal) => {
      if (pending.current) return;
      pending.current = true;
      try {
        const api = createWorldApi(serverUrl);
        const caps = validatedCapabilities(
          await api.request<Capabilities>("/customization/capabilities", { signal }),
        );
        if (signal.aborted) return;
        setCapabilities(caps);
        if (caps.configured) {
          const result = await api.request<{ jobs: Job[] }>("/customization/jobs", { signal });
          if (!Array.isArray(result?.jobs))
            throw new Error("The customization worker returned invalid jobs.");
          const records = result.jobs.map(validatedJob);
          if (!signal.aborted) setJobs(records);
        } else setJobs([]);
        setError("");
      } finally {
        pending.current = false;
      }
    },
    [serverUrl],
  );
  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => {
      if (document.visibilityState !== "hidden")
        void load(controller.signal).catch((failure: unknown) => {
          if (!controller.signal.aborted)
            setError(
              failure instanceof Error ? failure.message : "Customization service is unavailable.",
            );
        });
    };
    refresh();
    const timer = setInterval(refresh, 15_000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [load]);
  async function perform(path: string, body: unknown) {
    if (busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = validatedJob(
        await createWorldApi(serverUrl).request<Job>(path, {
          method: "POST",
          body: JSON.stringify(body),
          signal: controller.current.signal,
        }),
      );
      if (controller.current.signal.aborted) return;
      setConfirmed(false);
      setMessage(result.message || `Job ${result.status}.`);
      await load();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Customization operation failed.");
    } finally {
      setBusy(false);
    }
  }
  const selected = capabilities?.profiles.find((p) => p.id === profileId);
  return (
    <section className="wi-panel" aria-label="Model customization">
      <h3>
        <FlaskConical size={17} />
        Model customization
      </h3>
      <p className="wi-muted">
        Verified upstream staged training recipes for ForgeWM and SANA-WM. These are full
        distributed training/distillation jobs, not generic character or style LoRA. Datasets stay
        on the configured training host.
      </p>
      {capabilities && <p className="wi-notice">{capabilities.message}</p>}
      <button
        type="button"
        disabled={busy}
        onClick={() => {
          void load().catch((failure: unknown) =>
            setError(
              failure instanceof Error ? failure.message : "Could not refresh customization.",
            ),
          );
        }}
      >
        <RefreshCw size={14} />
        Refresh customization
      </button>
      {capabilities?.configured && (
        <>
          <label className="wi-field">
            Operator-approved recipe and dataset
            <select
              value={profileId}
              onChange={(e) => {
                setProfileId(e.target.value);
                setConfirmed(false);
              }}
            >
              <option value="">Choose a local profile</option>
              {capabilities.profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.modelId} · {p.stage}
                </option>
              ))}
            </select>
          </label>
          {selected && (
            <p className="wi-muted">
              {selected.datasetLabel} · requires {selected.devices} visible CUDA GPUs.{" "}
              {selected.activationCompatible
                ? "Final distilled student can be validated and installed."
                : "Intermediate checkpoint; complete the required later stages before inference activation."}
            </p>
          )}
          <button
            type="button"
            disabled={!selected || busy}
            onClick={() => void perform("/customization/jobs", { profileId })}
          >
            Prepare recipe without running training
          </button>
          <p className="wi-muted">
            Maximum training runtime: {Math.round(capabilities.maxTrainingSeconds / 3600)} hours.
            Only one training job runs at a time. Failed or cancelled runs retain local artifacts
            and logs; prepare a new approved job to retry.
          </p>
          <label className="wi-check">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
            I authorize a training job on the existing eight-GPU host. It may incur ongoing provider
            charges. No new compute is allocated here.
          </label>
          {!capabilities.trainingEnabled && (
            <p className="wi-notice">
              Execution is disabled by the worker operator. Preparation and existing artifacts
              remain inspectable.
            </p>
          )}
        </>
      )}
      {jobs.map((job) => (
        <article className="wi-result" key={job.id}>
          <strong>
            {job.modelId} · {job.stage} · {job.status}
          </strong>
          <p className="wi-muted">{job.message}</p>
          <p className="wi-muted">
            {job.gpuValidated
              ? "Worker reports GPU validation."
              : "Checkpoint inference has not been GPU-validated by this application."}
          </p>
          <div className="wi-row">
            {job.status === "prepared" && (
              <button
                type="button"
                disabled={busy || !confirmed || !capabilities?.trainingEnabled}
                onClick={() =>
                  void perform(`/customization/jobs/${encodeURIComponent(job.id)}/run`, {
                    confirmTraining: true,
                  })
                }
              >
                Run approved training
              </button>
            )}
            {["running", "cancelling", "unknown"].includes(job.status) && (
              <button
                type="button"
                disabled={busy || job.status === "cancelling"}
                onClick={() =>
                  void perform(`/customization/jobs/${encodeURIComponent(job.id)}/cancel`, {})
                }
              >
                <Square size={14} />
                Cancel training
              </button>
            )}
            {job.selectedArtifactId && (
              <button
                type="button"
                disabled={busy}
                onClick={() =>
                  void perform(`/customization/jobs/${encodeURIComponent(job.id)}/disable`, {})
                }
              >
                Select base checkpoint
              </button>
            )}
          </div>
          {job.artifacts.map((artifact) => (
            <div className="wi-row" key={artifact.id}>
              <span>
                {artifact.name} · {(artifact.size / 1024 / 1024).toFixed(1)} MB
              </span>
              {job.installed.includes(artifact.id) ? (
                <button
                  type="button"
                  disabled={busy || job.selectedArtifactId === artifact.id}
                  onClick={() =>
                    void perform(`/customization/jobs/${encodeURIComponent(job.id)}/enable`, {
                      artifactId: artifact.id,
                    })
                  }
                >
                  <Check size={14} />
                  {job.selectedArtifactId === artifact.id
                    ? "Selected for next worker"
                    : "Select for next worker"}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={busy || job.status !== "completed" || !job.activationCompatible}
                  onClick={() =>
                    void perform(`/customization/jobs/${encodeURIComponent(job.id)}/install`, {
                      artifactId: artifact.id,
                    })
                  }
                >
                  Validate and install
                </button>
              )}
            </div>
          ))}
        </article>
      ))}
      <p className="wi-muted">
        Selection writes a worker-local environment fragment for the operator’s next inference
        start. It does not reload a running model, deploy a worker, or prove checkpoint quality.
      </p>
      {error && (
        <p className="wi-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="wi-muted" role="status">
          {message}
        </p>
      )}
    </section>
  );
}
