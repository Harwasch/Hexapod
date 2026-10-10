import { useEffect, useRef, useState } from "react";
import type { MediaAsset, Reconstruction, Replay, WorldProject } from "../core/types";
import {
  createReconstructionBundle,
  downloadBlob,
  manifestFor,
  type ReconstructionSource,
} from "./bundle";
import { extractReplayFrames } from "./frames";
import {
  reconstructionGateway,
  type ReconstructionCapabilities,
  type ReconstructionJob,
} from "./gateway";
import { LocalSplatViewer } from "./LocalSplatViewer";
import { boundedArtifactBlob, VIEWER_MAX_BYTES } from "./media";
import "./reconstruction.css";

export interface ReconstructionPanelProps {
  projects: WorldProject[];
  replays: Replay[];
  assets: MediaAsset[];
  reconstructions: Reconstruction[];
  getAssetBlob: (id: string) => Promise<Blob | undefined>;
  onSave: (record: Reconstruction) => Promise<void>;
  apiBaseUrl?: string;
}

function FrameChoice({
  source,
  selected,
  onToggle,
}: {
  source: ReconstructionSource;
  selected: boolean;
  onToggle: () => void;
}) {
  const image = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const next = URL.createObjectURL(source.blob);
    if (image.current) image.current.src = next;
    return () => URL.revokeObjectURL(next);
  }, [source.blob]);
  return (
    <label className="worlds-reconstruction-frame">
      {source.blob.type.startsWith("image/") ? (
        <img ref={image} alt={source.name} />
      ) : (
        <span className="worlds-reconstruction-video">Recorded video</span>
      )}
      <span>
        <input type="checkbox" checked={selected} onChange={onToggle} />
        {source.timestampMs === undefined
          ? source.name
          : `${(source.timestampMs / 1000).toFixed(1)} s`}
      </span>
    </label>
  );
}

function safeArtifactUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (url.username || url.password || url.hash) return undefined;
    if (
      url.protocol === "https:" ||
      (url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname))
    )
      return value;
  } catch {
    /* A corrupt imported record must not become a navigable link. */
  }
  return undefined;
}

function recordFromJob(job: ReconstructionJob, record: Reconstruction): Reconstruction {
  return {
    ...record,
    jobId: job.id,
    status: job.status,
    progress: job.progress ?? undefined,
    stage: job.stage ?? undefined,
    error: job.error ?? undefined,
    artifacts: job.artifacts,
  };
}

export function ReconstructionPanel({
  projects,
  replays,
  assets,
  reconstructions,
  getAssetBlob,
  onSave,
  apiBaseUrl = "",
}: ReconstructionPanelProps) {
  const [sources, setSources] = useState<ReconstructionSource[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [replayId, setReplayId] = useState("");
  const [busy, setBusy] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [remoteConsent, setRemoteConsent] = useState(false);
  const [capabilities, setCapabilities] = useState<ReconstructionCapabilities | null>(null);
  const [capabilityError, setCapabilityError] = useState("");
  const [viewFile, setViewFile] = useState<File | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [reports, setReports] = useState<
    Record<string, { diagnostics?: string; cameras?: string }>
  >({});
  const extraction = useRef<AbortController | null>(null);
  const records = useRef(reconstructions);
  const save = useRef(onSave);
  useEffect(() => {
    records.current = reconstructions;
    save.current = onSave;
  }, [reconstructions, onSave]);
  const replay = replays.find((item) => item.id === replayId);
  const project = projects.find((item) => item.id === replay?.projectId);
  const chosen = sources.filter((source) => selected.includes(source.id));
  const bytes = chosen.reduce((total, source) => total + source.blob.size, 0);

  useEffect(() => {
    let live = true;
    void reconstructionGateway
      .capabilities(apiBaseUrl)
      .then((value) => {
        if (live) {
          setCapabilities(value);
          setCapabilityError("");
        }
      })
      .catch(() => {
        if (live) {
          setCapabilities(null);
          setCapabilityError(
            "Reconstruction service is offline or requires an access token in Settings. Local export still works.",
          );
        }
      });
    return () => {
      live = false;
    };
  }, [apiBaseUrl]);
  useEffect(() => () => extraction.current?.abort(), []);
  useEffect(() => {
    if (!activeId) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async (): Promise<void> => {
      const record = records.current.find((item) => item.id === activeId);
      if (!record?.jobId) return;
      try {
        const job = await reconstructionGateway.status(apiBaseUrl, record.jobId);
        if (!live) return;
        setReports((current) => ({
          ...current,
          [record.id]: {
            diagnostics: safeArtifactUrl(job.diagnosticsUrl ?? ""),
            cameras: safeArtifactUrl(job.camerasUrl ?? ""),
          },
        }));
        await save.current(recordFromJob(job, record));
        if (job.status === "queued" || job.status === "running")
          timer = setTimeout(() => {
            void poll();
          }, 4000);
        else setActiveId(null);
      } catch (cause) {
        if (live) {
          setError(
            cause instanceof Error ? cause.message : "Could not refresh reconstruction progress.",
          );
          setActiveId(null);
        }
      }
    };
    timer = setTimeout(() => {
      void poll();
    }, 500);
    return () => {
      live = false;
      if (timer) clearTimeout(timer);
    };
  }, [activeId, apiBaseUrl]);

  const replaceSources = (next: ReconstructionSource[]): void => {
    setSources(next);
    setSelected(next.map((source) => source.id));
    setRemoteConsent(false);
  };
  const loadReplay = async (id: string): Promise<void> => {
    setReplayId(id);
    setError("");
    setMessage("");
    const next = replays.find((item) => item.id === id);
    if (!next) {
      replaceSources([]);
      return;
    }
    setBusy(true);
    try {
      const blob = await getAssetBlob(next.assetId);
      if (!blob) throw new Error("The recording is missing from this browser's local library.");
      replaceSources([{ id: next.assetId, assetId: next.assetId, blob, name: next.name }]);
    } catch (cause) {
      replaceSources([]);
      setError(cause instanceof Error ? cause.message : "Could not load recording.");
    } finally {
      setBusy(false);
    }
  };
  const extract = async (): Promise<void> => {
    const source = sources.find((item) => item.blob.type.startsWith("video/"));
    if (!source) return;
    const controller = new AbortController();
    extraction.current = controller;
    setExtracting(true);
    setBusy(true);
    setError("");
    try {
      const frames = await extractReplayFrames(source.blob, {
        durationMs: replay?.durationMs,
        signal: controller.signal,
        onProgress: (done, total) =>
          setMessage(`Extracting frame ${done} of ${total} on this device…`),
      });
      replaceSources(frames);
      setMessage("Review the frames below. Keep overlapping views of a stable scene.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not extract replay frames.");
    } finally {
      extraction.current = null;
      setExtracting(false);
      setBusy(false);
    }
  };
  const submit = async (): Promise<void> => {
    if (!remoteConsent || !capabilities?.configured || !chosen.length) return;
    if (bytes > capabilities.maxUploadBytes) {
      setError("Select fewer frames or a smaller segment; the upload limit is 64 MiB.");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("Uploading selected sources to the configured reconstruction worker…");
    try {
      const manifest = manifestFor(chosen, replay?.modelId);
      const body = new FormData();
      chosen.forEach((source, index) =>
        body.append(
          "files",
          source.blob,
          manifest.frames[index]?.path.split("/").pop() ?? `source-${index}`,
        ),
      );
      body.set("metadata", JSON.stringify(manifest));
      const job = await reconstructionGateway.submit(apiBaseUrl, body);
      const record = recordFromJob(job, {
        id: crypto.randomUUID(),
        name: `${replay?.name ?? "Explored scene"} · 3D`,
        projectId: project?.id ?? "local-import",
        createdAt: Date.now(),
        status: job.status,
        format: "ply",
        sourceReplayId: replay?.id,
        sourceAssetIds: chosen.flatMap((source) => (source.assetId ? [source.assetId] : [])),
      });
      await onSave(record);
      setReports((current) => ({
        ...current,
        [record.id]: {
          diagnostics: safeArtifactUrl(job.diagnosticsUrl ?? ""),
          cameras: safeArtifactUrl(job.camerasUrl ?? ""),
        },
      }));
      setMessage("Reconstruction submitted. Progress comes directly from the worker.");
      if (job.status === "queued" || job.status === "running") setActiveId(record.id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not submit reconstruction.");
      setMessage("");
    } finally {
      setBusy(false);
    }
  };
  const openArtifact = async (url: string, format: string): Promise<void> => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(url, {
        credentials: "omit",
        referrerPolicy: "no-referrer",
        redirect: "error",
        signal: AbortSignal.timeout(60_000),
      });
      if (!response.ok)
        throw new Error("The download link expired. Refresh reconstruction status for a new link.");
      const blob = await boundedArtifactBlob(response);
      setViewFile(
        new File([blob], `reconstructed-world.${format === "point-cloud" ? "ply" : format}`),
      );
      setMessage("Reconstructed geometry is open below.");
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Download the file and use Open 3D file to view it locally.",
      );
    } finally {
      setBusy(false);
    }
  };
  const deleteWorkerData = async (record: Reconstruction): Promise<void> => {
    if (!record.jobId) return;
    setBusy(true);
    setError("");
    setActiveId(null);
    try {
      const result = await reconstructionGateway.delete(apiBaseUrl, record.jobId);
      if (result.deleted !== true)
        throw new Error("The worker did not confirm deletion. Its data may still exist.");
      await onSave({
        ...record,
        jobId: undefined,
        artifacts: [],
        status: "failed",
        progress: undefined,
        stage: "Worker data deleted",
        error: undefined,
      });
      setReports((current) => ({ ...current, [record.id]: {} }));
      setViewFile(null);
      setMessage(
        "The worker confirmed deletion of this job's input and output files. Your local source media remains in your library.",
      );
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Deletion failed. Contact the worker operator to remove its data.",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="worlds-reconstruction" aria-label="3D worlds">
      <header>
        <span className="worlds-reconstruction-eyebrow">FROM EXPLORATION TO A PLACE</span>
        <h2>Save this place in 3D</h2>
        <p>
          Choose the views worth keeping. Export them locally, or reconstruct with your connected
          worker.
        </p>
      </header>
      <div className="worlds-reconstruction-columns">
        <div className="worlds-reconstruction-card">
          <h3>01 / Choose your source</h3>
          <label className="worlds-reconstruction-field">
            Saved replay
            <select
              value={replayId}
              disabled={busy}
              onChange={(event) => {
                void loadReplay(event.target.value);
              }}
            >
              <option value="">Choose a replay</option>
              {replays.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                  {item.previewOnly ? " (interaction preview)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="worlds-reconstruction-file">
            Add frames or a recording
            <input
              type="file"
              multiple
              accept="image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime"
              disabled={busy}
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                setReplayId("");
                setError("");
                replaceSources(
                  files
                    .slice(0, 120)
                    .map((file) => ({ id: crypto.randomUUID(), blob: file, name: file.name })),
                );
                if (files.length > 120) setMessage("The first 120 files were selected.");
                event.target.value = "";
              }}
            />
          </label>
          {assets.some((asset) => asset.kind === "image") && (
            <label className="worlds-reconstruction-field">
              Or add a saved image
              <select
                value=""
                disabled={busy}
                onChange={(event) => {
                  const id = event.target.value;
                  const asset = assets.find((item) => item.id === id);
                  if (!asset) return;
                  void getAssetBlob(id)
                    .then((blob) => {
                      if (!blob) throw new Error("This image is missing from local storage.");
                      setSources((previous) =>
                        previous.some((item) => item.id === id)
                          ? previous
                          : [...previous, { id, assetId: id, blob, name: asset.name }],
                      );
                      setSelected((previous) => [...new Set([...previous, id])]);
                      setRemoteConsent(false);
                    })
                    .catch((cause: unknown) =>
                      setError(cause instanceof Error ? cause.message : "Could not load image."),
                    );
                }}
              >
                <option value="">Choose an image</option>
                {assets
                  .filter((asset) => asset.kind === "image")
                  .map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.name}
                    </option>
                  ))}
              </select>
            </label>
          )}
          {sources.some((source) => source.blob.type.startsWith("video/")) && (
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                void extract();
              }}
            >
              Extract 24 candidate frames locally
            </button>
          )}
          {extracting && (
            <button type="button" onClick={() => extraction.current?.abort()}>
              Cancel extraction
            </button>
          )}
          {replay?.previewOnly && (
            <p className="worlds-reconstruction-note">
              This replay is an interaction preview, not world-model output. It is unlikely to
              reconstruct a coherent place.
            </p>
          )}
          {sources.length > 0 && (
            <>
              <div className="worlds-reconstruction-selection">
                <span>
                  {chosen.length} selected · {(bytes / 1024 / 1024).toFixed(1)} MiB
                </span>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    setSelected(selected.length ? [] : sources.map((source) => source.id))
                  }
                >
                  {selected.length ? "Clear selection" : "Select all"}
                </button>
              </div>
              <div className="worlds-reconstruction-frames">
                {sources.map((source) => (
                  <FrameChoice
                    key={source.id}
                    source={source}
                    selected={selected.includes(source.id)}
                    onToggle={() =>
                      setSelected((previous) =>
                        previous.includes(source.id)
                          ? previous.filter((id) => id !== source.id)
                          : [...previous, source.id],
                      )
                    }
                  />
                ))}
              </div>
            </>
          )}
        </div>
        <div className="worlds-reconstruction-card">
          <h3>02 / Make it portable</h3>
          <p>
            Download a source bundle containing your selected media, timestamps and reconstruction
            manifest. Everything stays on this device.
          </p>
          <button
            type="button"
            disabled={!chosen.length || busy}
            onClick={() => {
              downloadBlob(
                createReconstructionBundle(chosen, replay?.modelId),
                "worlds-reconstruction-sources.tar",
              );
              setMessage(
                "Source bundle exported. It contains source media, not reconstructed geometry.",
              );
            }}
          >
            Export source bundle .tar
          </button>
          <hr />
          <h3>Reconstruct with a worker</h3>
          <p>
            {capabilityError
              ? capabilityError
              : (capabilities?.message ?? "Checking the reconstruction service…")}
          </p>
          <p className="worlds-reconstruction-note">
            Generated views can drift. Reconstruction estimates geometry; it does not make the world
            persistent. PLY, SPZ and mesh exports appear only when the worker produces them.
          </p>
          {capabilities?.configured && (
            <label className="worlds-reconstruction-consent">
              <input
                type="checkbox"
                checked={remoteConsent}
                onChange={(event) => setRemoteConsent(event.target.checked)}
              />
              Send only selected media and camera metadata to my configured worker. GPU charges may
              apply.
            </label>
          )}
          <button
            type="button"
            className="worlds-reconstruction-primary"
            disabled={busy || !chosen.length || !remoteConsent || !capabilities?.configured}
            onClick={() => {
              void submit();
            }}
          >
            Reconstruct selected views
          </button>
        </div>
      </div>
      {message && (
        <p role="status" className="worlds-reconstruction-note">
          {message}
        </p>
      )}
      {error && (
        <p role="alert" className="worlds-reconstruction-error">
          {error}
        </p>
      )}
      {reconstructions.length > 0 && (
        <div className="worlds-reconstruction-card">
          <h3>Your reconstructions</h3>
          {reconstructions.map((record) => (
            <article key={record.id} className="worlds-reconstruction-job">
              <div>
                <strong>{record.name}</strong>
                <p>
                  {record.stage ?? record.status}
                  {record.progress !== undefined ? ` · ${Math.round(record.progress * 100)}%` : ""}
                </p>
                {record.error && <p role="alert">{record.error}</p>}
              </div>
              {record.progress !== undefined && (
                <progress value={record.progress} max={1} aria-label={`${record.name} progress`} />
              )}
              {record.jobId && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    void deleteWorkerData(record);
                  }}
                >
                  Delete worker data
                </button>
              )}
              {record.jobId && (
                <button
                  type="button"
                  disabled={busy || activeId === record.id}
                  onClick={() => {
                    setError("");
                    setActiveId(record.id);
                  }}
                >
                  {activeId === record.id ? "Following progress…" : "Refresh status"}
                </button>
              )}
              {record.artifacts?.map((artifact, index) => {
                const url = safeArtifactUrl(artifact.url);
                return url ? (
                  <span key={index}>
                    <a href={url} target="_blank" rel="noopener noreferrer">
                      Download {artifact.format.toUpperCase()}
                    </a>
                    {["ply", "spz", "point-cloud"].includes(artifact.format) && (
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => {
                          void openArtifact(url, artifact.format);
                        }}
                      >
                        Open in viewer
                      </button>
                    )}
                  </span>
                ) : null;
              })}
              {reports[record.id]?.diagnostics && (
                <a href={reports[record.id]?.diagnostics} target="_blank" rel="noopener noreferrer">
                  Download reconstruction diagnostics
                </a>
              )}
              {reports[record.id]?.cameras && (
                <a href={reports[record.id]?.cameras} target="_blank" rel="noopener noreferrer">
                  Download estimated cameras
                </a>
              )}
            </article>
          ))}
        </div>
      )}
      <div className="worlds-reconstruction-card">
        <h3>Explore a reconstructed place</h3>
        <p>
          Open a Gaussian splat (.ply, .spz) or PLY point cloud in this browser. The file stays
          local.
        </p>
        <label className="worlds-reconstruction-file">
          Open 3D file
          <input
            type="file"
            accept=".ply,.spz"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file && file.size <= VIEWER_MAX_BYTES) {
                setError("");
                setViewFile(file);
              } else if (file) setError("Choose a file below 128 MiB for the embedded viewer.");
              event.target.value = "";
            }}
          />
        </label>
        {viewFile && (
          <>
            <p>{viewFile.name}</p>
            <LocalSplatViewer key={`${viewFile.name}-${viewFile.lastModified}`} file={viewFile} />
          </>
        )}
      </div>
    </section>
  );
}
