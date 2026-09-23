import {
  Ban,
  Box,
  CheckCircle2,
  Film,
  FileText,
  Images,
  MapPin,
  Play,
  RotateCcw,
  ScanLine,
} from "lucide-react";
import { useState } from "react";

import type { Capture, CaptureFile, Job, JobStep, RunStatus } from "@twin/contracts";
import { GlassBadge, GlassButton, GlassProgress, type GlassProgressTone } from "@twin/ui";

import { useCancelJob, useRetryJob, useStepLog } from "@/api/queries";
import { formatBytes, formatDate, formatDuration } from "@/lib/format";
import { uploadsForCapture, type UploadItem } from "@/state/uploads";

import { PhoneHandoff } from "./PhoneHandoff";
import { classify, type Proposal } from "./recipes";

/**
 * Plain words for the API's status values. The raw value stays on `data-status`, where
 * tests and anyone inspecting the page can still read it exactly.
 */
const CAPTURE_STATUS: Record<string, string> = {
  "awaiting-files": "Waiting for files",
  "not-started": "Ready",
  "in-progress": "Processing",
  complete: "Done",
  error: "Failed",
  cancelled: "Cancelled",
};

const JOB_STATUS: Record<string, string> = {
  "not-started": "Queued",
  "in-progress": "Running",
  complete: "Done",
  error: "Failed",
  cancelled: "Cancelled",
};

const KIND: Record<string, { label: string; icon: typeof Film }> = {
  video: { label: "Video", icon: Film },
  images: { label: "Photos", icon: Images },
  "gaussian-splat": { label: "Splat", icon: Box },
  "point-cloud": { label: "Point cloud", icon: ScanLine },
};

/** The recipe's verb, which is what a person asked for: "Reconstruct", "Package and place". */
function recipeLabel(recipe: string): string {
  if (recipe === "photo-reconstruct") return "Reconstruct";
  if (recipe === "splat-ingest") return "Package and place";
  return recipe
    .split(/[-_]/)
    .map((part, i) => (i === 0 ? part.charAt(0).toUpperCase() + part.slice(1) : part))
    .join(" ");
}

/** Stage ids the two shipped recipes walk through, in the order they run. */
const STAGE_LABELS: Record<string, string> = {
  normalize: "Normalize",
  frames: "Extract frames",
  pose: "Pose",
  train: "Train",
  package: "Package",
  register: "Register",
};

function stageLabel(stageId: string): string {
  return (
    STAGE_LABELS[stageId] ??
    stageId
      .split(/[-_]/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

function statusTone(status: string): "neutral" | "accent" | "success" | "danger" | "warning" {
  switch (status) {
    case "complete":
      return "success";
    case "error":
      return "danger";
    case "in-progress":
      return "accent";
    case "cancelled":
      return "warning";
    default:
      return "neutral";
  }
}

function progressTone(status: string): GlassProgressTone {
  switch (status) {
    case "complete":
      return "success";
    case "error":
      return "danger";
    case "in-progress":
      return "run";
    default:
      return "idle";
  }
}

function elapsed(step: { startedAt: string | null; finishedAt: string | null }): string | null {
  if (!step.startedAt) return null;
  const end = step.finishedAt ? Date.parse(step.finishedAt) : Date.now();
  const start = Date.parse(step.startedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return formatDuration((end - start) / 1000);
}

function uploadLabel(item: UploadItem): string {
  switch (item.phase) {
    case "queued":
      return "Waiting";
    case "registering":
      return "Starting";
    case "uploading":
      return `${formatBytes(item.uploaded)} of ${formatBytes(item.bytes)}`;
    case "completing":
      return "Finishing";
    case "complete":
      return formatBytes(item.bytes);
    case "cancelled":
      return "Cancelled";
    case "error":
      return "Failed";
  }
}

/** What the API says about a file, for rows this session did not upload itself. */
function serverProgress(file: CaptureFile): { value: number; max: number; text: string } {
  if (file.status === "complete") return { value: 1, max: 1, text: formatBytes(file.bytes) };
  const total = file.partsTotal ?? 0;
  if (total > 0) {
    return {
      value: file.partsCompleted,
      max: total,
      text: `part ${file.partsCompleted} of ${total}`,
    };
  }
  return { value: 0, max: 1, text: file.status };
}

export interface CaptureCardProps {
  capture: Capture;
  job: Job | undefined;
  uploads: Record<string, UploadItem>;
  onProcess: (recipe: string) => void;
  onRetryUpload: (uploadId: string) => void;
  onCancelUpload: (uploadId: string) => void;
  onFlyTo: (siteId: string) => void;
  processing: boolean;
  /** Its QR code is already open at the top of the panel, so the card does not offer one. */
  handoffShown?: boolean;
}

export function CaptureCard({
  capture,
  job,
  uploads,
  onProcess,
  onRetryUpload,
  onCancelUpload,
  onFlyTo,
  processing,
  handoffShown = false,
}: CaptureCardProps) {
  const mine = uploadsForCapture(uploads, capture.id);
  const byFileId = new Map(mine.filter((item) => item.fileId).map((item) => [item.fileId, item]));
  const uploading = mine.some(
    (item) => item.phase === "uploading" || item.phase === "registering" || item.phase === "queued",
  );
  const failedUpload = mine.find((item) => item.phase === "error");
  const filesReady =
    capture.files.length > 0 && capture.files.every((file) => file.status === "complete");
  const proposal: Proposal = classify(
    capture.files.map((file) => ({ name: file.filename, size: file.bytes ?? 0 })),
  );
  // The files decide, not what was proposed when the capture was made: a phone can add a
  // video to a capture that began as a dropped splat, and `splat-ingest` would then fail
  // at `normalize` on a file it cannot read. The stored recipe is only the fallback for a
  // capture whose files have not arrived.
  const recipe =
    capture.files.length > 0
      ? proposal.recipe
      : typeof capture.metadata.recipe === "string"
        ? capture.metadata.recipe
        : proposal.recipe;
  // A job that failed before it produced any step can only be started over; one that got
  // somewhere offers "Retry from <stage>" instead, which keeps the work already done.
  const canProcess =
    filesReady && !uploading && (!job || (job.status === "error" && job.steps.length === 0));

  // A phone capture is created before anything is picked, so its stored kind is a guess;
  // once files land, what they are is the better answer.
  const kindId =
    capture.metadata.origin === "phone" && capture.files.length > 0 ? proposal.kind : capture.kind;
  const kind = KIND[kindId] ?? { label: kindId, icon: Box };
  const KindIcon = kind.icon;

  return (
    <li>
      <div className="card capture" data-testid={`capture-card-${capture.slug}`}>
        <div className="card__row">
          <KindIcon className="capture__kind-icon" size={16} aria-hidden="true" />
          <h4 className="card__title">{capture.name}</h4>
          <GlassBadge
            tone={statusTone(capture.status)}
            data-testid="capture-status"
            data-status={capture.status}
          >
            {CAPTURE_STATUS[capture.status] ?? capture.status}
          </GlassBadge>
        </div>
        <div className="card__meta">
          <span>{kind.label}</span>
          <span>{formatDate(capture.createdAt)}</span>
          {capture.metadata.origin === "phone" && <span>From phone</span>}
        </div>

        <ul className="capture__files">
          {capture.files.map((file) => {
            const item = byFileId.get(file.id);
            const server = serverProgress(file);
            return (
              <li key={file.id} className="capture__file" data-testid="capture-file">
                <div className="capture__file-row">
                  <span className="capture__file-name">{file.filename}</span>
                  <span className="capture__file-meta">
                    {item ? uploadLabel(item) : server.text}
                  </span>
                </div>
                <GlassProgress
                  className={
                    file.status === "complete" || item?.phase === "complete"
                      ? "capture__file-bar capture__file-bar--done"
                      : "capture__file-bar"
                  }
                  label={`Uploading ${file.filename}`}
                  value={item ? item.uploaded : server.value}
                  max={item ? Math.max(item.bytes, 1) : server.max}
                  valueText={item ? uploadLabel(item) : server.text}
                  tone={
                    item?.phase === "error"
                      ? "danger"
                      : file.status === "complete" || item?.phase === "complete"
                        ? "success"
                        : "accent"
                  }
                />
                {item && (item.phase === "uploading" || item.phase === "registering") && (
                  <div className="capture__file-actions">
                    <GlassButton
                      size="sm"
                      variant="ghost"
                      onClick={() => onCancelUpload(item.id)}
                      leadingIcon={<Ban size={13} aria-hidden="true" />}
                    >
                      Cancel
                    </GlassButton>
                  </div>
                )}
              </li>
            );
          })}
        </ul>

        {failedUpload && (
          <>
            <p className="card__error" data-testid="capture-upload-error">
              {failedUpload.error ?? "The upload failed."}
            </p>
            <div className="capture__file-actions">
              <GlassButton
                size="sm"
                onClick={() => onRetryUpload(failedUpload.id)}
                leadingIcon={<RotateCcw size={13} aria-hidden="true" />}
                data-testid="capture-retry-upload"
              >
                Resume upload
              </GlassButton>
            </div>
          </>
        )}

        {capture.files.length === 0 && capture.status === "awaiting-files" && (
          <p className="capture__job-note">No files yet.</p>
        )}

        {job ? <JobStages job={job} /> : null}

        <div className="capture__actions">
          {canProcess && (
            <GlassButton
              size="sm"
              variant="primary"
              onClick={() => onProcess(recipe)}
              loading={processing}
              leadingIcon={<Play size={13} aria-hidden="true" />}
              data-testid="capture-process"
            >
              {job?.status === "error" ? "Start again" : recipeLabel(recipe)}
            </GlassButton>
          )}
          {capture.siteId && (
            <GlassButton
              size="sm"
              variant={canProcess ? "ghost" : "primary"}
              onClick={() => capture.siteId && onFlyTo(capture.siteId)}
              leadingIcon={<MapPin size={13} aria-hidden="true" />}
              data-testid="capture-fly-to"
            >
              Show on map
            </GlassButton>
          )}
          {/* Only while the capture can still take files: a handoff link to a capture
              that is already processing would mint a credential with nothing to do. */}
          {!handoffShown &&
            (capture.status === "awaiting-files" || capture.status === "not-started") && (
              <PhoneHandoff captureId={capture.id} />
            )}
        </div>
      </div>
    </li>
  );
}

function jobProgressPct(job: Job): number {
  if (job.status === "complete") return 100;
  if (job.steps.length === 0) return 0;
  const done = job.steps.filter((step) => step.status === "complete").length;
  return Math.round((done / job.steps.length) * 100);
}

function JobStages({ job }: { job: Job }) {
  const queued: boolean = job.status === "not-started" && job.steps.length === 0;
  const status: RunStatus = job.status;
  const running = status === "not-started" || status === "in-progress";
  const cancel = useCancelJob();
  const retry = useRetryJob();
  const ordered = [...job.steps].sort((a, b) => a.ordinal - b.ordinal);
  // Where a retry picks up: the stage that failed, or the first one that never finished.
  const resumeAt =
    ordered.find((step) => step.status === "error" || step.status === "cancelled") ??
    ordered.find((step) => step.status !== "complete");
  const canRetry = (status === "error" || status === "cancelled") && resumeAt !== undefined;

  const current = ordered.find((step) => step.status !== "complete") ?? ordered.at(-1);
  const stubbed = ordered.filter((step) => step.impl === "stub").length;
  const summary =
    status === "complete"
      ? `${ordered.length} stages${job.durationS ? ` · ${formatDuration(job.durationS)}` : ""}`
      : current
        ? `Stage ${ordered.indexOf(current) + 1} of ${Math.max(ordered.length, 1)} · ${stageLabel(current.stageId)}`
        : "Stages";

  return (
    <div className="capture__job" data-testid="capture-job">
      <div className="card__row card__row--between">
        <span className="capture__job-title">{recipeLabel(job.recipe)}</span>
        <GlassBadge tone={statusTone(status)} data-testid="capture-job-status" data-status={status}>
          {JOB_STATUS[status] ?? status}
        </GlassBadge>
      </div>
      <GlassProgress
        label={`${recipeLabel(job.recipe)} progress`}
        value={jobProgressPct(job)}
        valueText={queued ? "Queued" : `${jobProgressPct(job)}%`}
        tone={progressTone(status)}
      />
      {queued && (
        <p className="capture__job-note" data-testid="capture-job-queued">
          Waiting for a free worker.
        </p>
      )}
      {stubbed > 0 && (
        <p className="capture__job-note capture__job-note--warn" data-testid="capture-job-stub">
          {stubbed === ordered.length ? "All stages" : `${stubbed} of ${ordered.length} stages`} ran
          as stand-ins: the output is placeholder, not a real reconstruction.
        </p>
      )}
      {ordered.length > 0 && (
        // Live while it runs or needs attention; folded away once it is done.
        <details className="disclosure" open={status !== "complete"}>
          <summary className="disclosure__summary">{summary}</summary>
          <ul className="capture__stages">
            {ordered.map((step) => (
              <StageRow key={step.id} jobId={job.id} step={step} />
            ))}
          </ul>
          <p className="capture__job-version">
            {job.recipe} {job.recipeVersion}
          </p>
        </details>
      )}
      {job.error && (
        <p className="card__error" data-testid="capture-job-error">
          {job.error}
        </p>
      )}
      <div className="capture__file-actions">
        {running && (
          <GlassButton
            size="sm"
            variant="ghost"
            loading={cancel.isPending}
            onClick={() => cancel.mutate(job.id)}
            leadingIcon={<Ban size={13} aria-hidden="true" />}
            data-testid="capture-job-cancel"
          >
            Cancel
          </GlassButton>
        )}
        {canRetry && (
          <GlassButton
            size="sm"
            loading={retry.isPending}
            onClick={() => retry.mutate({ jobId: job.id, fromStage: resumeAt.stageId })}
            leadingIcon={<RotateCcw size={13} aria-hidden="true" />}
            data-testid="capture-job-retry"
          >
            Retry from {stageLabel(resumeAt.stageId)}
          </GlassButton>
        )}
      </div>
    </div>
  );
}

function stageDot(status: RunStatus): string {
  if (status === "complete") return "done";
  if (status === "in-progress") return "run";
  if (status === "error") return "warn";
  return "idle";
}

/** One stage, with its log behind a disclosure — logs are in object storage, not the row. */
function StageRow({ jobId, step }: { jobId: string; step: JobStep }) {
  const [open, setOpen] = useState(false);
  const log = useStepLog(jobId, step.id, open && step.logKey !== null);
  const label = stageLabel(step.stageId);

  return (
    <li className="capture__stage-item" data-testid="capture-stage">
      <div className="capture__stage">
        <span className={`mc-dot mc-dot--${stageDot(step.status)}`} aria-hidden="true" />
        <span className="capture__stage-name">
          {label}
          {step.impl === "stub" && (
            <span className="capture__stub" title="Ran as a stand-in; its output is placeholder">
              stub
            </span>
          )}
          {step.attempt > 1 && (
            <span className="capture__job-version" data-testid="capture-stage-attempt">
              {" "}
              attempt {step.attempt}
            </span>
          )}
        </span>
        <span className="capture__stage-meta">
          {step.status === "complete" && <CheckCircle2 size={12} aria-hidden="true" />}
          {elapsed(step) ?? step.status}
        </span>
        {step.logKey !== null && (
          <GlassButton
            size="sm"
            variant="ghost"
            iconOnly
            aria-expanded={open}
            aria-label={`${open ? "Hide" : "Show"} the ${label} log`}
            onClick={() => setOpen((was) => !was)}
            leadingIcon={<FileText size={12} aria-hidden="true" />}
            data-testid="capture-stage-log-toggle"
          />
        )}
      </div>
      {open && (
        <pre className="capture__stage-log" data-testid="capture-stage-log">
          {log.data?.text ?? (log.isError ? "The log could not be read." : "Loading\u2026")}
        </pre>
      )}
    </li>
  );
}
