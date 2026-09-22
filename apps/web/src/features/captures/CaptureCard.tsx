import { Ban, CheckCircle2, MapPin, Play, RotateCcw } from "lucide-react";

import type { Capture, CaptureFile, Job, RunStatus } from "@twin/contracts";
import { PhoneHandoff } from "./PhoneHandoff";
import { GlassBadge, GlassButton, GlassProgress, type GlassProgressTone } from "@twin/ui";

import { formatBytes, formatDate, formatDuration } from "@/lib/format";
import { uploadsForCapture, type UploadItem } from "@/state/uploads";

import { classify, type Proposal } from "./recipes";

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
  const recipe =
    typeof capture.metadata.recipe === "string" ? capture.metadata.recipe : proposal.recipe;
  const canProcess = filesReady && !uploading && (!job || job.status === "error");

  return (
    <li>
      <div className="card" data-testid={`capture-card-${capture.slug}`}>
        <div className="card__row">
          <h3 className="card__title">{capture.name}</h3>
          <GlassBadge tone={statusTone(capture.status)} data-testid="capture-status">
            {capture.status}
          </GlassBadge>
        </div>
        <div className="card__meta">
          <span>{capture.kind}</span>
          <span>{formatDate(capture.createdAt)}</span>
          <span>{proposal.action}</span>
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

        {job ? <JobStages job={job} /> : null}

        <div className="capture__actions">
          {canProcess && (
            <GlassButton
              size="sm"
              onClick={() => onProcess(recipe)}
              loading={processing}
              leadingIcon={<Play size={13} aria-hidden="true" />}
              data-testid="capture-process"
            >
              {job?.status === "error" ? "Retry" : proposal.action}
            </GlassButton>
          )}
          {capture.siteId && (
            <GlassButton
              size="sm"
              variant="ghost"
              onClick={() => capture.siteId && onFlyTo(capture.siteId)}
              leadingIcon={<MapPin size={13} aria-hidden="true" />}
              data-testid="capture-fly-to"
            >
              Fly to it
            </GlassButton>
          )}
          {/* Only while the capture can still take files: a handoff link to a capture
              that is already processing would mint a credential with nothing to do. */}
          {(capture.status === "awaiting-files" || capture.status === "not-started") && (
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
  return (
    <div className="capture__job" data-testid="capture-job">
      <div className="card__row card__row--between">
        <span className="capture__job-title">
          {job.recipe} <span className="capture__job-version">{job.recipeVersion}</span>
        </span>
        <GlassBadge tone={statusTone(status)} data-testid="capture-job-status">
          {status}
        </GlassBadge>
      </div>
      <GlassProgress
        label={`${job.recipe} progress`}
        value={jobProgressPct(job)}
        valueText={queued ? "Queued" : `${jobProgressPct(job)}%`}
        tone={progressTone(status)}
      />
      {queued && (
        <p className="capture__job-note" data-testid="capture-job-queued">
          Queued. A worker claims the job and runs its stages — nothing has picked it up yet.
        </p>
      )}
      {job.steps.length > 0 && (
        <ul className="capture__stages">
          {[...job.steps]
            .sort((a, b) => a.ordinal - b.ordinal)
            .map((step) => (
              <li key={step.id} className="capture__stage" data-testid="capture-stage">
                <span
                  className={`mc-dot mc-dot--${step.status === "complete" ? "done" : step.status === "in-progress" ? "run" : step.status === "error" ? "warn" : "idle"}`}
                  aria-hidden="true"
                />
                <span className="capture__stage-name">{stageLabel(step.stageId)}</span>
                <span className="capture__stage-meta">
                  {step.status === "complete" && <CheckCircle2 size={12} aria-hidden="true" />}
                  {elapsed(step) ?? step.status}
                </span>
              </li>
            ))}
        </ul>
      )}
      {job.error && (
        <p className="card__error" data-testid="capture-job-error">
          {job.error}
        </p>
      )}
    </div>
  );
}
