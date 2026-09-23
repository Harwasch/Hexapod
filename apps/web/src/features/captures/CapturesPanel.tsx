import { Boxes } from "lucide-react";

import { EmptyState, GlassBadge, GlassButton, Spinner } from "@twin/ui";

import { latestJobByCapture, useCaptures, useJobs, useProcessCapture } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useUploads } from "@/state/uploads";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { CaptureCard } from "./CaptureCard";
import { DropZone } from "./DropZone";
import { WriteTokenField } from "./WriteTokenField";
import { useCaptureUploads } from "./useCaptureUploads";

/**
 * Captures: drop files, watch them upload, watch the pipeline run.
 *
 * A panel rather than a sheet, because an upload runs for minutes and the camera has to
 * stay usable the whole time.
 */
export function CapturesPanel() {
  const open = useUi((s) => s.activePanel === "captures");
  const setPanel = useUi((s) => s.setPanel);
  const tokenPrompt = useUi((s) => s.writeTokenPrompt);
  const scene = useScene();
  const captures = useCaptures(open);
  const jobs = useJobs(open);
  const items = useUploads((s) => s.items);
  const clearSettled = useUploads((s) => s.clearSettled);
  const uploads = useCaptureUploads();
  const process = useProcessCapture();

  // `CatalogResult.data` is undefined until the first response lands, exactly as it is
  // for sites; the panel renders its empty state rather than throwing.
  const byCapture = latestJobByCapture(jobs.data ?? []);
  const list = captures.data ?? [];

  return (
    <FloatingPanel
      open={open}
      title="Captures"
      onClose={() => setPanel(null)}
      testId="captures-panel"
      actions={
        <GlassButton size="sm" variant="ghost" onClick={clearSettled}>
          Clear
        </GlassButton>
      }
    >
      <div className="glass-stack">
        {captures.builtin && (
          <GlassBadge tone="warning" data-testid="captures-offline">
            API offline — uploads are unavailable
          </GlassBadge>
        )}
        {tokenPrompt && <WriteTokenField onSaved={() => void uploads.retryLastDrop()} />}
        <DropZone
          onFiles={(files) => void uploads.start(files)}
          disabled={captures.builtin}
          busy={uploads.busy}
        />
        {uploads.error && (
          <p className="card__error" data-testid="capture-error">
            {uploads.error}
          </p>
        )}
        {uploads.error && uploads.canRetryDrop && !tokenPrompt && (
          <GlassButton size="sm" onClick={() => void uploads.retryLastDrop()}>
            Try again
          </GlassButton>
        )}
        {captures.isLoading && (
          <div className="glass-row" style={{ justifyContent: "center", padding: "1rem" }}>
            <Spinner label="Loading captures" />
          </div>
        )}
        {!captures.isLoading && list.length === 0 && (
          <EmptyState
            icon={<Boxes size={28} />}
            title="No captures yet"
            body="Drop a video, a folder of photos, or a splat file. It uploads straight to storage and the pipeline takes it from there."
          />
        )}
        <ul className="glass-list">
          {list.map((capture) => (
            <CaptureCard
              key={capture.id}
              capture={capture}
              job={byCapture[capture.id]}
              uploads={items}
              processing={process.isPending && process.variables?.captureId === capture.id}
              onProcess={(recipe) => process.mutate({ captureId: capture.id, recipe })}
              onRetryUpload={(id) => void uploads.retry(id)}
              onCancelUpload={uploads.cancel}
              onFlyTo={(siteId) => void scene?.sites.flyTo(siteId)}
            />
          ))}
        </ul>
      </div>
    </FloatingPanel>
  );
}
