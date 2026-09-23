import { Boxes, RotateCcw, Smartphone } from "lucide-react";
import { useState } from "react";

import { EmptyState, GlassBadge, GlassButton, Spinner } from "@twin/ui";

import { latestJobByCapture, useCaptures, useJobs, useProcessCapture } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useUploads } from "@/state/uploads";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { CaptureCard } from "./CaptureCard";
import { DropZone } from "./DropZone";
import { PhoneHandoff } from "./PhoneHandoff";
import { WriteTokenField } from "./WriteTokenField";
import { useCaptureUploads } from "./useCaptureUploads";

/**
 * Captures: add one (drop files here, or send them from a phone), then watch it upload and
 * process.
 *
 * The panel reads top to bottom as the steps do: a way in, then the list. The phone is the
 * second way in, not an extra: its QR code opens in the same place as the button, so there
 * is never more than one "add" affordance on screen.
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
  // The capture "New capture from phone" made, whose QR code is showing at the top.
  const [phoneCapture, setPhoneCapture] = useState<string | null>(null);

  // `CatalogResult.data` is undefined until the first response lands, exactly as it is
  // for sites; the panel renders its empty state rather than throwing.
  const byCapture = latestJobByCapture(jobs.data ?? []);
  const list = captures.data ?? [];
  const settled = Object.values(items).some(
    (item) => item.phase === "complete" || item.phase === "cancelled",
  );

  return (
    <FloatingPanel
      open={open}
      title="Captures"
      onClose={() => setPanel(null)}
      testId="captures-panel"
      actions={
        settled && (
          <GlassButton size="sm" variant="ghost" onClick={clearSettled}>
            Clear finished
          </GlassButton>
        )
      }
    >
      <div className="glass-stack">
        {captures.builtin && (
          <GlassBadge tone="warning" data-testid="captures-offline">
            Offline — uploads unavailable
          </GlassBadge>
        )}
        {tokenPrompt && <WriteTokenField onSaved={() => void uploads.retryLastDrop()} />}

        <section className="capture-add" aria-label="Add a capture">
          <DropZone
            onFiles={(files) => void uploads.start(files)}
            disabled={captures.builtin}
            busy={uploads.busy}
          />
          {phoneCapture ? (
            <PhoneHandoff
              key={phoneCapture}
              captureId={phoneCapture}
              autoOpen
              onClose={() => setPhoneCapture(null)}
            />
          ) : (
            <GlassButton
              block
              disabled={captures.builtin || uploads.busy}
              leadingIcon={<Smartphone size={15} aria-hidden="true" />}
              data-testid="capture-from-phone"
              onClick={() => {
                void uploads.startFromPhone().then((id) => {
                  if (id) setPhoneCapture(id);
                });
              }}
            >
              New capture from phone
            </GlassButton>
          )}
          {uploads.error && (
            <div className="capture-add__error">
              <p className="card__error" data-testid="capture-error">
                {uploads.error}
              </p>
              {uploads.canRetryDrop && !tokenPrompt && (
                <GlassButton
                  size="sm"
                  leadingIcon={<RotateCcw size={13} aria-hidden="true" />}
                  onClick={() => void uploads.retryLastDrop()}
                >
                  Try again
                </GlassButton>
              )}
            </div>
          )}
        </section>

        {captures.isLoading && (
          <div className="glass-row" style={{ justifyContent: "center", padding: "1rem" }}>
            <Spinner label="Loading captures" />
          </div>
        )}
        {!captures.isLoading && list.length === 0 && !captures.builtin && (
          <EmptyState
            icon={<Boxes size={24} />}
            title="No captures yet"
            body="Your uploads and their progress appear here."
          />
        )}
        {list.length > 0 && (
          <section className="capture-list" aria-label="Captures">
            <h3 className="panel__eyebrow">Recent</h3>
            <ul className="glass-list">
              {list.map((capture) => (
                <CaptureCard
                  key={capture.id}
                  capture={capture}
                  job={byCapture[capture.id]}
                  uploads={items}
                  processing={process.isPending && process.variables?.captureId === capture.id}
                  handoffShown={capture.id === phoneCapture}
                  onProcess={(recipe) => process.mutate({ captureId: capture.id, recipe })}
                  onRetryUpload={(id) => void uploads.retry(id)}
                  onCancelUpload={uploads.cancel}
                  onFlyTo={(siteId) => void scene?.sites.flyTo(siteId)}
                />
              ))}
            </ul>
          </section>
        )}
      </div>
    </FloatingPanel>
  );
}
