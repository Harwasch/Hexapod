import { KeyRound, X } from "lucide-react";

import { formatAltitude, formatResolution, SCALE_BAND_LABELS } from "@twin/geo";

import { representationLabel } from "@/lib/format";
import { useLayers } from "@/state/layers";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewer } from "@/state/viewer";

import { useDemoKeyNotice } from "../notices/connection";

const RENDERER_NAMES: Record<SplatRenderer, string> = {
  playcanvas: "PlayCanvas",
  spark: "Spark",
  cesium: "CesiumJS",
};

/**
 * Camera and renderer telemetry for whoever is building or deploying this: altitude, scale,
 * metres per pixel, which world and which renderer draws the model, and the setup note about
 * the shared demo map key. Settings › Advanced › Show developer readouts; off by default.
 */
export function DevReadouts() {
  const enabled = useSettings((s) => s.devReadouts);
  const camera = useViewer((s) => s.camera);
  const worldLabel = useViewer((s) => s.worldLabel);
  const units = useSettings((s) => s.units);
  const splatRenderer = useSettings((s) => s.splatRenderer);
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const assetsLoading = useSites((s) =>
    Object.values(s.assets).some((a) => a.loadState === "loading" || a.progress.pending > 0),
  );
  const layersLoading = useLayers((s) =>
    Object.values(s.runtime).some((l) => l.loadState === "loading"),
  );
  const loading = assetsLoading || layersLoading;
  const demoKey = useDemoKeyNotice();
  if (!enabled) return null;
  return (
    <div className="glass glass--strong dev-readouts mc-mono" data-testid="status-bar">
      <span
        className={`mc-dot ${loading ? "mc-dot--amber mc-dot--pulse" : "mc-dot--teal"}`}
        aria-hidden="true"
      />
      <span>
        Alt <strong data-testid="status-altitude">{formatAltitude(camera.altitude, units)}</strong>
      </span>
      <span className="dev-readouts__sep" />
      <span>{SCALE_BAND_LABELS[camera.scaleBand]}</span>
      <span className="mc-muted">{formatResolution(camera.metersPerPixel, units)}</span>
      {/* The default world goes without saying; only a different one is news. */}
      {worldLabel !== "Open world" && (
        <>
          <span className="dev-readouts__sep" />
          <span className="mc-muted">{worldLabel}</span>
        </>
      )}
      {representation && (
        <>
          <span className="dev-readouts__sep" />
          <span>
            {representationLabel(representation)}
            {representation === "gaussian-splat" && (
              <span className="mc-muted"> · {RENDERER_NAMES[splatRenderer]}</span>
            )}
          </span>
        </>
      )}
      {demoKey.show && (
        <span className="dev-readouts__note" data-testid="notice-default-token">
          <KeyRound size={12} aria-hidden="true" />
          <span>Shared demo map key: tiles may load slowly</span>
          <button
            type="button"
            className="dev-readouts__dismiss"
            aria-label="Dismiss the evaluation ion token notice"
            onClick={demoKey.dismiss}
            data-testid="notice-default-token-dismiss"
          >
            <X size={12} aria-hidden="true" />
          </button>
        </span>
      )}
    </div>
  );
}
