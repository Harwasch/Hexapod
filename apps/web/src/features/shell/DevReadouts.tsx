import { KeyRound, X } from "lucide-react";

import { formatAltitude, formatResolution, SCALE_BAND_LABELS } from "@twin/geo";

import { representationLabel } from "@/lib/format";
import { rendererReadout, useScanRendererStatus } from "@/lib/rendererReadout";
import { useLayers } from "@/state/layers";
import { useSettings, useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewer } from "@/state/viewer";

import { useDemoKeyNotice } from "../notices/connection";

/**
 * Camera and renderer telemetry for whoever is building or deploying this: altitude, scale,
 * metres per pixel, which world and which renderer draws the model -- for a splat scan the
 * engine and graphics API, how fast it drew while the camera last moved, and why the WebGPU
 * trial fell back to WebGL2 when it did (lib/rendererReadout.ts) -- and the setup note about
 * the shared demo map key. Settings › Advanced › Show developer readouts; off by default.
 */
export function DevReadouts() {
  const enabled = useSettings((s) => s.devReadouts);
  const camera = useViewer((s) => s.camera);
  const worldLabel = useViewer((s) => s.worldLabel);
  const units = useSettings((s) => s.units);
  const splatRenderer = useSplatRenderer();
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
  const splat = representation === "gaussian-splat";
  const scan = useScanRendererStatus(enabled && splat && splatRenderer !== "cesium");
  const renderer = rendererReadout(splatRenderer, scan);
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
            {splat && (
              <span className="mc-muted" data-testid="status-splat-renderer">
                {" "}
                · {renderer.label}
              </span>
            )}
          </span>
          {splat && renderer.meter && (
            <span className="mc-muted" data-testid="status-splat-meter">
              {renderer.meter}
            </span>
          )}
        </>
      )}
      {/* The WebGPU trial on WebGL2, and why: one line, the reason in full on hover. */}
      {splat && renderer.notice && (
        <span
          className="dev-readouts__note"
          title={renderer.notice}
          data-testid="status-splat-notice"
        >
          <span>{renderer.notice}</span>
        </span>
      )}
      {demoKey.show && (
        <span
          className="dev-readouts__note"
          title="Tiles may load slowly. Set VITE_CESIUM_ION_ACCESS_TOKEN to your own Cesium ion token."
          data-testid="notice-default-token"
        >
          <KeyRound size={12} aria-hidden="true" />
          <span>Shared demo map key</span>
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
