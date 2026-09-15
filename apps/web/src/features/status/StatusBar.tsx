import { formatAltitude, formatResolution, SCALE_BAND_LABELS } from "@twin/geo";
import { GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { representationLabel } from "@/lib/format";
import { useLayers } from "@/state/layers";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewer } from "@/state/viewer";

/** Bottom-centre context: altitude, scale, world representation and the local site representation. */
export function StatusBar() {
  const camera = useViewer((s) => s.camera);
  const worldLabel = useViewer((s) => s.worldLabel);
  const units = useSettings((s) => s.units);
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const assets = useSites((s) => s.assets);
  const layers = useLayers((s) => s.runtime);
  const sites = useSiteCatalog();
  const site = sites.data?.find((s) => s.id === activeSiteId);
  const loading =
    Object.values(assets).some((a) => a.loadState === "loading" || a.progress.pending > 0) ||
    Object.values(layers).some((l) => l.loadState === "loading");

  return (
    <GlassPanel pill soft className="status" role="status" aria-live="off" data-testid="status-bar">
      <span className="status__item" title="Loading state">
        <span
          className={`status__dot ${loading ? "status__dot--loading" : ""}`}
          aria-hidden="true"
        />
        <span className="sr-only">{loading ? "Loading data" : "Idle"}</span>
      </span>
      <span className="status__sep" aria-hidden="true" />
      <span className="status__item">
        Alt <strong data-testid="status-altitude">{formatAltitude(camera.altitude, units)}</strong>
      </span>
      <span className="status__sep" aria-hidden="true" />
      <span className="status__item">
        <strong>{SCALE_BAND_LABELS[camera.scaleBand]}</strong>
        <span className="glass-subtle">{formatResolution(camera.metersPerPixel, units)}</span>
      </span>
      <span className="status__sep" aria-hidden="true" />
      <span className="status__item" title="Active global representation">
        {worldLabel}
      </span>
      {site && representation && (
        <>
          <span className="status__sep" aria-hidden="true" />
          <span className="status__item" title="Local reality model">
            {site.name} · <strong>{representationLabel(representation)}</strong>
          </span>
        </>
      )}
    </GlassPanel>
  );
}
