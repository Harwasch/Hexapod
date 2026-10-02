import { Loader2, RotateCw, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";

import { GlassPanel } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useSites, type AssetRuntime } from "@/state/sites";

import { fromSiteRecord, outstanding, retrySiteLoad, shownAsset, siteLoad } from "./siteLoad";

interface Progress {
  assetId: string | null;
  peak: number;
  settled: boolean;
}

const START: Progress = { assetId: null, peak: 0, settled: false };

/** Folds one store update into the load's progress; a fresh load starts the count over. */
function advance(progress: Progress, assetId: string, runtime: AssetRuntime | undefined): Progress {
  const base = progress.assetId === assetId ? progress : { ...START, assetId };
  if (!runtime) return base;
  if (runtime.loadState === "loading" || runtime.loadState === "error")
    return base.peak === 0 && !base.settled ? base : { ...START, assetId };
  if (runtime.loadState !== "ready" || base.settled) return base;
  const left = outstanding(runtime);
  if (left === 0) return base.peak > 0 ? { ...base, settled: true } : base;
  return left > base.peak ? { ...base, peak: left } : base;
}

/**
 * "Loading 3D model 42%" beside the Splat / Mesh / Points switch while the site's model comes
 * in, and "Couldn't load the 3D model · Retry" when it fails — the model is what a site visit
 * is for, so its state is said where the operator is looking, not only in a toast.
 */
export function SiteLoadStatus() {
  const scene = useScene();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const temporal = useSites((s) => (activeSiteId ? s.temporalAsset[activeSiteId] : undefined));
  const site = useSite(activeSiteId).data;
  const asset = site ? shownAsset(site.assets, representation, temporal) : null;
  const assetId = asset?.id ?? null;
  const runtime = useSites((s) => (assetId ? s.assets[assetId] : undefined));

  const [progress, setProgress] = useState<Progress>(START);
  useEffect(() => {
    if (!assetId) return;
    const fold = (next: AssetRuntime | undefined) => setProgress((p) => advance(p, assetId, next));
    queueMicrotask(() => fold(useSites.getState().assets[assetId]));
    return useSites.subscribe((state, prev) => {
      if (state.assets[assetId] !== prev.assets[assetId]) fold(state.assets[assetId]);
    });
  }, [assetId]);

  // The site's own record first: during a fly-to it covers the wait for the site's details
  // and the model's creation, before any asset has a runtime to read.
  const record = useSites((s) => (activeSiteId ? s.siteLoads[activeSiteId] : undefined));
  const retrySite = useSites((s) => s.retrySiteLoad);
  const fromSite = fromSiteRecord(record);

  const current = progress.assetId === assetId ? progress : START;
  const load =
    fromSite ?? (activeSiteId && assetId ? siteLoad(runtime, current.peak, current.settled) : null);
  if (!load || !activeSiteId) return null;
  const retry = () => {
    if (fromSite && record?.retryable !== false) retrySite(activeSiteId);
    else if (assetId) void retrySiteLoad(scene, assetId);
  };

  return (
    <GlassPanel
      strong
      pill
      className={`site-load ${load.phase === "error" ? "site-load--error" : ""}`}
      role="status"
      data-testid="site-load"
    >
      {load.phase === "loading" ? (
        <>
          <Loader2 size={14} className="site-load__spin" aria-hidden="true" />
          <span>
            Loading 3D model
            {load.percent === null ? "…" : <span className="mc-mono"> {load.percent}%</span>}
          </span>
        </>
      ) : (
        <>
          <TriangleAlert size={14} className="site-load__icon" aria-hidden="true" />
          <span title={load.message}>Couldn’t load the 3D model</span>
          <span className="site-load__sep" aria-hidden="true">
            ·
          </span>
          <button
            type="button"
            className="site-load__retry"
            onClick={retry}
            data-testid="site-load-retry"
          >
            <RotateCw size={12} aria-hidden="true" />
            Retry
          </button>
        </>
      )}
    </GlassPanel>
  );
}
