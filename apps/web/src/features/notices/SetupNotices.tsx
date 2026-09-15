import { CloudOff, KeyRound } from "lucide-react";

import { GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useViewer } from "@/state/viewer";

/** Small, dismiss-free notices for missing credentials and an unreachable API. */
export function SetupNotices() {
  const tokenState = useViewer((s) => s.tokenState);
  const status = useViewer((s) => s.status);
  const sites = useSiteCatalog();
  return (
    <div className="glass-stack" style={{ gap: "0.5rem" }}>
      {tokenState === "invalid" && (
        <GlassPanel strong compact className="notice" role="status" data-testid="notice-token">
          <KeyRound className="notice__icon" size={18} aria-hidden="true" />
          <div>
            <strong>Cesium ion rejected the access token.</strong> Set{" "}
            <code>VITE_CESIUM_ION_ACCESS_TOKEN</code> in <code>.env</code> to a token with{" "}
            <code>assets:read</code> and <code>geocode</code> scopes, or leave it empty to use the
            CesiumJS evaluation token.
          </div>
        </GlassPanel>
      )}
      {tokenState === "default" && (
        <GlassPanel
          compact
          soft
          className="notice notice--info"
          role="status"
          data-testid="notice-default-token"
        >
          <KeyRound className="notice__icon" size={18} aria-hidden="true" />
          <div className="glass-muted">
            Evaluation ion token · set <code>VITE_CESIUM_ION_ACCESS_TOKEN</code>
          </div>
        </GlassPanel>
      )}
      {sites.builtin && (
        <GlassPanel compact soft className="notice" role="status" data-testid="notice-api-offline">
          <CloudOff className="notice__icon" size={18} aria-hidden="true" />
          <div className="glass-muted">
            Catalog API offline — showing the built-in demo site and a minimal layer set. Start the
            API (<code>pnpm dev:api</code>) to load your catalog.
          </div>
        </GlassPanel>
      )}
      {status === "context-lost" && (
        <GlassPanel strong compact className="notice" role="alert">
          <CloudOff className="notice__icon" size={18} aria-hidden="true" />
          <div>
            The graphics context was lost. The globe will resume if the browser restores it;
            otherwise reload the page.
          </div>
        </GlassPanel>
      )}
    </div>
  );
}
