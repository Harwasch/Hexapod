import { CloudOff, KeyRound, TriangleAlert, X } from "lucide-react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useSettings } from "@/state/settings";
import { useViewer } from "@/state/viewer";

/**
 * Notices for missing credentials and an unreachable API.
 *
 * Only the evaluation-token hint can be dismissed, and it stays dismissed: it tells whoever
 * deployed this build why their tiles are rate-limited, which is a one-time thing to learn,
 * while an operator who reads it can do nothing about it. A rejected token is a real fault
 * and stays until it is fixed.
 *
 * Each notice is one line of plain language. The variable to set is named only where a
 * person can act on it, and then as a detail, not as the headline.
 */
export function SetupNotices() {
  const tokenState = useViewer((s) => s.tokenState);
  const status = useViewer((s) => s.status);
  const tokenNoticeDismissed = useSettings((s) => s.ionTokenNoticeDismissed);
  const setSettings = useSettings((s) => s.set);
  const sites = useSiteCatalog();
  const showDefaultToken = tokenState === "default" && !tokenNoticeDismissed;
  const contextLost = status === "context-lost";
  if (tokenState !== "invalid" && !showDefaultToken && !sites.builtin && !contextLost) return null;
  return (
    <>
      {tokenState === "invalid" && (
        <GlassPanel
          strong
          compact
          className="notice notice--warning"
          role="status"
          data-testid="notice-token"
        >
          <KeyRound className="notice__icon" size={16} aria-hidden="true" />
          <div className="notice__text">
            <strong>Map key rejected</strong>
            <span>
              Cesium ion tiles won’t load. Set <code>VITE_CESIUM_ION_ACCESS_TOKEN</code> to a valid
              token.
            </span>
          </div>
        </GlassPanel>
      )}
      {showDefaultToken && (
        <GlassPanel
          strong
          compact
          className="notice notice--info"
          role="status"
          data-testid="notice-default-token"
        >
          <KeyRound className="notice__icon" size={16} aria-hidden="true" />
          <div className="notice__text">
            <strong>Shared demo map key</strong>
            <span>Tiles may load slowly.</span>
          </div>
          <GlassButton
            iconOnly
            variant="ghost"
            size="sm"
            className="notice__dismiss"
            aria-label="Dismiss the evaluation ion token notice"
            onClick={() => setSettings({ ionTokenNoticeDismissed: true })}
            data-testid="notice-default-token-dismiss"
          >
            <X size={14} aria-hidden="true" />
          </GlassButton>
        </GlassPanel>
      )}
      {sites.builtin && (
        <GlassPanel
          strong
          compact
          className="notice notice--warning"
          role="status"
          data-testid="notice-api-offline"
        >
          <CloudOff className="notice__icon" size={16} aria-hidden="true" />
          <div className="notice__text">
            <strong>Offline</strong>
            <span>Showing the built-in demo. Uploads are unavailable.</span>
          </div>
        </GlassPanel>
      )}
      {contextLost && (
        <GlassPanel strong compact className="notice notice--warning" role="alert">
          <TriangleAlert className="notice__icon" size={16} aria-hidden="true" />
          <div className="notice__text">
            <strong>Graphics paused</strong>
            <span>The browser dropped the 3D view. Reload if it doesn’t come back.</span>
          </div>
        </GlassPanel>
      )}
    </>
  );
}
