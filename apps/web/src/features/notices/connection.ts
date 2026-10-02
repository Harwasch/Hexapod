import { useSites as useSiteCatalog } from "@/api/queries";
import { useSettings } from "@/state/settings";
import { useViewer } from "@/state/viewer";

export interface ConnectionIssue {
  id: "api-offline" | "token" | "context-lost";
  /** Kept from the notices these replaced, so tests and tools still find them. */
  testId: string;
  /** The few words on the status line. */
  title: string;
  /** The plain sentence behind them, in the activity log and the tooltip. */
  detail: string;
}

/**
 * What is wrong with the connection, if anything — degraded states only.
 *
 * These were a stack of cards in the right dock; now each is a few words on the status line,
 * with its sentence in the activity log. Healthy states say nothing. The evaluation-token
 * hint is not here: it tells whoever deployed this build why tiles are slow, which an operator
 * can do nothing about, so it lives with the developer readouts (`useDemoKeyNotice`).
 */
export function useConnectionIssues(): ConnectionIssue[] {
  const tokenState = useViewer((s) => s.tokenState);
  const status = useViewer((s) => s.status);
  const sites = useSiteCatalog();
  const issues: ConnectionIssue[] = [];
  if (sites.builtin)
    issues.push({
      id: "api-offline",
      testId: "notice-api-offline",
      title: "Catalog API offline",
      detail: "Showing the built-in demo. Uploads are unavailable.",
    });
  if (tokenState === "invalid")
    issues.push({
      id: "token",
      testId: "notice-token",
      title: "Map key rejected",
      detail: "Cesium ion tiles won’t load. Set VITE_CESIUM_ION_ACCESS_TOKEN to a valid token.",
    });
  if (status === "context-lost")
    issues.push({
      id: "context-lost",
      testId: "notice-context-lost",
      title: "Graphics paused",
      detail: "The browser dropped the 3D view. Reload if it doesn’t come back.",
    });
  return issues;
}

/**
 * The evaluation-token hint, for whoever deployed this build: shown with the developer
 * readouts until dismissed, and dismissed for good (it rides the persisted settings).
 */
export function useDemoKeyNotice(): { show: boolean; dismiss: () => void } {
  const tokenState = useViewer((s) => s.tokenState);
  const dismissed = useSettings((s) => s.ionTokenNoticeDismissed);
  const set = useSettings((s) => s.set);
  return {
    show: tokenState === "default" && !dismissed,
    dismiss: () => set({ ionTokenNoticeDismissed: true }),
  };
}
