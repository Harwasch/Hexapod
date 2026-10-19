import { useMediaQuery } from "@/lib/media";
import { useLand } from "@/state/land";
import { useUi } from "@/state/ui";

import { AgentCard } from "./agent/AgentCard";

/** Below this width the right dock folds into the bottom sheet (`styles/app.css`). */
export const COMPACT_MEDIA = "(max-width: 899px)";

/** Saved land, open and being looked at (not drawn or reviewed): when the agent card shows. */
function useAgentLand() {
  const open = useUi((s) => s.activePanel === "land");
  const active = useLand((s) => s.active);
  const draft = useLand((s) => s.draft);
  const mode = useLand((s) => s.mode);
  return open && active && !draft && mode === "browse" ? active : null;
}

/**
 * The agent, in the right dock: ask about the land, watch it work with a line to where it is
 * looking, and see what it found pinned on the map. Runs only when the person asks.
 *
 * On narrow screens the right dock and the left dock share one bottom sheet and only one of
 * them shows, so there the card lives in the left sheet instead (`LandAgentInline`). Exactly
 * one of the two renders at any width.
 */
export function LandAgent() {
  const land = useAgentLand();
  const compact = useMediaQuery(COMPACT_MEDIA);
  if (!land || compact) return null;
  return <AgentCard key={land.id} land={land} />;
}

/** The agent card for the left sheet on narrow screens; renders nothing on wide ones. */
export function LandAgentInline() {
  const land = useAgentLand();
  const compact = useMediaQuery(COMPACT_MEDIA);
  if (!land || !compact) return null;
  return <AgentCard key={land.id} land={land} variant="inline" />;
}
