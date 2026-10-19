import { useLand } from "@/state/land";
import { useUi } from "@/state/ui";

import { LandDrawBar } from "./LandDrawBar";
import { LensDock } from "./LensDock";

/**
 * Land's controls along the bottom of the map: the draw bar while outlining or adjusting, the
 * lens dock while looking at saved land.
 */
export function LandStrip() {
  const open = useUi((s) => s.activePanel === "land");
  const mode = useLand((s) => s.mode);
  const active = useLand((s) => s.active);
  const draft = useLand((s) => s.draft);
  if (!open) return null;
  if (mode !== "browse" && mode !== "candidates") return <LandDrawBar />;
  if (active && !draft && mode === "browse") return <LensDock />;
  return null;
}
