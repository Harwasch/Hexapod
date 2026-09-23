import { Wind } from "lucide-react";

import { GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useLiving } from "@/state/living";

/**
 * The ambient "Simulated motion" badge.
 *
 * **Why ambient.** Gaussian splats never write depth and the splat primitive early-returns on
 * the pick pass, so a swaying tree cannot be clicked — there is no hover, no selection, no
 * tooltip to hang a disclosure off. If the label is not on screen the whole time the motion is,
 * it does not exist. So this sits in the corner of the HUD from the first displaced frame to
 * the last, next to the project badge, where the mission layer already says "· simulated fleet"
 * about its demo machines. Same product, same voice, same corner.
 *
 * **What it hangs off.** `status.animating`, which the scene derives from wind × attachment —
 * not from the last frame's `displaced` flag. A badge that blinked off between two texture
 * uploads would be worse than no badge, because a viewer would learn it comes and goes and
 * stop reading it.
 *
 * **What it does not claim.** It says the *motion* is modelled. It says nothing about whether
 * the geometry under it was measured, because that varies by site and this badge cannot see the
 * difference — the synthetic tree is a procedural fixture, and a badge that called its geometry
 * "measured" would do exactly the damage this badge exists to prevent. The per-site
 * Observed/Simulated split belongs in the Inspector, which has the catalog metadata to state it
 * honestly. What it can say without qualification, for every site and every wind, is that the
 * data underneath is untouched: the deformer computes each frame from an immutable copy of the
 * canonical positions, so calm restores the loaded geometry byte for byte. It says "the data",
 * not "the geometry", on purpose — a viewer watching a tree bend would read "the geometry never
 * changes" as a contradiction of what is in front of them, and stop trusting the sentence.
 *
 * Meaning is carried by words, never by the amber. `role="status"` also announces it once when
 * the motion starts, which is the non-visual equivalent of noticing a tree begin to sway.
 */
export function SimulatedBadge() {
  const status = useLiving((s) => s.status);
  const catalog = useSiteCatalog();
  if (!status.animating) return null;
  const moving = status.sites.filter((site) => site.phase === "ready");
  if (moving.length === 0) return null;
  const names = moving.map((site) => {
    const match = (catalog.data ?? []).find(
      (entry) => entry.id === site.siteId || entry.slug === site.siteSlug,
    );
    return match?.name ?? site.siteSlug;
  });
  const subject = names.length === 1 ? names[0] : `${names.length} sites`;
  return (
    <GlassPanel
      strong
      compact
      className="living-badge"
      role="status"
      aria-label="Simulated motion"
      data-testid="simulated-badge"
    >
      <Wind className="living-badge__icon" size={16} aria-hidden="true" />
      <div>
        <strong className="living-badge__title">Simulated motion</strong>
        <span className="living-badge__body">
          Modelled wind on {subject}. The data underneath is never altered.
        </span>
      </div>
    </GlassPanel>
  );
}
