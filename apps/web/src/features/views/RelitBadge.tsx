import { JulianDate } from "cesium";
import { Sun } from "lucide-react";
import { useEffect } from "react";

import { GlassPanel } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { holdClock } from "@/cesium/sunClock";
import { pickShownAsset, siteEpochs } from "@/lib/epochs";
import { captureTimeOf, relitSentence, SUNLIGHT_TEXT } from "@/lib/sunlightText";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";
import { useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useSunReport } from "@/state/sunReport";

/**
 * Says the splat scan is relit (ADR 0019 "Sunlight"): by which sun, over the shadows the
 * capture baked in, and when that was; "Sun below horizon" when the frame drawn had the sun
 * down. Loaded by `RelitSlot` (ViewsSlot.tsx) once the sun is on; it says anything only while
 * the relight can be drawn -- As captured off, CesiumJS drawing the active site's splat scan --
 * and only what the scene reports drawing for the shown model (state/sunReport.ts), so it never
 * claims a relight that is not on screen.
 *
 * "At capture" is held here: while it is chosen and the shown date has a capture time, the
 * scene clock stands at that time (cesium/sunClock.ts), and it is given back, exactly, when
 * this goes -- the sun turned off or to Now, As captured on, another renderer, another site
 * -- or moves to another date's time when the date changes.
 */
export function RelitBadge() {
  const sun = useLighting((s) => s.sun);
  const held = useAsCaptured((s) => s.on);
  const cesium = useSplatRenderer() === "cesium";
  const siteId = useSites((s) => s.activeSiteId);
  const splat = useSites((s) => siteId !== null && s.representation[siteId] === "gaussian-splat");
  const drawn = sun !== "off" && !held && cesium && splat;
  const scene = useScene();
  const site = useSite(drawn ? siteId : null).data;
  const shownEpoch = useSites((s) => (siteId === null ? null : (s.epoch[siteId] ?? null)));
  const epochs = site ? siteEpochs(site) : [];
  const asset = site
    ? pickShownAsset(site.assets, epochs, "gaussian-splat", { epochId: shownEpoch })
    : null;
  const captured = captureTimeOf(asset, epochs);
  const holding = drawn && sun === "capture" && captured !== null;
  const report = useSunReport((s) => (asset ? s.reports[asset.id] : undefined));

  useEffect(() => {
    if (!holding || !scene || captured === null) return;
    const release = holdClock(scene.viewer.clock, JulianDate.fromIso8601(captured));
    scene.scene.requestRender();
    return () => {
      release();
      scene.scene.requestRender();
    };
  }, [holding, captured, scene]);

  if (!drawn || !report) return null;
  return (
    <GlassPanel
      strong
      compact
      className="living-badge"
      role="status"
      aria-label={SUNLIGHT_TEXT.badge}
      data-testid="relit-badge"
    >
      <Sun className="notice__icon" size={16} aria-hidden="true" />
      <div className="notice__text">
        <strong>{relitSentence(holding ? "at capture" : "now", captured)}</strong>
        {sun === "capture" && !holding && <span>{SUNLIGHT_TEXT.nowInstead}</span>}
        {!(report.sine > 0) && <span data-testid="relit-night">{SUNLIGHT_TEXT.night}</span>}
      </div>
    </GlassPanel>
  );
}
