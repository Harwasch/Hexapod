import { AnimatePresence, motion, useReducedMotionConfig } from "motion/react";
import { useCallback, useEffect, useRef } from "react";

import type { SiteSummary } from "@twin/contracts";
import { GlassButton, GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { SCAN_SHOWCASE_SLUG } from "@/api/fallback";
import { useScene } from "@/cesium/SceneContext";
import { BUILTIN_RANCH_ID, DEMO_RANCH, DEMO_RANCH_SLUG } from "@/missions/demoRanchData";
import { useSettings } from "@/state/settings";
import { useSites as useSiteState } from "@/state/sites";
import { useViewer } from "@/state/viewer";

import { DescentForm } from "../descent/DescentForm";

const SPRING = { type: "spring", stiffness: 300, damping: 30 } as const;

const isSplat = (site: SiteSummary) => site.representations.includes("gaussian-splat");

/**
 * The scan to show someone who asked for one: the Redmond showcase when its splat is readable
 * here, else any site that has a splat. None at all hides the link.
 */
function scanSite(sites: readonly SiteSummary[]): SiteSummary | undefined {
  return (
    sites.find((site) => site.slug === SCAN_SHOWCASE_SLUG && isSplat(site)) ?? sites.find(isSplat)
  );
}

export function OnboardingCard() {
  const dismissed = useSettings((s) => s.onboardingDismissed);
  const setSettings = useSettings((s) => s.set);
  const status = useViewer((s) => s.status);
  // The operating system's setting, or the app's own switch (App.tsx's MotionConfig).
  const reduceMotion = useReducedMotionConfig();
  const scene = useScene();
  const sites = useSiteCatalog();
  // The demo is the ranch and nothing else: a catalog without it disables the button rather
  // than flying somewhere else under its name.
  const ranch = sites.data?.find((s) => s.slug === DEMO_RANCH_SLUG || s.id === BUILTIN_RANCH_ID);
  const scan = scanSite(sites.data ?? []);

  const root = useRef<HTMLDivElement>(null);

  // The exit animation is finished by animation frames, a frame every few seconds on software
  // WebGL while the globe flies off; until then the card is still in the page. Inert from the
  // click, it no longer takes the clicks meant for the map under it.
  const dismiss = useCallback(() => {
    if (root.current) root.current.inert = true;
    setSettings({ onboardingDismissed: true });
  }, [setSettings]);
  const shown = !dismissed && status === "ready";

  // A site flown to some other way (the switcher, search, a link) is somewhere the person
  // chose to go: the card would only cover it. Not merely a site that became active, which
  // the camera passing near one does by itself. Escape closes it as it closes the app's
  // other panels.
  const flightSiteId = useSiteState((s) => s.flightSiteId);
  useEffect(() => {
    if (shown && flightSiteId !== null) dismiss();
  }, [shown, flightSiteId, dismiss]);
  useEffect(() => {
    if (!shown) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented) dismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shown, dismiss]);
  const exploreEarth = () => {
    dismiss();
    scene?.camera.flyHome(-95, 32);
  };
  const viewDemo = () => {
    dismiss();
    if (ranch) void scene?.sites.flyTo(ranch.id);
  };
  const viewScan = () => {
    dismiss();
    if (scan) void scene?.sites.flyTo(scan.id);
  };

  const card = shown ? (
    <motion.div
      // It waits a beat for the globe to come in first; going, it answers the click at once.
      initial={reduceMotion ? false : { opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1, transition: { ...SPRING, delay: 0.4 } }}
      exit={{ opacity: 0, y: 12, scale: 0.98, transition: SPRING }}
      ref={root}
      className="onboarding"
    >
      <GlassPanel
        strong
        padding="none"
        style={{ padding: "1.25rem 1.25rem 1rem" }}
        role="dialog"
        aria-labelledby="onboarding-title"
        data-testid="onboarding"
      >
        <h2 id="onboarding-title">Explore the living world</h2>
        <p>From the whole planet down to the machines working a field.</p>
        <p className="onboarding__note">
          A real place on {DEMO_RANCH.region}. The fleet, zones and plans are simulated.
        </p>
        <DescentForm
          reducedMotion={Boolean(reduceMotion)}
          onDescend={() => {
            if (root.current) root.current.inert = true;
          }}
        />
        <div className="onboarding__actions">
          <GlassButton
            variant="glass"
            size="lg"
            onClick={exploreEarth}
            data-testid="onboarding-explore"
          >
            Explore Earth
          </GlassButton>
          <GlassButton
            variant="primary"
            size="lg"
            onClick={viewDemo}
            disabled={!ranch}
            title={ranch ? undefined : "The demo ranch is not in this catalog."}
            data-testid="onboarding-demo"
          >
            Open the demo ranch
          </GlassButton>
        </div>
        {scan && (
          <button
            type="button"
            className="mc-link onboarding__scan"
            onClick={viewScan}
            data-testid="onboarding-scan"
          >
            or see a 3D scan
          </button>
        )}
      </GlassPanel>
    </motion.div>
  ) : null;

  // With reduced motion it neither fades in nor out: it is there, and once dismissed it is
  // gone in the same render. An exit, even an opacity-only one, is finished by animation
  // frames, and the click that dismisses it also flies the globe somewhere new, whose first
  // frames compile shaders and read back depth: on software WebGL that is a frame every few
  // seconds, and the dismissed card stayed in the page that long, invisible but still taking
  // the clicks meant for the map under it.
  return reduceMotion ? card : <AnimatePresence>{card}</AnimatePresence>;
}
