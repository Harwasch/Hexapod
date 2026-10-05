import { AnimatePresence, motion, useReducedMotionConfig } from "motion/react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { DEMO_SITE_SLUG } from "@/api/fallback";
import { useScene } from "@/cesium/SceneContext";
import { useSettings } from "@/state/settings";
import { useViewer } from "@/state/viewer";

const SPRING = { type: "spring", stiffness: 300, damping: 30 } as const;

export function OnboardingCard() {
  const dismissed = useSettings((s) => s.onboardingDismissed);
  const setSettings = useSettings((s) => s.set);
  const status = useViewer((s) => s.status);
  // The operating system's setting, or the app's own switch (App.tsx's MotionConfig).
  const reduceMotion = useReducedMotionConfig();
  const scene = useScene();
  const sites = useSiteCatalog();
  const demo = sites.data?.find((s) => s.slug === DEMO_SITE_SLUG) ?? sites.data?.[0];

  const dismiss = () => setSettings({ onboardingDismissed: true });
  const exploreEarth = () => {
    dismiss();
    scene?.camera.flyHome(-95, 32);
  };
  const viewDemo = () => {
    dismiss();
    if (demo) void scene?.sites.flyTo(demo.id);
  };

  const card =
    !dismissed && status === "ready" ? (
      <motion.div
        // It waits a beat for the globe to come in first; going, it answers the click at once.
        initial={reduceMotion ? false : { opacity: 0, y: 16, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1, transition: { ...SPRING, delay: 0.4 } }}
        exit={{ opacity: 0, y: 12, scale: 0.98, transition: SPRING }}
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
              disabled={!demo}
              data-testid="onboarding-demo"
            >
              Open the demo site
            </GlassButton>
          </div>
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
