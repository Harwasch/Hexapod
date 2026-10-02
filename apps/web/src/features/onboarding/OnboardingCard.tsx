import { AnimatePresence, motion } from "motion/react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { DEMO_SITE_SLUG } from "@/api/fallback";
import { useScene } from "@/cesium/SceneContext";
import { useSettings } from "@/state/settings";
import { useViewer } from "@/state/viewer";

export function OnboardingCard() {
  const dismissed = useSettings((s) => s.onboardingDismissed);
  const setSettings = useSettings((s) => s.set);
  const status = useViewer((s) => s.status);
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

  return (
    <AnimatePresence>
      {!dismissed && status === "ready" && (
        <motion.div
          initial={{ opacity: 0, y: 16, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 12, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 300, damping: 30, delay: 0.4 }}
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
      )}
    </AnimatePresence>
  );
}
