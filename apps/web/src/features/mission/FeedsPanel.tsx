import { X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { GlassPanel } from "@twin/ui";

import { useMission } from "@/state/mission";

const STATE_CLASS = {
  live: "mc-dot--teal",
  weak: "mc-dot--amber",
  idle: "mc-dot--neutral",
} as const;

/** Camera feed dashboard (design: CAMERA FEED DASHBOARD). Feeds are placeholders until a robot bridge streams video. */
export function FeedsPanel() {
  const project = useMission((s) => s.project);
  const open = useMission((s) => s.feedsOpen);
  const setOpen = useMission((s) => s.setFeedsOpen);
  const streaming = project?.feeds.filter((f) => f.state !== "idle").length ?? 0;
  return (
    <AnimatePresence>
      {open && project && (
        <motion.div
          key="feeds"
          initial={{ opacity: 0, y: 10, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 380, damping: 32 }}
        >
          <GlassPanel
            strong
            className="mc-feeds"
            role="region"
            aria-label="Live feeds"
            data-testid="feeds-panel"
          >
            <div className="mc-window__head mc-window__head--tight">
              <div>
                <div className="mc-card__title">Cameras</div>
                <div className="mc-card__meta">
                  {streaming} of {project.machines.length} online
                  {project.simulated ? " · simulated fleet, no video" : ""}
                </div>
              </div>
              <button
                type="button"
                className="mc-close"
                onClick={() => setOpen(false)}
                aria-label="Close feeds"
              >
                <X size={13} aria-hidden="true" />
              </button>
            </div>
            <div className="mc-feeds__grid">
              {project.feeds.map((feed) => (
                <div
                  key={feed.id}
                  className="mc-feed"
                  role="img"
                  aria-label={`${feed.id} ${feed.camera} camera, ${feed.state}`}
                >
                  <span className="mc-feed__id">{feed.id}</span>
                  <span className="mc-feed__state">
                    <span className={`mc-dot ${STATE_CLASS[feed.state]}`} aria-hidden="true" />{" "}
                    <span className="mc-mono">{feed.state}</span>
                  </span>
                  <span className="mc-feed__placeholder">No video</span>
                  <span className="mc-feed__time mc-mono">{feed.time}</span>
                  <span className="mc-feed__cam">{feed.camera}</span>
                </div>
              ))}
            </div>
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
