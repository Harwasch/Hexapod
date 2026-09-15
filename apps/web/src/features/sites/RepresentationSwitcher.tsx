import { Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import type { Representation } from "@twin/contracts";
import { GlassPanel, GlassSegmentedControl } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { representationLabel } from "@/lib/format";
import { useSites } from "@/state/sites";

const ORDER: Representation[] = ["gaussian-splat", "mesh", "point-cloud"];

/** [Splat] [Mesh] [Points] — appears when the camera is near a loaded site. Switching keeps the camera. */
export function RepresentationSwitcher() {
  const scene = useScene();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const nearSiteId = useSites((s) => s.nearSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const assets = useSites((s) => s.assets);
  const site = useSite(activeSiteId).data;
  const visible = Boolean(site && nearSiteId === activeSiteId);
  const available = ORDER.filter((rep) => site?.assets.some((a) => a.representation === rep));
  const options = ORDER.map((rep) => {
    const asset = site?.assets.find((a) => a.representation === rep);
    const loading = asset ? assets[asset.id]?.loadState === "loading" : false;
    return {
      value: rep,
      label: representationLabel(rep),
      disabled: !asset,
      ariaLabel: asset
        ? `${representationLabel(rep)} representation`
        : `${representationLabel(rep)} not available for this site`,
      icon: loading ? (
        <Loader2
          size={12}
          aria-hidden="true"
          style={{ animation: "glass-spin 0.8s linear infinite" }}
        />
      ) : undefined,
    };
  });
  return (
    <AnimatePresence>
      {visible && site && representation && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ type: "spring", stiffness: 380, damping: 30 }}
        >
          <GlassPanel strong pill className="rep-switch" data-testid="representation-switcher">
            <span className="rep-switch__label" title={site.name}>
              {site.name}
            </span>
            <GlassSegmentedControl
              aria-label="Reality model representation"
              value={representation}
              onValueChange={(rep) => void scene?.sites.setRepresentation(rep)}
              options={options}
            />
            {available.length === 1 && (
              <span className="sr-only">Only one representation is available for this site.</span>
            )}
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
