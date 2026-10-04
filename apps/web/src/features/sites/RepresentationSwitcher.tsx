import { Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import type { Representation } from "@twin/contracts";
import { GlassPanel, GlassSegmentedControl, GlassSwitch } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { representationLabel } from "@/lib/format";
import { describeEvidence } from "@/lib/inferred";
import { useInferred } from "@/state/inferred";
import { useSites } from "@/state/sites";

import { InstanceSearch } from "./InstanceSearch";
import { siteDisplayName } from "./siteNames";

const ORDER: Representation[] = ["gaussian-splat", "mesh", "point-cloud"];

/**
 * [Splat] [Mesh] [Points] — appears while the camera is near a site that has a model or still
 * frames it (SiteManager `framesSite`), beside the site's one display name (`siteNames.ts`),
 * never the record of the tileset it came from. Switching keeps the camera. Which engine
 * draws a splat is a comparison for developers, so that choice lives in Settings › Advanced,
 * not here.
 */
export function RepresentationSwitcher() {
  const scene = useScene();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const nearSiteId = useSites((s) => s.nearSiteId);
  const inViewSiteId = useSites((s) => s.inViewSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const assets = useSites((s) => s.assets);
  const site = useSite(activeSiteId).data;
  const inferred = useInferred((s) => s.layers);
  const showInferred = useInferred((s) => s.show);
  const setShowInferred = useInferred((s) => s.setShow);
  const shownAsset = site?.assets.find((a) => a.representation === representation);
  const evidence = shownAsset ? (inferred[shownAsset.id] ?? []) : [];
  const available = ORDER.filter((rep) => site?.assets.some((a) => a.representation === rep));
  // Up while the active site is near or still framed (a pitched view kilometres out).
  const visible = Boolean(
    site &&
    activeSiteId &&
    (nearSiteId === activeSiteId || inViewSiteId === activeSiteId) &&
    available.length > 0,
  );
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
            <span className="rep-switch__label" title={siteDisplayName(site)}>
              {siteDisplayName(site)}
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
            {representation === "gaussian-splat" && evidence.length > 0 && (
              <span
                className="rep-switch__inferred"
                data-testid="inferred-toggle"
                title={evidence.map(describeEvidence).join("\n")}
              >
                <span id="inferred-label">Inferred fill</span>
                <GlassSwitch
                  aria-labelledby="inferred-label"
                  checked={showInferred}
                  onCheckedChange={setShowInferred}
                />
              </span>
            )}
            {representation === "gaussian-splat" && shownAsset && (
              <InstanceSearch assetId={shownAsset.id} />
            )}
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
