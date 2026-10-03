import { Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import type { Representation } from "@twin/contracts";
import { GlassPanel, GlassSegmentedControl, GlassSwitch } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { representationLabel } from "@/lib/format";
import { describeEvidence } from "@/lib/inferred";
import { useInferred } from "@/state/inferred";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";

import { InstanceSearch } from "./InstanceSearch";

const ORDER: Representation[] = ["gaussian-splat", "mesh", "point-cloud"];

/** Who draws the splat, side by side for comparison (settings `splatRenderer`). */
const RENDERERS: { value: SplatRenderer; label: string; ariaLabel: string }[] = [
  { value: "playcanvas", label: "PlayCanvas", ariaLabel: "Draw splats with PlayCanvas" },
  { value: "spark", label: "Spark", ariaLabel: "Draw splats with Spark" },
  { value: "cesium", label: "Cesium", ariaLabel: "Draw splats with CesiumJS" },
];

/**
 * [Splat] [Mesh] [Points] — appears while the camera is near a loaded site or still frames it
 * (SiteManager `framesSite`). Switching keeps the camera.
 */
export function RepresentationSwitcher() {
  const scene = useScene();
  const splatRenderer = useSettings((s) => s.splatRenderer);
  const setSettings = useSettings((s) => s.set);
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
  // Up while the active site is near or still framed (a pitched view kilometres out).
  const visible = Boolean(
    site && activeSiteId && (nearSiteId === activeSiteId || inViewSiteId === activeSiteId),
  );
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
            {representation === "gaussian-splat" && (
              <GlassSegmentedControl
                aria-label="Splat renderer"
                data-testid="splat-renderer"
                value={splatRenderer}
                onValueChange={(next) => setSettings({ splatRenderer: next })}
                options={RENDERERS}
              />
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
