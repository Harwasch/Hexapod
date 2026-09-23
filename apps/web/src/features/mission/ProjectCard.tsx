import { ChevronDown, ChevronUp, Plus } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { formatArea } from "@twin/geo";
import { GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useMission } from "@/state/mission";
import { representationLabel } from "@/lib/format";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/** Top-left project badge and switcher (design: PROJECT BADGE / SWITCHER). */
export function ProjectCard() {
  const scene = useScene();
  const project = useMission((s) => s.project);
  const open = useMission((s) => s.projectsOpen);
  const setOpen = useMission((s) => s.setProjectsOpen);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const units = useSettings((s) => s.units);
  const sites = useSiteCatalog();
  const title = project?.name ?? "Land Ops";
  const meta = project
    ? `${project.meta}${project.simulated ? " · simulated fleet" : ""}`
    : "Pick a site to start";

  return (
    <div className="mc-project" data-testid="project-card">
      <button
        type="button"
        className="glass mc-project__badge"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={`Project ${title}`}
      >
        <span className="mc-project__mark" aria-hidden="true" />
        <span className="mc-project__text">
          <span className="mc-project__name">{title}</span>
          <span className="mc-project__meta">{meta}</span>
        </span>
        {open ? (
          <ChevronUp size={15} aria-hidden="true" />
        ) : (
          <ChevronDown size={15} aria-hidden="true" />
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 420, damping: 34 }}
            className="mc-project__pop"
          >
            <GlassPanel strong className="mc-project__menu" role="menu" aria-label="Projects">
              <div className="mc-project__menu-head">
                <span className="mc-eyebrow">PROJECTS</span>
                <button
                  type="button"
                  className="mc-link"
                  onClick={() => {
                    setOpen(false);
                    setAddDataOpen(true);
                  }}
                >
                  <Plus size={12} aria-hidden="true" /> New
                </button>
              </div>
              <div className="mc-project__menu-list">
                {(sites.data ?? []).map((site) => (
                  <button
                    key={site.id}
                    type="button"
                    role="menuitem"
                    className="mc-project__row"
                    onClick={() => {
                      setOpen(false);
                      void scene?.sites.flyTo(site.id);
                    }}
                  >
                    <span className="mc-dot mc-dot--teal" aria-hidden="true" />
                    <span className="mc-project__row-text">
                      <span className="mc-project__row-name">{site.name}</span>
                      <span className="mc-project__row-meta">
                        {formatArea(site.areaM2, units)} ·{" "}
                        {site.representations.map(representationLabel).join(", ")}
                      </span>
                    </span>
                    <span className="mc-mono mc-project__row-count">
                      {site.slug === project?.siteId || site.id === project?.siteId
                        ? (project?.machines.length ?? "")
                        : ""}
                    </span>
                  </button>
                ))}
                {sites.data?.length === 0 && (
                  <div className="mc-project__row mc-muted">No sites yet.</div>
                )}
              </div>
              <div className="mc-project__menu-foot">
                <button
                  type="button"
                  className="mc-link"
                  onClick={() => {
                    setOpen(false);
                    scene?.camera.flyHome();
                  }}
                >
                  Back to the whole Earth
                </button>
              </div>
            </GlassPanel>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
