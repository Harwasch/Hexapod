import {
  Bookmark,
  ChevronDown,
  ChevronUp,
  Database,
  Earth,
  Images,
  Plus,
  Save,
  Trash2,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useId, useRef, useState } from "react";

import { formatArea } from "@twin/geo";
import { GlassButton, GlassPanel } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { representationLabel } from "@/lib/format";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

import { useSavedViews } from "../bookmarks/savedViews";
import { siteDisplayName, useSiteName } from "../sites/siteNames";

/**
 * Top left: the site you are on, and the switcher behind it.
 *
 * The badge names the site (its project's name when it has one, `siteNames.ts`) and what is
 * happening there. Its menu is the one place to change where you are: every catalog site to
 * fly to, the current site's saved views (open, save the current camera, delete), "Add a
 * site", and the way to the scan gallery and the data console. It replaced the Sites and
 * Bookmarks panels; both are still command-box actions (`s`, `b`).
 */
export function ProjectCard() {
  const scene = useScene();
  const project = useMission((s) => s.project);
  const open = useMission((s) => s.projectsOpen);
  const setOpen = useMission((s) => s.setProjectsOpen);
  const focusOn = useUi((s) => s.switcherFocus);
  // A project is named after its site (`siteNames.ts`), so this is the site's one name.
  const title = project?.name ?? "Land Ops";
  const meta = project
    ? `${project.meta}${project.simulated ? " · simulated fleet" : ""}`
    : "Pick a site to start";
  const root = useRef<HTMLDivElement>(null);
  const badge = useRef<HTMLButtonElement>(null);
  const menuId = useId();

  // A popover: a press anywhere else closes it, and closing hands the keyboard back.
  useEffect(() => {
    if (!open) return;
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    const host = root.current;
    const button = badge.current;
    return () => {
      document.removeEventListener("pointerdown", onDown);
      if (host?.contains(document.activeElement)) button?.focus();
    };
  }, [open, setOpen]);

  return (
    <div className="mc-project" data-testid="project-card" ref={root}>
      <button
        ref={badge}
        type="button"
        className="glass mc-project__badge"
        onClick={() => {
          if (!open) useUi.getState().setSwitcherFocus("sites");
          setOpen(!open);
        }}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={open ? menuId : undefined}
        aria-label={`${title} — switch site`}
      >
        <span className="mc-project__mark" aria-hidden="true" />
        <span className="mc-project__text">
          {/* The name keeps only the room it needs, so a status chip can sit beside it. */}
          <span className="mc-project__title" data-testid="project-title">
            <span className="mc-project__name">{title}</span>
          </span>
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
            data-hud-popover=""
          >
            <GlassPanel
              strong
              className="mc-project__menu"
              role="dialog"
              aria-label="Switch site"
              id={menuId}
              data-testid="site-switcher"
              onKeyDown={(event) => {
                if (event.key !== "Escape") return;
                // Escape closes it from anywhere inside, the "Name this view" field included
                // (`b` opens the switcher with the keyboard there), and hands the keyboard back
                // to the badge. The app's own Escape ignores keys typed into a field, and the
                // switcher has no close button: from that field, Escape did nothing at all.
                event.preventDefault();
                event.stopPropagation();
                setOpen(false);
                badge.current?.focus();
              }}
            >
              <SiteList focus={focusOn === "sites"} onDone={() => setOpen(false)} />
              <ViewList focus={focusOn === "views"} onDone={() => setOpen(false)} />
              <div className="mc-project__menu-foot">
                <a className="mc-link" href="/view.html" data-testid="switcher-scans">
                  <Images size={13} aria-hidden="true" /> Scan gallery
                </a>
                <a className="mc-link" href="/admin.html" data-testid="switcher-console">
                  <Database size={13} aria-hidden="true" /> Data console
                </a>
                <button
                  type="button"
                  className="mc-link"
                  onClick={() => {
                    setOpen(false);
                    scene?.camera.flyHome();
                  }}
                >
                  <Earth size={13} aria-hidden="true" /> Whole Earth
                </button>
              </div>
            </GlassPanel>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SiteList({ focus, onDone }: { focus: boolean; onDone: () => void }) {
  const scene = useScene();
  const sites = useSiteCatalog();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const units = useSettings((s) => s.units);
  const openAdd = useUi((s) => s.openAdd);
  const list = useRef<HTMLUListElement>(null);
  const headingId = useId();

  useEffect(() => {
    if (!focus) return;
    const rows = list.current?.querySelectorAll<HTMLButtonElement>("button");
    const here = list.current?.querySelector<HTMLButtonElement>('[aria-current="true"]');
    (here ?? rows?.[0])?.focus();
  }, [focus]);

  return (
    <section aria-labelledby={headingId}>
      <div className="mc-project__menu-head">
        <span className="mc-eyebrow" id={headingId}>
          Sites
        </span>
        <button
          type="button"
          className="mc-link"
          onClick={() => {
            onDone();
            openAdd("link");
          }}
          data-testid="switcher-add-site"
        >
          <Plus size={12} aria-hidden="true" /> Add a site
        </button>
      </div>
      {sites.builtin && <p className="mc-project__note mc-muted">Built-in demo (API offline)</p>}
      <ul className="mc-project__menu-list" ref={list}>
        {(sites.data ?? []).map((site) => {
          const here = site.id === activeSiteId;
          return (
            <li key={site.id}>
              <button
                type="button"
                className={`mc-project__row ${here ? "is-here" : ""}`}
                aria-current={here ? "true" : undefined}
                onClick={() => {
                  onDone();
                  void scene?.sites.flyTo(site.id);
                }}
                data-testid={`site-row-${site.slug}`}
              >
                <span className={`mc-dot ${here ? "mc-dot--teal" : ""}`} aria-hidden="true" />
                <span className="mc-project__row-text">
                  <span className="mc-project__row-name">{siteDisplayName(site)}</span>
                  <span className="mc-project__row-meta">
                    {formatArea(site.areaM2, units)} ·{" "}
                    {site.representations.map(representationLabel).join(", ")}
                  </span>
                </span>
                {here && <span className="mc-project__row-count">Here</span>}
              </button>
            </li>
          );
        })}
        {sites.data?.length === 0 && <li className="mc-project__empty mc-muted">No sites yet.</li>}
      </ul>
    </section>
  );
}

function ViewList({ focus, onDone }: { focus: boolean; onDone: () => void }) {
  const scene = useScene();
  const saved = useSavedViews();
  const siteName = useSiteName(saved.siteId);
  const [name, setName] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const headingId = useId();
  const hintId = useId();

  useEffect(() => {
    if (focus) input.current?.focus();
  }, [focus]);

  return (
    <section className="mc-project__views" aria-labelledby={headingId}>
      <div className="mc-project__menu-head">
        <span className="mc-eyebrow" id={headingId}>
          Saved views{siteName ? ` · ${siteName}` : ""}
        </span>
      </div>
      {saved.views.length > 0 ? (
        <ul className="mc-project__menu-list" aria-label="Saved views">
          {saved.views.map((view) => (
            <li key={view.id} className="mc-project__view">
              <button
                type="button"
                className="mc-project__row"
                onClick={() => {
                  onDone();
                  scene?.camera.flyToBookmark(view);
                }}
                data-testid={`saved-view-${view.name}`}
              >
                <Bookmark size={13} aria-hidden="true" className="mc-muted" />
                <span className="mc-project__row-text">
                  <span className="mc-project__row-name">
                    {view.name}
                    {view.isDefault && <span className="mc-muted"> · default</span>}
                  </span>
                </span>
              </button>
              <GlassButton
                iconOnly
                size="sm"
                variant="ghost"
                aria-label={`Delete ${view.name}`}
                onClick={() => saved.remove(view)}
              >
                <Trash2 size={13} aria-hidden="true" />
              </GlassButton>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mc-project__note mc-muted">
          No saved views yet. Save the camera to come back to it.
        </p>
      )}
      <form
        className="mc-project__save"
        onSubmit={(event) => {
          event.preventDefault();
          void saved.save(name).then((ok) => ok && setName(""));
        }}
      >
        <input
          ref={input}
          className="mc-input mc-project__save-input"
          placeholder="Name this view"
          aria-label="View name"
          aria-describedby={hintId}
          value={name}
          onChange={(event) => setName(event.target.value)}
          data-testid="view-name"
        />
        <GlassButton
          type="submit"
          size="sm"
          variant="primary"
          loading={saved.saving}
          leadingIcon={<Save size={13} aria-hidden="true" />}
          data-testid="save-view"
        >
          Save current view
        </GlassButton>
      </form>
      <p className="mc-project__note mc-muted" id={hintId} role={saved.error ? "alert" : undefined}>
        {saved.error ??
          (saved.destination === "site"
            ? `Saved to ${siteName ?? "this site"}.`
            : "Saved in this browser.")}
      </p>
    </section>
  );
}
