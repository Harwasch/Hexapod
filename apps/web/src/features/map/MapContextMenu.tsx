import {
  ClipboardList,
  CloudFog,
  Info,
  LandPlot,
  Navigation,
  Ruler,
  type LucideIcon,
} from "lucide-react";
import { AnimatePresence, motion, useIsPresent } from "motion/react";
import { useEffect, useLayoutEffect, useRef, type KeyboardEvent } from "react";

import { footprintContains } from "@twin/geo";
import { GlassPanel } from "@twin/ui";

import { env } from "@/app/env";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { useScene } from "@/cesium/SceneContext";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";
import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";
import { useUi, type MapMenuAt } from "@/state/ui";

import { hasUnsavedOutline, outlineLandAt } from "../land/landEntry";
import { planHere } from "../mission/planFlow";
import { placeMenu } from "./placeMenu";

interface MapAction {
  id: string;
  label: string;
  icon: LucideIcon;
  run: (scene: CesiumSceneManager, at: MapMenuAt) => void;
}

const ACTIONS: readonly MapAction[] = [
  {
    id: "inspect",
    label: "What's here",
    icon: Info,
    run: (scene, at) => void scene.selection.inspectAt(at),
  },
  {
    id: "measure",
    label: "Measure from here",
    icon: Ruler,
    run: (scene, at) => {
      // The point waits in the scene for the tool the store is about to start (SceneBridge).
      scene.measurement.seed(at);
      useUi.getState().setMeasureMode("distance");
    },
  },
  { id: "plan", label: "Plan here", icon: ClipboardList, run: (scene, at) => planHere(at, scene) },
  {
    id: "fly",
    label: "Fly here",
    icon: Navigation,
    run: (scene, at) => void scene.selection.flyTowards(at),
  },
];

/**
 * Into the Land tool at the point, finding mapped parcels and lines there (landEntry.ts). With
 * an unsaved outline open, landEntry only reopens the tool on it, and the item says so.
 */
function outline(unsaved: boolean): MapAction {
  return {
    id: "outline",
    label: unsaved ? "Return to your unsaved outline" : "Outline land here",
    icon: LandPlot,
    run: (scene, at) => void outlineLandAt(at, scene),
  };
}

/**
 * The Fog of Knowing at the point: the fog shows over the selected land and the card opens on
 * the cell there, listing every stored record that covers it (`KnowledgeFogBridge`).
 */
function knowledgeHere(landId: string): MapAction {
  return {
    id: "knowledge",
    label: "What do we know here",
    icon: CloudFog,
    run: (_scene, at) => useKnowledge.getState().selectAt(landId, at.longitude, at.latitude),
  };
}

/**
 * The menu's items. The Land item comes after Plan here, and only where land exploration is
 * on, the workspace is ready and the user's role can add land, as the inspector's "Outline
 * this land" (PlaceFacts): otherwise it would clear the selected land for an outline that
 * could not be saved. "What do we know here" follows What's here, on the selected land only.
 */
function actions(land: {
  offered: boolean;
  unsaved: boolean;
  knowledgeOf: string | null;
}): readonly MapAction[] {
  const items = [...ACTIONS];
  if (land.offered) {
    const plan = items.findIndex((action) => action.id === "plan") + 1;
    items.splice(plan, 0, outline(land.unsaved));
  }
  if (land.knowledgeOf) items.splice(1, 0, knowledgeHere(land.knowledgeOf));
  return items;
}

/**
 * The map's menu at a point. A plain click on the map selects things on it and does nothing on
 * empty ground; asking about a place is a right-click (Ctrl+click on a one-button Mac, a long
 * press on touch; `SelectionManager` raises it), which opens this: What's here (the inspector,
 * the site's card on a scan's surface), Measure from here, Plan here, Outline land here (with
 * land exploration on and a role that can add land) and Fly here.
 *
 * A menu (`role="menu"`): the first item takes the keyboard, the arrows, Home and End move it,
 * and Escape or Tab closes it and gives the keyboard back. A press anywhere else or the camera
 * moving closes it too: it is about a point on the screen, and the map has moved from under it.
 */
export function MapContextMenu() {
  const at = useUi((s) => s.mapMenu);
  return (
    <AnimatePresence>
      {at && <MapMenu key={`${at.x},${at.y},${at.longitude}`} at={at} />}
    </AnimatePresence>
  );
}

function MapMenu({ at }: { at: MapMenuAt }) {
  const scene = useScene();
  const setMapMenu = useUi((s) => s.setMapMenu);
  const landReady = useLandAccessReady();
  const canAddLand = useLandCanEdit();
  const unsaved = useLand(hasUnsavedOutline);
  const offerLand = env.landExplorationEnabled && landReady && canAddLand;
  // The fog is about the selected land: offered at points inside it.
  const knowledgeOf = useLand((s) =>
    env.landExplorationEnabled &&
    landReady &&
    s.active &&
    footprintContains(s.active.boundary, { longitude: at.longitude, latitude: at.latitude })
      ? s.active.id
      : null,
  );
  // Leaving (its exit fade), it takes no more keys or presses.
  const present = useIsPresent();
  const root = useRef<HTMLDivElement>(null);
  const before = useRef<Element | null>(null);

  // Placed before it is painted, on the screen, so it never shows hanging off an edge; the
  // press is in canvas px.
  useLayoutEffect(() => {
    const menu = root.current;
    if (!menu) return;
    const canvas = scene?.viewer.canvas.getBoundingClientRect();
    const { left, top } = placeMenu(
      { x: (canvas?.left ?? 0) + at.x, y: (canvas?.top ?? 0) + at.y },
      { width: menu.offsetWidth, height: menu.offsetHeight },
      { width: window.innerWidth, height: window.innerHeight },
    );
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }, [scene, at]);

  // The keyboard goes to the first item; `close` gives it back.
  useEffect(() => {
    before.current = document.activeElement;
    root.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus({ preventScroll: true });
  }, []);

  // A press anywhere else, or the camera moving, closes it. Only the menu that is present
  // listens: one still playing its exit would otherwise close a menu opened in its place.
  useEffect(() => {
    if (!present) return;
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setMapMenu(null);
    };
    document.addEventListener("pointerdown", onDown, { capture: true });
    const offMotion = scene?.events.on("motion", (moving) => {
      if (moving) setMapMenu(null);
    });
    return () => {
      document.removeEventListener("pointerdown", onDown, { capture: true });
      offMotion?.();
    };
  }, [present, scene, setMapMenu]);

  const close = () => {
    setMapMenu(null);
    const previous = before.current;
    if (previous instanceof HTMLElement && root.current?.contains(document.activeElement))
      previous.focus({ preventScroll: true });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    const items = Array.from(
      root.current?.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)') ?? [],
    );
    const index = items.indexOf(document.activeElement as HTMLElement);
    let next: number;
    if (event.key === "ArrowDown") next = (index + 1) % items.length;
    else if (event.key === "ArrowUp") next = (index - 1 + items.length) % items.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = items.length - 1;
    else if (event.key === "Escape" || event.key === "Tab") {
      // Handled here, so the app's own Escape (`stepBack`) takes no second step.
      event.preventDefault();
      event.stopPropagation();
      close();
      return;
    } else return;
    event.preventDefault();
    items[next]?.focus();
  };

  return (
    <motion.div
      ref={root}
      className="map-menu"
      inert={!present}
      // A whole `transform`, played by the compositor (as the site switcher's).
      initial={{ opacity: 0, transform: "scale(0.96)" }}
      animate={{ opacity: 1, transform: "scale(1)" }}
      exit={{ opacity: 0, transform: "scale(0.96)" }}
      transition={{ type: "spring", stiffness: 520, damping: 36 }}
      data-hud-popover=""
      data-testid="map-menu"
    >
      <GlassPanel
        strong
        compact
        className="map-menu__panel"
        role="menu"
        aria-label="Map"
        onKeyDown={onKeyDown}
      >
        {actions({ offered: offerLand, unsaved, knowledgeOf }).map(
          ({ id, label, icon: Icon, run }) => (
            <button
              key={id}
              type="button"
              role="menuitem"
              tabIndex={-1}
              className="map-menu__item"
              disabled={!scene}
              data-testid={`map-menu-${id}`}
              onClick={() => {
                close();
                if (scene) run(scene, at);
              }}
            >
              <Icon size={15} aria-hidden="true" />
              {label}
            </button>
          ),
        )}
      </GlassPanel>
    </motion.div>
  );
}
