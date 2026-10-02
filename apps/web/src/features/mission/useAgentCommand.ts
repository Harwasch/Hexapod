import { useCallback, useEffect, useRef, useState } from "react";

import {
  useLayers as useLayerCatalog,
  usePlannerStatus,
  useSites as useSiteCatalog,
} from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { runIntent } from "@/lib/intents";
import { describeError } from "@/lib/log";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

import { flyToPlace } from "../search/places";
import { siteDisplayName } from "../sites/siteNames";
import { startPlanDraft } from "./planDrafting";
import { planFromText } from "./planFlow";
import { useMissionActions } from "./useMissionActions";

/** Words for the quick layers, so "show vegetation" finds the land-cover layer. */
const LAYER_ALIASES: Record<string, string> = {
  vegetation: "esa-worldcover-2021",
  "land cover": "esa-worldcover-2021",
  buildings: "cesium-osm-buildings",
  terrain: "cesium-world-terrain",
  imagery: "bing-maps-aerial",
  satellite: "bing-maps-aerial",
  osm: "openstreetmap",
  streets: "openstreetmap",
  water: "usgs-nhd-hydrography",
  hydrography: "usgs-nhd-hydrography",
  rivers: "usgs-nhd-hydrography",
};

/**
 * Asking the agent: what the command box's "Ask the agent" row runs.
 *
 * The words go to `lib/intents.ts`, which maps the phrases it knows (machines, zones, views,
 * layers, measuring, the camera, sites, places) onto the app and hands a sentence of work to
 * the planner (`planFlow.ts`). The exchange is written to the agent log, where the status line
 * reads its latest line and the activity log keeps the thread.
 *
 * It also keeps the store's `plannerConfigured` in step with the API, which tells the agent
 * whether it can look at imagery; the command box is always mounted, so that happens once.
 */
export function useAgentCommand(): { ask: (text: string) => Promise<void>; busy: boolean } {
  const scene = useScene();
  const sites = useSiteCatalog();
  const layers = useLayerCatalog();
  const { selectMachine, selectZone } = useMissionActions();
  const planner = usePlannerStatus();
  const setPlannerConfigured = useMission((s) => s.setPlannerConfigured);
  useEffect(() => {
    setPlannerConfigured(planner.data?.provider === "claude");
  }, [planner.data?.provider, setPlannerConfigured]);
  const [busy, setBusy] = useState(false);
  // Read at call time: whether an answer is on its way, and what was asked meanwhile.
  const busyRef = useRef(false);
  const waiting = useRef<string[]>([]);

  /** One exchange: the words to the app or the planner, the reply to the agent log. */
  const answer = useCallback(
    async (input: string) => {
      const { appendLog } = useMission.getState();
      try {
        const runtime = useLayers.getState().runtime;
        const reply = await runIntent(input, {
          flyToPlace: async (query) => {
            if (!scene) return "The world is still starting.";
            const results = await scene.geocoder.search(query);
            const hit = results[0];
            if (!hit) return `I couldn't find “${query}”.`;
            flyToPlace(scene, hit);
            return `Flying to ${hit.label}.`;
          },
          flyToSite: (query) => {
            // A site answers to its display name ("Blackrock Mesa") and to its catalog record.
            const site = (sites.data ?? []).find(
              (s) =>
                siteDisplayName(s).toLowerCase().includes(query) ||
                s.name.toLowerCase().includes(query) ||
                s.slug.includes(query.replace(/\s+/g, "-")),
            );
            if (!site) return null;
            void scene?.sites.flyTo(site.id);
            return `Flying to ${siteDisplayName(site)}.`;
          },
          toggleLayer: (query, visible) => {
            if (/^(zones?)$/.test(query) || /^(tracks?)$/.test(query)) {
              const key = query.startsWith("zone") ? "zones" : "tracks";
              const on = visible ?? !useMission.getState().layers[key];
              if (useMission.getState().layers[key] !== on) useMission.getState().toggleLayer(key);
              scene?.mission.setLayer(key, on);
              return `${key === "zones" ? "Zones" : "Tracks"} ${on ? "on" : "off"}.`;
            }
            const slug = LAYER_ALIASES[query];
            const layer = (layers.data ?? []).find(
              (l) => l.slug === slug || l.name.toLowerCase().includes(query),
            );
            if (!layer) return null;
            const on = visible ?? !runtime[layer.id]?.visible;
            void scene?.layers.setVisible(layer.id, on);
            return `${on ? "Showing" : "Hiding"} ${layer.name}.`;
          },
          selectMachine: (id) => {
            const machine = useMission.getState().project?.machines.find((m) => m.id === id);
            if (!machine) return null;
            selectMachine(id, { fly: true });
            return `Locating ${machine.name} — ${machine.task}, battery ${machine.batteryPct}%.`;
          },
          selectZone: (id) => {
            const zone = useMission.getState().project?.zones.find((z) => z.id === id);
            if (!zone) return null;
            selectZone(id, { fly: true });
            return `${zone.name}: ${zone.task}, ${zone.progressPct}% done. ${zone.note}`;
          },
          openPlan: (query) => {
            const state = useMission.getState();
            const plan = query
              ? state.project?.plans.find((p) => p.title.toLowerCase().includes(query))
              : undefined;
            state.openPlan(plan?.id ?? null);
            return plan ? `Opening “${plan.title}”.` : "Showing plans.";
          },
          draftPlan: (goal) => {
            const state = useMission.getState();
            if (!state.project) return "The world is still starting; try again in a moment.";
            if (!goal) {
              if (!state.composer) state.openComposer();
              return "Tell me what to do and where, in one sentence.";
            }
            return planFromText(goal, scene);
          },
          refinePlan: (change) => {
            const composer = useMission.getState().composer;
            if (!composer?.draft) return null;
            void startPlanDraft(composer.goal, { zoneIds: composer.zoneIds, refinement: change });
            return "Redrafting with that.";
          },
          setView: (view) => {
            useMission.getState().setView(view);
            return `Showing ${view}.`;
          },
          startMeasure: (mode) => {
            useUi.getState().setMeasureMode(mode);
            return `Measuring ${mode} — click in the world; Esc to stop.`;
          },
          camera: (action) => {
            if (action === "north") scene?.camera.resetNorth();
            if (action === "top-down") scene?.camera.topDown();
            if (action === "home") scene?.camera.flyHome();
            if (action === "explore") useUi.getState().setExploreMode(true);
            return {
              north: "North is up.",
              "top-down": "Looking straight down.",
              home: "Back to Earth.",
              explore: "Explore mode: WASD to move, drag to look, Esc to exit.",
            }[action];
          },
          openSettings: () => {
            useUi.getState().setSettingsOpen(true);
            return "Settings are open.";
          },
        });
        appendLog("agent", reply);
      } catch (error) {
        appendLog("agent", `That didn't work: ${describeError(error)}`);
      }
    },
    [scene, sites.data, layers.data, selectMachine, selectZone],
  );

  /**
   * Asks, or, while an answer is on its way, waits its turn: the words are logged at once (the
   * operator sees they were heard) and answered in order. A second Enter used to be dropped
   * without a word -- after the command box had already closed and cleared it -- and a
   * sentence of work keeps the agent busy until its draft is in, so a refinement typed while
   * the plan was being drafted simply vanished.
   */
  const ask = useCallback(
    async (text: string) => {
      const input = text.trim();
      if (!input) return;
      useMission.getState().appendLog("you", input);
      waiting.current.push(input);
      if (busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      try {
        for (let next = waiting.current.shift(); next !== undefined; next = waiting.current.shift())
          await answer(next);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [answer],
  );

  return { ask, busy };
}
