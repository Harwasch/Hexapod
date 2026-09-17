import { Sparkles } from "lucide-react";
import { useState } from "react";

import { formatAltitude, formatResolution, SCALE_BAND_LABELS } from "@twin/geo";
import { GlassPanel, GlassTooltip } from "@twin/ui";

import { useLayers as useLayerCatalog, useSites as useSiteCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { runIntent } from "@/lib/intents";
import { describeError } from "@/lib/log";
import { MOD_LABEL } from "@/lib/hotkeys";
import { representationLabel } from "@/lib/format";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

import { startPlanDraft } from "./planDrafting";
import { useMissionActions } from "./useMissionActions";

/** Bottom command bar: "Ask or instruct the agent" plus live camera readouts (design: COMMAND BAR). */
export function CommandBar() {
  const scene = useScene();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const sites = useSiteCatalog();
  const layers = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const appendLog = useMission((s) => s.appendLog);
  const setStreamOpen = useMission((s) => s.setStreamOpen);
  const setPaletteOpen = useUi((s) => s.setPaletteOpen);
  const { selectMachine, selectZone } = useMissionActions();

  const camera = useViewer((s) => s.camera);
  const worldLabel = useViewer((s) => s.worldLabel);
  const units = useSettings((s) => s.units);
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const loading =
    useSites((s) =>
      Object.values(s.assets).some((a) => a.loadState === "loading" || a.progress.pending > 0),
    ) || Object.values(runtime).some((l) => l.loadState === "loading");

  const submit = async () => {
    const input = text.trim();
    if (!input || busy) return;
    setBusy(true);
    appendLog("you", input);
    setStreamOpen(true);
    try {
      const reply = await runIntent(input, {
        flyToPlace: async (query) => {
          if (!scene) return "The world is still starting.";
          const results = await scene.geocoder.search(query);
          const hit = results[0];
          if (!hit) return `I couldn't find “${query}”.`;
          const d = hit.destination;
          if (d.kind === "rectangle") scene.camera.flyToRectangle(d.west, d.south, d.east, d.north);
          else scene.camera.flyTo(d.longitude, d.latitude, 2500, { pitch: -45 });
          return `Flying to ${hit.label}.`;
        },
        flyToSite: (query) => {
          const site = (sites.data ?? []).find(
            (s) =>
              s.name.toLowerCase().includes(query) || s.slug.includes(query.replace(/\s+/g, "-")),
          );
          if (!site) return null;
          void scene?.sites.flyTo(site.id);
          return `Flying to ${site.name}.`;
        },
        toggleLayer: (query, visible) => {
          const alias: Record<string, string> = {
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
          if (/^(zones?)$/.test(query)) {
            const on = visible ?? !useMission.getState().layers.zones;
            if (useMission.getState().layers.zones !== on)
              useMission.getState().toggleLayer("zones");
            scene?.mission.setLayer("zones", on);
            return `Zones ${on ? "on" : "off"}.`;
          }
          if (/^(tracks?)$/.test(query)) {
            const on = visible ?? !useMission.getState().layers.tracks;
            if (useMission.getState().layers.tracks !== on)
              useMission.getState().toggleLayer("tracks");
            scene?.mission.setLayer("tracks", on);
            return `Tracks ${on ? "on" : "off"}.`;
          }
          const slug = alias[query];
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
          if (!state.project)
            return "Load a project with zones and machines first, then I can plan.";
          if (!goal) {
            state.openComposer();
            return "Tell me the goal in the plan window and I'll draft it.";
          }
          state.openComposer({ goal });
          void startPlanDraft(goal);
          return "Drafting the plan — it will appear in the Plans window for your review.";
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
    } finally {
      setBusy(false);
      setText("");
    }
  };

  return (
    <GlassPanel className="mc-bar" role="search" data-testid="command-bar">
      <span className="mc-bar__mark" aria-hidden="true" />
      <form
        className="mc-bar__form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <input
          className="mc-bar__input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask or instruct the agent — “plan: mow Z-14 this week”, “where is TR-07”, “fly to Yosemite”"
          aria-label="Ask or instruct the agent"
          disabled={busy}
          data-testid="command-input"
        />
      </form>
      <div className="mc-bar__readout mc-mono" data-testid="status-bar" aria-live="off">
        <span
          className={`mc-dot ${loading ? "mc-dot--amber mc-dot--pulse" : "mc-dot--teal"}`}
          aria-hidden="true"
        />
        <span>
          Alt{" "}
          <strong data-testid="status-altitude">{formatAltitude(camera.altitude, units)}</strong>
        </span>
        <span className="mc-bar__sep" />
        <span>{SCALE_BAND_LABELS[camera.scaleBand]}</span>
        <span className="mc-muted">{formatResolution(camera.metersPerPixel, units)}</span>
        <span className="mc-bar__sep" />
        <span className="mc-muted">{worldLabel}</span>
        {representation && (
          <>
            <span className="mc-bar__sep" />
            <span>{representationLabel(representation)}</span>
          </>
        )}
      </div>
      <GlassTooltip content="Command palette" shortcut={`${MOD_LABEL} K`} side="top">
        <button
          type="button"
          className="mc-bar__button"
          onClick={() => setPaletteOpen(true)}
          aria-label="Open command palette"
        >
          <Sparkles size={16} aria-hidden="true" />
        </button>
      </GlassTooltip>
    </GlassPanel>
  );
}
