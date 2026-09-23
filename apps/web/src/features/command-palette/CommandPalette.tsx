import { Command } from "cmdk";
import {
  Bookmark,
  Boxes,
  Columns2,
  Compass,
  Layers,
  MapPin,
  MousePointerSquareDashed,
  Plus,
  Ruler,
  Search,
  Settings2,
  Square,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { GlassSheet, Kbd } from "@twin/ui";

import { useLayers as useLayerCatalog, useSites as useSiteCatalog } from "@/api/queries";
import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import type { GeocodeResult } from "@/cesium/types";
import { useHotkey } from "@/lib/hotkeys";
import { useLayers } from "@/state/layers";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/** ⌘K / Ctrl+K palette: every major action, plus places and sites. Later, the natural-language entry point. */
export function CommandPalette() {
  const open = useUi((s) => s.paletteOpen);
  const setOpen = useUi((s) => s.setPaletteOpen);

  useHotkey(
    "mod+k",
    (e) => {
      e.preventDefault();
      setOpen(!open);
    },
    { allowInInputs: true },
  );

  return (
    <GlassSheet
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      hideTitle
      side="center"
      className="palette"
      testId="command-palette"
    >
      <PaletteBody close={() => setOpen(false)} />
    </GlassSheet>
  );
}

/** Mounted only while the sheet is open, so its query state resets naturally on close. */
function PaletteBody({ close }: { close: () => void }) {
  const scene = useScene();
  const ui = useUi();
  const settings = useSettings();
  const sites = useSiteCatalog();
  const layers = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const [query, setQuery] = useState("");
  const [places, setPlaces] = useState<GeocodeResult[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    const trimmed = query.trim();
    const timer = setTimeout(
      () => {
        if (!scene || trimmed.length < 3) {
          setPlaces([]);
          return;
        }
        scene.geocoder
          .search(trimmed, controller.signal)
          .then((results) => !controller.signal.aborted && setPlaces(results.slice(0, 4)))
          .catch(() => !controller.signal.aborted && setPlaces([]));
      },
      trimmed.length < 3 ? 0 : 300,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, scene]);

  const run = useCallback(
    (action: () => void) => {
      close();
      action();
    },
    [close],
  );

  return (
    <Command label="Command palette" shouldFilter loop>
      <Command.Input
        value={query}
        onValueChange={setQuery}
        placeholder="Type a command or place…"
        data-testid="palette-input"
      />
      <Command.List>
        <Command.Empty>No matching commands.</Command.Empty>
        {places.length > 0 && (
          <Command.Group heading="Places">
            {places.map((place, i) => (
              <Command.Item
                key={`place-${i}`}
                value={`place ${place.label}`}
                onSelect={() =>
                  run(() => {
                    const d = place.destination;
                    if (d.kind === "rectangle")
                      scene?.camera.flyToRectangle(d.west, d.south, d.east, d.north);
                    else scene?.camera.flyTo(d.longitude, d.latitude, 2500, { pitch: -45 });
                  })
                }
              >
                <Search size={14} aria-hidden="true" />
                {place.label}
              </Command.Item>
            ))}
          </Command.Group>
        )}
        <Command.Group heading="Sites">
          {(sites.data ?? []).map((site) => (
            <Command.Item
              key={site.id}
              value={`fly to site ${site.name}`}
              onSelect={() => run(() => void scene?.sites.flyTo(site.id))}
            >
              <MapPin size={14} aria-hidden="true" />
              Fly to {site.name}
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Layers">
          {(layers.data ?? []).map((layer) => (
            <Command.Item
              key={layer.id}
              value={`toggle layer ${layer.name}`}
              onSelect={() =>
                run(() => void scene?.layers.setVisible(layer.id, !runtime[layer.id]?.visible))
              }
            >
              <Layers size={14} aria-hidden="true" />
              {runtime[layer.id]?.visible ? "Hide" : "Show"} {layer.name}
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Actions">
          <Command.Item value="add data" onSelect={() => run(() => ui.setAddDataOpen(true))}>
            <Plus size={14} aria-hidden="true" />
            Add data
          </Command.Item>
          <Command.Item
            value="upload a capture"
            onSelect={() => run(() => ui.setPanel("captures"))}
          >
            <Boxes size={14} aria-hidden="true" />
            Upload a capture <Kbd>U</Kbd>
          </Command.Item>
          <Command.Item value="top-down view" onSelect={() => run(() => scene?.camera.topDown())}>
            <Square size={14} aria-hidden="true" />
            Top-down view <Kbd>T</Kbd>
          </Command.Item>
          <Command.Item value="reset north" onSelect={() => run(() => scene?.camera.resetNorth())}>
            <Compass size={14} aria-hidden="true" />
            Reset north <Kbd>N</Kbd>
          </Command.Item>
          <Command.Item
            value="ground explore mode"
            onSelect={() => run(() => ui.setExploreMode(!ui.exploreMode))}
          >
            <MousePointerSquareDashed size={14} aria-hidden="true" />
            {ui.exploreMode ? "Exit" : "Enter"} ground explore mode <Kbd>G</Kbd>
          </Command.Item>
          <Command.Item
            value="start measuring distance"
            onSelect={() => run(() => ui.setMeasureMode("distance"))}
          >
            <Ruler size={14} aria-hidden="true" />
            Start measuring <Kbd>M</Kbd>
          </Command.Item>
          <Command.Item value="compare layers" onSelect={() => run(() => ui.setPanel("compare"))}>
            <Columns2 size={14} aria-hidden="true" />
            Compare layers <Kbd>C</Kbd>
          </Command.Item>
          <Command.Item
            value="bookmarks saved views"
            onSelect={() => run(() => ui.setPanel("bookmarks"))}
          >
            <Bookmark size={14} aria-hidden="true" />
            Saved views <Kbd>B</Kbd>
          </Command.Item>
          <Command.Item value="open settings" onSelect={() => run(() => ui.setSettingsOpen(true))}>
            <Settings2 size={14} aria-hidden="true" />
            Open settings <Kbd>,</Kbd>
          </Command.Item>
          {env.devToolsEnabled && (
            <Command.Item
              value="toggle developer tools"
              onSelect={() => run(() => settings.set({ devToolsOpen: !settings.devToolsOpen }))}
            >
              <Terminal size={14} aria-hidden="true" />
              Toggle developer tools <Kbd>D</Kbd>
            </Command.Item>
          )}
        </Command.Group>
      </Command.List>
    </Command>
  );
}
