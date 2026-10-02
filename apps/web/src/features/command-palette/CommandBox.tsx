import {
  ClipboardList,
  CornerDownLeft,
  Hexagon,
  Layers,
  Loader2,
  MapPin,
  Search,
  Sparkles,
  X,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { GlassButton, GlassPanel, Kbd } from "@twin/ui";

import { useLayers as useLayerCatalog, useSites as useSiteCatalog } from "@/api/queries";
import { HOTKEYS, hotkeyKeys } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";

import { useQuickLayers } from "../mission/quickLayers";
import { useAgentCommand } from "../mission/useAgentCommand";
import { useMissionActions } from "../mission/useMissionActions";
import { flyToPlace, usePlaceSearch } from "../search/places";
import { siteDisplayName } from "../sites/siteNames";
import {
  buildCommandGroups,
  defaultActiveId,
  moveActive,
  type CommandGroupId,
  type CommandRow,
} from "./commandResults";
import { useAppActions } from "./useAppActions";

const GROUP_ICONS: Record<CommandGroupId, LucideIcon> = {
  places: Search,
  sites: MapPin,
  zones: Hexagon,
  plans: ClipboardList,
  layers: Layers,
  actions: Sparkles,
  agent: Sparkles,
};

/**
 * The one text box: search, actions and the agent, top centre (⌘K / Ctrl+K, or `/`).
 *
 * It replaced three inputs — the search pill, the ⌘K palette and the agent bar — and keeps
 * what each of them did: places from the geocoder, catalog sites, the project's zones and
 * plans, layers, every action with its key, and a last row that asks the agent with the words
 * as typed (`useAgentCommand`, the old bar's flow). Which row Enter runs follows the words
 * (`commandResults.ts`): a name runs its best match, a sentence asks the agent.
 *
 * A WAI-ARIA combobox: focus stays in the input, the highlight moves with the arrow keys and
 * is announced through `aria-activedescendant`, Enter runs it and Escape closes the list.
 */
export function CommandBox() {
  const scene = useScene();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  // The row the operator moved to; null follows the default for the words.
  const [chosen, setChosen] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listId = useId();

  const { places, busy: searching, error } = usePlaceSearch(open ? query : "", 4);
  const { ask, busy: asking } = useAgentCommand();
  const actions = useAppActions();
  const quickLayers = useQuickLayers();
  const sites = useSiteCatalog();
  const layerCatalog = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const project = useMission((s) => s.project);
  const draftOpen = useMission((s) => Boolean(s.composer?.draft));
  const { selectZone } = useMissionActions();

  const focus = useCallback((text?: string) => {
    if (text !== undefined) setQuery(text);
    setChosen(null);
    setOpen(true);
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    if (text === undefined) input.select();
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    setChosen(null);
    inputRef.current?.blur();
  }, []);

  // On a phone the field is folded into a search button and the box opens full screen; the
  // input is only focusable once it is shown, so focus follows the open state.
  useEffect(() => {
    if (open && document.activeElement !== inputRef.current) inputRef.current?.focus();
  }, [open]);

  useHotkey(
    HOTKEYS.command.combo,
    (event) => {
      event.preventDefault();
      if (document.activeElement === inputRef.current) close();
      else focus();
    },
    { allowInInputs: true },
  );
  useHotkey(HOTKEYS.search.combo, (event) => {
    event.preventDefault();
    focus();
  });

  // Cards put words here ("twin:bar": a suggested goal, or "" to start one), and the box takes
  // focus with them so the operator can finish the sentence.
  useEffect(() => {
    const onSay = (event: Event) => focus((event as CustomEvent<string | undefined>).detail ?? "");
    window.addEventListener("twin:bar", onSay);
    return () => window.removeEventListener("twin:bar", onSay);
  }, [focus]);

  const groups = useMemo(() => {
    const withIcon = (id: CommandGroupId, rows: CommandRow[]): CommandRow[] =>
      rows.map((row) => ({ ...row, icon: GROUP_ICONS[id] }));
    return buildCommandGroups(
      query,
      {
        places: withIcon(
          "places",
          places.map((place, i) => ({
            id: `place-${i}`,
            label: place.label,
            sub: place.attribution ?? "Place",
            run: () => flyToPlace(scene, place),
          })),
        ),
        sites: withIcon(
          "sites",
          (sites.data ?? []).map((site) => ({
            id: `site-${site.id}`,
            label: siteDisplayName(site),
            sub: "Site · fly to its reality model",
            // Still found by the catalog's own name and slug, which the label no longer shows.
            keywords: `${site.name} ${site.slug.replace(/-/g, " ")}`,
            run: () => void scene?.sites.flyTo(site.id),
          })),
        ),
        zones: withIcon(
          "zones",
          (project?.zones ?? []).map((zone) => ({
            id: `zone-${zone.id}`,
            // Demo zone names lead with their id already ("Z-21 North fence"); drawn areas do not.
            label: zone.name.startsWith(zone.id) ? zone.name : `${zone.id} ${zone.name}`,
            sub: `${zone.task} · ${zone.progressPct}% done`,
            keywords: `zone ${zone.short}`,
            run: () => selectZone(zone.id, { fly: true }),
          })),
        ),
        plans: withIcon(
          "plans",
          (project?.plans ?? []).map((plan) => ({
            id: `plan-${plan.id}`,
            label: plan.title,
            sub: `Plan · ${plan.state}`,
            keywords: `plan ${plan.zoneIds.join(" ")}`,
            run: () => useMission.getState().openPlan(plan.id),
          })),
        ),
        layers: withIcon("layers", [
          ...quickLayers
            .filter((layer) => !layer.disabled)
            .map((layer) => ({
              id: `quick-${layer.id}`,
              label: `${layer.on ? "Hide" : "Show"} ${layer.label.toLowerCase()}`,
              sub: "Quick layer",
              keywords: "layer toggle",
              run: layer.toggle,
            })),
          ...(layerCatalog.data ?? []).map((layer) => {
            const on = Boolean(runtime[layer.id]?.visible);
            return {
              id: `layer-${layer.id}`,
              label: `${on ? "Hide" : "Show"} ${layer.name}`,
              sub: `Layer · ${layer.category}`,
              keywords: `layer toggle ${layer.slug.replace(/-/g, " ")} ${layer.category}`,
              run: () => void scene?.layers.setVisible(layer.id, !on),
            };
          }),
        ]),
        actions: actions.map((action) => ({
          id: `action-${action.id}`,
          label: action.label,
          keywords: action.keywords,
          ...(action.shortcut ? { shortcut: action.shortcut } : {}),
          icon: action.icon,
          run: action.run,
        })),
      },
      (text) => void ask(text),
    );
  }, [
    query,
    places,
    sites.data,
    project,
    quickLayers,
    layerCatalog.data,
    runtime,
    actions,
    scene,
    selectZone,
    ask,
  ]);

  const rows = groups.flatMap((group) => group.rows);
  const activeId =
    chosen && rows.some((row) => row.id === chosen)
      ? chosen
      : defaultActiveId(query, groups, { draftOpen });
  const optionId = (id: string) => `${listId}-${id}`;
  const showList = open && rows.length > 0;

  useEffect(() => {
    if (!showList || !activeId) return;
    document.getElementById(optionId(activeId))?.scrollIntoView({ block: "nearest" });
    // optionId is derived from listId, which is stable for the component's life.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showList, activeId]);

  const run = (row: CommandRow) => {
    // Close first: a row may hand focus straight back (New plan puts the box in "goal" mode).
    close();
    setQuery("");
    row.run();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setOpen(true);
      setChosen(moveActive(rows, activeId, event.key === "ArrowDown" ? 1 : -1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const row = rows.find((r) => r.id === activeId);
      if (row) run(row);
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  };

  const busy = searching || asking;
  const placesHint =
    query.trim().length >= 2 && open
      ? error
        ? `Places unavailable: ${error}`
        : places.length > 0 && scene
          ? scene.geocoder.attribution
          : null
      : null;

  return (
    <>
      <GlassButton
        iconOnly
        variant="glass"
        className="command-trigger"
        aria-label="Search, run an action or ask the agent"
        aria-expanded={open}
        onClick={() => {
          // Shown and focused inside the tap itself, so a phone raises its keyboard.
          flushSync(() => setOpen(true));
          focus();
        }}
        data-testid="command-trigger"
      >
        <Search size={18} aria-hidden="true" />
      </GlassButton>
      <div className={`command ${open ? "is-open" : ""}`} data-testid="command-box">
        <GlassPanel strong pill className="command__field">
          {busy ? (
            <Loader2 size={16} className="glass-muted command__spin" aria-hidden="true" />
          ) : (
            <Search size={16} className="glass-muted" aria-hidden="true" />
          )}
          <input
            ref={inputRef}
            className="command__input"
            type="text"
            placeholder="Search places, sites, actions — or ask the agent"
            aria-label="Search, run an action or ask the agent"
            role="combobox"
            aria-expanded={showList}
            aria-controls={listId}
            aria-haspopup="listbox"
            aria-autocomplete="list"
            aria-activedescendant={showList && activeId ? optionId(activeId) : undefined}
            autoComplete="off"
            spellCheck={false}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setChosen(null);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onKeyDown={onKeyDown}
            data-testid="command-input"
          />
          {query ? (
            <GlassButton
              iconOnly
              size="sm"
              variant="ghost"
              aria-label="Clear"
              className="command__clear"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => focus("")}
            >
              <X size={14} aria-hidden="true" />
            </GlassButton>
          ) : (
            <span className="command__keys glass-subtle" aria-hidden="true">
              {hotkeyKeys(HOTKEYS.command).map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </span>
          )}
          {/* Full screen on a phone, the box needs its own way out; elsewhere a click away does. */}
          <button type="button" className="command__close" onClick={close}>
            Cancel
          </button>
        </GlassPanel>
        <div className="sr-only" role="status" aria-live="polite">
          {showList ? `${rows.length} ${rows.length === 1 ? "result" : "results"}` : ""}
        </div>
        {showList && (
          <GlassPanel
            strong
            className="command__results"
            role="listbox"
            id={listId}
            aria-label="Results"
            data-hud-popover=""
            data-testid="command-results"
            onMouseDown={(event) => event.preventDefault()}
          >
            {groups.map((group) => (
              <div
                key={group.id}
                role="group"
                aria-labelledby={`${listId}-heading-${group.id}`}
                className="command__group"
              >
                <div
                  id={`${listId}-heading-${group.id}`}
                  className="command__heading"
                  aria-hidden="true"
                >
                  {group.heading}
                </div>
                {group.rows.map((row) => {
                  const Icon = row.icon ?? GROUP_ICONS[group.id];
                  const active = row.id === activeId;
                  return (
                    // Options are not focusable: focus stays in the input and the highlight is
                    // announced through aria-activedescendant; the keyboard path is the input's.
                    // eslint-disable-next-line jsx-a11y/click-events-have-key-events
                    <div
                      key={row.id}
                      id={optionId(row.id)}
                      role="option"
                      tabIndex={-1}
                      aria-selected={active}
                      className={`command__row ${group.id === "agent" ? "command__row--agent" : ""}`}
                      onClick={() => run(row)}
                      onMouseMove={() => row.id !== activeId && setChosen(row.id)}
                      data-testid={`command-row-${row.id}`}
                    >
                      <Icon size={15} className="command__row-icon" aria-hidden="true" />
                      <span className="command__row-text">
                        <span className="command__row-label">{row.label}</span>
                        {row.sub && <span className="command__row-sub">{row.sub}</span>}
                      </span>
                      {row.shortcut && (
                        <span
                          className="command__row-keys"
                          aria-label={`Shortcut ${row.shortcut.join(" ")}`}
                        >
                          {row.shortcut.map((key) => (
                            <Kbd key={key}>{key}</Kbd>
                          ))}
                        </span>
                      )}
                      {active && (
                        <CornerDownLeft
                          size={13}
                          className="command__row-enter"
                          aria-hidden="true"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
            {placesHint && <div className="command__hint">{placesHint}</div>}
            <div className="command__foot" aria-hidden="true">
              <span>
                <Kbd>↑</Kbd>
                <Kbd>↓</Kbd> move
              </span>
              <span>
                <Kbd>↵</Kbd> run
              </span>
              <span>
                <Kbd>Esc</Kbd> close
              </span>
            </div>
          </GlassPanel>
        )}
      </div>
    </>
  );
}
