import { Boxes, Eye, EyeOff } from "lucide-react";
import { useId, useRef, useState, type KeyboardEvent } from "react";

import { GlassButton, GlassInput, GlassPopover, GlassSwitch } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { instanceSphere } from "@/cesium/splatInstances";
import { formatSplats, type SearchResult } from "@/lib/instances";
import { useInstances } from "@/state/instances";
import { useSettings } from "@/state/settings";

import { MotionRendererNote } from "./MotionRendererNote";

/** Moves focus between the rows' main buttons with the arrow keys; Escape goes back up. */
function moveFocus(event: KeyboardEvent<HTMLElement>, list: HTMLElement | null): void {
  if (!list) return;
  const buttons = [...list.querySelectorAll<HTMLButtonElement>("[data-result]")];
  if (buttons.length === 0) return;
  const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
  if (event.key === "ArrowDown") {
    event.preventDefault();
    buttons[Math.min(buttons.length - 1, at + 1)]?.focus();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    if (at <= 0) list.parentElement?.querySelector<HTMLInputElement>("input")?.focus();
    else buttons[at - 1]?.focus();
  }
}

/** The search panel for one scan's objects (lib/instances.ts, state/instances.ts). */
export function InstancePanel({ assetId }: { assetId: string }) {
  const scene = useScene();
  const entry = useInstances((s) => s.assets[assetId]);
  const dimOthers = useInstances((s) => s.dimOthers);
  const setQuery = useInstances((s) => s.setQuery);
  const toggleHidden = useInstances((s) => s.toggleHidden);
  const hideMatches = useInstances((s) => s.hideMatches);
  const showOnlyMatches = useInstances((s) => s.showOnlyMatches);
  const gap = useInstances((s) => s.gaps[assetId]);
  const setSettings = useSettings((s) => s.set);
  const showAll = useInstances((s) => s.showAll);
  const highlight = useInstances((s) => s.highlight);
  const setDimOthers = useInstances((s) => s.setDimOthers);
  const inputId = useId();
  const listId = useId();
  const dimId = useId();
  const listRef = useRef<HTMLUListElement>(null);
  if (!entry) return null;

  const select = (result: SearchResult): void => {
    highlight(assetId, [result.id]);
    const sphere = instanceSphere(assetId, result.id);
    if (sphere && scene) scene.camera.flyToBoundingSphere(sphere, { pitch: -35 });
  };

  const onInputKey = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Enter") {
      const first = entry.results[0];
      if (first) select(first);
    } else if (event.key === "ArrowDown") {
      moveFocus(event, listRef.current);
    }
  };

  return (
    <div className="instance-panel" data-testid="instance-panel">
      <label className="instance-panel__label" htmlFor={inputId}>
        Find objects
      </label>
      <GlassInput
        id={inputId}
        type="search"
        autoComplete="off"
        placeholder="tree, table, vegetation > 0.5"
        value={entry.query}
        aria-controls={listId}
        aria-describedby={`${inputId}-hint`}
        onChange={(event) => setQuery(assetId, event.target.value)}
        onKeyDown={onInputKey}
      />
      <span id={`${inputId}-hint`} className="sr-only">
        Words match each object&apos;s tags; filters like vegetation &gt; 0.5 or behaviour:movable
        narrow them. Enter goes to the best match; the arrow keys move through the results.
      </span>
      {entry.filters.length > 0 && (
        <div className="instance-panel__filters" role="group" aria-label="Quick filters">
          {entry.filters.map((filter) => (
            <GlassButton
              key={filter.query}
              size="sm"
              variant="ghost"
              active={entry.query === filter.query}
              aria-pressed={entry.query === filter.query}
              title={`${filter.query} (${String(filter.count)})`}
              onClick={() => setQuery(assetId, entry.query === filter.query ? "" : filter.query)}
            >
              {filter.label}
            </GlassButton>
          ))}
        </div>
      )}
      {gap && (
        <div className="instance-panel__gap" role="note" data-testid="instance-renderer-gap">
          <p>Highlight and hide need the Cesium renderer. {gap.reason}</p>
          <GlassButton
            size="sm"
            variant="ghost"
            onClick={() => setSettings({ splatRenderer: "cesium" })}
          >
            Use the Cesium renderer
          </GlassButton>
        </div>
      )}
      <MotionRendererNote assetId={assetId} />
      {entry.matches.length > 0 && (
        <div className="instance-panel__matches" role="group" aria-label="All matches">
          <span className="instance-panel__count" role="status" data-testid="instance-count">
            {entry.matches.length > entry.results.length
              ? `${String(entry.results.length)} of ${String(entry.matches.length)}`
              : `${String(entry.matches.length)} ${entry.matches.length === 1 ? "match" : "matches"}`}
          </span>
          <GlassButton size="sm" variant="ghost" onClick={() => hideMatches(assetId)}>
            {entry.matches.length === 1
              ? "Hide the match"
              : `Hide all ${String(entry.matches.length)} matches`}
          </GlassButton>
          <GlassButton size="sm" variant="ghost" onClick={() => showOnlyMatches(assetId)}>
            Show only matches
          </GlassButton>
        </div>
      )}
      <ul
        id={listId}
        ref={listRef}
        className="instance-panel__results"
        aria-label="Matching objects"
      >
        {entry.results.map((result) => {
          const hidden = entry.hidden.has(result.id);
          const lit = entry.highlighted.has(result.id);
          return (
            <li key={result.id} className="instance-panel__row" data-hidden={hidden || undefined}>
              <button
                type="button"
                data-result
                className="instance-panel__result"
                aria-current={lit || undefined}
                onClick={() => select(result)}
                onKeyDown={(event) => moveFocus(event, listRef.current)}
              >
                <span className="instance-panel__name">{result.label}</span>
                <span className="instance-panel__meta">
                  #{result.id} · {formatSplats(result.splats)} splats ·{" "}
                  {Math.round(result.score * 100)}% · {result.behaviour}
                </span>
              </button>
              <GlassButton
                iconOnly
                size="sm"
                variant="ghost"
                aria-label={`${hidden ? "Show" : "Hide"} ${result.label} #${String(result.id)}`}
                aria-pressed={hidden}
                onClick={() => toggleHidden(assetId, result.id)}
              >
                {hidden ? (
                  <EyeOff size={14} aria-hidden="true" />
                ) : (
                  <Eye size={14} aria-hidden="true" />
                )}
              </GlassButton>
            </li>
          );
        })}
      </ul>
      {entry.query.trim() !== "" && entry.results.length === 0 && (
        <p className="instance-panel__empty" role="status">
          No object matches.
        </p>
      )}
      <div className="instance-panel__footer">
        <span className="instance-panel__dim">
          <span id={dimId}>Dim the rest</span>
          <GlassSwitch aria-labelledby={dimId} checked={dimOthers} onCheckedChange={setDimOthers} />
        </span>
        {entry.highlighted.size > 0 && (
          <GlassButton size="sm" variant="ghost" onClick={() => highlight(assetId, [])}>
            Clear highlight
          </GlassButton>
        )}
        {entry.hidden.size > 0 && (
          <GlassButton size="sm" variant="ghost" onClick={() => showAll(assetId)}>
            Show all ({entry.hidden.size} hidden)
          </GlassButton>
        )}
      </div>
    </div>
  );
}

/** A compact button beside the representation switcher that opens the object search. */
export function InstanceSearch({ assetId }: { assetId: string }) {
  const count = useInstances((s) => s.assets[assetId]?.instances.length ?? 0);
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <GlassPopover
      open={open}
      onOpenChange={setOpen}
      side="top"
      aria-label="Objects in this scan"
      className="instance-popover"
      trigger={
        <GlassButton
          size="sm"
          variant="ghost"
          data-testid="instance-search"
          leadingIcon={<Boxes size={14} aria-hidden="true" />}
          aria-label={`Objects: search, highlight and hide (${String(count)})`}
        >
          Objects
        </GlassButton>
      }
    >
      <InstancePanel assetId={assetId} />
    </GlassPopover>
  );
}
