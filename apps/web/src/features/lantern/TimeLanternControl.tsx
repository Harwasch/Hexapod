import { History } from "lucide-react";
import { useEffect, useRef, type KeyboardEvent } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import type { LanternSource } from "@/cesium/lanternTypes";
import { LANTERN_MAX_ALTITUDE_M, useLantern, type LanternKind } from "@/state/lantern";
import { useViewer } from "@/state/viewer";

import "./lantern.css";
import { LANTERN_HINT, UNAVAILABLE } from "./lanternWords";
import { kindOf, useDatedSources } from "./useDatedSources";

const KINDS: { kind: LanternKind; label: string }[] = [
  { kind: "aerial", label: "Aerial" },
  { kind: "maps", label: "Maps" },
  { kind: "satellite", label: "Satellite" },
];

const metres = (value: number) =>
  value >= 10 ? `${Math.round(value)} m` : `${value.toFixed(1).replace(/\.0$/, "")} m`;

/** The rail's small line: what kind of record, and how fine. */
function railMeta(source: LanternSource): string {
  if (source.mapScale) return `Map 1:${source.mapScale.toLocaleString("en-US")}`;
  if (kindOf(source) === "satellite") return `Satellite · ${metres(source.resolutionM)}`;
  return `Aerial · ${metres(source.resolutionM)}`;
}

/**
 * The Time Lantern in the strip: "Look back" opens the lens (cesium/TimeLantern.ts) and a rail
 * of every dated record found here, newest first, to step through with the arrows, [ and ], or
 * the wheel over the map. Kind chips filter it; satellite scenes start off (10 m is coarse at
 * ranch scale). Providers that could not answer are listed, so an empty rail is never read as
 * "nothing exists". Where the lens cannot show, the button says why.
 */
export function TimeLanternControl() {
  const altitude = useViewer((s) => s.camera.altitude);
  const availability = useLantern((s) => s.availability);
  const mode = useLantern((s) => s.mode);
  const index = useLantern((s) => s.index);
  const kinds = useLantern((s) => s.kinds);
  const { sources, providers, status, error } = useDatedSources();
  const rail = useRef<HTMLDivElement>(null);
  const open = mode !== "off";

  // The chosen record in view along the rail, and focused when the rail has the keyboard.
  useEffect(() => {
    const item = rail.current?.querySelector<HTMLElement>(`[data-index="${index}"]`);
    if (!item) return;
    item.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    if (rail.current?.contains(document.activeElement)) item.focus();
  }, [index, sources]);

  if (altitude > LANTERN_MAX_ALTITUDE_M) return null;
  if (!availability.ok && (availability.reason === "orbit" || availability.reason === "altitude"))
    return null;

  if (!availability.ok)
    return (
      <GlassPanel strong pill className="lantern-control" data-testid="lantern-control">
        <GlassButton
          size="sm"
          variant="ghost"
          disabled
          aria-describedby="lantern-unavailable"
          data-testid="lantern-toggle"
        >
          <History size={14} aria-hidden="true" />
          Look back
        </GlassButton>
        <span id="lantern-unavailable" className="lantern-control__reason">
          {UNAVAILABLE[availability.reason]}
        </span>
      </GlassPanel>
    );

  const current = sources[index];
  const step = (delta: number) => useLantern.getState().step(delta, sources.length);
  const onRailKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const older = ["ArrowDown", "ArrowRight", "]"].includes(event.key);
    const newer = ["ArrowUp", "ArrowLeft", "["].includes(event.key);
    if (older || newer) {
      event.preventDefault();
      step(older ? 1 : -1);
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      useLantern.getState().select(event.key === "Home" ? 0 : Math.max(0, sources.length - 1));
    }
  };
  const missing = providers.filter((p) => p.status === "unavailable" || p.status === "empty");

  return (
    <GlassPanel
      strong
      className={open ? "lantern-control lantern-control--open" : "lantern-control"}
      pill={!open}
      role="group"
      aria-label="Time Lantern"
      data-testid="lantern-control"
    >
      <div className="lantern-control__head">
        <GlassButton
          size="sm"
          variant={open ? "primary" : "ghost"}
          aria-pressed={open}
          onClick={() => useLantern.getState().toggle()}
          title="Time Lantern: dated imagery under the cursor (Y)"
          data-testid="lantern-toggle"
        >
          <History size={14} aria-hidden="true" />
          Look back
        </GlassButton>
        {open && (
          <div className="lantern-control__kinds" role="group" aria-label="Kinds of record">
            {KINDS.map(({ kind, label }) => (
              <button
                key={kind}
                type="button"
                className="lantern-chip"
                aria-pressed={kinds[kind]}
                onClick={() => useLantern.getState().toggleKind(kind)}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
      {open && (
        <>
          {status === "loading" && (
            <p className="lantern-control__note" data-testid="lantern-loading">
              Looking for dated imagery here…
            </p>
          )}
          {status === "error" && (
            <p className="lantern-control__note lantern-control__note--error" role="alert">
              The dated imagery list could not be loaded{error ? ` (${error})` : ""}. Close the
              lantern and open it again to retry.
            </p>
          )}
          {status === "ready" && sources.length === 0 && (
            <p className="lantern-control__note" data-testid="lantern-empty">
              No dated imagery found for this area
              {Object.values(kinds).some((on) => !on) ? " with these kinds" : ""}.
            </p>
          )}
          {sources.length > 0 && (
            <div
              ref={rail}
              className="lantern-rail"
              role="radiogroup"
              aria-label="Dated imagery, newest first"
              data-testid="lantern-rail"
            >
              {sources.map((source, i) => (
                <button
                  key={source.id}
                  type="button"
                  role="radio"
                  aria-checked={i === index}
                  aria-label={`${source.dateLabel}, ${source.title}`}
                  tabIndex={i === index ? 0 : -1}
                  data-index={i}
                  className="lantern-rail__item"
                  onClick={() => useLantern.getState().select(i)}
                  onKeyDown={onRailKey}
                >
                  <span className="lantern-rail__year">{source.sortDate.slice(0, 4)}</span>
                  <span className="lantern-rail__meta">{railMeta(source)}</span>
                </button>
              ))}
            </div>
          )}
          <p className="lantern-control__live" aria-live="polite" data-testid="lantern-live">
            {current ? `${current.dateLabel} · ${current.title} · ${current.license}` : ""}
          </p>
          <p className="lantern-control__hint">{LANTERN_HINT}</p>
          {missing.length > 0 && (
            <details className="lantern-control__providers" data-testid="lantern-providers">
              <summary>
                {missing.length === 1
                  ? "1 source has nothing to show here"
                  : `${missing.length} sources have nothing to show here`}
              </summary>
              <ul>
                {missing.map((provider) => (
                  <li key={provider.provider}>{provider.summary}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </GlassPanel>
  );
}
