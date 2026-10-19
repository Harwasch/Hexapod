import { ExternalLink, History } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import type { LanternSource } from "@/cesium/lanternTypes";
import { useScene } from "@/cesium/SceneContext";
import { LANTERN_MAX_ALTITUDE_M, useLantern, type LanternKind } from "@/state/lantern";
import { useViewer } from "@/state/viewer";

import "./lantern.css";
import { LANTERN_HINT, providersLine, UNAVAILABLE } from "./lanternWords";
import { kindOf, useDatedSources, type LanternAreaName } from "./useDatedSources";

const KINDS: { kind: LanternKind; label: string }[] = [
  { kind: "aerial", label: "Aerial" },
  { kind: "maps", label: "Maps" },
  { kind: "satellite", label: "Satellite" },
];

const metres = (value: number) =>
  value >= 10 ? `${Math.round(value)} m` : `${value.toFixed(1).replace(/\.0$/, "")} m`;

/** The rail's small line: what kind of record, and how fine, when its records say. */
function railMeta(source: LanternSource): string {
  if (source.mapScale) return `Map 1:${source.mapScale.toLocaleString("en-US")}`;
  const kind = kindOf(source) === "satellite" ? "Satellite" : "Aerial";
  return source.resolutionM ? `${kind} · ${metres(source.resolutionM)}` : kind;
}

/** Where the list is for, as "Move over … to look back" says it. */
function areaWords(area: LanternAreaName | null): string {
  if (area?.kind === "land") return area.name;
  if (area?.kind === "chosen") return "the place you chose";
  return "the area around the view's centre";
}

/**
 * What the lens shows that the strip should say beside it: whether the cursor's ground is outside
 * the area listed, and the page of the record under the lens's centre. From `events.lantern`,
 * rendered only when one of them changes, not every frame the lens moves.
 */
function useLensHere(): { outside: boolean; recordUrl: string | null } {
  const scene = useScene();
  const [here, setHere] = useState<{ outside: boolean; recordUrl: string | null }>({
    outside: false,
    recordUrl: null,
  });
  const last = useRef(here);
  useEffect(() => {
    if (!scene) return;
    return scene.events.on("lantern", (frame) => {
      const outside = frame.outside;
      const recordUrl = frame.part?.recordUrl ?? null;
      if (last.current.outside === outside && last.current.recordUrl === recordUrl) return;
      last.current = { outside, recordUrl };
      setHere(last.current);
    });
  }, [scene]);
  return here;
}

/**
 * The Time Lantern in the strip: "Look back" opens the lens (cesium/TimeLantern.ts) and a rail
 * of every dated record found here, newest first, to step through with the arrows, [ and ], or
 * the wheel over the map. Kind chips filter it; satellite scenes start off (10 m is coarse at
 * ranch scale). Under the rail, what the record's date means and every caveat the catalog gives
 * (a datum shift, a trimmed sheet edge, cloud), with a link to the record. Providers that could
 * not be reached are counted apart from those with nothing here, so an empty rail is never read
 * as "nothing exists", and they can be asked again. Where the lens cannot show, the button says
 * why.
 */
export function TimeLanternControl() {
  const tooHigh = useViewer((s) => s.camera.altitude > LANTERN_MAX_ALTITUDE_M);
  const availability = useLantern((s) => s.availability);
  const mode = useLantern((s) => s.mode);
  const index = useLantern((s) => s.index);
  const kinds = useLantern((s) => s.kinds);
  const { sources, providers, status, error, area, incomplete, refreshing, retry, refreshIfStale } =
    useDatedSources();
  const here = useLensHere();
  const rail = useRef<HTMLDivElement>(null);
  const open = mode !== "off";

  // The chosen record in view along the rail, and focused when the rail has the keyboard.
  useEffect(() => {
    const item = rail.current?.querySelector<HTMLElement>(`[data-index="${index}"]`);
    if (!item) return;
    item.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    if (rail.current?.contains(document.activeElement)) item.focus();
  }, [index, sources]);

  // Opening the lantern asks again for a list that has gone stale (an incomplete one after 30 s).
  useEffect(() => {
    if (open) refreshIfStale();
  }, [open, refreshIfStale]);

  if (tooHigh) return null;
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
  // The record under the lens, else the source's only one: never one of many picked for it.
  const recordUrl =
    here.recordUrl ?? (current?.parts.length === 1 ? (current.parts[0]?.recordUrl ?? null) : null);
  const tryAgain = (
    <GlassButton
      size="sm"
      variant="glass"
      onClick={retry}
      disabled={refreshing}
      data-testid="lantern-retry"
    >
      {refreshing ? "Asking again…" : "Try again"}
    </GlassButton>
  );

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
            <div className="lantern-control__row" role="alert">
              <p className="lantern-control__note lantern-control__note--error">
                The dated imagery list could not be loaded{error ? ` (${error})` : ""}.
              </p>
              {tryAgain}
            </div>
          )}
          {status === "ready" && sources.length === 0 && (
            <div className="lantern-control__row" data-testid="lantern-empty">
              <p className="lantern-control__note">
                {incomplete
                  ? "Some sources could not be reached, so this list may be incomplete."
                  : `No dated imagery found for this area${
                      Object.values(kinds).some((on) => !on) ? " with these kinds" : ""
                    }.`}
              </p>
              {incomplete && tryAgain}
            </div>
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
          {current && here.outside && (
            <p className="lantern-control__note" data-testid="lantern-outside">
              Dated imagery is listed for {areaWords(area)}. Move over it to look back.
            </p>
          )}
          {current && (
            <section
              className="lantern-about"
              aria-label="About this record"
              data-testid="lantern-about"
            >
              <ul>
                <li>{current.dateMeaning}</li>
                {current.warnings.map((warning) => (
                  <li key={warning} className="lantern-about__warning">
                    {warning}
                  </li>
                ))}
              </ul>
              {recordUrl && (
                <a
                  className="lantern-about__link"
                  href={recordUrl}
                  target="_blank"
                  rel="noreferrer"
                  data-testid="lantern-record"
                >
                  Open the record
                  <ExternalLink size={12} aria-hidden="true" />
                </a>
              )}
            </section>
          )}
          <p className="lantern-control__hint">{LANTERN_HINT}</p>
          {missing.length > 0 && (
            <details className="lantern-control__providers" data-testid="lantern-providers">
              <summary>{providersLine(providers)}</summary>
              <ul>
                {missing.map((provider) => (
                  <li key={provider.provider}>{provider.summary}</li>
                ))}
              </ul>
              {incomplete && sources.length > 0 && tryAgain}
            </details>
          )}
        </>
      )}
    </GlassPanel>
  );
}
