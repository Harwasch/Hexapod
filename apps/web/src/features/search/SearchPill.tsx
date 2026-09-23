import { Loader2, Search, X } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";

import { GlassButton, GlassPanel, Kbd } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import type { GeocodeResult } from "@/cesium/types";
import { MOD_LABEL, useHotkey } from "@/lib/hotkeys";
import { describeError } from "@/lib/log";

interface Result {
  id: string;
  label: string;
  sub: string;
  run: () => void;
}

/** Top-centre floating search: places (geocoder) and catalog sites. */
export function SearchPill() {
  const scene = useScene();
  const sites = useSiteCatalog();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listId = useId();

  useHotkey("/", (e) => {
    e.preventDefault();
    inputRef.current?.focus();
  });

  const flyToGeocode = useCallback(
    (result: GeocodeResult) => {
      if (!scene) return;
      const d = result.destination;
      if (d.kind === "rectangle") scene.camera.flyToRectangle(d.west, d.south, d.east, d.north);
      else scene.camera.flyTo(d.longitude, d.latitude, 2500, { pitch: -45 });
    },
    [scene],
  );

  useEffect(() => {
    const trimmed = query.trim();
    const controller = new AbortController();
    const timer = setTimeout(
      async () => {
        if (trimmed.length < 2) {
          setResults([]);
          setError(null);
          return;
        }
        setBusy(true);
        setError(null);
        const siteMatches: Result[] = (sites.data ?? [])
          .filter((s) => s.name.toLowerCase().includes(trimmed.toLowerCase()))
          .slice(0, 3)
          .map((s) => ({
            id: `site-${s.id}`,
            label: s.name,
            sub: "Site · fly to reality model",
            run: () => void scene?.sites.flyTo(s.id),
          }));
        try {
          const geocoded = scene ? await scene.geocoder.search(trimmed, controller.signal) : [];
          if (controller.signal.aborted) return;
          const places: Result[] = geocoded.slice(0, 6).map((g, i) => ({
            id: `place-${i}`,
            label: g.label,
            sub: g.attribution ?? "Place",
            run: () => flyToGeocode(g),
          }));
          setResults([...siteMatches, ...places]);
          setActive(0);
        } catch (err) {
          if (controller.signal.aborted) return;
          setResults(siteMatches);
          setError(describeError(err));
        } finally {
          if (!controller.signal.aborted) setBusy(false);
        }
      },
      trimmed.length < 2 ? 0 : 280,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, scene, sites.data, flyToGeocode]);

  const choose = (result: Result) => {
    result.run();
    setOpen(false);
    setQuery(result.label);
    inputRef.current?.blur();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((a) => Math.min(results.length - 1, a + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (event.key === "Enter") {
      const result = results[active];
      if (result) choose(result);
    } else if (event.key === "Escape") {
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  const showList = open && query.trim().length >= 2;

  return (
    <div className="search" role="search">
      <GlassPanel strong pill className="search__field">
        {busy ? (
          <Loader2
            size={16}
            className="glass-muted"
            aria-hidden="true"
            style={{ animation: "glass-spin 0.8s linear infinite" }}
          />
        ) : (
          <Search size={16} className="glass-muted" aria-hidden="true" />
        )}
        <input
          ref={inputRef}
          className="search__input"
          type="search"
          placeholder="Search anywhere…"
          aria-label="Search places and sites"
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-activedescendant={
            showList && results[active] ? `${listId}-${results[active].id}` : undefined
          }
          aria-autocomplete="list"
          autoComplete="off"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={onKeyDown}
          data-testid="search-input"
        />
        {query ? (
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label="Clear search"
            onClick={() => setQuery("")}
          >
            <X size={14} aria-hidden="true" />
          </GlassButton>
        ) : (
          <span
            className="glass-subtle"
            style={{ display: "inline-flex", gap: 2, paddingRight: 4 }}
            aria-hidden="true"
          >
            <Kbd>{MOD_LABEL}</Kbd>
            <Kbd>K</Kbd>
          </span>
        )}
      </GlassPanel>
      {showList && (
        <GlassPanel
          strong
          className="search__results"
          role="listbox"
          id={listId}
          aria-label="Search results"
        >
          {results.map((result, index) => (
            <button
              key={result.id}
              id={`${listId}-${result.id}`}
              type="button"
              role="option"
              aria-selected={index === active}
              className="search__result"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => choose(result)}
              onMouseEnter={() => setActive(index)}
            >
              <span>{result.label}</span>
              <span className="search__result-sub">{result.sub}</span>
            </button>
          ))}
          {results.length === 0 && !busy && (
            <div className="search__hint">
              {error ? `Search unavailable: ${error}` : "No matches"}
            </div>
          )}
          {results.length > 0 && scene && (
            <div className="search__hint">{scene.geocoder.attribution}</div>
          )}
        </GlassPanel>
      )}
    </div>
  );
}
