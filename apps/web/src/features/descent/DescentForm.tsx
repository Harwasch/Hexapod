import { MapPin } from "lucide-react";
import { useEffect, useId, useRef, useState, type SubmitEvent } from "react";

import { GlassButton, GlassField, GlassInput } from "@twin/ui";

import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import type { GeocodeResult } from "@/cesium/types";
import { describeError } from "@/lib/log";

import { descend, looksLikeCoordinates, parseCoordinates, targetFromPlace } from "./descent";
import "./descent.css";

/** Places offered when a search finds more than one. */
const MAX_CHOICES = 5;

type Answer =
  | { kind: "idle" }
  | { kind: "busy" }
  | { kind: "choices"; query: string; places: GeocodeResult[] }
  | { kind: "error"; message: string };

/**
 * "Where is your land?": an address, a place or coordinates, then one flight down to it (the
 * Descent, ./descent.ts). The geocoder is asked once, when the form is sent, never as the person
 * types: Nominatim, the fallback, forbids autocomplete. One place flies there at once; several
 * are offered as choices; none says so in place.
 */
export function DescentForm({
  reducedMotion,
  onDescend,
}: {
  reducedMotion: boolean;
  /** Called as the flight leaves (the onboarding card makes itself inert). */
  onDescend?: () => void;
}) {
  const scene = useScene();
  const id = useId();
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<Answer>({ kind: "idle" });
  const search = useRef<AbortController | null>(null);
  useEffect(() => () => search.current?.abort(), []);

  const go = (place: GeocodeResult) => {
    if (!scene) return;
    onDescend?.();
    descend(scene, targetFromPlace(place, scene.geocoder.attribution), { reducedMotion });
  };

  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = query.trim();
    if (!text || !scene) return;
    search.current?.abort();
    const coordinates = parseCoordinates(text);
    if (coordinates) {
      onDescend?.();
      descend(scene, coordinates, { reducedMotion });
      return;
    }
    if (looksLikeCoordinates(text)) {
      setAnswer({
        kind: "error",
        message:
          "Latitude runs from -90 to 90 and longitude from -180 to 180. Write latitude first.",
      });
      return;
    }
    const controller = new AbortController();
    search.current = controller;
    setAnswer({ kind: "busy" });
    scene.geocoder
      .search(text, controller.signal)
      .then((places) => {
        if (controller.signal.aborted) return;
        const [only] = places;
        if (places.length === 1 && only) {
          setAnswer({ kind: "idle" });
          go(only);
        } else if (places.length === 0)
          setAnswer({
            kind: "error",
            message: `No place found for “${text}”. Try a nearby town, or coordinates such as 34.698, -120.04.`,
          });
        else setAnswer({ kind: "choices", query: text, places: places.slice(0, MAX_CHOICES) });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setAnswer({
          kind: "error",
          message: `Places could not be searched: ${describeError(error)}. Try coordinates instead.`,
        });
      });
  };

  const inputId = `${id}-place`;
  const label = env.landExplorationEnabled ? "Where is your land?" : "Where do you want to go?";
  return (
    <form className="descent-form" onSubmit={submit} role="search" data-testid="descent-form">
      <GlassField
        label={label}
        htmlFor={inputId}
        error={answer.kind === "error" ? answer.message : undefined}
        hint={scene ? scene.geocoder.attribution : undefined}
      >
        <div className="descent-form__row">
          <GlassInput
            id={inputId}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              if (answer.kind !== "busy") setAnswer({ kind: "idle" });
            }}
            placeholder="Address, place or coordinates"
            autoComplete="off"
            spellCheck={false}
            enterKeyHint="go"
            invalid={answer.kind === "error"}
            data-testid="descent-input"
          />
          <GlassButton
            type="submit"
            variant="primary"
            disabled={!query.trim() || answer.kind === "busy" || !scene}
            data-testid="descent-submit"
          >
            {answer.kind === "busy" ? "Searching…" : "Take me there"}
          </GlassButton>
        </div>
      </GlassField>
      {answer.kind === "choices" && (
        <ul
          className="descent-form__choices"
          aria-label={`Places for ${answer.query}`}
          data-testid="descent-choices"
        >
          {answer.places.map((place, index) => (
            <li key={`${place.label}-${index}`}>
              <button type="button" className="descent-form__choice" onClick={() => go(place)}>
                <MapPin size={14} aria-hidden="true" />
                <span>{place.label}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </form>
  );
}
