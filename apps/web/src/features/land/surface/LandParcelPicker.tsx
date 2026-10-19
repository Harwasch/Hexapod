import { useEffect, useRef, useState } from "react";

import type {
  CandidateResult,
  Footprint,
  LandCandidate,
  SelectionInterpretation,
} from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import {
  GlassButton,
  GlassPanel,
  GlassSegmentedControl,
  GlassSelect,
  GlassTextarea,
} from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

import { drawingReferences } from "../boundarySources";
import { areaLabel } from "./landFlow";
import { MapLink } from "./mapLinks";

type Kind = "parcel" | "line" | "building";
const KINDS: { value: Kind; label: string }[] = [
  { value: "parcel", label: "Parcels" },
  { value: "line", label: "Roads & rivers" },
  { value: "building", label: "Buildings" },
];

/**
 * Pick the land from records instead of drawing it: click near it, the parcels around the
 * click outline on the map, and clicking them (on the map or here) selects them. Each new
 * click on open ground looks again there and keeps what is already chosen.
 */
export function LandParcelPicker() {
  const points = useLand((s) => s.points);
  const candidates = useLandContext((s) => s.candidates);
  const selectedIds = useLandContext((s) => s.selectedIds);
  const [kind, setKind] = useState<Kind>("parcel");
  const [answer, setAnswer] = useState<{
    point: [number, number];
    kind: Kind;
    result: CandidateResult | null;
  } | null>(null);
  const [instruction, setInstruction] = useState("");
  const [width, setWidth] = useState(100);
  const [unit, setUnit] = useState<"ft" | "m">("ft");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarification, setClarification] = useState<string | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const point = points[0];
  useEffect(() => {
    if (!point) return;
    const controller = new AbortController();
    void unwrap(
      api.POST("/api/v1/land/selection/candidates", {
        body: { point: { type: "Point", coordinates: point }, kind, radiusM: 250 },
        signal: controller.signal,
      }),
    )
      .then((response) => {
        if (controller.signal.aborted) return;
        setAnswer({ point, kind, result: response });
        setError(null);
        useLandContext.getState().addCandidates(response.candidates);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setAnswer({ point, kind, result: null });
        setError(describeError(cause));
      });
    return () => controller.abort();
  }, [point, kind]);
  const answered = answer !== null && answer.point === point && answer.kind === kind;
  const result = answered ? answer.result : null;
  const searching = Boolean(point) && !answered;

  const selected = selectedIds
    .map((id) => candidates.find((candidate) => candidate.id === id))
    .filter((candidate): candidate is LandCandidate => Boolean(candidate));
  const line =
    selected.length === 1 && selected[0]?.geometry.type === "LineString" ? selected[0] : null;
  const selectedArea = selected.reduce(
    (sum, candidate) =>
      candidate.geometry.type === "Polygon" || candidate.geometry.type === "MultiPolygon"
        ? sum + footprintAreaM2(candidate.geometry)
        : sum,
    0,
  );

  const propose = async (interpretation?: SelectionInterpretation) => {
    if (!alive.current || useLand.getState().mode !== "candidates") return;
    setBusy(true);
    setError(null);
    setClarification(null);
    const session = useLand.getState().session;
    try {
      if (interpretation?.operation === "clarify") {
        setClarification(interpretation.question ?? interpretation.explanation);
        return;
      }
      const chosen = interpretation
        ? (interpretation.candidateIds ?? [])
            .map((id) => candidates.find((candidate) => candidate.id === id))
            .filter((candidate): candidate is LandCandidate => Boolean(candidate))
        : selected;
      const first = chosen[0];
      if (!first) throw new Error("Click a highlighted parcel or feature first.");
      let boundary: Footprint;
      if (first.geometry.type === "LineString") {
        if (chosen.length !== 1) throw new Error("Choose one line for a corridor.");
        const totalWidth = interpretation?.widthM ?? (unit === "ft" ? width * 0.3048 : width);
        boundary = (
          await unwrap(
            api.POST("/api/v1/land/corridor", {
              body: {
                coordinates: first.geometry.coordinates,
                widthM: totalWidth,
                cap: interpretation?.cap ?? "round",
              },
            }),
          )
        ).boundary;
      } else {
        if (first.geometry.type === "Point") throw new Error("Choose an area or a line.");
        boundary = first.geometry;
        for (const next of chosen.slice(1)) {
          if (next.geometry.type !== "Polygon" && next.geometry.type !== "MultiPolygon")
            throw new Error("Only areas can be combined.");
          boundary = (
            await unwrap(
              api.POST("/api/v1/land/operations", {
                body: { operation: "union", left: boundary, right: next.geometry },
              }),
            )
          ).boundary;
        }
      }
      if (
        !alive.current ||
        useLand.getState().session !== session ||
        useLand.getState().mode !== "candidates"
      )
        return;
      useLand.getState().propose({
        name: chosen.length > 1 ? `${chosen.length} parcels` : first.label,
        description: interpretation?.explanation ?? "",
        boundary,
        source: {
          ...first.source,
          ...(first.geometry.type === "LineString"
            ? { method: "corridor", meaning: "study-area", label: `Corridor along ${first.label}` }
            : {}),
          ...(chosen.length > 1
            ? { label: "Combined parcels", recordId: undefined, meaning: "study-area" }
            : {}),
        },
      });
    } catch (cause) {
      if (alive.current) setError(describeError(cause));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const status = !point
    ? "Click near your land on the map."
    : searching
      ? "Looking for records around that spot…"
      : (result?.message ?? "");

  return (
    <GlassPanel strong className="land-card land-card--scroll" data-testid="land-parcel-picker">
      <div className="land-eyebrow-row">
        <span className="land-lens-label">Pick your land</span>
      </div>
      <GlassSegmentedControl
        aria-label="What to pick"
        value={kind}
        options={KINDS}
        block
        onValueChange={(value: Kind) => {
          setKind(value);
          setAnswer(null);
          useLandContext.getState().setCandidates([]);
        }}
      />
      <p className="land-lede" aria-live="polite">
        {status}
      </p>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      {result?.truncated && (
        <p className="land-note">Only the nearest records are shown. Click closer to your land.</p>
      )}
      {candidates.length > 0 && (
        <ul className="land-facts" aria-label="Records near your click">
          {candidates.map((candidate) => {
            const chosen = selectedIds.includes(candidate.id);
            return (
              <MapLink
                key={candidate.id}
                as="li"
                className={`land-fact land-candidate ${chosen ? "is-chosen" : ""}`}
                target={{ key: `candidate:${candidate.id}`, geometry: candidate.geometry }}
                tone={chosen ? "warn" : "accent"}
                fly={false}
              >
                <label className="land-candidate__row" htmlFor={`land-candidate-${candidate.id}`}>
                  <input
                    id={`land-candidate-${candidate.id}`}
                    type="checkbox"
                    checked={chosen}
                    onChange={() => useLandContext.getState().toggleCandidate(candidate.id)}
                  />
                  {candidate.label}
                </label>
                <span className="land-fact__note">
                  {candidate.source.meaning === "recorded-parcel"
                    ? "Parcel record"
                    : "Mapped feature"}
                  {candidate.distanceM >= 1 ? ` · ${Math.round(candidate.distanceM)} m away` : ""}
                </span>
              </MapLink>
            );
          })}
        </ul>
      )}
      {line && (
        <div className="land-row">
          <label className="land-visually-hidden" htmlFor="land-parcel-width">
            Total corridor width
          </label>
          <input
            id="land-parcel-width"
            className="glass-input land-number"
            type="number"
            min="0.1"
            max={unit === "ft" ? 328084 : 100000}
            value={width}
            onChange={(event) => setWidth(Number(event.target.value))}
          />
          <GlassSelect
            aria-label="Corridor width unit"
            value={unit}
            onChange={(event) => setUnit(event.target.value === "m" ? "m" : "ft")}
          >
            <option value="ft">ft wide</option>
            <option value="m">m wide</option>
          </GlassSelect>
        </div>
      )}
      {selected.length > 0 && (
        <div className="land-row">
          <GlassButton variant="primary" disabled={busy} onClick={() => void propose()}>
            {busy
              ? "Preparing…"
              : line
                ? "Make corridor"
                : selected.length > 1
                  ? `Use ${selected.length} parcels · ${areaLabel(selectedArea)}`
                  : `Use this · ${areaLabel(selectedArea)}`}
          </GlassButton>
          <GlassButton
            disabled={busy}
            onClick={() => {
              const references = drawingReferences(selected.map((candidate) => candidate.source));
              useLandContext.getState().setLayer({
                id: "drawing-guides",
                title: "Drawing guides",
                drawingGuideSources: references,
                features: selected.map((candidate) => ({
                  id: candidate.id,
                  label: candidate.label,
                  geometry: candidate.geometry,
                })),
              });
              const land = useLand.getState();
              land.setSnapMapped(true);
              land.begin(line ? "corridor" : "draw");
              land.setTraceSources(references);
            }}
          >
            Trace over them
          </GlassButton>
        </div>
      )}
      {candidates.length > 0 && (
        <details className="land-details">
          <summary>Describe it instead</summary>
          <form
            className="land-details__body"
            onSubmit={(event) => {
              event.preventDefault();
              const session = useLand.getState().session;
              setBusy(true);
              setError(null);
              void unwrap(
                api.POST("/api/v1/land/selection/interpret", {
                  body: { instruction, candidates, selectedIds },
                }),
              )
                .then((interpretation) => {
                  if (alive.current && useLand.getState().session === session)
                    return propose(interpretation);
                })
                .catch((cause: unknown) => {
                  if (!alive.current || useLand.getState().session !== session) return;
                  setError(describeError(cause));
                  setBusy(false);
                });
            }}
          >
            <label className="land-visually-hidden" htmlFor="land-selection-instruction">
              Describe your selection
            </label>
            <GlassTextarea
              id="land-selection-instruction"
              rows={2}
              maxLength={5000}
              value={instruction}
              placeholder="The two parcels north of the road"
              onChange={(event) => setInstruction(event.target.value)}
            />
            <GlassButton type="submit" size="sm" disabled={busy || !instruction.trim()}>
              Use my description
            </GlassButton>
          </form>
        </details>
      )}
      {clarification && (
        <p className="land-note" role="status">
          {clarification}
        </p>
      )}
      <div className="land-row">
        <GlassButton variant="ghost" size="sm" onClick={() => useLand.getState().begin("browse")}>
          Cancel
        </GlassButton>
      </div>
    </GlassPanel>
  );
}
