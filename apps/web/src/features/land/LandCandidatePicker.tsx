import { useEffect, useRef, useState } from "react";
import type {
  CandidateResult,
  Footprint,
  LandCandidate,
  SelectionInterpretation,
} from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { describeError } from "@/lib/log";
import { drawingReferences } from "./boundarySources";

export function LandCandidatePicker() {
  const mode = useLand((state) => state.mode);
  const points = useLand((state) => state.points);
  const candidates = useLandContext((state) => state.candidates);
  const selectedIds = useLandContext((state) => state.selectedIds);
  const [kind, setKind] = useState<"parcel" | "line" | "building">("parcel");
  const [radius, setRadius] = useState(250);
  const [result, setResult] = useState<CandidateResult | null>(null);
  const [instruction, setInstruction] = useState("");
  const [width, setWidth] = useState(100);
  const [unit, setUnit] = useState<"ft" | "m">("ft");
  const [cap, setCap] = useState<"round" | "flat" | "square">("round");
  const [startIndex, setStartIndex] = useState(0);
  const [endIndex, setEndIndex] = useState<number | null>(null);
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
  useEffect(() => {
    if (mode !== "candidates" || points.length !== 1 || !points[0]) return;
    const controller = new AbortController();
    const coordinates = points[0];
    void unwrap(
      api.POST("/api/v1/land/selection/candidates", {
        body: { point: { type: "Point", coordinates }, kind, radiusM: radius },
        signal: controller.signal,
      }),
    )
      .then((response) => {
        if (!controller.signal.aborted) {
          setResult(response);
          setError(null);
          useLandContext.getState().setCandidates(response.candidates);
          setStartIndex(0);
          setEndIndex(null);
        }
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(describeError(cause));
      });
    return () => controller.abort();
  }, [points, mode, kind, radius]);
  const selected = selectedIds
    .map((id) => candidates.find((candidate) => candidate.id === id))
    .filter((candidate): candidate is LandCandidate => Boolean(candidate));
  const line =
    selected.length === 1 && selected[0]?.geometry.type === "LineString" ? selected[0] : null;
  const lineCoordinates = line?.geometry.type === "LineString" ? line.geometry.coordinates : [];
  const finalIndex = Math.min(
    endIndex ?? Math.max(1, lineCoordinates.length - 1),
    Math.max(1, lineCoordinates.length - 1),
  );
  if (mode !== "candidates") return null;

  const propose = async (interpretation?: SelectionInterpretation) => {
    if (!alive.current || useLand.getState().mode !== "candidates") return;
    setBusy(true);
    setError(null);
    setClarification(null);
    const session = useLand.getState().session;
    try {
      const chosen = interpretation
        ? (interpretation.candidateIds ?? [])
            .map((id) => candidates.find((candidate) => candidate.id === id))
            .filter((candidate): candidate is LandCandidate => Boolean(candidate))
        : selected;
      if (interpretation?.operation === "clarify") {
        setClarification(interpretation.question ?? interpretation.explanation);
        return;
      }
      const first = chosen[0];
      if (!first) throw new Error("Choose a highlighted feature first.");
      let boundary: Footprint;
      if (first.geometry.type === "LineString") {
        if (chosen.length !== 1) throw new Error("Choose one line segment for this corridor.");
        const coordinates =
          first.id === line?.id
            ? first.geometry.coordinates.slice(startIndex, finalIndex + 1)
            : first.geometry.coordinates;
        if (coordinates.length < 2) throw new Error("The corridor needs at least two line points.");
        const totalWidth = interpretation?.widthM ?? (unit === "ft" ? width * 0.3048 : width);
        const buffered = await unwrap(
          api.POST("/api/v1/land/corridor", {
            body: { coordinates, widthM: totalWidth, cap: interpretation?.cap ?? cap },
          }),
        );
        boundary = buffered.boundary;
      } else {
        if (first.geometry.type === "Point") throw new Error("Choose an area or line.");
        boundary = first.geometry;
        for (const next of chosen.slice(1)) {
          if (next.geometry.type !== "Polygon" && next.geometry.type !== "MultiPolygon")
            throw new Error("Combine polygon areas only.");
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
        name: chosen.length > 1 ? `${chosen.length} selected parcels or areas` : first.label,
        description: interpretation?.explanation ?? "",
        boundary,
        source: {
          ...first.source,
          ...(first.geometry.type === "LineString"
            ? {
                method: "corridor",
                meaning: "study-area",
                label: `Corridor along ${first.label}`,
              }
            : {}),
          ...(chosen.length > 1
            ? {
                label: "Combined selected boundaries",
                recordId: undefined,
                meaning: "study-area",
              }
            : {}),
        },
      });
    } catch (cause) {
      if (alive.current) setError(describeError(cause));
    } finally {
      if (alive.current) setBusy(false);
    }
  };
  return (
    <section className="land-selection" aria-label="Select from map records">
      <span className="land-eyebrow">Find your exact place</span>
      <h3>Pick the land, not every vertex.</h3>
      <label className="land-name">
        Find
        <select
          value={kind}
          onChange={(event) => {
            setKind(event.target.value as typeof kind);
            useLandContext.getState().removeLayer("candidates");
            setResult(null);
          }}
        >
          <option value="parcel">Recorded parcels</option>
          <option value="line">Lines, roads and waterways</option>
          <option value="building">Mapped buildings</option>
        </select>
      </label>
      <label className="land-name">
        Search distance
        <select value={radius} onChange={(event) => setRadius(Number(event.target.value))}>
          <option value={100}>100 m</option>
          <option value={250}>250 m</option>
          <option value={1000}>1 km</option>
          <option value={2000}>2 km</option>
        </select>
      </label>
      <p aria-live="polite">
        {!points.length
          ? "Click near your land on the map."
          : result
            ? result.message
            : "Looking for mapped records near your point…"}
      </p>
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
      {result?.truncated && (
        <p className="land-notice">
          Only the first candidates are shown. Narrow the search or choose a closer point.
        </p>
      )}
      {candidates.length > 0 && (
        <div className="land-candidates">
          {candidates.map((candidate) => (
            <button
              type="button"
              key={candidate.id}
              aria-pressed={selectedIds.includes(candidate.id)}
              onClick={() => {
                useLandContext.getState().toggleCandidate(candidate.id);
                setStartIndex(0);
                setEndIndex(null);
              }}
            >
              <strong>{candidate.label}</strong>
              <span>
                {candidate.distanceM < 1
                  ? "At your point"
                  : `${Math.round(candidate.distanceM)} m away`}{" "}
                ·{" "}
                {candidate.source.meaning === "recorded-parcel"
                  ? "Recorded parcel"
                  : "Mapped feature"}
              </span>
            </button>
          ))}
        </div>
      )}
      {line && (
        <>
          <label className="land-name">
            Total corridor width
            <div className="land-width-controls">
              <input
                type="number"
                min="0.1"
                max={unit === "ft" ? 328084 : 100000}
                value={width}
                onChange={(event) => setWidth(Number(event.target.value))}
              />
              <select
                aria-label="Corridor width unit"
                value={unit}
                onChange={(event) => setUnit(event.target.value as "ft" | "m")}
              >
                <option value="ft">feet</option>
                <option value="m">meters</option>
              </select>
            </div>
          </label>
          <label className="land-name">
            End caps
            <select value={cap} onChange={(event) => setCap(event.target.value as typeof cap)}>
              <option value="round">Rounded</option>
              <option value="flat">Flat</option>
              <option value="square">Square</option>
            </select>
          </label>
          <label className="land-name">
            Start along mapped line
            <input
              type="range"
              min="0"
              max={Math.max(0, finalIndex - 1)}
              value={startIndex}
              onChange={(event) => setStartIndex(Number(event.target.value))}
            />
          </label>
          <label className="land-name">
            End along mapped line
            <input
              type="range"
              min={startIndex + 1}
              max={Math.max(1, lineCoordinates.length - 1)}
              value={finalIndex}
              onChange={(event) => setEndIndex(Number(event.target.value))}
            />
          </label>
          <p className="land-footnote">
            Uses mapped vertices {startIndex + 1}–{finalIndex + 1}. Adjust the resulting boundary in
            the next step.
          </p>
        </>
      )}
      {selected.length > 0 && (
        <div className="land-actions">
          <button
            type="button"
            className="land-primary"
            disabled={busy}
            onClick={() => void propose()}
          >
            {busy
              ? "Preparing…"
              : line
                ? "Preview corridor"
                : selected.length > 1
                  ? "Combine selected areas"
                  : "Preview this boundary"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              try {
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
                useLand.getState().setSnapMapped(true);
                useLand.getState().begin(line ? "corridor" : "draw");
                useLand.getState().setTraceSources(references);
              } catch (cause) {
                setError(describeError(cause));
              }
            }}
          >
            Trace using selected features
          </button>
        </div>
      )}
      {candidates.length > 0 && (
        <form
          className="land-ask"
          onSubmit={(event) => {
            event.preventDefault();
            const requestSession = useLand.getState().session;
            setBusy(true);
            setError(null);
            void unwrap(
              api.POST("/api/v1/land/selection/interpret", {
                body: { instruction, candidates, selectedIds },
              }),
            )
              .then((interpretation) => {
                if (alive.current && useLand.getState().session === requestSession)
                  return propose(interpretation);
              })
              .catch((cause: unknown) => {
                if (!alive.current || useLand.getState().session !== requestSession) return;
                setError(describeError(cause));
                setBusy(false);
              });
          }}
        >
          <label htmlFor="land-selection-instruction">Or describe your selection</label>
          <textarea
            id="land-selection-instruction"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            maxLength={5000}
            placeholder="A corridor 100 feet wide along the selected transmission line"
            rows={2}
          />
          <button disabled={busy || !instruction.trim()}>Preview my instruction</button>
        </form>
      )}
      {clarification && (
        <p className="land-notice" role="status">
          {clarification}
        </p>
      )}
      <div className="land-actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            setResult(null);
            useLandContext.getState().removeLayer("candidates");
            useLand.getState().begin("candidates");
          }}
        >
          Choose another point
        </button>
        <button type="button" onClick={() => useLand.getState().begin("browse")}>
          Cancel
        </button>
      </div>
    </section>
  );
}
