import { useId, useState } from "react";
import { Check, Download, Focus, PencilLine, Plus, Trash2, Undo2 } from "lucide-react";

import type { Footprint, LandArea } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassInput, GlassSelect, GlassTextarea } from "@twin/ui";

import { DESIGN_LABELS, METHOD_LABELS } from "./ecologyData";
import { SurveySummaryView } from "./EcologySurveyView";
import { STRATA, type SurveyDraft, type SurveyDraftState } from "./useEcologySurveys";
import { areaLabel } from "../../landFlow";
import { MapLink } from "../../mapLinks";
import { useFlyTo } from "../../mapHooks";

type Step = "basics" | "plots" | "review";
const STEPS: readonly [Step, string][] = [
  ["basics", "Basics"],
  ["plots", "Plots"],
  ["review", "Review"],
];

function basicsReady(draft: SurveyDraft): boolean {
  return Boolean(
    draft.name.trim() &&
    draft.observer.trim() &&
    draft.observedOn &&
    draft.methodNotes.trim() &&
    draft.assessedStrata.length,
  );
}

function sketchArea(corners: number[][]): number | null {
  const first = corners[0];
  if (corners.length < 3 || !first) return null;
  const polygon: Footprint = { type: "Polygon", coordinates: [[...corners, first]] };
  return footprintAreaM2(polygon);
}

/**
 * Recording a field survey, one step at a time: the basics, then each plot drawn on the map
 * with the species seen in it, then a review the server checks before the survey is saved.
 */
export function EcologySurveyRecorder({
  land,
  survey,
}: {
  land: LandArea;
  survey: SurveyDraftState;
}) {
  const draft = survey.draft;
  const [step, setStep] = useState<Step>(() =>
    survey.corners.length || draft?.plots.length ? "plots" : "basics",
  );
  const [index, setIndex] = useState(() =>
    survey.corners.length ? (draft?.plots.length ?? 0) : 0,
  );
  if (!draft) return null;
  const ready = basicsReady(draft);
  const at = Math.min(index, draft.plots.length);

  const goPlots = () => {
    setStep("plots");
    if (!draft.plots.length) {
      setIndex(0);
      if (survey.canDraw && !survey.drawing) survey.draw();
    }
  };
  const newPlot = () => {
    setIndex(draft.plots.length);
    setStep("plots");
    if (survey.canDraw && !survey.drawing) survey.draw();
  };
  const review = async () => {
    survey.stopDrawing();
    if (await survey.preview()) setStep("review");
  };

  return (
    <div className="land-eco-recorder">
      <ol className="land-eco-steps" aria-label="Survey steps">
        {STEPS.map(([key, label], number) => {
          const enabled =
            key === "basics" ||
            (key === "plots" && ready) ||
            (key === "review" && ready && draft.plots.length > 0);
          return (
            <li key={key}>
              <button
                type="button"
                aria-current={step === key ? "step" : undefined}
                disabled={!enabled}
                onClick={() => (key === "review" ? void review() : setStep(key))}
              >
                <span aria-hidden="true">{number + 1}</span>
                {label}
              </button>
            </li>
          );
        })}
      </ol>

      {step === "basics" && (
        <SurveyBasics land={land} survey={survey} draft={draft} ready={ready} onNext={goPlots} />
      )}
      {step === "plots" && (
        <SurveyPlots
          land={land}
          survey={survey}
          draft={draft}
          at={at}
          setIndex={setIndex}
          onNewPlot={newPlot}
          onReview={() => void review()}
        />
      )}
      {step === "review" &&
        (survey.review ? (
          <>
            <SurveySummaryView summary={survey.review} />
            <div className="land-row">
              <GlassButton
                variant="primary"
                disabled={survey.busy}
                leadingIcon={<Check size={16} aria-hidden="true" />}
                onClick={survey.save}
              >
                {survey.busy ? "Saving…" : "Save immutable survey"}
              </GlassButton>
              <GlassButton variant="ghost" onClick={() => setStep("plots")}>
                Back to plots
              </GlassButton>
            </div>
            <p className="land-muted">
              A saved survey cannot be changed. Corrections are new copies.
            </p>
          </>
        ) : (
          <div className="land-row">
            <GlassButton
              variant="primary"
              disabled={survey.busy || !draft.plots.length}
              onClick={() => void review()}
            >
              {survey.busy ? "Checking…" : "Review survey"}
            </GlassButton>
            <GlassButton variant="ghost" onClick={() => setStep("plots")}>
              Back to plots
            </GlassButton>
          </div>
        ))}

      {survey.error && (
        <p role="alert" className="land-alert">
          {survey.error}
        </p>
      )}
      {survey.storageError && (
        <p role="alert" className="land-alert">
          This draft could not be stored in this browser. Export a copy before leaving.
        </p>
      )}
      <div className="land-row land-eco-recorder__footer">
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<Download size={14} aria-hidden="true" />}
          onClick={survey.exportDraft}
        >
          Export draft
        </GlassButton>
        <GlassButton size="sm" variant="ghost" onClick={survey.discard}>
          Discard survey draft
        </GlassButton>
      </div>
    </div>
  );
}

function SurveyBasics({
  land,
  survey,
  draft,
  ready,
  onNext,
}: {
  land: LandArea;
  survey: SurveyDraftState;
  draft: SurveyDraft;
  ready: boolean;
  onNext: () => void;
}) {
  const id = useId();
  return (
    <fieldset className="land-eco-form" disabled={survey.busy}>
      <legend className="land-visually-hidden">Survey basics</legend>
      <label className="land-field" htmlFor={`${id}-name`}>
        Survey name
        <GlassInput
          id={`${id}-name`}
          required
          maxLength={200}
          value={draft.name}
          onChange={(event) => survey.change({ name: event.target.value })}
        />
      </label>
      <div className="land-eco-pair">
        <label className="land-field" htmlFor={`${id}-date`}>
          Observation date
          <GlassInput
            id={`${id}-date`}
            required
            type="date"
            max={new Date().toISOString().slice(0, 10)}
            value={draft.observedOn}
            onChange={(event) => survey.change({ observedOn: event.target.value })}
          />
        </label>
        <label className="land-field" htmlFor={`${id}-observer`}>
          Observer
          <GlassInput
            id={`${id}-observer`}
            required
            maxLength={200}
            value={draft.observer}
            onChange={(event) => survey.change({ observer: event.target.value })}
          />
        </label>
      </div>
      <div className="land-eco-pair">
        <label className="land-field" htmlFor={`${id}-method`}>
          Method
          <GlassSelect
            id={`${id}-method`}
            value={draft.method}
            onChange={(event) => survey.setMethod(event.target.value as SurveyDraft["method"])}
          >
            {Object.entries(METHOD_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </GlassSelect>
        </label>
        <label className="land-field" htmlFor={`${id}-design`}>
          Sampling design
          <GlassSelect
            id={`${id}-design`}
            value={draft.design}
            onChange={(event) =>
              survey.change({ design: event.target.value as SurveyDraft["design"] })
            }
          >
            {Object.entries(DESIGN_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </GlassSelect>
        </label>
      </div>
      <p className="land-muted">
        Changing the method clears the numbers entered. For point intercept, count points with the
        species, not repeated contacts at one point.
      </p>
      <fieldset className="land-eco-fieldset">
        <legend className="land-muted">Vegetation layers assessed</legend>
        <div className="land-eco-chips">
          {STRATA.map((stratum) => (
            <label key={stratum} className="land-eco-chip">
              <input
                type="checkbox"
                checked={draft.assessedStrata.includes(stratum)}
                onChange={(event) =>
                  survey.change({
                    assessedStrata: event.target.checked
                      ? [...draft.assessedStrata, stratum]
                      : draft.assessedStrata.filter((value) => value !== stratum),
                  })
                }
              />
              {stratum}
            </label>
          ))}
        </div>
      </fieldset>
      <label className="land-field" htmlFor={`${id}-notes`}>
        Survey method and limitations
        <GlassTextarea
          id={`${id}-notes`}
          required
          rows={2}
          maxLength={5000}
          placeholder="How you looked, and what you might have missed"
          value={draft.methodNotes}
          onChange={(event) => survey.change({ methodNotes: event.target.value })}
        />
      </label>
      {draft.supersedesId && (
        <p className="land-muted">This corrected copy keeps the earlier survey as it was.</p>
      )}
      {draft.boundaryRevision !== land.revision && (
        <p className="land-note">
          Made for an earlier boundary.{" "}
          <button
            type="button"
            className="land-link"
            onClick={() => survey.change({ boundaryRevision: land.revision })}
          >
            Check against current boundary
          </button>
        </p>
      )}
      <div className="land-row">
        <GlassButton variant="primary" disabled={!ready} onClick={onNext}>
          Next: map the plots
        </GlassButton>
        {!ready && <span className="land-muted">Add the observer and how you surveyed.</span>}
      </div>
    </fieldset>
  );
}

function SurveyPlots({
  land,
  survey,
  draft,
  at,
  setIndex,
  onNewPlot,
  onReview,
}: {
  land: LandArea;
  survey: SurveyDraftState;
  draft: SurveyDraft;
  at: number;
  setIndex: (index: number) => void;
  onNewPlot: () => void;
  onReview: () => void;
}) {
  const plot = draft.plots[at];
  return (
    <div className="land-eco-form">
      <div className="land-eco-tabs" role="group" aria-label="Plots">
        {draft.plots.map((item, number) => (
          <button
            key={number}
            type="button"
            aria-pressed={number === at}
            onClick={() => {
              survey.stopDrawing();
              setIndex(number);
            }}
          >
            {item.label || `Plot ${number + 1}`}
          </button>
        ))}
        <button
          type="button"
          aria-pressed={at === draft.plots.length}
          disabled={draft.plots.length >= 100}
          onClick={onNewPlot}
        >
          <Plus size={13} aria-hidden="true" /> New plot
        </button>
      </div>
      {plot ? (
        <PlotEditor
          survey={survey}
          draft={draft}
          index={at}
          onRemoved={() => setIndex(Math.max(0, at - 1))}
        />
      ) : (
        <PlotSketch
          land={land}
          survey={survey}
          draft={draft}
          onFinished={(index) => setIndex(index)}
        />
      )}
      <div className="land-row">
        <GlassButton
          variant={plot ? "primary" : "default"}
          disabled={survey.busy || !draft.plots.length}
          onClick={onReview}
        >
          {survey.busy ? "Checking…" : "Review survey"}
        </GlassButton>
        {plot && (
          <GlassButton
            leadingIcon={<Plus size={15} aria-hidden="true" />}
            disabled={draft.plots.length >= 100}
            onClick={onNewPlot}
          >
            Add another plot
          </GlassButton>
        )}
      </div>
    </div>
  );
}

/** The plot being drawn: corners are clicked on the map; the card shows the count and size. */
function PlotSketch({
  land,
  survey,
  draft,
  onFinished,
}: {
  land: LandArea;
  survey: SurveyDraftState;
  draft: SurveyDraft;
  onFinished: (index: number) => void;
}) {
  const corners = survey.corners;
  const area = sketchArea(corners);
  const status = survey.drawing
    ? corners.length
      ? `Click corner ${corners.length + 1} on the map, or finish the plot`
      : "Click the plot's first corner on the map"
    : corners.length
      ? `${corners.length} ${corners.length === 1 ? "corner" : "corners"} placed`
      : "Draw the plot's corners on the map";
  return (
    <div className="land-eco-sketch">
      <p className="land-eco-sketch__status" aria-live="polite">
        {status}
        {area !== null && <b> · ≈ {areaLabel(area)}</b>}
      </p>
      <div className="land-row">
        {survey.drawing ? (
          <GlassButton size="sm" onClick={survey.stopDrawing}>
            Pause drawing
          </GlassButton>
        ) : (
          <GlassButton
            size="sm"
            variant={corners.length ? "default" : "primary"}
            disabled={!survey.canDraw || corners.length >= 1000}
            leadingIcon={<PencilLine size={14} aria-hidden="true" />}
            onClick={survey.draw}
          >
            {corners.length ? "Keep drawing" : "Draw on the map"}
          </GlassButton>
        )}
        <GlassButton
          size="sm"
          variant="ghost"
          disabled={!corners.length}
          leadingIcon={<Undo2 size={14} aria-hidden="true" />}
          onClick={survey.undoCorner}
        >
          Undo corner
        </GlassButton>
        <GlassButton
          size="sm"
          variant="primary"
          disabled={corners.length < 3}
          onClick={() => {
            const index = survey.finishPlot();
            if (index !== null) onFinished(index);
          }}
        >
          Finish plot ({corners.length} corners)
        </GlassButton>
      </div>
      <p className="land-muted">
        Map only the area you assessed. Plots sit inside the land and must not overlap. Esc pauses
        drawing.
      </p>
      {!draft.plots.length && !corners.length && (
        <button
          type="button"
          className="land-link"
          onClick={() => {
            survey.stopDrawing();
            survey.addPlot(land.boundary);
            onFinished(0);
          }}
        >
          Use entire land as the surveyed plot
        </button>
      )}
    </div>
  );
}

/** One drawn plot: its name, then the species seen in it. */
function PlotEditor({
  survey,
  draft,
  index,
  onRemoved,
}: {
  survey: SurveyDraftState;
  draft: SurveyDraft;
  index: number;
  onRemoved: () => void;
}) {
  const id = useId();
  const fly = useFlyTo();
  const plot = draft.plots[index];
  if (!plot) return null;
  const visual = draft.method === "visual-cover";
  const layers = draft.assessedStrata.length ? draft.assessedStrata : (["herb"] as const);
  const setObservation = (j: number, value: Partial<(typeof plot.observations)[number]>) =>
    survey.updatePlot(index, {
      observations: plot.observations.map((item, k) => (k === j ? { ...item, ...value } : item)),
    });
  return (
    <fieldset className="land-eco-form" disabled={survey.busy}>
      <legend className="land-visually-hidden">{plot.label || `Plot ${index + 1}`}</legend>
      <MapLink
        target={{ key: `ecology-draft-plot-${index}`, geometry: plot.boundary }}
        fly={false}
        className="land-fact land-eco-plotcard"
      >
        <label className="land-field" htmlFor={`${id}-label`}>
          Plot label
          <GlassInput
            id={`${id}-label`}
            required
            maxLength={100}
            value={plot.label}
            onChange={(event) => survey.updatePlot(index, { label: event.target.value })}
          />
        </label>
        <div className="land-row">
          <span className="land-muted">≈ {areaLabel(footprintAreaM2(plot.boundary))}</span>
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<Focus size={14} aria-hidden="true" />}
            onClick={() => fly(plot.boundary)}
          >
            Frame plot
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<Trash2 size={14} aria-hidden="true" />}
            onClick={() => {
              survey.removePlot(index);
              onRemoved();
            }}
          >
            Remove plot
          </GlassButton>
        </div>
      </MapLink>
      {!visual && (
        <label className="land-field" htmlFor={`${id}-points`}>
          Sampled points
          <GlassInput
            id={`${id}-points`}
            className="land-number"
            required
            type="number"
            min={1}
            max={100000}
            step={1}
            value={plot.samplePoints ?? ""}
            onChange={(event) =>
              survey.updatePlot(index, {
                samplePoints: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </label>
      )}
      <ul className="land-eco-observations" aria-label={`Species in ${plot.label || "this plot"}`}>
        {plot.observations.map((observation, j) => (
          <li key={j} className="land-eco-obs">
            <label className="land-field land-eco-obs__taxon" htmlFor={`${id}-taxon-${j}`}>
              Taxon or field identifier
              <GlassInput
                id={`${id}-taxon-${j}`}
                required
                maxLength={200}
                value={observation.taxon}
                onChange={(event) => setObservation(j, { taxon: event.target.value })}
              />
            </label>
            <label className="land-field" htmlFor={`${id}-cover-${j}`}>
              {visual ? "Cover (%)" : "Points with taxon"}
              <GlassInput
                id={`${id}-cover-${j}`}
                required
                type="number"
                min={0}
                max={visual ? 100 : (plot.samplePoints ?? 100000)}
                step={visual ? "any" : 1}
                value={(visual ? observation.percentCover : observation.hits) ?? ""}
                onChange={(event) =>
                  setObservation(j, {
                    [visual ? "percentCover" : "hits"]:
                      event.target.value === "" ? null : Number(event.target.value),
                  })
                }
              />
            </label>
            <label className="land-field" htmlFor={`${id}-layer-${j}`}>
              Vegetation layer
              <GlassSelect
                id={`${id}-layer-${j}`}
                value={observation.stratum}
                onChange={(event) =>
                  setObservation(j, {
                    stratum: event.target.value as typeof observation.stratum,
                  })
                }
              >
                {[...new Set([...layers, observation.stratum])].map((stratum) => (
                  <option key={stratum} value={stratum}>
                    {stratum}
                  </option>
                ))}
              </GlassSelect>
            </label>
            <label className="land-field" htmlFor={`${id}-id-${j}`}>
              Identification
              <GlassSelect
                id={`${id}-id-${j}`}
                value={observation.identification}
                onChange={(event) =>
                  setObservation(j, {
                    identification: event.target.value as typeof observation.identification,
                  })
                }
              >
                <option value="tentative">Tentative</option>
                <option value="verified">Verified by observer</option>
                <option value="unidentified">Unidentified field taxon</option>
              </GlassSelect>
            </label>
            <label className="land-field land-eco-obs__notes" htmlFor={`${id}-note-${j}`}>
              Observation notes
              <GlassInput
                id={`${id}-note-${j}`}
                maxLength={1000}
                value={observation.notes ?? ""}
                onChange={(event) => setObservation(j, { notes: event.target.value })}
              />
            </label>
            <button
              type="button"
              className="land-icon-button land-eco-obs__remove"
              aria-label={`Remove observation ${j + 1}`}
              onClick={() =>
                survey.updatePlot(index, {
                  observations: plot.observations.filter((_, k) => k !== j),
                })
              }
            >
              <Trash2 size={15} aria-hidden="true" />
            </button>
          </li>
        ))}
      </ul>
      <div className="land-row">
        <GlassButton
          size="sm"
          leadingIcon={<Plus size={14} aria-hidden="true" />}
          disabled={plot.observations.length >= 200}
          onClick={() =>
            survey.updatePlot(index, {
              observations: [
                ...plot.observations,
                {
                  taxon: "",
                  stratum: draft.assessedStrata[0] ?? "herb",
                  identification: "tentative",
                  percentCover: null,
                  hits: null,
                },
              ],
            })
          }
        >
          Add species observation
        </GlassButton>
      </div>
      <label className="land-eco-check">
        <input
          type="checkbox"
          checked={plot.completeInventory ?? false}
          onChange={(event) =>
            survey.updatePlot(index, { completeInventory: event.target.checked })
          }
        />
        Complete inventory: every species in the assessed layers is listed, so unlisted ones were
        not seen
      </label>
    </fieldset>
  );
}
