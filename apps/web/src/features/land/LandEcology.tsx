import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, Footprint, LandArea } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { api, ApiError, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";
import { describeError } from "@/lib/log";
import { parseSurveyCorners, parseSurveyDraft } from "./surveyDraft";
import { EcologyResearch } from "./EcologyResearch";
import { SurveySummaryView } from "./SurveySummaryView";
import "./ecology.css";

type Draft = components["schemas"]["SurveyCreate"];
type Plot = components["schemas"]["SurveyPlot"];
type Summary = components["schemas"]["SurveySummary"];
const strata = ["canopy", "shrub", "herb", "ground", "aquatic"] as const;

function fresh(land: LandArea): Draft {
  return {
    requestKey: crypto.randomUUID(),
    name: "Field survey",
    boundaryRevision: land.revision,
    observedOn: new Date().toISOString().slice(0, 10),
    observer: "",
    method: "visual-cover",
    design: "purposive",
    assessedStrata: ["herb"],
    methodNotes: "",
    plots: [],
  };
}
function download(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : JSON.stringify(value, null, 2)], {
      type: "application/json",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function LandEcology({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <EcologyEditor key={`${scope}:${land.id}`} land={land} scope={scope} />;
}
function EcologyEditor({ land, scope }: { land: LandArea; scope: string }) {
  const ready = useLandAccessReady(),
    canEdit = useLandCanEdit(),
    scene = useScene(),
    cache = useQueryClient();
  const section = useLandContext((s) => s.section),
    panel = useUi((s) => s.activePanel);
  const selectedId = useLandContext((s) => s.selectedSurveyId);
  const picker = useLandContext((s) => s.pointPicker);
  const visible = ready && section === "ecology" && panel === "land";
  const owner = `survey:${land.id}`;
  const key = `living-world-land-draft:${encodeURIComponent(scope)}:survey:${land.id}`;
  const [draft, setDraft] = useState<Draft | null>(null);
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  });
  const [points, storePoints] = useState<number[][]>(() => {
    try {
      return parseSurveyCorners(localStorage.getItem(key + ":corners"));
    } catch {
      return [];
    }
  });
  const [offset, setOffset] = useState(0);
  const [review, setReview] = useState<{ signature: string; summary: Summary } | null>(null);
  const [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const ticket = useRef(0),
    pickTicket = useRef(0);
  const pointsRef = useRef(points);
  const setPoints = (change: number[][] | ((previous: number[][]) => number[][])) => {
    const next = typeof change === "function" ? change(pointsRef.current) : change;
    pointsRef.current = next;
    storePoints(next);
    try {
      localStorage.setItem(key + ":corners", JSON.stringify(next));
    } catch {
      setStorageError(true);
    }
  };
  const signature = JSON.stringify(draft);
  const queryKey = ["land-surveys", scope, land.id];
  const catalog = useQuery({
    queryKey: [...queryKey, offset, land.revision],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys", {
          params: { path: { land_id: land.id }, query: { limit: 20, offset } },
        }),
      ),
    retry: false,
  });
  const selected = useQuery({
    queryKey: ["land-survey", scope, land.id, selectedId, land.revision],
    enabled: ready && !!selectedId,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys/{survey_id}", {
          params: { path: { land_id: land.id, survey_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const cancelPick = () => {
    pickTicket.current += 1;
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
  };
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );
  useEffect(() => {
    if (!visible) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === owner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    };
  }, [visible, owner, scene]);
  useEffect(() => {
    if (!visible) return;
    const plots = draft?.plots ?? selected.data?.plots ?? [];
    useLandContext.getState().setLayer({
      id: owner,
      title: "Survey plots",
      features: [
        ...plots.map((p, i) => ({ id: `plot-${i}`, label: p.label, geometry: p.boundary })),
        ...points.map((p, i) => ({
          id: `corner-${i}`,
          label: `Corner ${i + 1}`,
          geometry: { type: "Point" as const, coordinates: p },
        })),
      ],
    });
    return () => useLandContext.getState().removeLayer(owner);
  }, [visible, draft, selected.data, points, owner]);
  const update = (value: Draft | null) => {
    ticket.current++;
    setBusy(false);
    setError(null);
    setReview(null);
    setDraft(value);
    try {
      if (value) localStorage.setItem(key, JSON.stringify(value));
      else {
        localStorage.removeItem(key);
        localStorage.removeItem(key + ":corners");
      }
      setStorageError(false);
      setRecovery(null);
    } catch {
      setStorageError(true);
    }
  };
  const updatePlot = (index: number, value: Partial<Plot>) => {
    if (draft)
      update({
        ...draft,
        plots: draft.plots.map((p, i) => (i === index ? { ...p, ...value } : p)),
      });
  };
  const addPlot = (boundary: Footprint) => {
    if (draft)
      update({
        ...draft,
        plots: [
          ...draft.plots,
          {
            label: `Plot ${draft.plots.length + 1}`,
            boundary,
            completeInventory: false,
            samplePoints: null,
            observations: [],
          },
        ],
      });
  };
  const pick = async () => {
    if (!scene || !draft) return;
    const current = ++pickTicket.current;
    useLandContext.getState().setPointPicker(owner);
    const point = await scene.areas.pickGround();
    if (current !== pickTicket.current) return;
    useLandContext.getState().setPointPicker(null);
    if (point) setPoints((p) => [...p, [point.longitude, point.latitude]]);
  };
  const calculate = async (save: boolean) => {
    if (!draft) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (save) {
        const result = await unwrap(
          api.POST("/api/v1/land/{land_id}/surveys", {
            params: { path: { land_id: land.id } },
            body: draft,
          }),
        );
        if (current === ticket.current) {
          update(null);
          setPoints([]);
          useLandContext.getState().selectSurvey(result.id);
        }
        await cache.invalidateQueries({ queryKey });
      } else {
        const result = await unwrap(
          api.POST("/api/v1/land/{land_id}/surveys/preview", {
            params: { path: { land_id: land.id } },
            body: draft,
          }),
        );
        if (current === ticket.current) setReview({ signature, summary: result });
      }
    } catch (e) {
      if (current === ticket.current)
        setError(
          e instanceof ApiError && e.fieldErrors.length
            ? e.fieldErrors.join(". ")
            : describeError(e),
        );
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  };
  const restore = async (raw: string) => {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (raw.length > 2_000_000) throw new Error("Survey files are limited to 2 MB.");
      const value = parseSurveyDraft(raw);
      const normalized: Draft = {
        ...value,
        requestKey: crypto.randomUUID(),
        boundaryRevision: land.revision,
        supersedesId: null,
      };
      await unwrap(
        api.POST("/api/v1/land/{land_id}/surveys/preview", {
          params: { path: { land_id: land.id } },
          body: normalized,
        }),
      );
      if (current === ticket.current) update(normalized);
    } catch (e) {
      if (current === ticket.current) setError(describeError(e));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  };
  const frame = (boundary: Footprint) => {
    const b = boundsOf(boundary);
    scene?.camera.flyToRectangle(b.west, b.south, b.east, b.north);
  };
  return (
    <section className="land-ecology" aria-label="Field ecology">
      <span className="land-eyebrow">Ground the picture in observations</span>
      <h3>Species and field surveys.</h3>
      <p>
        Map sampled plots, record the species you observed, and preserve the method behind each
        estimate.
      </p>
      <EcologyResearch
        land={land}
        surveyNames={(draft?.plots ?? selected.data?.plots ?? []).flatMap((plot) =>
          plot.observations.map((observation) => observation.taxon),
        )}
      />
      {error && (
        <p role="alert" className="land-error">
          {error}
        </p>
      )}
      {storageError && (
        <p role="alert">
          This draft could not be stored in this browser. Export a copy before leaving.
        </p>
      )}
      {recovery && !draft && (
        <div className="land-notice">
          <p>An unfinished survey is stored in this browser.</p>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              try {
                const value = parseSurveyDraft(recovery);
                setDraft(value);
                setRecovery(null);
              } catch {
                setError("The saved draft could not be read. Export it before discarding.");
              }
            }}
          >
            Resume survey
          </button>
          <button type="button" onClick={() => download(recovery, "survey-draft-recovery.json")}>
            Export draft
          </button>
          <button type="button" onClick={() => update(null)}>
            Discard draft
          </button>
        </div>
      )}
      {!draft && canEdit && !recovery && (
        <div className="land-actions">
          <button
            type="button"
            onClick={() => {
              useLandContext.getState().selectSurvey(null);
              update(fresh(land));
            }}
          >
            Record a field survey
          </button>
          <label>
            Import survey JSON
            <input
              type="file"
              accept=".json,application/json"
              disabled={busy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  if (file.size > 2_000_000) setError("Survey files are limited to 2 MB.");
                  else
                    void file
                      .text()
                      .then(restore)
                      .catch((cause: unknown) => setError(describeError(cause)));
                }
                e.target.value = "";
              }}
            />
          </label>
        </div>
      )}
      {draft && (
        <form
          className="land-survey-form"
          onSubmit={(e) => {
            e.preventDefault();
            void calculate(false);
          }}
        >
          <fieldset disabled={busy || !canEdit}>
            <label>
              Survey name
              <input
                required
                maxLength={200}
                value={draft.name}
                onChange={(e) => update({ ...draft, name: e.target.value })}
              />
            </label>
            <label>
              Observation date
              <input
                required
                type="date"
                max={new Date().toISOString().slice(0, 10)}
                value={draft.observedOn}
                onChange={(e) => update({ ...draft, observedOn: e.target.value })}
              />
            </label>
            <label>
              Observer
              <input
                required
                maxLength={200}
                value={draft.observer}
                onChange={(e) => update({ ...draft, observer: e.target.value })}
              />
            </label>
            <label>
              Method
              <select
                value={draft.method}
                onChange={(e) => {
                  const method = e.target.value as Draft["method"];
                  update({
                    ...draft,
                    method,
                    plots: draft.plots.map((p) => ({
                      ...p,
                      samplePoints: null,
                      observations: p.observations.map((o) => ({
                        ...o,
                        percentCover: null,
                        hits: null,
                      })),
                    })),
                  });
                }}
              >
                <option value="visual-cover">Visual percent cover</option>
                <option value="point-intercept">Point intercept</option>
              </select>
            </label>
            <p className="land-footnote">
              Changing the method resets numeric observations. For point intercept, count points
              with a taxon, not repeated contacts at the same point.
            </p>
            <label>
              Sampling design
              <select
                value={draft.design}
                onChange={(e) => update({ ...draft, design: e.target.value as Draft["design"] })}
              >
                <option value="purposive">Purposive plots</option>
                <option value="random">Random plots</option>
                <option value="systematic">Systematic plots</option>
                <option value="census">Whole-boundary census</option>
              </select>
            </label>
            <fieldset>
              <legend>Vegetation layers assessed</legend>
              {strata.map((s) => (
                <label key={s} className="land-check">
                  <input
                    type="checkbox"
                    checked={draft.assessedStrata.includes(s)}
                    onChange={(e) =>
                      update({
                        ...draft,
                        assessedStrata: e.target.checked
                          ? [...draft.assessedStrata, s]
                          : draft.assessedStrata.filter((v) => v !== s),
                      })
                    }
                  />
                  {s}
                </label>
              ))}
            </fieldset>
            <label>
              Survey method and limitations
              <textarea
                required
                rows={3}
                maxLength={5000}
                value={draft.methodNotes}
                onChange={(e) => update({ ...draft, methodNotes: e.target.value })}
              />
            </label>
            <p>
              Boundary revision {draft.boundaryRevision}.{" "}
              {draft.supersedesId && "This correction preserves the earlier survey."}
            </p>
            {draft.boundaryRevision !== land.revision && (
              <button
                type="button"
                onClick={() => update({ ...draft, boundaryRevision: land.revision })}
              >
                Check against current boundary
              </button>
            )}
            <h4>Surveyed plots</h4>
            <p className="land-footnote">
              Plots must be inside this land and must not overlap. Map only the area actually
              assessed.
            </p>
            <div className="land-actions">
              <button
                type="button"
                disabled={!scene || points.length >= 1000 || draft.plots.length >= 100}
                onClick={() => void pick()}
              >
                Pick next plot corner
              </button>
              <button
                type="button"
                disabled={!points.length}
                onClick={() => setPoints((p) => p.slice(0, -1))}
              >
                Undo corner
              </button>
              <button
                type="button"
                disabled={points.length < 3}
                onClick={() => {
                  const first = points[0];
                  if (first) addPlot({ type: "Polygon", coordinates: [[...points, first]] });
                  setPoints([]);
                  cancelPick();
                }}
              >
                Finish plot ({points.length} corners)
              </button>
              {picker === owner && (
                <button type="button" onClick={cancelPick}>
                  Cancel map pick
                </button>
              )}
            </div>
            <button
              type="button"
              disabled={draft.plots.length > 0}
              onClick={() => addPlot(land.boundary)}
            >
              Use entire land as the surveyed plot
            </button>
            {draft.plots.map((plot, index) => (
              <fieldset key={index} className="land-survey-plot">
                <legend>{plot.label || `Plot ${index + 1}`}</legend>
                <label>
                  Plot label
                  <input
                    required
                    maxLength={100}
                    value={plot.label}
                    onChange={(e) => updatePlot(index, { label: e.target.value })}
                  />
                </label>
                <div className="land-actions">
                  <button type="button" onClick={() => frame(plot.boundary)}>
                    Frame plot
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      update({ ...draft, plots: draft.plots.filter((_, i) => i !== index) })
                    }
                  >
                    Remove plot
                  </button>
                </div>
                {draft.method === "point-intercept" && (
                  <label>
                    Sampled points
                    <input
                      required
                      type="number"
                      min={1}
                      max={100000}
                      step={1}
                      value={plot.samplePoints ?? ""}
                      onChange={(e) =>
                        updatePlot(index, {
                          samplePoints: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                    />
                  </label>
                )}
                <label className="land-check">
                  <input
                    type="checkbox"
                    checked={plot.completeInventory ?? false}
                    onChange={(e) => updatePlot(index, { completeInventory: e.target.checked })}
                  />
                  Complete inventory for all assessed layers; unlisted taxa are recorded as not
                  detected
                </label>
                {plot.observations.map((o, i) => (
                  <div key={i} className="land-species-form">
                    <label>
                      Taxon or field identifier
                      <input
                        required
                        maxLength={200}
                        value={o.taxon}
                        onChange={(e) =>
                          updatePlot(index, {
                            observations: plot.observations.map((v, j) =>
                              j === i ? { ...v, taxon: e.target.value } : v,
                            ),
                          })
                        }
                      />
                    </label>
                    <label>
                      Vegetation layer
                      <select
                        value={o.stratum}
                        onChange={(e) =>
                          updatePlot(index, {
                            observations: plot.observations.map((v, j) =>
                              j === i ? { ...v, stratum: e.target.value as typeof o.stratum } : v,
                            ),
                          })
                        }
                      >
                        {strata.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Identification
                      <select
                        value={o.identification}
                        onChange={(e) =>
                          updatePlot(index, {
                            observations: plot.observations.map((v, j) =>
                              j === i
                                ? {
                                    ...v,
                                    identification: e.target.value as typeof o.identification,
                                  }
                                : v,
                            ),
                          })
                        }
                      >
                        <option value="tentative">Tentative identification</option>
                        <option value="verified">Verified by observer</option>
                        <option value="unidentified">Unidentified field taxon</option>
                      </select>
                    </label>
                    <label>
                      {draft.method === "visual-cover" ? "Cover (%)" : "Points with taxon"}
                      <input
                        required
                        type="number"
                        min={0}
                        max={draft.method === "visual-cover" ? 100 : (plot.samplePoints ?? 100000)}
                        step={draft.method === "visual-cover" ? "any" : 1}
                        value={
                          draft.method === "visual-cover" ? (o.percentCover ?? "") : (o.hits ?? "")
                        }
                        onChange={(e) =>
                          updatePlot(index, {
                            observations: plot.observations.map((v, j) =>
                              j === i
                                ? {
                                    ...v,
                                    [draft.method === "visual-cover" ? "percentCover" : "hits"]:
                                      e.target.value === "" ? null : Number(e.target.value),
                                  }
                                : v,
                            ),
                          })
                        }
                      />
                    </label>
                    <label>
                      Observation notes
                      <input
                        maxLength={1000}
                        value={o.notes ?? ""}
                        onChange={(e) =>
                          updatePlot(index, {
                            observations: plot.observations.map((v, j) =>
                              j === i ? { ...v, notes: e.target.value } : v,
                            ),
                          })
                        }
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() =>
                        updatePlot(index, {
                          observations: plot.observations.filter((_, j) => j !== i),
                        })
                      }
                    >
                      Remove observation
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  disabled={plot.observations.length >= 200}
                  onClick={() =>
                    updatePlot(index, {
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
                </button>
              </fieldset>
            ))}
            <div className="land-actions">
              <button type="submit" disabled={!draft.plots.length}>
                Review survey
              </button>
              <button type="button" onClick={() => download(draft, "survey-draft.json")}>
                Export draft
              </button>
              <button
                type="button"
                onClick={() => {
                  cancelPick();
                  setPoints([]);
                  update(null);
                }}
              >
                Discard survey draft
              </button>
            </div>
          </fieldset>
          {review?.signature === signature && (
            <>
              <SurveySummaryView summary={review.summary} />
              <button
                type="button"
                disabled={busy || !canEdit}
                onClick={() => void calculate(true)}
              >
                Save immutable survey
              </button>
            </>
          )}
        </form>
      )}
      {!draft && (
        <>
          <h4>Saved surveys</h4>
          {catalog.isError && (
            <p role="alert">
              Surveys could not be loaded.{" "}
              <button type="button" onClick={() => void catalog.refetch()}>
                Retry
              </button>
            </p>
          )}
          {catalog.data?.map((s) => (
            <button
              className="land-survey-choice"
              key={s.id}
              type="button"
              aria-pressed={selectedId === s.id}
              onClick={() => useLandContext.getState().selectSurvey(s.id)}
            >
              {s.name} · {s.observedOn}
              {s.stale ? " · older boundary" : ""}
              {s.supersedesId ? " · correction" : ""}
            </button>
          ))}
          {catalog.data?.length === 0 && <p>No field surveys have been saved.</p>}
          <div className="land-actions">
            <button
              type="button"
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 20))}
            >
              Previous surveys
            </button>
            <button
              type="button"
              disabled={(catalog.data?.length ?? 0) < 20}
              onClick={() => setOffset(offset + 20)}
            >
              More surveys
            </button>
          </div>
          {selected.isError && <p role="alert">The selected survey could not be loaded.</p>}
          {selected.data && (
            <article>
              <h4>{selected.data.name}</h4>
              <p>
                {selected.data.observedOn} · {selected.data.observer} · {selected.data.method} ·{" "}
                {selected.data.design}
              </p>
              <p>{selected.data.methodNotes}</p>
              {selected.data.stale && (
                <p className="land-notice">This survey uses an earlier boundary revision.</p>
              )}
              <SurveySummaryView summary={selected.data.summary} />
              <div className="land-actions">
                <button type="button" onClick={() => download(selected.data, "field-survey.json")}>
                  Export field survey
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const ctx = useLandContext.getState();
                    ctx.setResearchQuestion(
                      `Interpret field survey ${selected.data.id} (${selected.data.name}), including sampling and identification limitations, and suggest restoration questions.`,
                    );
                    ctx.setSection("discover");
                  }}
                >
                  Explore with agent
                </button>
                {canEdit && (
                  <button
                    type="button"
                    onClick={() =>
                      update({
                        ...parseSurveyDraft(JSON.stringify(selected.data)),
                        requestKey: crypto.randomUUID(),
                        supersedesId: selected.data.id,
                      })
                    }
                  >
                    Create corrected copy
                  </button>
                )}
              </div>
              <details>
                <summary>Source record</summary>
                <p>
                  Boundary revision {selected.data.boundaryRevision}. SHA-256{" "}
                  <code>{selected.data.sha256}</code>.
                </p>
                {selected.data.plots.map((p, i) => (
                  <div key={p.label}>
                    <strong>{p.label}</strong> · {selected.data.summary.plotAreasM2[i]?.toFixed(1)}{" "}
                    m²{" "}
                    <button type="button" onClick={() => frame(p.boundary)}>
                      Frame {p.label}
                    </button>
                    <p className="land-footnote">
                      {p.completeInventory
                        ? "Inventory marked complete"
                        : "Unlisted taxa remain unknown"}
                      {p.samplePoints != null ? ` · ${p.samplePoints} sampled points` : ""}.
                    </p>
                    <ul>
                      {p.observations.map((o) => (
                        <li key={`${o.taxon}:${o.stratum}`}>
                          {o.taxon} · {o.stratum} ·{" "}
                          {o.percentCover != null
                            ? `${o.percentCover}% cover`
                            : `${o.hits} points with taxon`}{" "}
                          · {o.identification}
                          {o.notes ? ` — ${o.notes}` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </details>
            </article>
          )}
        </>
      )}
    </section>
  );
}
