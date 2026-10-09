import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, BookOpen, Pin, Search, Square, X } from "lucide-react";
import type {
  components,
  LandArea,
  LandEvidence,
  LandFinding,
  ResearchEvent,
} from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { boundsOf } from "@twin/geo";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandScope, useLandCanEdit } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import {
  beginInvestigation,
  cancelResearch,
  setFindingDisposition,
  startResearch,
  useInvestigation,
  useInvestigations,
  useResearchStatus,
} from "./researchApi";
import { ResearchArtifactView } from "./ResearchArtifacts";
import { LandDocumentViewer } from "./LandDocumentViewer";
import { VegetationStart } from "./VegetationStart";
import "./research.css";

function eventText(event: ResearchEvent): string {
  const value =
    event.payload.message ?? event.payload.title ?? event.payload.provider ?? event.payload.query;
  return typeof value === "string" ? value : "";
}

export function EvidenceView({
  evidence,
  onClose,
}: {
  evidence: LandEvidence;
  onClose: () => void;
}) {
  return (
    <section className="land-evidence" aria-label="Source evidence">
      <div className="land-place-heading">
        <span className="land-eyebrow">Source evidence</span>
        <button type="button" aria-label="Close source evidence" onClick={onClose}>
          <X size={16} />
        </button>
      </div>
      <h4>{evidence.title}</h4>
      <p>{evidence.relevanceNote}</p>
      <dl>
        <dt>Spatial match</dt>
        <dd>{evidence.spatialRelevance}</dd>
        <dt>Retrieved</dt>
        <dd>{new Date(evidence.retrievedAt).toLocaleDateString()}</dd>
        {evidence.observedAt && (
          <>
            <dt>Observed</dt>
            <dd>{new Date(evidence.observedAt).toLocaleDateString()}</dd>
          </>
        )}
        <dt>License</dt>
        <dd>{evidence.license}</dd>
      </dl>
      <blockquote>{evidence.excerpt}</blockquote>
      <p className="land-footnote">{evidence.attribution}</p>
      {evidence.document && (
        <LandDocumentViewer
          key={`${evidence.document.documentId}:${evidence.document.page}`}
          landId={evidence.document.landId}
          documentId={evidence.document.documentId}
          initialPage={evidence.document.page}
          pinnedHash={evidence.document.sha256}
          pinnedOcrId={evidence.document.ocrId ?? undefined}
        />
      )}
      {evidence.survey && (
        <button
          type="button"
          onClick={() => {
            const context = useLandContext.getState();
            context.selectSurvey(evidence.survey?.surveyId ?? null);
            context.setSection("ecology");
          }}
        >
          Open cited field survey
        </button>
      )}
      {evidence.inventory && (
        <div className="land-actions">
          <p>Saved asset revision {evidence.inventory.revision} · {evidence.inventory.section}</p>
          <button
            type="button"
            onClick={() => useLandContext.getState().selectInventory(evidence.inventory?.featureId ?? null)}
          >
            Open current asset
          </button>
        </div>
      )}
      {evidence.url && (
        <a href={evidence.url} target="_blank" rel="noopener noreferrer">
          Open original source <ArrowUpRight size={14} />
        </a>
      )}
    </section>
  );
}

export function LandResearch({ land }: { land: LandArea }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const scene = useScene();
  const catalog = useInvestigations(land.id);
  const status = useResearchStatus();
  const chosen = useLandContext((s) => s.selectedInvestigationId);
  const setChosen = useLandContext((s) => s.selectInvestigation);
  const selected =
    chosen ??
    catalog.data?.find((item) => item.boundaryRevision === land.revision)?.id ??
    catalog.data?.[0]?.id ??
    null;
  const [offset, setOffset] = useState(0);
  const detail = useInvestigation(selected, offset);
  const [tab, setTab] = useState<"overview" | "conversation" | "visuals">("overview");
  const question = useLandContext((state) => state.researchQuestion);
  const setQuestion = useLandContext((state) => state.setResearchQuestion);
  const questionInput = useRef<HTMLTextAreaElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [evidence, setEvidence] = useState<LandEvidence | null>(null);
  const [showDismissed, setShowDismissed] = useState(false);
  const [newTopic, setNewTopic] = useState(false);
  const [budget, setBudget] = useState<components["schemas"]["ResearchBudget"]>({
    maxSteps: 16,
    maxSeconds: 180,
    maxOutputTokens: 12000,
    maxWebSearches: 6,
  });
  const pending = useRef<{
    investigationId: string;
    question: string;
    key: string;
    budget: components["schemas"]["ResearchBudget"];
  } | null>(null);
  const pendingTerrain = useRef<{
    investigationId: string;
    key: string;
    revision: number;
    signature: string;
  } | null>(null);
  const pendingArchive = useRef<{ investigationId: string; key: string; revision: number } | null>(
    null,
  );
  const lastRun = detail.data?.runs.at(-1);
  const running = lastRun?.status === "queued" || lastRun?.status === "running";
  const progress = useQuery({
    queryKey: ["land-research", scope, "events", lastRun?.id],
    enabled: Boolean(lastRun) && ready,
    queryFn: async () => {
      const events: ResearchEvent[] = [];
      for (let page = 0; page < 20; page++) {
        const batch = await unwrap(
          api.GET("/api/v1/research/runs/{run_id}/events", {
            params: {
              path: { run_id: lastRun?.id ?? "" },
              query: { after: events.at(-1)?.sequence ?? 0 },
            },
          }),
        );
        events.push(...batch);
        if (batch.length < 200) break;
      }
      return events;
    },
    retry: false,
    refetchInterval: running ? 1500 : false,
  });

  useEffect(() => {
    if (!lastRun?.id || (lastRun?.status !== "succeeded" && lastRun?.status !== "partial")) return;
    void cache.invalidateQueries({ queryKey: ["land-scenarios", scope, land.id] });
    void cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] });
  }, [lastRun?.id, lastRun?.status, cache, scope, land.id]);
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-research", scope] });
  useEffect(() => {
    if (!ready || !canEdit || !catalog.isSuccess || catalog.data.length > 0) return;
    let current = true;
    void unwrap(
      api.POST("/api/v1/land/{land_id}/overview", {
        params: { path: { land_id: land.id } },
        body: { boundaryRevision: land.revision },
      }),
    )
      .then((result) => {
        if (current) {
          setChosen(result.investigation.id);
          void cache.invalidateQueries({ queryKey: ["land-research", scope, "list", land.id] });
        }
      })
      .catch((cause: unknown) => {
        if (current) setError(describeError(cause));
      });
    return () => {
      current = false;
    };
  }, [
    ready,
    canEdit,
    catalog.isSuccess,
    catalog.data,
    cache,
    scope,
    land.id,
    land.revision,
    setChosen,
  ]);

  const startCurrentOverview = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await unwrap(
        api.POST("/api/v1/land/{land_id}/overview", {
          params: { path: { land_id: land.id } },
          body: { boundaryRevision: land.revision },
        }),
      );
      setChosen(result.investigation.id);
      setOffset(0);
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const analyzeRaster = async (
    dataset: "cop-dem-glo-30" | "esa-worldcover-2021" | "sentinel-2-ndvi",
    periods?: components["schemas"]["VegetationPeriod"][],
  ) => {
    if (!canEdit) return;
    const analysis: components["schemas"]["RasterRequest"] = {
      dataset,
      resolutionM: dataset === "esa-worldcover-2021" ? 10 : dataset === "sentinel-2-ndvi" ? 20 : 30,
      maxDimension: 512,
      ...(periods ? { periods } : {}),
    };
    const signature = JSON.stringify(analysis);
    const title =
      dataset === "sentinel-2-ndvi"
        ? "Vegetation through time"
        : dataset === "esa-worldcover-2021"
          ? "Land cover in 2021"
          : "Surface elevation and slope";
    setBusy(true);
    setError(null);
    try {
      let operation = pendingTerrain.current;
      if (operation?.revision !== land.revision || operation.signature !== signature) {
        const investigation = await beginInvestigation(land, title);
        operation = {
          investigationId: investigation.id,
          key: crypto.randomUUID(),
          revision: land.revision,
          signature,
        };
        pendingTerrain.current = operation;
      }
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: operation.investigationId } },
          body: {
            kind: "raster",
            question:
              dataset === "sentinel-2-ndvi"
                ? "Compare dated vegetation signals on common clear cells inside this land."
                : dataset === "esa-worldcover-2021"
                  ? "Analyze the broad land-cover classes mapped in 2021 inside this land."
                  : "Analyze surface elevation and slope inside this land.",
            requestKey: operation.key,
            analysis,
          },
        }),
      );
      setChosen(operation.investigationId);
      setOffset(0);
      setTab("visuals");
      pendingTerrain.current = null;
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const discoverArchives = async () => {
    if (!canEdit) return;
    setBusy(true);
    setError(null);
    try {
      let operation = pendingArchive.current;
      if (operation?.revision !== land.revision) {
        const investigation = await beginInvestigation(land, "Photographs and historical maps");
        operation = {
          investigationId: investigation.id,
          key: crypto.randomUUID(),
          revision: land.revision,
        };
        pendingArchive.current = operation;
      }
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: operation.investigationId } },
          body: {
            kind: "archive",
            question:
              "Discover openly licensed photographs and historical map sheets relevant to this land.",
            requestKey: operation.key,
          },
        }),
      );
      setChosen(operation.investigationId);
      setOffset(0);
      setTab("visuals");
      pendingArchive.current = null;
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const ask = async () => {
    if (!canEdit || !question.trim()) return;
    setBusy(true);
    setError(null);
    try {
      let operation = pending.current;
      if (
        operation?.question !== question ||
        (selected && operation.investigationId !== selected && !newTopic)
      ) {
        const id = selected && !newTopic ? selected : (await beginInvestigation(land, question)).id;
        operation = { investigationId: id, question, key: crypto.randomUUID(), budget };
        pending.current = operation;
      }
      await startResearch(
        operation.investigationId,
        operation.question,
        "investigation",
        operation.key,
        operation.budget,
      );
      setChosen(operation.investigationId);
      setQuestion("");
      setNewTopic(false);
      setOffset(0);
      setTab("conversation");
      pending.current = null;
      await refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const viewEvidence = async (id: string | undefined) => {
    if (!id) return;
    setError(null);
    const loaded = detail.data?.evidence.find((item) => item.id === id);
    if (loaded) {
      setEvidence(loaded);
      return;
    }
    try {
      setEvidence(
        await unwrap(
          api.GET("/api/v1/research/evidence/{evidence_id}", {
            params: { path: { evidence_id: id } },
          }),
        ),
      );
    } catch (cause) {
      setError(describeError(cause));
    }
  };
  const updateFinding = (finding: LandFinding, value: "visible" | "pinned" | "dismissed") => {
    if (!canEdit) return;
    void setFindingDisposition(finding.id, value)
      .then(refresh)
      .catch((cause: unknown) => setError(describeError(cause)));
  };
  const findings =
    detail.data?.findings
      .filter((finding) => showDismissed || finding.disposition !== "dismissed")
      .sort((a, b) => Number(b.disposition === "pinned") - Number(a.disposition === "pinned")) ??
    [];
  const maxTotal = Math.max(0, ...Object.values(detail.data?.page.totals ?? {}));
  return (
    <section className="land-research" aria-label="Explore your land">
      <div className="land-research-heading">
        <div>
          <span className="land-eyebrow">Look closer</span>
          <h3>Explore this land</h3>
        </div>
        <BookOpen size={24} />
      </div>
      <p>
        Explore the evidence, uncover a story, or investigate an idea. Findings stay connected to
        this land.
      </p>
      <div className="land-terrain-start">
        <strong>Read the terrain</strong>
        <p className="land-footnote">
          Map surface elevation and slope from Copernicus GLO-30. Buildings and vegetation can
          affect this model; small plots may be below its resolution.
        </p>
        <button
          type="button"
          disabled={busy || running || !ready || !canEdit}
          onClick={() => void analyzeRaster("cop-dem-glo-30")}
        >
          {busy ? "Starting…" : "Analyze terrain"}
        </button>
      </div>
      <div className="land-terrain-start">
        <strong>What covers this land?</strong>
        <p className="land-footnote">
          Explore tree cover, grassland, water and other broad classes mapped in 2021. Use these as
          a starting point for field observations and restoration questions.
        </p>
        <button
          type="button"
          disabled={busy || running || !ready || !canEdit}
          onClick={() => void analyzeRaster("esa-worldcover-2021")}
        >
          {busy ? "Starting…" : "Analyze land cover"}
        </button>
      </div>
      <VegetationStart
        disabled={busy || running || !ready || !canEdit}
        busy={busy}
        onStart={(periods) => void analyzeRaster("sentinel-2-ndvi", periods)}
      />
      <div className="land-terrain-start">
        <strong>A place with a past</strong>
        <p className="land-footnote">
          Find openly licensed photographs nearby and historical USGS map sheets. Every source keeps
          its dates, creator, reuse terms and location limitations.
        </p>
        <button
          type="button"
          disabled={busy || running || !ready || !canEdit}
          onClick={() => void discoverArchives()}
        >
          {busy ? "Starting…" : "Discover photos and maps"}
        </button>
      </div>
      {catalog.data && catalog.data.length > 0 && (
        <label className="land-name">
          Investigation
          <select
            value={selected ?? ""}
            onChange={(event) => {
              setChosen(event.target.value);
              setOffset(0);
              setEvidence(null);
              pending.current = null;
            }}
          >
            {catalog.data.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title} · boundary {item.boundaryRevision}
                {item.stale ? " · older boundary" : ""}
              </option>
            ))}
          </select>
        </label>
      )}
      <nav className="land-research-tabs" aria-label="Land research views">
        {(["overview", "conversation", "visuals"] as const).map((value) => (
          <button
            type="button"
            key={value}
            aria-pressed={tab === value}
            onClick={() => setTab(value)}
          >
            {value === "overview"
              ? "Discoveries"
              : value === "conversation"
                ? "Investigation"
                : "Visuals"}
          </button>
        ))}
      </nav>
      {detail.data?.investigation.stale && (
        <div className="land-notice">
          These results use boundary revision {detail.data.investigation.boundaryRevision}. Your
          current boundary is revision {land.revision}.{" "}
          <button
            type="button"
            disabled={busy || !canEdit}
            onClick={() => void startCurrentOverview()}
          >
            Explore the current boundary
          </button>
        </div>
      )}
      {(error !== null || catalog.isError || detail.isError) && (
        <div className="land-error" role="alert">
          {error ?? "Research could not be loaded. Your saved land is retained."}
          <button
            type="button"
            onClick={() => {
              setError(null);
              void refresh();
            }}
          >
            Retry loading
          </button>
        </div>
      )}
      {lastRun && (
        <div className="land-run" role="status">
          <span className={`land-run-dot ${running ? "is-running" : ""}`} />
          <span>
            {lastRun.status === "queued"
              ? "Research is queued"
              : lastRun.status === "running"
                ? "Investigating your land"
                : lastRun.status === "partial"
                  ? "Partial results available"
                  : lastRun.status === "failed"
                    ? lastRun.error
                    : lastRun.status === "cancelled"
                      ? "Research stopped; completed results retained"
                      : "Research complete"}
          </span>
          {running && (
            <button
              type="button"
              aria-label="Stop research"
              onClick={() =>
                void cancelResearch(lastRun.id)
                  .then(refresh)
                  .catch((cause: unknown) => setError(describeError(cause)))
              }
            >
              <Square size={13} /> Stop
            </button>
          )}
        </div>
      )}
      {progress.data && progress.data.length > 0 && (
        <details className="land-progress">
          <summary>Research activity</summary>
          <ol>
            {progress.data
              .filter((event) => event.kind !== "progress" || event.payload.message)
              .map((event) => (
                <li key={event.sequence}>
                  <span>{event.kind}</span> {eventText(event)}
                </li>
              ))}
          </ol>
        </details>
      )}
      {tab === "overview" && (
        <>
          {findings.map((finding) => (
            <article
              key={finding.id}
              className={`land-finding ${finding.disposition === "pinned" ? "is-pinned" : ""}`}
            >
              <div className="land-finding-meta">
                <span>{finding.category}</span>
                <span>{finding.confidence}</span>
              </div>
              <h4>{finding.title}</h4>
              <p>{finding.summary}</p>
              <details>
                <summary>How certain is this?</summary>
                <p>{finding.uncertainty}</p>
              </details>
              <div className="land-actions">
                <button type="button" onClick={() => void viewEvidence(finding.evidenceIds[0])}>
                  <BookOpen size={14} />
                  Evidence ({finding.evidenceIds.length})
                </button>
                <button
                  type="button"
                  aria-pressed={finding.disposition === "pinned"}
                  onClick={() =>
                    updateFinding(finding, finding.disposition === "pinned" ? "visible" : "pinned")
                  }
                >
                  <Pin size={13} />
                  {finding.disposition === "pinned" ? "Unpin" : "Pin"}
                </button>
                <button
                  type="button"
                  onClick={() =>
                    updateFinding(
                      finding,
                      finding.disposition === "dismissed" ? "visible" : "dismissed",
                    )
                  }
                >
                  {finding.disposition === "dismissed" ? "Restore" : "Dismiss"}
                </button>
                {finding.boundary && (
                  <button
                    type="button"
                    onClick={() => {
                      if (finding.boundary) {
                        const b = boundsOf(finding.boundary);
                        scene?.camera.flyToRectangle(b.west, b.south, b.east, b.north);
                      }
                    }}
                  >
                    Show on map
                  </button>
                )}
              </div>
              {finding.evidenceIds.length > 1 && (
                <details>
                  <summary>All supporting records</summary>
                  {finding.evidenceIds.map((id, i) => (
                    <button
                      className="land-evidence-link"
                      type="button"
                      key={id}
                      onClick={() => void viewEvidence(id)}
                    >
                      Source {i + 1}
                    </button>
                  ))}
                </details>
              )}
              {finding.suggestedQuestions?.map((prompt) => (
                <button
                  type="button"
                  key={prompt}
                  className="land-question"
                  onClick={() => {
                    setQuestion(prompt);
                    setTab("conversation");
                  }}
                >
                  {prompt}
                  <ArrowUpRight size={14} />
                </button>
              ))}
            </article>
          ))}
          {!findings.length && (
            <p className="land-footnote">
              {running
                ? "Findings will appear as sources are checked."
                : "No findings on this page yet. Review source coverage or start an investigation."}
            </p>
          )}
          <label className="land-dismissed">
            <input
              type="checkbox"
              checked={showDismissed}
              onChange={(event) => setShowDismissed(event.target.checked)}
            />{" "}
            Show dismissed findings
          </label>
        </>
      )}
      {tab === "conversation" && (
        <div className="land-conversation">
          {detail.data?.messages.map((message) => (
            <article key={message.id} data-role={message.role}>
              <span>{message.role === "user" ? "You" : "Land research"}</span>
              <p>
                {message.content
                  .split(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi)
                  .map((part, index) =>
                    index % 2 === 1 ? (
                      <button
                        key={`${index}:${part}`}
                        type="button"
                        className="land-evidence-link"
                        onClick={() => void viewEvidence(part)}
                      >
                        Source {Math.ceil(index / 2)}
                      </button>
                    ) : (
                      part
                    ),
                  )}
              </p>
            </article>
          ))}
        </div>
      )}
      {tab === "visuals" && (
        <>
          {detail.data?.artifacts.map((artifact) => (
            <div key={artifact.id}>
              <ResearchArtifactView
                artifact={artifact}
                evidence={detail.data.evidence}
                onAsk={() => {
                  setTab("conversation");
                  requestAnimationFrame(() => questionInput.current?.focus());
                }}
              />
              <div className="land-actions">
                <button type="button" onClick={() => void viewEvidence(artifact.evidenceIds[0])}>
                  Inspect source evidence
                </button>
              </div>
            </div>
          ))}
          {!detail.data?.artifacts.length && (
            <p className="land-footnote">
              Charts, tables, maps and comparisons will collect here as the investigation produces
              them.
            </p>
          )}
        </>
      )}
      {evidence && <EvidenceView evidence={evidence} onClose={() => setEvidence(null)} />}
      {maxTotal > 100 && (
        <div className="land-actions">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - 100))}
          >
            Earlier results
          </button>
          <span>Page {offset / 100 + 1}</span>
          <button
            type="button"
            disabled={offset + 100 >= maxTotal}
            onClick={() => setOffset(offset + 100)}
          >
            More results
          </button>
        </div>
      )}
      {status.data && !status.data.modelConfigured && (
        <p className="land-notice">
          AI investigation is not configured on this server. The open-data overview can still
          produce sourced findings and charts.
        </p>
      )}
      <form
        className="land-ask"
        onSubmit={(event) => {
          event.preventDefault();
          void ask();
        }}
      >
        <label htmlFor="land-question">Follow your curiosity</label>
        <textarea
          id="land-question"
          ref={questionInput}
          placeholder="What has changed here? What could this land become?"
          value={question}
          maxLength={10000}
          onChange={(event) => setQuestion(event.target.value)}
          rows={3}
        />
        <details className="land-budget">
          <summary>Research limits</summary>
          <label className="land-name">
            Time budget
            <select
              value={budget.maxSeconds}
              onChange={(event) => setBudget({ ...budget, maxSeconds: Number(event.target.value) })}
            >
              <option value={60}>1 minute</option>
              <option value={180}>3 minutes</option>
              <option value={600}>10 minutes</option>
            </select>
          </label>
          <label className="land-name">
            Public web searches
            <select
              value={budget.maxWebSearches}
              onChange={(event) =>
                setBudget({ ...budget, maxWebSearches: Number(event.target.value) })
              }
            >
              <option value={0}>Registered data sources only</option>
              <option value={6}>Up to 6 searches</option>
              <option value={12}>Up to 12 searches</option>
            </select>
          </label>
          <p className="land-footnote">
            Search may find useful leads outside the registered datasets. Each source still needs a
            location match and appropriate reuse rights.
          </p>
        </details>
        <div>
          <label>
            <input
              type="checkbox"
              checked={newTopic}
              onChange={(event) => setNewTopic(event.target.checked)}
            />{" "}
            Start a separate investigation
          </label>
          <button
            type="submit"
            disabled={
              !canEdit || busy || running || !question.trim() || !status.data?.modelConfigured
            }
          >
            <Search size={15} />
            {busy ? "Starting…" : "Investigate"}
          </button>
        </div>
      </form>
    </section>
  );
}
