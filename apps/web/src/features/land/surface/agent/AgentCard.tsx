import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import { RotateCcw, Square, X } from "lucide-react";

import type { LandArea, LandEvidence, LandFinding, ResearchEvent } from "@twin/contracts";
import { GlassButton, GlassPanel } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";
import { useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { EvidenceView } from "../../EvidenceView";
import {
  cancelResearch,
  setFindingDisposition,
  useInvestigation,
  useInvestigations,
  useResearchStatus,
} from "../../researchApi";
import { useResearchProgress } from "../../useResearchProgress";
import { useAgentBus } from "../agentBus";
import { useLeader } from "../mapHooks";
import {
  citationParts,
  findingLocations,
  findingPlace,
  runProgress,
  splitAnswer,
  stepsFromEvents,
} from "./agentEvents";
import { AgentAskForm } from "./AgentAskForm";
import { AgentDetails, EarlierQuestions, ResearchLimits } from "./AgentDetails";
import { AgentFindings } from "./AgentFindings";
import { AgentSelection } from "./AgentSelection";
import { AgentSteps } from "./AgentSteps";
import { isAgentInvestigation, useAgentUi } from "./agentStore";
import { useAgentFindingsOnMap, useAgentLookingOnMap } from "./useAgentMap";
import { useAgentAsk } from "./useAgentAsk";
import "./agent.css";

const EMPTY_EVIDENCE: LandEvidence[] = [];
const EMPTY_EVENTS: ResearchEvent[] = [];

/**
 * The agent card. Idle, it is one line: "Ask about this land…". Sent, it shows the question,
 * a progress bar and the steps, with a line from the step it is on to where on the map it is
 * looking; findings drop onto the map as they arrive. Done, it gives a short answer, the
 * findings (each tied to its place), follow-up questions and "New question". It runs only
 * when the person sends a question.
 */
export function AgentCard({
  land,
  variant = "dock",
}: {
  land: LandArea;
  /** "inline" is the narrow-screen sheet: fewer steps, the same card. */
  variant?: "dock" | "inline";
}) {
  const scope = useLandScope();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const status = useResearchStatus();
  const list = useInvestigations(land.id);
  const chosen = useLandContext((state) => state.selectedInvestigationId);
  const selectInvestigation = useLandContext((state) => state.selectInvestigation);
  const agentList = useMemo(
    () =>
      (list.data ?? [])
        .filter(isAgentInvestigation)
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    [list.data],
  );
  // A chosen analysis (a lens's terrain run, a saved view's) is not the agent's to show.
  const shownId =
    chosen &&
    (agentList.some((item) => item.id === chosen) ||
      !(list.data ?? []).some((item) => item.id === chosen))
      ? chosen
      : null;

  // Coming back to land whose last question is still being worked on shows that work, once.
  const latest = agentList[0] ?? null;
  const autoShown = useAgentUi((state) => state.autoShown);
  const probe = useInvestigation(
    !shownId && latest && autoShown !== land.id ? latest.id : null,
    0,
    1,
    3000,
  );
  useEffect(() => {
    const run = probe.data?.runs.at(-1);
    if (!latest || shownId || autoShown === land.id || probe.data?.investigation.id !== latest.id)
      return;
    useAgentUi.getState().setAutoShown(land.id);
    if (
      (run?.status === "queued" || run?.status === "running") &&
      useAgentUi.getState().closed !== latest.id
    )
      selectInvestigation(latest.id);
  }, [probe.data, latest, shownId, land.id, autoShown, selectInvestigation]);

  const [offset, setOffset] = useState(0);
  const detail = useInvestigation(shownId, offset, 100, 1500);
  const data = detail.data?.investigation.id === shownId ? detail.data : null;
  const run = data?.runs.at(-1) ?? null;
  const running = run?.status === "queued" || run?.status === "running";
  const done = Boolean(run) && !running;
  const progress = useResearchProgress(run?.id, running);
  const events = progress.data ?? EMPTY_EVENTS;
  const steps = useMemo(() => stepsFromEvents(events, running), [events, running]);
  const places = useMemo(() => findingLocations(events), [events]);
  const evidence = data?.evidence ?? EMPTY_EVIDENCE;
  const runFindings = useMemo(
    () => (data?.findings ?? []).filter((finding) => finding.runId === run?.id),
    [data?.findings, run?.id],
  );
  const visibleFindings = useMemo(
    () => runFindings.filter((finding) => finding.disposition !== "dismissed"),
    [runFindings],
  );
  const active = running ? steps.at(-1) : undefined;
  useAgentFindingsOnMap(visibleFindings, places, evidence);
  useAgentLookingOnMap(active?.location ?? null, evidence);

  // Tick while running so the bar follows the time budget between events.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 2000);
    return () => clearInterval(timer);
  }, [running]);

  // A finished run changes what other lenses show (scenarios, plans built from findings).
  useEffect(() => {
    if (!run?.id || (run.status !== "succeeded" && run.status !== "partial")) return;
    void cache.invalidateQueries({ queryKey: ["land-scenarios", scope, land.id] });
    void cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] });
  }, [run?.id, run?.status, cache, scope, land.id]);

  const followUp = done ? shownId : null;
  const ask = useAgentAsk(land, followUp);

  // "Ask about this" from any lens: the question waits in the box; the person sends it.
  const request = useAgentBus((state) => state.request);
  useEffect(() => {
    if (!request) return;
    const context = useLandContext.getState();
    context.setResearchQuestion(request.question);
    context.setResearchFocus(
      request.focus
        ? {
            artifactId: request.focus.artifactId,
            featureIndex: request.focus.featureIndex,
            label: request.focusLabel ?? "the selected map feature",
          }
        : null,
    );
    context.selectInvestigation(null);
    useAgentUi.getState().focusInput();
    useAgentBus.getState().consume(request.id);
  }, [request]);

  const [shownEvidence, setShownEvidence] = useState<LandEvidence | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-research", scope] });
  const viewEvidence = async (id: string) => {
    setError(null);
    const loaded = evidence.find((item) => item.id === id);
    if (loaded) {
      setShownEvidence(loaded);
      return;
    }
    try {
      setShownEvidence(
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
  const updateFinding = (finding: LandFinding, value: LandFinding["disposition"]) => {
    if (!canEdit) return;
    void setFindingDisposition(finding.id, value)
      .then(refresh)
      .catch((cause: unknown) => setError(describeError(cause)));
  };
  const choose = (id: string | null) => {
    setOffset(0);
    setShownEvidence(null);
    useAgentUi.getState().setHighlight(null);
    selectInvestigation(id);
  };
  const close = () => {
    if (running && shownId) useAgentUi.getState().setClosed(shownId);
    choose(null);
  };
  const askAgain = (question: string) => {
    ask.setQuestion(question);
    choose(null);
    useAgentUi.getState().focusInput();
  };

  const answer = useMemo(() => {
    if (!done || !data) return null;
    const last = data.messages.at(-1);
    return last?.role === "assistant" ? splitAnswer(last.content) : null;
  }, [done, data]);
  const followUps = useMemo(
    () =>
      [...new Set(runFindings.flatMap((finding) => finding.suggestedQuestions ?? []))].slice(0, 3),
    [runFindings],
  );
  const lead = visibleFindings
    .map((finding) => findingPlace(finding, places, evidence))
    .find(Boolean);
  const answerRef = useRef<HTMLDivElement>(null);
  useLeader(answerRef, lead?.anchor, "agent", done && Boolean(answer));

  const history = (
    <EarlierQuestions
      items={agentList}
      current={shownId}
      revision={land.revision}
      onChoose={(id) => {
        useAgentUi.getState().setClosed(null);
        choose(id);
      }}
    />
  );
  const limits = <ResearchLimits budget={ask.budget} onChange={ask.setBudget} />;
  const modelConfigured = status.data?.modelConfigured;
  const inline = variant === "inline";

  return (
    <GlassPanel
      strong
      className={clsx(
        "land-card land-card--scroll land-agent",
        running && "is-running",
        inline && "land-agent--inline",
      )}
      role="region"
      aria-label="Agent"
      data-testid="land-agent"
    >
      <div className="land-eyebrow-row">
        <span className="land-agent__who">
          <span className="land-agent__pulse" aria-hidden="true" />
          Agent
        </span>
        {running && canEdit && run ? (
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<Square size={12} aria-hidden="true" />}
            onClick={() =>
              void cancelResearch(run.id)
                .then(refresh)
                .catch((cause: unknown) => setError(describeError(cause)))
            }
          >
            Cancel
          </GlassButton>
        ) : shownId ? (
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label="Close this answer"
            onClick={close}
          >
            <X size={15} aria-hidden="true" />
          </GlassButton>
        ) : null}
      </div>

      <AgentSelection land={land} canAsk={ask.canAsk && modelConfigured !== false} />

      {shownId ? (
        <>
          {!data && detail.isPending && <p className="land-muted">Loading the agent's work…</p>}
          {detail.isError && (
            <p className="land-alert" role="alert">
              The agent's work couldn't be loaded.{" "}
              <button type="button" className="land-link" onClick={() => void detail.refetch()}>
                Try again
              </button>
            </p>
          )}
          {data && (
            <>
              <p className="land-agent__question">{run?.question ?? data.investigation.question}</p>
              {run?.focusLabel && <p className="land-muted">About {run.focusLabel}</p>}
              {running && (
                <>
                  <div
                    className="land-agent__progress"
                    role="progressbar"
                    aria-label="Time and steps used"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(runProgress(run, events, now) * 100)}
                  >
                    <i style={{ width: `${runProgress(run, events, now) * 100}%` }} />
                  </div>
                  <AgentSteps steps={steps} evidence={evidence} limit={inline ? 2 : undefined} />
                  <p className="land-visually-hidden" aria-live="polite">
                    {active?.text ?? "Working"}
                  </p>
                  {progress.phase === "unavailable" && (
                    <p className="land-alert" role="alert">
                      {progress.message}{" "}
                      <button type="button" className="land-link" onClick={progress.retry}>
                        Retry updates
                      </button>
                    </p>
                  )}
                  {progress.phase === "reconnecting" && (
                    <p className="land-muted">{progress.message}</p>
                  )}
                </>
              )}
              {done && run?.status === "failed" && (
                <p className="land-alert" role="alert">
                  {run.error ?? "The agent couldn't finish."}
                </p>
              )}
              {done && run?.status === "cancelled" && (
                <p className="land-muted">Stopped. What it found so far is kept.</p>
              )}
              {done && answer && run?.status !== "failed" && (
                <div ref={answerRef} className="land-agent-answer">
                  <Answer text={answer.text} onEvidence={(id) => void viewEvidence(id)} />
                  {answer.evidenceIds.length > 0 && (
                    <p className="land-row land-muted">
                      Sources
                      {answer.evidenceIds.map((id, index) => (
                        <button
                          key={id}
                          type="button"
                          className="land-link"
                          onClick={() => void viewEvidence(id)}
                        >
                          {index + 1}
                        </button>
                      ))}
                    </p>
                  )}
                  {run?.status === "partial" && (
                    <p className="land-muted">
                      It ran out of time or steps; this is what it found.
                    </p>
                  )}
                </div>
              )}
              {data.investigation.stale && (
                <p className="land-note">Asked about an earlier outline of this land.</p>
              )}
              {done && (
                <AgentFindings
                  findings={visibleFindings}
                  places={places}
                  evidence={evidence}
                  canEdit={canEdit}
                  onEvidence={(id) => void viewEvidence(id)}
                  onDisposition={updateFinding}
                />
              )}
              {running && visibleFindings.length > 0 && (
                <p className="land-muted">
                  {visibleFindings.length === 1
                    ? "1 finding so far, pinned on the map."
                    : `${visibleFindings.length} findings so far, pinned on the map.`}
                </p>
              )}
              {done && followUps.length > 0 && canEdit && modelConfigured && (
                <div className="land-agent-chips" role="group" aria-label="Follow-up questions">
                  {followUps.map((question) => (
                    <button
                      key={question}
                      type="button"
                      className="land-agent-chip"
                      onClick={() => {
                        ask.setQuestion(question);
                        useAgentUi.getState().focusInput();
                      }}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              )}
              {shownEvidence && (
                <EvidenceView evidence={shownEvidence} onClose={() => setShownEvidence(null)} />
              )}
              {done && canEdit && (
                <div className="land-row">
                  <GlassButton
                    size="sm"
                    onClick={() => {
                      choose(null);
                      useAgentUi.getState().focusInput();
                    }}
                  >
                    New question
                  </GlassButton>
                  {(run?.status === "failed" || data.investigation.stale) && run && (
                    <GlassButton
                      size="sm"
                      variant="ghost"
                      leadingIcon={<RotateCcw size={13} aria-hidden="true" />}
                      onClick={() => askAgain(run.question)}
                    >
                      Ask again
                    </GlassButton>
                  )}
                </div>
              )}
              {done && (
                <AgentAskForm
                  land={land}
                  ask={ask}
                  placeholder="Ask a follow-up…"
                  modelConfigured={modelConfigured}
                  onViewPending={(id) => choose(id)}
                />
              )}
              {error && (
                <p className="land-alert" role="alert">
                  {error}
                </p>
              )}
              <AgentDetails
                detail={data}
                steps={done ? steps : []}
                runId={run?.id ?? null}
                places={places}
                canEdit={canEdit}
                onEvidence={(id) => void viewEvidence(id)}
                onDisposition={updateFinding}
                offset={offset}
                onOffset={setOffset}
                history={history}
                limits={limits}
              />
            </>
          )}
        </>
      ) : (
        <>
          <AgentAskForm
            land={land}
            ask={ask}
            placeholder="Ask about this land…"
            modelConfigured={modelConfigured}
            onViewPending={(id) => choose(id)}
          />
          {list.isError && (
            <p className="land-alert" role="alert">
              Earlier questions couldn't be loaded.{" "}
              <button type="button" className="land-link" onClick={() => void list.refetch()}>
                Try again
              </button>
            </p>
          )}
          {(agentList.length > 0 || canEdit) && (
            <details className="land-details">
              <summary>
                {agentList.length ? `Earlier questions · ${agentList.length}` : "Research limits"}
              </summary>
              <div className="land-details__body">
                {agentList.length > 0 && history}
                {canEdit && limits}
              </div>
            </details>
          )}
        </>
      )}
    </GlassPanel>
  );
}

/** The agent's answer, short by default, with its citations as small "source" links. */
function Answer({ text, onEvidence }: { text: string; onEvidence: (id: string) => void }) {
  const [more, setMore] = useState(false);
  const long = text.length > 320;
  return (
    <>
      <p className={clsx("land-agent-answer__text", long && !more && "is-clamped")}>
        {citationParts(text).map((part, index) =>
          part.kind === "source" ? (
            <button
              key={`${index}:${part.evidenceId}`}
              type="button"
              className="land-link"
              onClick={() => onEvidence(part.evidenceId)}
            >
              source
            </button>
          ) : (
            <span key={index}>{part.text}</span>
          ),
        )}
      </p>
      {long && (
        <button type="button" className="land-link" onClick={() => setMore(!more)}>
          {more ? "Less" : "More"}
        </button>
      )}
    </>
  );
}
