import { useState, type ReactNode } from "react";

import type {
  Investigation,
  InvestigationDetail,
  LandEvidence,
  LandFinding,
} from "@twin/contracts";

import { ResearchArtifactView } from "../../ResearchArtifacts";
import type { QuestionBudget } from "../../researchDraft";
import { citationParts, type AgentStep, type EventLocation } from "./agentEvents";
import { AgentFindings } from "./AgentFindings";
import { AgentSteps } from "./AgentSteps";
import { useAgentUi } from "./agentStore";

/** A fold whose contents mount only while it is open (maps and rasters fetch when shown). */
export function Fold({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details
      className="land-agent-fold"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>{summary}</summary>
      {open && <div className="land-agent-fold__body">{children}</div>}
    </details>
  );
}

export function EarlierQuestions({
  items,
  current,
  revision,
  onChoose,
}: {
  items: Investigation[];
  current: string | null;
  revision: number;
  onChoose: (id: string) => void;
}) {
  const others = items.filter((item) => item.id !== current);
  const [shown, setShown] = useState(8);
  if (!others.length) return <p className="land-muted">No earlier questions yet.</p>;
  return (
    <>
      <ul className="land-agent-history" aria-label="Earlier questions">
        {others.slice(0, shown).map((item) => (
          <li key={item.id}>
            <button type="button" onClick={() => onChoose(item.id)}>
              <span>{item.question}</span>
              <small>
                {new Date(item.createdAt).toLocaleDateString()}
                {item.boundaryRevision !== revision ? " · earlier outline" : ""}
              </small>
            </button>
          </li>
        ))}
      </ul>
      {others.length > shown && (
        <button type="button" className="land-link" onClick={() => setShown(shown + 8)}>
          Show more
        </button>
      )}
    </>
  );
}

export function ResearchLimits({
  budget,
  onChange,
}: {
  budget: QuestionBudget;
  onChange: (budget: QuestionBudget) => void;
}) {
  return (
    <div className="land-agent-limits">
      <label className="land-field">
        <span>Time budget</span>
        <select
          value={budget.maxSeconds}
          onChange={(event) => onChange({ ...budget, maxSeconds: Number(event.target.value) })}
        >
          <option value={60}>1 minute</option>
          <option value={180}>3 minutes</option>
          <option value={600}>10 minutes</option>
        </select>
      </label>
      <label className="land-field">
        <span>Public web searches</span>
        <select
          value={budget.maxWebSearches}
          onChange={(event) => onChange({ ...budget, maxWebSearches: Number(event.target.value) })}
        >
          <option value={0}>Registered data sources only</option>
          <option value={6}>Up to 6 searches</option>
          <option value={12}>Up to 12 searches</option>
        </select>
      </label>
      <p className="land-muted">
        Web search can find leads outside the registered datasets. Each source still needs a
        location match and reuse rights.
      </p>
    </div>
  );
}

/**
 * Everything behind the answer: each step it took, what it made (maps, charts, tables, map
 * layers, photo sets), the conversation with its citations, earlier findings and questions,
 * and the research limits.
 */
export function AgentDetails({
  detail,
  steps,
  runId,
  places,
  canEdit,
  onEvidence,
  onDisposition,
  offset,
  onOffset,
  history,
  limits,
}: {
  detail: InvestigationDetail;
  steps: AgentStep[];
  runId: string | null;
  places: Map<string, EventLocation>;
  canEdit: boolean;
  onEvidence: (id: string) => void;
  onDisposition: (finding: LandFinding, value: LandFinding["disposition"]) => void;
  offset: number;
  onOffset: (offset: number) => void;
  history: ReactNode;
  limits: ReactNode;
}) {
  const [showDismissed, setShowDismissed] = useState(false);
  const evidence: LandEvidence[] = detail.evidence;
  const earlier = detail.findings.filter(
    (finding) => finding.runId !== runId && (showDismissed || finding.disposition !== "dismissed"),
  );
  const dismissed = detail.findings.filter(
    (finding) => finding.runId === runId && finding.disposition === "dismissed",
  );
  const total = Math.max(0, ...Object.values(detail.page.totals));
  return (
    <details className="land-details">
      <summary>Details and sources</summary>
      <div className="land-details__body">
        {steps.length > 0 && (
          <Fold summary={`Every step · ${steps.length}`}>
            <AgentSteps steps={steps} evidence={evidence} label="Every step the agent took" />
          </Fold>
        )}
        {detail.artifacts.length > 0 && (
          <Fold summary={`What it made · ${detail.artifacts.length}`}>
            {detail.artifacts.map((artifact) => (
              <div key={artifact.id} className="land-agent-artifact">
                <ResearchArtifactView
                  artifact={artifact}
                  evidence={evidence}
                  onEvidence={onEvidence}
                  onAsk={() => useAgentUi.getState().focusInput()}
                />
                {artifact.evidenceIds.slice(0, 1).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="land-link"
                    onClick={() => onEvidence(id)}
                  >
                    Inspect source evidence
                  </button>
                ))}
              </div>
            ))}
          </Fold>
        )}
        {detail.messages.length > 0 && (
          <Fold summary="Conversation">
            <div className="land-agent-conversation">
              {detail.messages.map((message) => (
                <article key={message.id} data-role={message.role}>
                  <span className="land-fact__label">
                    {message.role === "user" ? "You" : "Agent"}
                  </span>
                  <p>
                    {citationParts(message.content).map((part, index) =>
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
                </article>
              ))}
            </div>
          </Fold>
        )}
        {(earlier.length > 0 || dismissed.length > 0) && (
          <Fold summary="Earlier and dismissed findings">
            <label className="land-row land-muted">
              <input
                type="checkbox"
                checked={showDismissed}
                onChange={(event) => setShowDismissed(event.target.checked)}
              />
              Show dismissed findings
            </label>
            <AgentFindings
              findings={[...(showDismissed ? dismissed : []), ...earlier]}
              places={places}
              evidence={evidence}
              canEdit={canEdit}
              onEvidence={onEvidence}
              onDisposition={onDisposition}
            />
          </Fold>
        )}
        {total > detail.page.limit && (
          <div className="land-row">
            <button
              type="button"
              className="land-link"
              disabled={offset === 0}
              onClick={() => onOffset(Math.max(0, offset - detail.page.limit))}
            >
              Earlier results
            </button>
            <span className="land-muted">Page {offset / detail.page.limit + 1}</span>
            <button
              type="button"
              className="land-link"
              disabled={offset + detail.page.limit >= total}
              onClick={() => onOffset(offset + detail.page.limit)}
            >
              More results
            </button>
          </div>
        )}
        <Fold summary="Earlier questions">{history}</Fold>
        <Fold summary="Research limits">{limits}</Fold>
      </div>
    </details>
  );
}
