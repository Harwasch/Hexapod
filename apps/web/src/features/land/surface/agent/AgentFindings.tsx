import { useEffect, useRef } from "react";
import { clsx } from "clsx";
import { BookOpen, Pin, X } from "lucide-react";

import type { LandEvidence, LandFinding } from "@twin/contracts";

import { useFlyTo } from "../mapHooks";
import { MapLink } from "../mapLinks";
import { findingPlace, type EventLocation } from "./agentEvents";
import { useAgentUi } from "./agentStore";

const CONFIDENCE: Record<LandFinding["confidence"], string> = {
  measured: "Measured",
  supported: "Backed by sources",
  inferred: "Inferred",
  uncertain: "Uncertain",
};

/**
 * What the agent found, each tied to its place: hovering one outlines it and draws a line to
 * it, clicking flies there. Pin keeps a finding at the top; dismiss hides it.
 */
export function AgentFindings({
  findings,
  places,
  evidence,
  canEdit,
  onEvidence,
  onDisposition,
}: {
  findings: LandFinding[];
  places: Map<string, EventLocation>;
  evidence: LandEvidence[];
  canEdit: boolean;
  onEvidence: (id: string) => void;
  onDisposition: (finding: LandFinding, value: LandFinding["disposition"]) => void;
}) {
  const highlight = useAgentUi((state) => state.highlight);
  const list = useRef<HTMLUListElement>(null);
  useEffect(() => {
    if (!highlight) return;
    list.current
      ?.querySelector(`[data-finding="${CSS.escape(highlight)}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [highlight]);
  if (!findings.length) return null;
  const ordered = [...findings].sort(
    (a, b) => Number(b.disposition === "pinned") - Number(a.disposition === "pinned"),
  );
  return (
    <ul ref={list} className="land-agent-findings" aria-label="What the agent found">
      {ordered.map((finding) => {
        const place = findingPlace(finding, places, evidence);
        return (
          <MapLink
            key={finding.id}
            as="li"
            fly={false}
            tone="agent"
            target={place ? { key: `agent-finding:${finding.id}`, ...place } : null}
            className={clsx(
              "land-agent-finding",
              finding.disposition === "pinned" && "is-pinned",
              finding.disposition === "dismissed" && "is-dismissed",
              highlight === finding.id && "is-highlighted",
            )}
          >
            <div data-finding={finding.id} className="land-agent-finding__body">
              <FindingTitle finding={finding} place={place} />
              <p className="land-agent-finding__summary">{finding.summary}</p>
              <div className="land-agent-finding__meta">
                <span>{CONFIDENCE[finding.confidence]}</span>
                <button
                  type="button"
                  className="land-link"
                  onClick={() => finding.evidenceIds[0] && onEvidence(finding.evidenceIds[0])}
                >
                  <BookOpen size={12} aria-hidden="true" /> Sources ({finding.evidenceIds.length})
                </button>
                {canEdit && (
                  <>
                    <button
                      type="button"
                      className="land-icon-button land-icon-button--sm"
                      aria-pressed={finding.disposition === "pinned"}
                      aria-label={finding.disposition === "pinned" ? "Unpin" : "Pin"}
                      title={finding.disposition === "pinned" ? "Unpin" : "Pin"}
                      onClick={() =>
                        onDisposition(
                          finding,
                          finding.disposition === "pinned" ? "visible" : "pinned",
                        )
                      }
                    >
                      <Pin size={13} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="land-icon-button land-icon-button--sm"
                      aria-label={finding.disposition === "dismissed" ? "Restore" : "Dismiss"}
                      title={finding.disposition === "dismissed" ? "Restore" : "Dismiss"}
                      onClick={() =>
                        onDisposition(
                          finding,
                          finding.disposition === "dismissed" ? "visible" : "dismissed",
                        )
                      }
                    >
                      <X size={13} aria-hidden="true" />
                    </button>
                  </>
                )}
              </div>
              {finding.evidenceIds.length > 1 && (
                <details className="land-agent-finding__more">
                  <summary>All sources and how sure it is</summary>
                  <p className="land-muted">{finding.uncertainty}</p>
                  <div className="land-row">
                    {finding.evidenceIds.map((id, index) => (
                      <button
                        key={id}
                        type="button"
                        className="land-link"
                        onClick={() => onEvidence(id)}
                      >
                        Source {index + 1}
                      </button>
                    ))}
                  </div>
                </details>
              )}
              {finding.evidenceIds.length <= 1 && (
                <details className="land-agent-finding__more">
                  <summary>How sure it is</summary>
                  <p className="land-muted">{finding.uncertainty}</p>
                </details>
              )}
            </div>
          </MapLink>
        );
      })}
    </ul>
  );
}

/** The finding's title; when it has a place, a button that flies there. */
function FindingTitle({
  finding,
  place,
}: {
  finding: LandFinding;
  place: ReturnType<typeof findingPlace>;
}) {
  const fly = useFlyTo();
  if (!place) return <span className="land-agent-finding__title">{finding.title}</span>;
  return (
    <button
      type="button"
      className="land-agent-finding__title land-agent-finding__title--place"
      aria-label={`${finding.title}: show on map`}
      onClick={() => fly(place.geometry)}
    >
      {finding.title}
      <span className="land-fact__go" aria-hidden="true">
        on map
      </span>
    </button>
  );
}
