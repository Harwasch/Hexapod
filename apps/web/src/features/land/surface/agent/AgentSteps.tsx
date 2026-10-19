import { useEffect, useRef } from "react";
import { clsx } from "clsx";

import type { LandEvidence } from "@twin/contracts";

import { MapLink } from "../mapLinks";
import { useLeader } from "../mapHooks";
import { locationAnchor, locationGeometry, type AgentStep } from "./agentEvents";

/**
 * The agent's steps, newest last. The step it is on is marked, and a line runs from it to
 * where on the map it is looking; any earlier step with a place points there on hover.
 */
export function AgentSteps({
  steps,
  evidence,
  limit,
  label = "What the agent is doing",
}: {
  steps: AgentStep[];
  evidence: LandEvidence[];
  /** Show only the last few (the phone sheet). */
  limit?: number;
  label?: string;
}) {
  const list = useRef<HTMLOListElement>(null);
  const shown = limit ? steps.slice(-limit) : steps;
  const lastKey = shown.at(-1)?.key;
  useEffect(() => {
    const element = list.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [lastKey]);
  if (!shown.length) return null;
  return (
    <ol ref={list} className="land-steps" aria-label={label}>
      {shown.map((step) => (
        <Step key={step.key} step={step} evidence={evidence} />
      ))}
    </ol>
  );
}

function Step({ step, evidence }: { step: AgentStep; evidence: LandEvidence[] }) {
  const row = useRef<HTMLLIElement>(null);
  const active = step.state === "active";
  const anchor = locationAnchor(step.location, evidence);
  // The step being worked on always points at its place; done steps only while hovered.
  useLeader(row, anchor, "agent", active);
  const geometry = active ? null : locationGeometry(step.location, evidence);
  const body = (
    <>
      <span className="land-step__mark" aria-hidden="true" />
      <span>
        <span>{step.text}</span>
        {(step.source ?? step.note) && (
          <span className="land-step__source">
            {[step.source, step.note].filter(Boolean).join(" · ")}
          </span>
        )}
      </span>
    </>
  );
  const className = clsx(
    "land-step",
    active && "is-active",
    step.state === "finding" && "is-finding",
  );
  if (geometry && anchor)
    return (
      <MapLink
        as="li"
        className={className}
        tone="agent"
        target={{ key: `agent-step:${step.key}`, geometry, anchor }}
        label={`${step.text}: show on map`}
      >
        {body}
      </MapLink>
    );
  return (
    <li ref={row} className={className} aria-current={active ? "step" : undefined}>
      {body}
    </li>
  );
}
