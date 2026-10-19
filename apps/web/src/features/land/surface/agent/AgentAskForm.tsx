import { useEffect, useRef } from "react";
import { ArrowUp, Sparkles, X } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import type { AgentAsk } from "./useAgentAsk";
import { useAgentUi } from "./agentStore";

/**
 * The question box: one line that grows as the person types, the map feature it is about,
 * and what to do when something got in the way (a lost send, a draft this browser couldn't
 * read or save). Enter sends; Shift+Enter starts a new line.
 */
export function AgentAskForm({
  land,
  ask,
  placeholder,
  modelConfigured,
  onSent,
  onViewPending,
}: {
  land: LandArea;
  ask: AgentAsk;
  placeholder: string;
  /** False when this server has no model; undefined while that is still loading. */
  modelConfigured: boolean | undefined;
  onSent?: (investigationId: string) => void;
  onViewPending: (investigationId: string) => void;
}) {
  const input = useRef<HTMLTextAreaElement>(null);
  const focusTick = useAgentUi((state) => state.focusTick);
  useEffect(() => {
    if (!focusTick) return;
    const frame = requestAnimationFrame(() => input.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [focusTick]);
  const send = async () => {
    const id = await ask.ask();
    if (id) onSent?.(id);
  };
  const blocked = Boolean(ask.pending) || Boolean(ask.recoveryError);
  const disabled =
    !ask.canAsk || ask.busy || blocked || !ask.question.trim() || modelConfigured !== true;
  return (
    <>
      {ask.recoveryError && (
        <section className="land-note" aria-label="Unreadable saved question">
          <p>{ask.recoveryError} The saved record is kept until you discard it.</p>
          <div className="land-row">
            <button type="button" className="land-link" onClick={ask.downloadUnreadable}>
              Download saved question
            </button>
            <button type="button" className="land-link" onClick={ask.discardUnreadable}>
              Discard unreadable record
            </button>
          </div>
        </section>
      )}
      {ask.pending && !ask.busy && (
        <section className="land-note" aria-label="Captured research request">
          <p>This question may not have reached the server.</p>
          <blockquote className="land-agent__quote">{ask.pending.question}</blockquote>
          {ask.pending.focus && <p className="land-muted">About {ask.pending.focus.label}</p>}
          <p className="land-muted">
            Retrying checks the same request and starts it only if it wasn't received.
          </p>
          <div className="land-row">
            <GlassButton
              size="sm"
              disabled={!ask.canAsk}
              onClick={() => {
                const pending = ask.pending;
                if (pending)
                  void ask.ask(pending).then((id) => {
                    if (id) onSent?.(id);
                  });
              }}
            >
              Retry captured request
            </GlassButton>
            <button
              type="button"
              className="land-link"
              onClick={() => ask.pending && onViewPending(ask.pending.investigationId)}
            >
              View its investigation
            </button>
            <button type="button" className="land-link" onClick={ask.discardPending}>
              Discard retry record
            </button>
          </div>
        </section>
      )}
      {ask.recovered !== null && (
        <p className="land-muted land-agent__notice">
          Restored from this browser; not sent yet.
          {ask.recovered !== land.revision && " It was written for an earlier outline."}{" "}
          <button type="button" className="land-link" onClick={ask.dismissRecovered}>
            Dismiss recovery notice
          </button>
        </p>
      )}
      {ask.storageError && (
        <p className="land-muted land-agent__notice">
          This browser could not save or restore your question. Keep this tab open to keep it.
        </p>
      )}
      {ask.focus && (
        <p className="land-agent__focus">
          <span>
            About <strong>{ask.focus.label}</strong>
          </span>
          <button
            type="button"
            className="land-icon-button land-icon-button--sm"
            aria-label="Clear question focus"
            onClick={ask.clearFocus}
          >
            <X size={13} aria-hidden="true" />
          </button>
        </p>
      )}
      <form
        className="land-agent-ask"
        onSubmit={(event) => {
          event.preventDefault();
          if (!disabled) void send();
        }}
      >
        <Sparkles className="land-agent-ask__icon" size={15} aria-hidden="true" />
        <label htmlFor={`land-agent-question-${land.id}`} className="land-visually-hidden">
          Ask about this land
        </label>
        <textarea
          id={`land-agent-question-${land.id}`}
          ref={input}
          rows={1}
          maxLength={10000}
          placeholder={placeholder}
          value={ask.question}
          disabled={!ask.canAsk || modelConfigured === false}
          onChange={(event) => ask.setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
            event.preventDefault();
            if (!disabled) void send();
          }}
        />
        <GlassButton
          type="submit"
          size="sm"
          variant="primary"
          iconOnly
          aria-label="Ask the agent"
          disabled={disabled}
          loading={ask.busy}
        >
          <ArrowUp size={15} aria-hidden="true" />
        </GlassButton>
      </form>
      {modelConfigured === false && (
        <p className="land-muted">The agent isn't set up on this server.</p>
      )}
      {!ask.canEdit && modelConfigured !== false && (
        <p className="land-muted">Asking the agent needs edit access to this land.</p>
      )}
      {ask.error && (
        <p className="land-alert" role="alert">
          {ask.error}
        </p>
      )}
    </>
  );
}
