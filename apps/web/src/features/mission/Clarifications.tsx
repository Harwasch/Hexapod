import { useEffect, useRef, useState } from "react";

import type { PlanAnswerValue, PlanClarification } from "@twin/contracts";

/** How long a chosen chip stays lit before the next question slides in. */
const ADVANCE_MS = 220;

/**
 * The agent's questions, one at a time: a choice is a row of chips (tapping one moves on), a
 * range is a slider with a Next button, an area choice hands off to the map. Back and Skip
 * step through the set; the last answer, or skipping past the end, completes the round and
 * the console redrafts with everything answered so far.
 */
export function ClarificationStepper({
  items,
  answers,
  onAnswer,
  onArea,
  onComplete,
  disabled,
}: {
  items: PlanClarification[];
  answers: Record<string, PlanAnswerValue>;
  onAnswer: (id: string, value: PlanAnswerValue) => void;
  onArea: (value: string) => void;
  /** The round is over; `extra` is the last answer, not yet in `answers`. */
  onComplete: (extra: Record<string, PlanAnswerValue>) => void;
  disabled?: boolean;
}) {
  const [index, setIndex] = useState(0);
  const [lit, setLit] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  const total = items.length;
  const item = items[index];
  if (!item) return null;
  const value = answers[item.id] ?? item.default ?? null;
  const last = index + 1 >= total;
  const advance = (extra: Record<string, PlanAnswerValue>) => {
    if (last) onComplete(extra);
    else setIndex(index + 1);
  };
  const choose = (option: string) => {
    if (disabled || lit) return;
    onAnswer(item.id, option);
    if (item.kind === "area") onArea(option);
    setLit(option);
    timer.current = setTimeout(() => {
      setLit(null);
      advance({ [item.id]: option });
    }, ADVANCE_MS);
  };
  const next = () => {
    if (disabled) return;
    const current = Number(value ?? item.min ?? 0);
    onAnswer(item.id, current);
    advance({ [item.id]: current });
  };
  return (
    <section className="mc-stepper" data-testid="plan-clarifications" aria-live="polite">
      <div className="mc-stepper__head">
        <span className="mc-stepper__dots" aria-hidden="true">
          {items.map((it, i) => (
            <i
              key={it.id}
              className={i < index ? "is-done" : i === index ? "is-on" : ""}
              data-testid={`clarify-dot-${it.id}`}
            />
          ))}
        </span>
        <span className="mc-mono mc-muted" data-testid="clarify-progress">
          {index + 1} of {total}
        </span>
      </div>
      <div key={item.id} className="mc-stepper__card" data-testid={`clarify-${item.id}`}>
        <div className="mc-stepper__question">{item.question}</div>
        {item.why && <div className="mc-clarify__why">{item.why}</div>}
        {item.kind === "range" ? (
          <div className="mc-stepper__range">
            <label className="mc-clarify__range">
              <input
                type="range"
                min={item.min ?? 0}
                max={item.max ?? 100}
                step={item.step ?? 1}
                value={Number(value ?? item.min ?? 0)}
                onChange={(e) => onAnswer(item.id, Number(e.target.value))}
                disabled={disabled}
                aria-label={item.question}
                data-testid={`clarify-${item.id}-range`}
              />
              <span className="mc-mono mc-clarify__value">
                {Number(value ?? item.min ?? 0)} {item.unit ?? ""}
              </span>
            </label>
            <button
              type="button"
              className="mc-btn mc-btn--sm mc-btn--accent"
              onClick={next}
              disabled={disabled}
              data-testid="clarify-next"
            >
              {last ? "Update the plan" : "Next"}
            </button>
          </div>
        ) : (
          <div className="mc-tags mc-stepper__choices">
            {(item.options ?? []).map((option) => {
              const on = lit ? lit === option.value : String(value) === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  className={`mc-tag ${on ? "is-on" : ""}`}
                  aria-pressed={on}
                  onClick={() => choose(option.value)}
                  disabled={disabled}
                  data-testid={`clarify-${item.id}-${option.value}`}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
        )}
      </div>
      <div className="mc-stepper__nav">
        <button
          type="button"
          className="mc-link"
          onClick={() => setIndex(Math.max(0, index - 1))}
          disabled={index === 0 || disabled}
          data-testid="clarify-back"
        >
          Back
        </button>
        <button
          type="button"
          className="mc-link"
          onClick={() => advance({})}
          disabled={disabled}
          data-testid="clarify-skip"
        >
          {last ? "Skip, I'm done" : "Skip"}
        </button>
      </div>
    </section>
  );
}

/** Short chips for what has already been answered, each removable to ask again. */
export function AnsweredChips({
  answers,
  labels,
  onClear,
}: {
  answers: Record<string, PlanAnswerValue>;
  labels: Record<string, string>;
  onClear: (id: string) => void;
}) {
  const entries = Object.entries(answers);
  if (entries.length === 0) return null;
  return (
    <div className="mc-tags" aria-label="Your answers">
      {entries.map(([id, value]) => (
        <button
          key={id}
          type="button"
          className="mc-tag is-on"
          onClick={() => onClear(id)}
          title="Ask again"
          data-testid={`answered-${id}`}
        >
          {labels[id] ?? id} · {String(value)} <span aria-hidden="true">×</span>
        </button>
      ))}
    </div>
  );
}
