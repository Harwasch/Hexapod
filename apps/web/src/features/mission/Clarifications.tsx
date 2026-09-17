import type { PlanAnswerValue, PlanClarification } from "@twin/contracts";

/**
 * The agent's questions, answered with chips and sliders rather than prose: a choice is a
 * row of options, a range is a slider with its value, an area choice hands off to the map.
 */
export function Clarifications({
  items,
  answers,
  onAnswer,
  onArea,
  disabled,
}: {
  items: PlanClarification[];
  answers: Record<string, PlanAnswerValue>;
  onAnswer: (id: string, value: PlanAnswerValue) => void;
  onArea: (value: string) => void;
  disabled?: boolean;
}) {
  if (items.length === 0) return null;
  return (
    <div className="mc-clarify" data-testid="plan-clarifications">
      {items.map((item) => {
        const value = answers[item.id] ?? item.default ?? null;
        return (
          <section key={item.id} className="mc-clarify__item" data-testid={`clarify-${item.id}`}>
            <div className="mc-clarify__question">{item.question}</div>
            {item.why && <div className="mc-clarify__why">{item.why}</div>}
            {item.kind === "range" ? (
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
            ) : (
              <div className="mc-tags">
                {(item.options ?? []).map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`mc-tag ${String(value) === option.value ? "is-on" : ""}`}
                    aria-pressed={String(value) === option.value}
                    onClick={() => {
                      onAnswer(item.id, option.value);
                      if (item.kind === "area") onArea(option.value);
                    }}
                    disabled={disabled}
                    data-testid={`clarify-${item.id}-${option.value}`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
          </section>
        );
      })}
    </div>
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
